/**
 * Gaze-Head Steering — in-browser demo for the project page.
 *
 * Runs Qwen3-VL-2B (ONNX, q4f16) fully client-side on WebGPU via
 * transformers.js. The decoder graph (baulab/Qwen3-VL-2B-Instruct-GazeHeads-ONNX)
 * carries two extra inputs that implement the paper's intervention:
 *   gaze_head_mask (28,16)        — selects the gaze heads
 *   gaze_sign (total_seq_len,)    — ±δ additive attention bias per KV position
 * Hovering a comic panel updates the bias between decode steps, so the
 * narration re-steers live, mid-generation (the paper's dynamic gaze switching).
 *
 * Loaded lazily by index.html when the visitor clicks "Try the demo";
 * `start(container)` builds the UI and immediately begins the model load.
 * Vision features are precomputed per comic (comics/*.embeds.bin) — the
 * vision tower never runs in the browser. The prompt KV is prefilled once
 * per comic and snapshotted, so each Generate streams its first word in
 * well under a second.
 */
import {
  AutoModelForImageTextToText,
  AutoProcessor,
  Tensor,
  TextStreamer,
  env,
  load_image,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js';

const MODEL_ID = 'baulab/Qwen3-VL-2B-Instruct-GazeHeads-ONNX';
const PROMPT = 'Tell what happens in this comic strip as one smooth, flowing story, '
  + 'like a storyteller reading aloud. Never mention the comic, panels, scenes, '
  + 'frames, images, or their order — no phrases like "in the first scene", '
  + '"the next panel", or "the scene shifts". Just tell the story itself, '
  + 'in plain prose paragraphs.';
const MAX_NEW_TOKENS = 500;
// Reading pace: minimum ms between decode steps. Throttles the decode loop
// itself (NOT a display buffer), so hover re-steering still shows up within
// one step — the text on screen is always the newest token.
const PACE_MS = 125;
const MIN_NEW_TOKENS = 400;
const TOP_K = 10;
const DELTA = 10000.0;
const N_LAYERS = 28, N_HEADS = 16;

const COMICS = [
  { name: 'mixed2', label: 'Variety of objects' },
  { name: 'robotour', label: 'Tiny robot world tour' },
  { name: 'balloonride', label: 'Red balloon journey' },
  { name: 'comic4', label: "Cat's treasure hunt" },
  { name: 'comic217', label: 'Dragon through the seasons' },
  { name: 'comic110', label: 'Dog earns cookies' },
  { name: 'comic281', label: 'Scientist at work' },
];

const asset = (p) => new URL(p, import.meta.url).href;

// ---------------------------------------------------------------------------
// Steering controller (per-step feeds for the modified decoder graph)
// ---------------------------------------------------------------------------
class GazeController {
  constructor() {
    this.headMask = new Float32Array(N_LAYERS * N_HEADS);
    this.boost = []; this.suppress = [];
    this.stats = { calls: 0, steered: 0 };
    this.paceMs = 0; this._lastStep = 0;
  }
  setHeads(heads) {
    this.headMask.fill(0);
    for (const { layer, head } of heads) this.headMask[layer * N_HEADS + head] = 1.0;
  }
  setTarget({ boost = [], suppress = [] }) { this.boost = boost; this.suppress = suppress; }
  clear() { this.boost = []; this.suppress = []; }
  get active() { return this.boost.length > 0 || this.suppress.length > 0; }

  attach(session) {
    const hidden = ['gaze_head_mask', 'gaze_sign'];
    const realNames = [...session.inputNames];
    if (!realNames.includes('gaze_sign')) throw new Error('decoder has no gaze inputs');
    Object.defineProperty(session, 'inputNames', {
      get: () => realNames.filter((n) => !hidden.includes(n)), configurable: true,
    });
    const origRun = session.run.bind(session);
    const ctl = this;
    session.run = async function (feeds, ...rest) {
      const am = feeds.attention_mask;
      const totalLen = Number(am.dims[am.dims.length - 1]);
      const seqLen = feeds.inputs_embeds ? Number(feeds.inputs_embeds.dims[1]) : 1;
      if (seqLen === 1 && ctl.paceMs > 0) {
        const wait = ctl._lastStep + ctl.paceMs - performance.now();
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
        ctl._lastStep = performance.now();
      }
      return origRun({ ...feeds, ...ctl.buildFeeds(totalLen, seqLen, am.constructor) }, ...rest);
    };
  }

  buildFeeds(totalLen, seqLen, TensorCtor) {
    const steer = this.active && seqLen === 1; // decode-only (paper's main setting)
    this.stats.calls += 1;
    if (steer) this.stats.steered += 1;
    const sign = new Float32Array(totalLen);
    const mask = steer ? this.headMask : new Float32Array(N_LAYERS * N_HEADS);
    if (steer) {
      for (const p of this.boost) if (p < totalLen) sign[p] = +DELTA;
      for (const p of this.suppress) if (p < totalLen) sign[p] = -DELTA;
    }
    return {
      gaze_head_mask: new TensorCtor('float16', f32ToF16Bits(mask), [N_LAYERS, N_HEADS]),
      gaze_sign: new TensorCtor('float16', f32ToF16Bits(sign), [totalLen]),
    };
  }
}

function f32ToF16Bits(f32) {
  const out = new Uint16Array(f32.length);
  const buf = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < f32.length; i++) {
    buf.setFloat32(0, f32[i]);
    const x = buf.getUint32(0);
    const sign = (x >>> 16) & 0x8000;
    const e = ((x >>> 23) & 0xff) - 127 + 15;
    const frac = x & 0x7fffff;
    out[i] = e >= 0x1f ? (sign | 0x7c00) : e <= 0 ? sign : (sign | (e << 10) | (frac >> 13));
  }
  return out;
}

function bboxToTokenPositions([x0, y0, x1, y1], { t, h, w }, [W, H], imgStart) {
  const positions = [];
  for (let ti = 0; ti < t; ti++) {
    for (let r = 0; r < h; r++) {
      const cy = (r + 0.5) * H / h;
      for (let c = 0; c < w; c++) {
        const cx = (c + 0.5) * W / w;
        if (x0 <= cx && cx <= x1 && y0 <= cy && cy <= y1) {
          positions.push(imgStart + ti * h * w + r * w + c);
        }
      }
    }
  }
  return positions;
}

// ---------------------------------------------------------------------------
// UI template + styles (scoped under .gazedemo)
// ---------------------------------------------------------------------------
const STYLES = `
.gazedemo { text-align: left; }
.gazedemo .gd-strip-wrap { position: relative; border: 1px solid #ccc;
  border-radius: 6px; overflow: hidden; user-select: none;
  /* magnifying-glass cursor: "inspect this panel" (hotspot = lens center).
     24px so it survives browser zoom (Chrome drops cursors scaled past 32px). */
  cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24'%3E%3Ccircle cx='10' cy='10' r='6' fill='white' fill-opacity='.3' stroke='%231d7a3a' stroke-width='2'/%3E%3Cline x1='14.5' y1='14.5' x2='21' y2='21' stroke='%23222' stroke-width='3' stroke-linecap='round'/%3E%3Cline x1='6.5' y1='7.5' x2='8.5' y2='5.5' stroke='white' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E") 10 10, crosshair; }
.gazedemo .gd-strip { display: block; width: 100%; }
.gazedemo .gd-hl { position: absolute; inset: 0; display: none;
  /* circular spotlight (UI only): JS paints a radial gradient that dims the
     strip except a round window following the cursor. The steered region is
     the bounding box of this circle (see applyTarget). */
  pointer-events: none; }
.gazedemo .gd-controls { margin: 12px 0; display: flex; gap: 10px; align-items: center;
  flex-wrap: wrap; }
.gazedemo button { padding: 5px 14px; font-size: 14px; border-radius: 6px;
  border: 1px solid #888; background: #f5f5f5; cursor: pointer; }
.gazedemo button:disabled { opacity: .45; cursor: default; }
.gazedemo .gd-gen { background: #1d7a3a; border-color: #1d7a3a; color: #fff; }
.gazedemo .gd-stop { background: #a33333; border-color: #a33333; color: #fff; }
.gazedemo .gd-badge { font-size: 13px; padding: 3px 10px; border-radius: 99px;
  background: #eee; border: 1px solid #ccc; }
.gazedemo .gd-out { border: 1px solid #ddd; border-radius: 6px; padding: 14px;
  min-height: 100px; max-height: 380px; overflow-y: auto; font-size: 18px;
  line-height: 1.6; white-space: pre-wrap; background: #fff; color: #000; }
.gazedemo .gd-out span.p0 { background: #ffe6e6; } .gazedemo .gd-out span.p1 { background: #fff2cc; }
.gazedemo .gd-out span.p2 { background: #e6ffe6; } .gazedemo .gd-out span.p3 { background: #e6f2ff; }
.gazedemo .gd-out span.p4 { background: #f2e6ff; } .gazedemo .gd-out span.p5 { background: #ffe6f7; }
.gazedemo .gd-status { color: #555; font-size: 13px; min-height: 18px; margin: 8px 0; }
.gazedemo .gd-perf { font-size: 12px; color: #777; margin-top: 6px; }
.gazedemo .gd-hint { font-size: 13px; color: #555; margin-top: 6px; }
.gazedemo .gd-hint .gd-key { font-weight: 700; font-size: 1.35em; color: #1a4f9c; }
.gazedemo .gd-fact { font-size: 13px; color: #444; margin: 0 0 6px; }
.gazedemo.loading .gd-status, .gazedemo.loading .gd-hint,
.gazedemo.loading .gd-fact,
.gazedemo.loading .gd-controls, .gazedemo.loading .gd-out,
.gazedemo.loading .gd-perf { display: none; }
.gazedemo .gd-loading { position: absolute; inset: 0; display: flex;
  align-items: center; justify-content: center;
  background: rgba(255,255,255,.55); }
.gazedemo .gd-loading-card { background: #fff; border: 1px solid #ccc;
  border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,.18);
  padding: 14px 22px; text-align: center; max-width: 86%; }
.gazedemo .gd-loading-msg { font-size: 14px; color: #333; margin-bottom: 8px; }
.gazedemo .gd-progress progress { width: min(380px, 60vw); height: 14px;
  vertical-align: middle; accent-color: #1d7a3a; }
.gazedemo .gd-progress-label { font-size: 12px; color: #555; margin-left: 8px; }
`;

const TEMPLATE = `
<div class="gazedemo loading">
  <div class="gd-status">Starting…</div>
  <div class="gd-fact"><b>A single image with a 6-panel layout, provided to
    the model as one image.</b> Hovering redirects just <b>10 attention
    heads</b>. Notice how the model still stitches a smooth story even when
    you change its gaze abruptly: we steer <b>where it looks</b>, not how it
    writes.</div>
  <div class="gd-strip-wrap">
    <img class="gd-strip" alt="comic strip" />
    <div class="gd-hl"></div>
    <div class="gd-loading">
      <div class="gd-loading-card">
        <div class="gd-loading-msg">Preparing the demo&hellip;</div>
        <div class="gd-progress"><progress max="100"></progress><span class="gd-progress-label"></span></div>
      </div>
    </div>
  </div>
  <div class="gd-hint"><span class="gd-key">Hover</span> the strip and the model
    writes about whatever is under your spotlight. Move to re-steer mid-sentence,
    <span class="gd-key">scroll</span> to resize the spotlight, and move off to stop.
    Text is tinted by the region steering it.</div>
  <div class="gd-controls">
    <select class="gd-comic" disabled></select>
    <button class="gd-gen" disabled>Generate without steering</button>
    <button class="gd-stop" disabled>Stop</button>
    <span class="gd-badge">target: —</span>
  </div>
  <div class="gd-out"></div>
  <div class="gd-perf"></div>
</div>`;

// ---------------------------------------------------------------------------
export async function start(container, opts = {}) {
  if (!document.getElementById('gazedemo-styles')) {
    const st = document.createElement('style');
    st.id = 'gazedemo-styles';
    st.textContent = STYLES;
    document.head.appendChild(st);
  }
  const mount = document.createElement('div');
  mount.innerHTML = TEMPLATE;
  container.appendChild(mount);
  const $ = (cls) => mount.querySelector('.' + cls);
  const status = (t) => { $('gd-status').textContent = t; };
  const modelId = opts.modelId || MODEL_ID; // recording pages can point at the all-layers build
  // Progress overlay over the strip — used for the initial model load AND for
  // every per-comic preparation (embeds fetch + tokenize + KV prefill).
  const showPrep = (initialMsg) => {
    let ov = mount.querySelector('.gd-loading');
    if (!ov) {
      ov = document.createElement('div');
      ov.className = 'gd-loading';
      ov.innerHTML = '<div class="gd-loading-card"><div class="gd-loading-msg"></div>'
        + '<div class="gd-progress"><progress max="100"></progress>'
        + '<span class="gd-progress-label"></span></div></div>';
      mount.querySelector('.gd-strip-wrap').appendChild(ov);
    }
    ov.querySelector('progress')?.removeAttribute('value'); // indeterminate pulse
    const m = ov.querySelector('.gd-loading-msg');
    if (m) m.textContent = initialMsg;
  };
  const hidePrep = () => { mount.querySelector('.gd-loading')?.remove(); };
  // During the initial load an overlay card covers the strip; route progress
  // messages there while it exists, to the status line afterwards.
  const msg = (t) => { const m = mount.querySelector('.gd-loading-msg'); if (m) m.textContent = t; else status(t); };

  // Show the real layout immediately: the comic renders right away and the
  // model download happens behind the overlay instead of a broken shell.
  $('gd-strip').src = asset(`comics/${COMICS[0].name}.png`);
  $('gd-comic').innerHTML = COMICS.map((c) => `<option value="${c.name}">${c.label}</option>`).join('');

  const state = {
    processor: null, model: null, gaze: new GazeController(), config: null,
    inputs: null, promptLen: 0, grid: null, imgStart: 0, allImage: null,
    meta: null, image: null, embeds: null, embedsDims: null,
    kvSnapshot: null, CacheCtor: null,
    generating: false, stopFlag: false, currentPanel: -1,
    hovering: false, spot: { x: 0, y: 0 }, radiusPx: 48,
    ready: false, hoverArmed: true, hoverTimer: null, leaveTimer: null,
    ranking: null, steerLayers: [], headMode: 'gaze',
  };

  // ---- model load (starts immediately) -------------------------------------
  try {
    msg('Preparing the demo…');
    state.processor = await AutoProcessor.from_pretrained(modelId);
    state.config = await (await fetch(
      `https://huggingface.co/${modelId}/resolve/main/config.json`)).json();

    const dlFiles = new Map();
    const bar = $('gd-progress');
    const updateBar = () => {
      let loaded = 0, total = 0;
      for (const f of dlFiles.values()) { loaded += f.loaded ?? 0; total += f.total ?? 0; }
      if (!total) return;
      bar.querySelector('progress').value = loaded / total * 100;
      bar.querySelector('.gd-progress-label').textContent =
        `${(loaded / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB`;
      msg('Downloading the model — one-time, cached for your next visit');
    };
    state.model = await AutoModelForImageTextToText.from_pretrained(modelId, {
      dtype: { embed_tokens: 'q4f16', vision_encoder: 'q4f16', decoder_model_merged: 'q4f16' },
      device: 'webgpu',
      progress_callback: (p) => {
        if (p.status === 'progress' && p.file && p.total) {
          dlFiles.set(p.file, p);
          updateBar();
        }
      },
    });
    // Download done — switch the bar to indeterminate for the warm-up phase.
    bar.querySelector('progress').removeAttribute('value');
    bar.querySelector('.gd-progress-label').textContent = '';
    msg('Warming up the model…');

    const decoder = Object.values(state.model.sessions).find(
      (s) => s.inputNames?.includes?.('gaze_sign'));
    if (!decoder) throw new Error('gaze inputs missing from decoder');
    state.gaze.attach(decoder);
    state.gaze.paceMs = PACE_MS;
    state.ranking = await (await fetch(asset('gaze_head_ranking_qwen3vl_2b.json'))).json();
    // Which layers actually honor the steering bias. The default (site) graph
    // only wired the 7 gaze-hosting layers; the all-layers build (opts.allLayers,
    // used by the recording control demos) wires every layer, so all 448 heads
    // are steerable and "random" can draw from anywhere in the network.
    state.steerLayers = opts.allLayers
      ? Array.from({ length: N_LAYERS }, (_, i) => i)
      : [...new Set(state.ranking.slice(0, 20).map((e) => e.layer))];
    applyHeadMode(opts.headSet || 'gaze');
    if (opts.headModes) buildHeadModeSelect();

    // Precomputed image features instead of the vision tower.
    state.model.encode_image = async () =>
      new Tensor('float32', state.embeds, state.embedsDims);

    const sel = $('gd-comic'); // options were populated at mount
    sel.onchange = async () => {
      if (state.generating) state.stopFlag = true;
      sel.disabled = true; $('gd-gen').disabled = true;
      await loadComic(sel.value);
      sel.disabled = false; $('gd-gen').disabled = false;
    };

    await loadComic(COMICS[0].name);
    // Everything is ready: drop the overlay and reveal the controls.
    mount.querySelector('.gd-loading')?.remove();
    mount.querySelector('.gazedemo').classList.remove('loading');
    sel.disabled = false;
    $('gd-gen').disabled = false;
  } catch (err) {
    $('gd-progress')?.remove();
    msg('The demo failed to start: ' + err.message);
    const m = mount.querySelector('.gd-loading-msg');
    if (m) m.style.color = '#a33333';
    console.error(err);
    return;
  }

  // ---- per-comic preparation ------------------------------------------------
  async function loadComic(name) {
    state.ready = false;
    showPrep('Preparing the comic…');
    state.meta = await (await fetch(asset(`comics/${name}.json`))).json();
    const buf = await (await fetch(asset(`comics/${state.meta.embeds.file}`))).arrayBuffer();
    state.embeds = new Float32Array(buf);
    state.embedsDims = state.meta.embeds.dims;
    $('gd-strip').src = asset(`comics/${name}.png`);
    state.image = await load_image(asset(`comics/${name}.png`));

    const messages = [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: PROMPT }] }];
    const text = state.processor.apply_chat_template(messages, { add_generation_prompt: true });
    state.inputs = await state.processor(text, state.image);
    state.promptLen = state.inputs.input_ids.dims[1];

    const ids = state.inputs.input_ids.data;
    const imageId = BigInt(state.config.image_token_id);
    let imgStart = -1, imgEnd = -1;
    for (let i = 0; i < ids.length; i++) {
      if (ids[i] === imageId) { if (imgStart < 0) imgStart = i; imgEnd = i + 1; }
    }
    const [t, gh, gw] = Array.from(state.inputs.image_grid_thw.data, Number);
    const merge = state.config.vision_config.spatial_merge_size ?? 2;
    state.grid = { t, h: gh / merge, w: gw / merge };
    state.imgStart = imgStart;
    state.allImage = new Set(Array.from({ length: imgEnd - imgStart }, (_, i) => imgStart + i));
    state.hovering = false;
    clearTarget();
    $('gd-out').textContent = '';
    $('gd-perf').textContent = '';

    msg('Warming up the model (one-time per comic)…');
    await snapshotPromptKV();
    state.ready = true;
    state.hoverArmed = true;
    hidePrep();
    status('ready — hover a comic panel and the model starts writing');
  }

  // Prefill prompt[:-1] once; each Generate resumes from CPU copies of the KV.
  async function snapshotPromptKV() {
    state.kvSnapshot = null;
    const N = state.inputs.input_ids.dims[1];
    const res = await state.model.generate({
      input_ids: state.inputs.input_ids.slice(null, [0, N - 1]),
      attention_mask: state.inputs.attention_mask.slice(null, [0, N - 1]),
      pixel_values: state.inputs.pixel_values,
      image_grid_thw: state.inputs.image_grid_thw,
      max_new_tokens: 1, do_sample: false, return_dict_in_generate: true,
    });
    const cache = res.past_key_values;
    state.CacheCtor = cache.constructor;
    const snap = {};
    for (const [k, t] of Object.entries(cache)) {
      if (!t?.dims) continue;
      let data;
      if (t.location === 'gpu-buffer') {
        const getData = t.getData?.bind(t) ?? t.ort_tensor?.getData?.bind(t.ort_tensor);
        data = (await getData()).slice(0);
        try { t.dispose?.(); } catch { /* released */ }
      } else {
        data = t.data.slice(0);
      }
      snap[k] = { type: t.type, data, dims: [...t.dims] };
    }
    state.kvSnapshot = snap;
  }

  function buildPastFromSnapshot() {
    const past = new state.CacheCtor();
    for (const [k, rec] of Object.entries(state.kvSnapshot)) {
      past[k] = new Tensor(rec.type, rec.data.slice(0), rec.dims);
    }
    return past;
  }

  // ---- hover-to-steer (continuous circular spotlight) ------------------------
  // Visual: a round "torch" follows the cursor (UI only). Backend: the bounding
  // box of that circle is fed to the same bboxToTokenPositions, so the steered
  // region glides continuously with the cursor instead of snapping to panels.
  function renderSpot() {
    const hl = $('gd-hl');
    if (!state.hovering) { hl.style.display = 'none'; return; }
    const { x, y } = state.spot, r = state.radiusPx, feather = 20;
    hl.style.display = 'block';
    hl.style.background = `radial-gradient(circle at ${x}px ${y}px, `
      + `rgba(0,0,0,0) 0px, rgba(0,0,0,0) ${r}px, rgba(0,0,0,0.6) ${r + feather}px)`;
  }
  function clearTarget() {
    state.currentPanel = -1;
    state.gaze.clear();
    $('gd-badge').textContent = 'target: —';
  }
  function applyTarget() {
    if (!state.meta) return;
    const rect = $('gd-strip').getBoundingClientRect();
    const W = state.meta.width, H = state.meta.height;
    const ix = state.spot.x / rect.width * W;       // cursor in image px
    const iy = state.spot.y / rect.height * H;
    const rImg = state.radiusPx / rect.width * W;    // radius in image px (uniform scale)
    const boost = bboxToTokenPositions(
      [ix - rImg, iy - rImg, ix + rImg, iy + rImg], state.grid, [W, H], state.imgStart);
    // tint colour: whichever panel the spotlight centre sits over
    const b = state.meta.panel_boundaries_px;
    let p = -1;
    for (let i = 0; i < b.length - 1; i++) if (ix >= b[i] && ix < b[i + 1]) { p = i; break; }
    state.currentPanel = p;
    if (!boost.length) { state.gaze.clear(); $('gd-badge').textContent = 'spotlight too small'; return; }
    const boostSet = new Set(boost);
    state.gaze.setTarget({ boost, suppress: [...state.allImage].filter((pos) => !boostSet.has(pos)) });
    $('gd-badge').textContent = `steering ${boost.length} image tokens`;
  }

  $('gd-strip-wrap').addEventListener('mousemove', (e) => {
    if (!state.meta) return;
    const rect = $('gd-strip').getBoundingClientRect();
    state.spot = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    state.hovering = true;
    renderSpot();
    applyTarget();
    if (state.leaveTimer) { clearTimeout(state.leaveTimer); state.leaveTimer = null; }
    // Entering the strip starts a run (small delay = hover intent).
    if (state.ready && state.hoverArmed && !state.generating && !state.hoverTimer) {
      state.hoverTimer = setTimeout(() => {
        state.hoverTimer = null;
        if (state.ready && state.hoverArmed && !state.generating && state.hovering) {
          state.hoverArmed = false;
          runGeneration();
        }
      }, 150);
    }
  });
  // Mouse wheel resizes the spotlight (UI radius -> larger/smaller steered box).
  $('gd-strip-wrap').addEventListener('wheel', (e) => {
    if (!state.meta) return;
    e.preventDefault();
    state.radiusPx = Math.max(30, Math.min(170, state.radiusPx + (e.deltaY < 0 ? 9 : -9)));
    if (state.hovering) { renderSpot(); applyTarget(); }
  }, { passive: false });
  $('gd-strip-wrap').addEventListener('mouseleave', () => {
    state.hovering = false;
    renderSpot();
    clearTarget();
    if (state.hoverTimer) { clearTimeout(state.hoverTimer); state.hoverTimer = null; }
    // Grace period so brushing past the strip edge doesn't kill the run.
    if (!state.leaveTimer) {
      state.leaveTimer = setTimeout(() => {
        state.leaveTimer = null;
        state.hoverArmed = true; // next hover starts a fresh run
        if (state.generating) state.stopFlag = true;
      }, 400);
    }
  });

  // ---- generation -------------------------------------------------------------
  $('gd-gen').onclick = () => {
    // Baseline: generate with no steering (hovering mid-run still re-steers).
    clearTarget();
    state.hoverArmed = false; // don't let the same hover restart a new run
    runGeneration();
  };

  async function runGeneration() {
    if (state.generating || !state.ready) return;
    state.generating = true;
    state.stopFlag = false;
    $('gd-gen').disabled = true;
    $('gd-stop').disabled = false;
    const out = $('gd-out');
    out.textContent = '';
    // The current target is already set by the latest mousemove (or cleared by
    // the baseline button), so no re-targeting needed here.

    const t0 = performance.now();
    let nTokens = 0;
    const streamer = new TextStreamer(state.processor.tokenizer, {
      skip_prompt: true,
      decode_kwargs: { skip_special_tokens: true },
      token_callback_function: () => { if (state.stopFlag) throw new Error('__stopped__'); },
      callback_function: (chunk) => {
        nTokens += 1;
        const span = document.createElement('span');
        if (state.gaze.active && state.currentPanel >= 0) span.className = 'p' + state.currentPanel;
        span.textContent = chunk;
        out.appendChild(span);
        out.scrollTop = out.scrollHeight;
        $('gd-perf').textContent = `${nTokens} chunks · ${((performance.now() - t0) / 1000).toFixed(1)}s`;
      },
    });

    try {
      const past = state.kvSnapshot ? { past_key_values: buildPastFromSnapshot() } : {};
      const result = await state.model.generate({
        ...state.inputs, ...past,
        max_new_tokens: MAX_NEW_TOKENS, min_new_tokens: MIN_NEW_TOKENS,
        do_sample: false, repetition_penalty: 1.1,
        streamer, return_dict_in_generate: true,
      });
      const outIds = result.sequences ?? result;
      try { result.past_key_values?.dispose?.(); } catch { /* best effort */ }
      const newTok = outIds.dims[1] - state.promptLen;
      const dt = (performance.now() - t0) / 1000;
      $('gd-perf').textContent = `${newTok} tokens · ${dt.toFixed(1)}s · ${(newTok / dt).toFixed(1)} tok/s`;
      if (!out.textContent) {
        out.textContent = state.processor.batch_decode(
          outIds.slice(null, [state.promptLen, null]), { skip_special_tokens: true })[0];
      }
      status('finished — move off the strip, then hover a panel for a new run');
    } catch (err) {
      if (err.message !== '__stopped__') {
        status('generation failed: ' + err.message);
        console.error(err);
      } else {
        status('stopped — hover a panel to start a new run');
      }
    } finally {
      state.generating = false;
      $('gd-gen').disabled = false;
      $('gd-stop').disabled = true;
    }
  }

  $('gd-stop').onclick = () => { state.stopFlag = true; };

  // ---- head-set modes (recording demo only; gated by opts.headModes) -------
  const headKey = (e) => e.layer * N_HEADS + e.head;
  function applyHeadMode(mode) {
    state.headMode = mode;
    if (!state.ranking) return;
    if (mode === 'all') {
      // Every head in the steerable layers at once (heads outside these layers
      // are inert in the graph). Over-steering — expected to degrade the story.
      const heads = [];
      for (const l of state.steerLayers) {
        for (let h = 0; h < N_HEADS; h++) heads.push({ layer: l, head: h });
      }
      state.gaze.setHeads(heads);
    } else if (mode === 'random') {
      // 10 random NON-gaze heads — a control showing arbitrary heads don't
      // carry gaze. The wired layers (16-22) are gaze-rich, so even non-top-10
      // heads there partially steer; restrict the pool to genuinely low-gaze
      // heads (rank >= 100, outside the paper's gaze set). Re-rolls on select.
      const rankOf = new Map(state.ranking.map((e, i) => [headKey(e), i]));
      const pool = [];
      for (const l of state.steerLayers) {
        for (let h = 0; h < N_HEADS; h++) {
          const e = { layer: l, head: h };
          if ((rankOf.get(headKey(e)) ?? 1e9) >= 100) pool.push(e);
        }
      }
      for (let i = pool.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [pool[i], pool[j]] = [pool[j], pool[i]];
      }
      state.gaze.setHeads(pool.slice(0, TOP_K));
    } else {
      state.gaze.setHeads(state.ranking.slice(0, TOP_K));
    }
  }
  function buildHeadModeSelect() {
    const sel = document.createElement('select');
    sel.className = 'gd-headmode';
    sel.title = 'Which heads to steer';
    sel.innerHTML = '<option value="gaze">Gaze heads (top 10)</option>'
      + '<option value="all">All heads</option>'
      + '<option value="random">Random 10 heads</option>';
    sel.onchange = () => applyHeadMode(sel.value);
    $('gd-controls').insertBefore(sel, $('gd-comic'));
  }

  window.__gazedemo = { state, applyHeadMode, clearTarget, applyTarget, renderSpot }; // console debugging hook

}
