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
const PROMPT = 'Describe this comic strip in detail.';
const MAX_NEW_TOKENS = 500;
const MIN_NEW_TOKENS = 400;
const TOP_K = 10;
const DELTA = 10000.0;
const N_LAYERS = 28, N_HEADS = 16;

const COMICS = [
  { name: 'comic217', label: 'Dragon through the seasons' },
  { name: 'comic110', label: 'Dog earns cookies' },
  { name: 'comic281', label: 'Scientist at work' },
  { name: 'comic4', label: "Cat's treasure hunt" },
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
.gazedemo .gd-strip-wrap { position: relative; cursor: crosshair; border: 1px solid #ccc;
  border-radius: 6px; overflow: hidden; user-select: none; }
.gazedemo .gd-strip { display: block; width: 100%; }
.gazedemo .gd-hl { position: absolute; top: 0; height: 100%; display: none;
  background: rgba(46,204,113,.25); border-left: 2px solid #2ecc71;
  border-right: 2px solid #2ecc71; pointer-events: none; }
.gazedemo .gd-controls { margin: 12px 0; display: flex; gap: 10px; align-items: center;
  flex-wrap: wrap; }
.gazedemo button { padding: 5px 14px; font-size: 14px; border-radius: 6px;
  border: 1px solid #888; background: #f5f5f5; cursor: pointer; }
.gazedemo button:disabled { opacity: .45; cursor: default; }
.gazedemo .gd-gen { background: #1d7a3a; border-color: #1d7a3a; color: #fff; }
.gazedemo .gd-stop { background: #a33333; border-color: #a33333; color: #fff; }
.gazedemo .gd-badge { font-size: 13px; padding: 3px 10px; border-radius: 99px;
  background: #eee; border: 1px solid #ccc; }
.gazedemo .gd-out { border: 1px solid #ddd; border-radius: 6px; padding: 12px;
  min-height: 100px; max-height: 320px; overflow-y: auto; font-size: 15px;
  line-height: 1.55; white-space: pre-wrap; background: #fff; }
.gazedemo .gd-out span.p0 { background: #ffe6e6; } .gazedemo .gd-out span.p1 { background: #fff2cc; }
.gazedemo .gd-out span.p2 { background: #e6ffe6; } .gazedemo .gd-out span.p3 { background: #e6f2ff; }
.gazedemo .gd-out span.p4 { background: #f2e6ff; } .gazedemo .gd-out span.p5 { background: #ffe6f7; }
.gazedemo .gd-status { color: #555; font-size: 13px; min-height: 18px; margin: 8px 0; }
.gazedemo .gd-perf { font-size: 12px; color: #777; margin-top: 6px; }
.gazedemo .gd-hint { font-size: 13px; color: #555; margin-top: 6px; }
.gazedemo.loading .gd-status, .gazedemo.loading .gd-hint,
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
  <div class="gd-hint">Hover a panel to steer the gaze heads onto it —
    <b>including while the model is writing</b>. Move off the strip to release.
    Text is tinted by the panel that was steering it.</div>
  <div class="gd-controls">
    <select class="gd-comic" disabled></select>
    <button class="gd-gen" disabled>Start generating</button>
    <button class="gd-stop" disabled>Stop</button>
    <label style="font-size:14px"><input type="checkbox" class="gd-steer" checked /> steering</label>
    <span class="gd-badge">target: —</span>
  </div>
  <div class="gd-out"></div>
  <div class="gd-perf"></div>
</div>`;

// ---------------------------------------------------------------------------
export async function start(container) {
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
  // During the initial load an overlay card covers the strip; route progress
  // messages there while it exists, to the status line afterwards.
  const msg = (t) => { const m = mount.querySelector('.gd-loading-msg'); if (m) m.textContent = t; else status(t); };

  // Show the real layout immediately: the comic renders right away and the
  // model download happens behind the overlay instead of a broken shell.
  $('gd-strip').src = asset(`comics/${COMICS[0].name}.png`);
  $('gd-comic').innerHTML = COMICS.map((c) => `<option value="${c.name}">${c.label}</option>`).join('');

  const state = {
    processor: null, model: null, gaze: new GazeController(), config: null,
    inputs: null, promptLen: 0, grid: null, panelPositions: [], allImage: null,
    meta: null, image: null, embeds: null, embedsDims: null,
    kvSnapshot: null, CacheCtor: null,
    generating: false, stopFlag: false, currentPanel: -1,
  };

  // ---- model load (starts immediately) -------------------------------------
  try {
    msg('Preparing the demo…');
    state.processor = await AutoProcessor.from_pretrained(MODEL_ID);
    state.config = await (await fetch(
      `https://huggingface.co/${MODEL_ID}/resolve/main/config.json`)).json();

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
    state.model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
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
    const ranking = await (await fetch(asset('gaze_head_ranking_qwen3vl_2b.json'))).json();
    state.gaze.setHeads(ranking.slice(0, TOP_K));

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
    msg('Preparing the comic…');
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
    state.panelPositions = [];
    for (let p = 0; p < state.meta.panel_widths.length; p++) {
      state.panelPositions.push(bboxToTokenPositions(
        [state.meta.panel_boundaries_px[p], 0, state.meta.panel_boundaries_px[p + 1], state.meta.height],
        state.grid, [state.meta.width, state.meta.height], imgStart));
    }
    state.allImage = new Set(Array.from({ length: imgEnd - imgStart }, (_, i) => imgStart + i));
    setTargetPanel(-1);
    $('gd-out').textContent = '';
    $('gd-perf').textContent = '';

    msg('Warming up the model (one-time per comic)…');
    await snapshotPromptKV();
    status('ready — press "Start generating", then hover panels to steer');
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

  // ---- hover-to-steer --------------------------------------------------------
  function setTargetPanel(p) {
    state.currentPanel = p;
    $('gd-badge').textContent = p >= 0 ? `target: panel ${p + 1}` : 'target: —';
    if (p < 0 || !$('gd-steer').checked) { state.gaze.clear(); return; }
    const boost = state.panelPositions[p];
    const boostSet = new Set(boost);
    state.gaze.setTarget({
      boost,
      suppress: [...state.allImage].filter((pos) => !boostSet.has(pos)),
    });
  }

  $('gd-strip-wrap').addEventListener('mousemove', (e) => {
    if (!state.meta) return;
    const img = $('gd-strip');
    const rect = img.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width * state.meta.width;
    const b = state.meta.panel_boundaries_px;
    let p = -1;
    for (let i = 0; i < b.length - 1; i++) if (px >= b[i] && px < b[i + 1]) { p = i; break; }
    if (p !== state.currentPanel) setTargetPanel(p);
    const hl = $('gd-hl');
    if (p >= 0) {
      hl.style.display = 'block';
      hl.style.left = (b[p] / state.meta.width * 100) + '%';
      hl.style.width = ((b[p + 1] - b[p]) / state.meta.width * 100) + '%';
    }
  });
  $('gd-strip-wrap').addEventListener('mouseleave', () => {
    $('gd-hl').style.display = 'none';
    setTargetPanel(-1);
  });

  // ---- generation -------------------------------------------------------------
  $('gd-gen').onclick = async () => {
    if (state.generating) return;
    state.generating = true;
    state.stopFlag = false;
    $('gd-gen').disabled = true;
    $('gd-stop').disabled = false;
    const out = $('gd-out');
    out.textContent = '';
    setTargetPanel(state.currentPanel);

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
      status('done — hover a different panel and press "Start generating" again');
    } catch (err) {
      if (err.message !== '__stopped__') {
        status('generation failed: ' + err.message);
        console.error(err);
      } else {
        status('stopped');
      }
    } finally {
      state.generating = false;
      $('gd-gen').disabled = false;
      $('gd-stop').disabled = true;
    }
  };

  $('gd-stop').onclick = () => { state.stopFlag = true; };

  window.__gazedemo = { state, setTargetPanel }; // console debugging hook

}
