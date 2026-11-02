/* ============================================================
   harness.js — the common frame every LARVAE page runs in.

   It owns time, randomness, the controls and the record. It knows
   nothing about larvae, glass or bowls: a page hands it a factory
   that builds a simulation, and the harness runs it.

   What a page gives the harness (the "sim" contract):
     create(params, rng)  -> sim
     sim.step(dt)                    advance one tick
     sim.draw(ctx, w, h)             paint the current state
     sim.status() -> {action, reason, detail}   what it is doing and why, in English
     sim.snapshot() -> plain object  what to keep in the log for this tick
     sim.invariants -> [{name, check(prev, cur) -> null | message}]
                                     prev/cur are snapshots; a message pauses the run
     sim.pointer(kind, x, y, extra)  optional: mouse in canvas pixels
     sim.done() -> null | {ok, why}  optional: the run has reached its end

   Determinism: fixed dt, every random number from a named generator
   seeded from one seed. Same seed + same parameters = same run, bit
   for bit, whatever ran before it.
   ============================================================ */
(function (root) {
  'use strict';

  const VERSION = '1.0';
  const DT = 1 / 30;            // one tick = 1/30 s of animal time, always
  const RECENT_TICKS = 300;     // ten seconds of snapshots kept for the log
  const MAX_TRANSITIONS = 400;

  /* ---------- randomness: one seed, many named streams ---------- */
  function hashString(s) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    return h >>> 0;
  }
  function mulberry32(a) {
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // rng('larva') and rng('target') never disturb each other
  function makeRng(seed) {
    const streams = {};
    return function rng(name) {
      if (!streams[name]) {
        const r = mulberry32((hashString(String(name)) ^ (seed >>> 0)) >>> 0);
        streams[name] = {
          next: r,
          range: (a, b) => a + r() * (b - a),
          sign: () => (r() < 0.5 ? -1 : 1)
        };
      }
      return streams[name];
    };
  }

  /* ---------- parameters: schema -> safe values ---------- */
  // A field: {key, label, type:'number'|'select', min, max, step, value, unit, status, hint, options}
  // status is the audit category: 'measured', 'body-derived', 'pose-derived', 'free'.
  function clampParams(schema, raw) {
    const out = {};
    for (const f of schema) {
      let v = raw && raw[f.key] !== undefined ? raw[f.key] : f.value;
      if (f.type === 'number') {
        v = parseFloat(v);
        if (!isFinite(v)) v = f.value;
        v = Math.min(f.max, Math.max(f.min, v));
        if (f.step) v = Math.round(v / f.step) * f.step;
        v = +v.toFixed(6);
      } else if (f.type === 'select') {
        const ok = f.options.some(o => o.value === v && !o.disabled);
        if (!ok) v = f.value;
      }
      out[f.key] = v;
    }
    return out;
  }

  /* ---------- the run itself, with no screen ---------- */
  function Run(factory, params, seed) {
    this.factory = factory;
    this.params = params;
    this.seed = seed >>> 0;
    this.sim = factory.create(params, makeRng(this.seed));
    this.tick = 0;
    this.transitions = [];
    this.violations = [];
    this.marks = [];
    this.recent = [];
    this.lastStatusKey = '';
    this.prev = this.sim.snapshot();
    this.finished = null;
    this.noteStatus();
  }
  Run.prototype.time = function () { return this.tick * DT; };
  Run.prototype.noteStatus = function () {
    const s = this.sim.status();
    const key = s.action + '|' + s.reason;
    if (key !== this.lastStatusKey) {
      this.lastStatusKey = key;
      this.transitions.push({ tick: this.tick, t: +this.time().toFixed(2), action: s.action, reason: s.reason });
      if (this.transitions.length > MAX_TRANSITIONS) this.transitions.shift();
    }
  };
  // returns null, or the violation that stopped it
  Run.prototype.step = function () {
    if (this.finished) return null;
    this.sim.step(DT);
    this.tick++;
    const cur = this.sim.snapshot();
    cur.tick = this.tick;
    this.recent.push(cur);
    if (this.recent.length > RECENT_TICKS) this.recent.shift();
    let hit = null;
    if (!cur.skipChecks) {
      for (const inv of this.sim.invariants || []) {
        const msg = inv.check(this.prev, cur);
        if (msg) { hit = { tick: this.tick, t: +this.time().toFixed(2), name: inv.name, message: msg }; break; }
      }
    }
    if (hit) this.violations.push(hit);
    this.prev = cur;
    this.noteStatus();
    if (this.sim.done) {
      const d = this.sim.done();
      if (d) this.finished = Object.assign({ tick: this.tick, t: +this.time().toFixed(2) }, d);
    }
    return hit;
  };
  Run.prototype.mark = function (text) {
    const s = this.sim.status();
    this.marks.push({
      tick: this.tick, t: +this.time().toFixed(2), text: text,
      action: s.action, reason: s.reason, detail: s.detail || '',
      around: this.recent.slice()   // the ten seconds before the mark
    });
  };
  Run.prototype.journal = function (page) {
    return {
      page: page, harness: VERSION, saved: new Date().toISOString(),
      seed: this.seed, params: this.params, tick: this.tick, t: +this.time().toFixed(2),
      finished: this.finished, violations: this.violations,
      transitions: this.transitions, marks: this.marks, recent: this.recent
    };
  };

  // for checks and for batch pages: run to the end with no drawing at all
  function runHeadless(factory, params, seed, maxTicks) {
    const r = new Run(factory, params, seed);
    let hit = null;
    while (!r.finished && r.tick < maxTicks && !hit) hit = r.step();
    return { run: r, violation: hit, finished: r.finished, tick: r.tick, t: r.time() };
  }

  /* ---------- the screen ---------- */
  const CSS = `
  .hz-wrap{max-width:1100px;margin:0 auto;padding:14px 16px 40px}
  .hz-wrap [hidden]{display:none!important}
  .hz-wrap h1{font-size:13px;font-weight:bold;letter-spacing:.14em;text-transform:uppercase;margin:0 0 12px}
  .hz-form{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px 18px;margin-bottom:12px}
  .hz-field label{display:block;margin-bottom:3px}
  .hz-field .hz-status{font-size:10px;letter-spacing:.06em;text-transform:uppercase}
  .hz-field .hz-hint{font-size:11px}
  .hz-field input,.hz-field select{font:inherit;width:100%;padding:5px 6px;border:1px solid #000;border-radius:0;
    background-color:#fff;background-image:linear-gradient(#fff,#fff);color:#000;-webkit-text-fill-color:#000;
    -webkit-appearance:none;appearance:none}
  .hz-bar{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
  .hz-bar button{font:inherit;padding:6px 14px;border:1px solid #000;border-radius:0;cursor:pointer;
    background-color:#fff;background-image:linear-gradient(#fff,#fff);color:#000;-webkit-text-fill-color:#000}
  .hz-bar button:disabled{border-color:#bbb;color:#bbb;-webkit-text-fill-color:#bbb;cursor:default}
  .hz-bar button.hz-go{font-weight:bold}
  .hz-markbox input{font:inherit;flex:1;min-width:0;padding:5px 6px;border:1px solid #000;border-radius:0;
    background-color:#fff;background-image:linear-gradient(#fff,#fff);color:#000;-webkit-text-fill-color:#000}
  .hz-bar select{font:inherit;padding:5px;border:1px solid #000;border-radius:0;background-image:linear-gradient(#fff,#fff);
    color:#000;-webkit-text-fill-color:#000}
  .hz-state{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;border-top:1px solid #000;
    border-bottom:1px solid #000;padding:6px 0;margin:6px 0;min-height:3.2em}
  .hz-state .hz-action{font-weight:bold}
  .hz-alert{color:#c00;-webkit-text-fill-color:#c00}
  .hz-mute{color:#666;-webkit-text-fill-color:#666}
  .hz-canvas{display:block;width:100%;height:auto;border:0;touch-action:none;
    background-image:linear-gradient(#fff,#fff)}
  .hz-meta{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;margin-top:4px}
  .hz-log{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;max-height:9.5em;overflow-y:auto;margin-top:8px}
  .hz-extra{margin-top:12px}
  `;

  function el(tag, attrs, kids) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'text') e.textContent = attrs[k];
      else if (k === 'html') e.innerHTML = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(c => c && e.appendChild(c));
    return e;
  }

  const STATUS_WORDS = {
    'measured': 'measured', 'body-derived': 'from the body', 'pose-derived': 'from the poses', 'free': 'free choice'
  };

  /*
    mount(config):
      host      element to build into
      title     page title
      page      file name, goes into the journal
      factory   {create}
      schema    parameter fields
      canvas    {width, height} in canvas pixels (logical)
      extra     optional function(container) for a page's own read-only panel
      onFrame   optional function(run) after each paint
  */
  function mount(cfg) {
    if (!document.getElementById('hz-style')) {
      const st = el('style', { id: 'hz-style' }); st.textContent = CSS; document.head.appendChild(st);
    }
    const host = cfg.host;
    const wrap = el('div', { class: 'hz-wrap' });
    host.appendChild(wrap);
    wrap.appendChild(el('h1', { text: cfg.title }));

    // 1. parameters
    const form = el('div', { class: 'hz-form' });
    const inputs = {};
    for (const f of cfg.schema) {
      let input;
      const id = 'hz-in-' + f.key.replace(/[^a-z0-9]+/gi, '-').toLowerCase();
      if (f.type === 'select') {
        input = el('select', { id });
        for (const o of f.options) {
          const op = el('option', { value: o.value, text: o.label });
          if (o.disabled) op.disabled = true;
          input.appendChild(op);
        }
        input.value = f.value;
      } else {
        input = el('input', { id, type: 'number', min: f.min, max: f.max, step: f.step || 'any', value: f.value });
        input.addEventListener('change', () => {
          const v = clampParams([f], { [f.key]: input.value })[f.key];
          input.value = v;
        });
      }
      inputs[f.key] = input;
      const range = f.type === 'number' ? ' (' + f.min + '–' + f.max + (f.unit ? ' ' + f.unit : '') + ')' : '';
      form.appendChild(el('div', { class: 'hz-field' }, [
        el('label', { for: id, text: f.label + range }),
        input,
        f.status ? el('div', { class: 'hz-status hz-mute', text: STATUS_WORDS[f.status] || f.status }) : null,
        f.hint ? el('div', { class: 'hz-hint hz-mute', text: f.hint }) : null
      ]));
    }
    // seed: empty = take a new one from the clock at Пуск
    const seedIn = el('input', { id: 'hz-in-seed', type: 'number', min: 0, max: 4294967295, step: 1, placeholder: 'auto' });
    form.appendChild(el('div', { class: 'hz-field' }, [
      el('label', { for: 'hz-in-seed', text: 'Seed' }), seedIn,
      el('div', { class: 'hz-hint hz-mute', text: 'Пусто — новое число при каждом «Пуске». Одно и то же число — тот же самый прогон.' })
    ]));
    wrap.appendChild(form);

    // 2. controls
    const bGo = el('button', { id: 'hz-btn-go', class: 'hz-go', text: 'Пуск' });
    const bPause = el('button', { id: 'hz-btn-pause', text: 'Пауза' });
    const bStep = el('button', { id: 'hz-btn-step', text: 'Шаг' });
    const bReset = el('button', { id: 'hz-btn-reset', text: 'Сброс' });
    const bMark = el('button', { id: 'hz-btn-mark', text: 'Отметить' });
    const bSave = el('button', { id: 'hz-btn-save', text: 'Скачать журнал' });
    const speed = el('select', { id: 'hz-speed', title: 'скорость показа' });
    [1, 2, 4, 8].forEach(k => speed.appendChild(el('option', { value: k, text: '×' + k })));
    wrap.appendChild(el('div', { class: 'hz-bar' }, [bGo, bPause, bStep, bReset, bMark, bSave, speed]));
    // the mark is typed in the page itself: some viewers never show prompt()
    const markIn = el('input', { id: 'hz-in-mark', type: 'text', placeholder: 'Что не так? Одна строка, Enter' });
    const markOk = el('button', { id: 'hz-btn-mark-ok', text: 'Записать' });
    const markBox = el('div', { class: 'hz-bar hz-markbox' }, [markIn, markOk]);
    markBox.hidden = true;
    wrap.appendChild(markBox);

    // 3. what it is doing
    const state = el('div', { class: 'hz-state', id: 'hz-state' });
    wrap.appendChild(state);

    // 4. picture
    const canvas = el('canvas', { class: 'hz-canvas', id: 'hz-canvas' });
    const W = cfg.canvas.width, H = cfg.canvas.height;
    const dpr = Math.min(2, (typeof window !== 'undefined' && window.devicePixelRatio) || 1);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    canvas.style.maxWidth = W + 'px';
    wrap.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    const meta = el('div', { class: 'hz-meta hz-mute', id: 'hz-meta' });
    wrap.appendChild(meta);
    const log = el('div', { class: 'hz-log', id: 'hz-log' });
    wrap.appendChild(log);
    const extra = el('div', { class: 'hz-extra' });
    wrap.appendChild(extra);
    if (cfg.extra) cfg.extra(extra);

    let run = null, paused = true, lastParams = null, alertMsg = '';

    function readParams() {
      const raw = {};
      for (const k in inputs) raw[k] = inputs[k].value;
      const p = clampParams(cfg.schema, raw);
      for (const k in inputs) inputs[k].value = p[k];
      return p;
    }
    function start(params, seed) {
      run = new Run(cfg.factory, params, seed);
      lastParams = params;
      alertMsg = '';
      paused = false;
      refresh();
    }
    function refresh() {
      bPause.textContent = paused ? 'Дальше' : 'Пауза';
      [bPause, bStep, bReset, bMark, bSave].forEach(b => (b.disabled = !run));
      if (!run) {
        state.innerHTML = '<span class="hz-mute">Задай параметры и нажми «Пуск».</span>';
        return;
      }
      const s = run.sim.status();
      let html = '<span class="hz-action">' + esc(s.action) + '</span> — ' + esc(s.reason);
      if (s.detail) html += '<br><span class="hz-mute">' + esc(s.detail) + '</span>';
      if (alertMsg) html += '<br><span class="hz-alert">' + esc(alertMsg) + '</span>';
      else if (run.finished) html += '<br><b>' + esc((run.finished.ok ? 'DONE — ' : 'FAILED — ') + run.finished.why) + '</b>';
      else if (paused) html += '<br><span class="hz-mute">PAUSED</span>';
      state.innerHTML = html;
      meta.textContent = 'seed ' + run.seed + ' · t ' + run.time().toFixed(1) + ' s · tick ' + run.tick +
        ' · marks ' + run.marks.length + ' · violations ' + run.violations.length;
      log.innerHTML = run.transitions.slice(-30).reverse()
        .map(x => x.t.toFixed(1) + ' s  ' + esc(x.action) + ' — ' + esc(x.reason)).join('<br>');
    }
    function esc(s) { return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }

    function paint() {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
      if (run) run.sim.draw(ctx, W, H);
      else if (cfg.factory.drawEmpty) cfg.factory.drawEmpty(ctx, W, H);
    }
    function tickOnce() {
      const hit = run.step();
      if (hit) { paused = true; alertMsg = 'PAUSED — ' + hit.name + ': ' + hit.message; }
      if (run.finished) paused = true;
    }

    bGo.onclick = () => {
      const p = readParams();
      let seed = parseInt(seedIn.value, 10);
      // a typed seed stays in the field; an empty field draws a new one each time
      if (!isFinite(seed)) seed = (Date.now() ^ Math.floor(Math.random() * 1e9)) >>> 0;
      start(p, seed);
    };
    bPause.onclick = () => { if (!run) return; if (run.finished) return; paused = !paused; if (!paused) alertMsg = ''; refresh(); };
    bStep.onclick = () => { if (!run) return; paused = true; alertMsg = ''; tickOnce(); paint(); refresh(); };
    bReset.onclick = () => { if (!run) return; const s = run.seed; start(lastParams, s); paused = true; paint(); refresh(); };
    bMark.onclick = () => {
      if (!run) return;
      paused = true; markBox.hidden = false; markIn.value = ''; markIn.focus(); refresh();
    };
    function commitMark() {
      const text = markIn.value.trim();
      if (text && run) { run.mark(text); alertMsg = ''; }
      markBox.hidden = true; refresh();
    }
    markOk.onclick = commitMark;
    markIn.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); commitMark(); }
      if (e.key === 'Escape') { markBox.hidden = true; }
    });
    bSave.onclick = () => {
      if (!run) return;
      const text = JSON.stringify(run.journal(cfg.page), null, 1);
      const name = cfg.page.replace(/\.html$/, '') + '-seed' + run.seed + '-t' + Math.round(run.time()) + '.json';
      const blob = new Blob([text], { type: 'application/json' });
      const a = el('a', { href: URL.createObjectURL(blob), download: name });
      document.body.appendChild(a); a.click(); a.remove();
      // inside an embedded viewer the download can be blocked silently: copy as well
      if (window.self !== window.top && navigator.clipboard) {
        navigator.clipboard.writeText(text).then(
          () => { alertMsg = 'Журнал скопирован в буфер — вставь его в чат (здесь файл может не скачаться).'; refresh(); },
          () => {});
      }
    };
    window.addEventListener('keydown', e => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
      if (e.target && e.target.isContentEditable) return;
      if (e.code === 'Space') { e.preventDefault(); bPause.click(); }
      else if (e.key === 's' || e.key === 'S') bStep.click();
    });

    // the mouse, in logical canvas pixels
    function pos(e) {
      const r = canvas.getBoundingClientRect();
      return { x: (e.clientX - r.left) * W / r.width, y: (e.clientY - r.top) * H / r.height };
    }
    let down = false;
    canvas.addEventListener('pointerdown', e => {
      if (!run || !run.sim.pointer) return;
      down = true; canvas.setPointerCapture(e.pointerId);
      const p = pos(e); run.sim.pointer('down', p.x, p.y); paint(); refresh();
    });
    canvas.addEventListener('pointermove', e => {
      if (!down || !run || !run.sim.pointer) return;
      const p = pos(e); run.sim.pointer('move', p.x, p.y); paint();
    });
    canvas.addEventListener('pointerup', e => {
      if (!down || !run || !run.sim.pointer) return;
      down = false; const p = pos(e); run.sim.pointer('up', p.x, p.y); paint(); refresh();
    });
    canvas.addEventListener('wheel', e => {
      if (!run || !run.sim.pointer) return;
      const p = pos(e);
      if (run.sim.pointer('wheel', p.x, p.y, { delta: e.deltaY > 0 ? 1 : -1 })) { e.preventDefault(); paint(); }
    }, { passive: false });

    // real time drives the clock, but the simulation only ever sees whole ticks
    let acc = 0, lastT = null;
    function frame(now) {
      if (lastT === null) lastT = now;
      let elapsed = (now - lastT) / 1000; lastT = now;
      if (elapsed > 0.25) elapsed = 0.25;   // a stalled tab must not make the animal jump
      if (run && !paused) {
        acc += elapsed * parseInt(speed.value, 10);
        let n = 0;
        while (acc >= DT && n < 64 && !paused) { tickOnce(); acc -= DT; n++; }
        if (n === 64) acc = 0;
        refresh();
      } else acc = 0;
      paint();
      if (cfg.onFrame && run) cfg.onFrame(run);
      requestAnimationFrame(frame);
    }
    refresh();
    requestAnimationFrame(frame);
    return { get run() { return run; } };
  }

  root.Harness = { VERSION, DT, makeRng, clampParams, Run, runHeadless, mount };
})(typeof window !== 'undefined' ? window : globalThis);
