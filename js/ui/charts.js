/* Weka Web — γραφήματα (SVG/Canvas) χωρίς εξωτερικές βιβλιοθήκες. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, App = window.App, h = App.h;
  const SVGNS = 'http://www.w3.org/2000/svg';
  const C = (App.charts = {});

  function svg(w, hgt, cls = 'chart') {
    const s = document.createElementNS(SVGNS, 'svg');
    s.setAttribute('viewBox', `0 0 ${w} ${hgt}`);
    s.setAttribute('class', cls);
    s.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    return s;
  }
  function el(name, attrs = {}, parent) {
    const e = document.createElementNS(SVGNS, name);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, str, attrs = {}) { const t = el('text', Object.assign({ x, y }, attrs), parent); t.textContent = str; return t; }
  C.svg = svg; C.el = el; C.text = text;

  /** "Ωραίες" υποδιαιρέσεις άξονα. */
  function ticks(min, max, n = 5) {
    if (!(isFinite(min) && isFinite(max))) return [];
    if (min === max) return [min];
    const span = max - min, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  C.ticks = ticks;
  const shortNum = (v) => { const a = Math.abs(v); if (a >= 1e6) return U.d2s(v / 1e6, 2) + 'M'; if (a >= 1e4) return U.d2s(v / 1e3, 1) + 'k'; return U.d2s(v, a < 1 ? 3 : 2); };
  C.shortNum = shortNum;

  // ------------------------------------------------------- tooltip
  let tipEl = null;
  C.tip = function (html, x, y) {
    if (!tipEl) { tipEl = h('div.tooltip'); document.body.appendChild(tipEl); }
    if (html === null) { tipEl.style.display = 'none'; return; }
    tipEl.innerHTML = html;
    tipEl.style.display = 'block';
    const r = tipEl.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
    tipEl.style.left = left + 'px'; tipEl.style.top = top + 'px';
  };

  // ------------------------------------------------------ histogram
  /**
   * Ιστόγραμμα χαρακτηριστικού, στοιβαγμένο ανά τιμή του colorAtt (αν nominal).
   */
  C.histogram = function (d, att, colorAtt, opts = {}) {
    const a = d.attributes[att];
    const Wd = opts.width || 520, Ht = opts.height || 220, m = { l: 44, r: 10, t: 12, b: a.type === 'nominal' ? 54 : 34 };
    const s = svg(Wd, Ht);
    const ca = colorAtt >= 0 ? d.attributes[colorAtt] : null;
    const colorNominal = ca && ca.type === 'nominal';
    const nCol = colorNominal ? ca.values.length : 1;
    let bins, labels, binOf;
    if (a.type === 'nominal' || a.type === 'string') {
      const nv = Math.min(a.values.length, 60);
      bins = Array.from({ length: nv }, () => U.zeros(nCol));
      labels = a.values.slice(0, nv);
      binOf = (v) => (v < nv ? v : -1);
    } else {
      let mn = Infinity, mx = -Infinity;
      for (const r of d.rows) { const v = r[att]; if (v === v) { if (v < mn) mn = v; if (v > mx) mx = v; } }
      if (!isFinite(mn)) { mn = 0; mx = 1; }
      const n = d.numInstances;
      let nb = Math.max(1, Math.min(opts.bins || 30, Math.ceil(Math.sqrt(n))));
      if (a.integer || d.rows.every((r) => r[att] !== r[att] || Number.isInteger(r[att]))) nb = Math.max(1, Math.min(nb, mx - mn + 1));
      const w = (mx - mn) / nb || 1;
      bins = Array.from({ length: nb }, () => U.zeros(nCol));
      labels = bins.map((_, i) => [mn + i * w, mn + (i + 1) * w]);
      binOf = (v) => Math.min(nb - 1, Math.floor((v - mn) / w));
      s.dataset.min = mn; s.dataset.max = mx;
    }
    for (let i = 0; i < d.numInstances; i++) {
      const r = d.rows[i], v = r[att];
      if (v !== v) continue;
      const b = binOf(v); if (b < 0) continue;
      const c = colorNominal ? r[colorAtt] : 0;
      bins[b][c === c ? c : 0] += d.weights[i];
    }
    const maxH = Math.max(1, ...bins.map((b) => U.sum(b)));
    const iw = Wd - m.l - m.r, ih = Ht - m.t - m.b;
    const g = el('g', { class: 'grid' }, s);
    for (const t of ticks(0, maxH, 4)) {
      const y = m.t + ih - (t / maxH) * ih;
      el('line', { x1: m.l, x2: Wd - m.r, y1: y, y2: y }, g);
      text(s, m.l - 6, y + 4, shortNum(t), { 'text-anchor': 'end' });
    }
    const bw = iw / bins.length;
    bins.forEach((b, i) => {
      let y = m.t + ih;
      const tot = U.sum(b);
      for (let c = 0; c < nCol; c++) {
        if (!b[c]) continue;
        const hh = (b[c] / maxH) * ih;
        y -= hh;
        const rect = el('rect', { x: m.l + i * bw + Math.min(2, bw * 0.12), y, width: Math.max(1, bw - Math.min(4, bw * 0.24)), height: Math.max(0.5, hh), rx: Math.min(3, bw / 4), fill: colorNominal ? App.color(c) : 'var(--primary)', opacity: 0.9 }, s);
        const lbl = Array.isArray(labels[i]) ? `[${U.d2s(labels[i][0], 3)}, ${U.d2s(labels[i][1], 3)})` : labels[i];
        rect.addEventListener('mousemove', (e) => C.tip(`<b>${App.esc(lbl)}</b><br>${colorNominal ? App.esc(ca.values[c]) + ': ' : ''}${U.d2s(b[c], 2)} ${colorNominal ? `<span style="color:var(--text-3)">/ ${U.d2s(tot, 2)}</span>` : ''}`, e.clientX, e.clientY));
        rect.addEventListener('mouseleave', () => C.tip(null));
      }
    });
    // x labels
    if (a.type === 'nominal' || a.type === 'string') {
      const step = Math.ceil(labels.length / Math.max(1, Math.floor(iw / 28)));
      labels.forEach((l, i) => {
        if (i % step) return;
        const x = m.l + i * bw + bw / 2, y = m.t + ih + 12;
        const t = text(s, x, y, l.length > 12 ? l.slice(0, 11) + '…' : l, { 'text-anchor': labels.length > 6 ? 'end' : 'middle', transform: labels.length > 6 ? `rotate(-35 ${x} ${y})` : null });
        t.appendChild(el('title')).textContent = l;
      });
    } else {
      const mn = +s.dataset.min, mx = +s.dataset.max;
      for (const t of ticks(mn, mx, 5)) { const x = m.l + ((t - mn) / (mx - mn || 1)) * iw; text(s, x, Ht - m.b + 16, shortNum(t), { 'text-anchor': 'middle' }); }
    }
    el('line', { x1: m.l, x2: Wd - m.r, y1: m.t + ih, y2: m.t + ih, stroke: 'var(--border-strong)' }, s);
    return s;
  };

  C.legend = function (labels, colorFn = App.color, onToggle, hidden = new Set()) {
    const lg = h('div.legend' + (onToggle ? '.interactive' : ''));
    labels.forEach((l, i) => {
      const sp = h('span' + (hidden.has(i) ? '.off' : ''), h('i', { style: { background: colorFn(i) } }), l);
      if (onToggle) sp.addEventListener('click', () => { onToggle(i); sp.classList.toggle('off'); });
      lg.appendChild(sp);
    });
    return lg;
  };

  // ------------------------------------------------------- bar chart
  C.bars = function (items, opts = {}) {
    // items: [{label, value, color?}]
    const Wd = opts.width || 520, rowH = opts.rowH || 26, m = { l: opts.labelW || 140, r: opts.right || 96, t: 6, b: 6 };
    const Ht = m.t + m.b + rowH * items.length;
    const s = svg(Wd, Ht);
    const mx = Math.max(1e-12, ...items.map((i) => Math.abs(i.value)));
    const iw = Wd - m.l - m.r;
    items.forEach((it, k) => {
      const y = m.t + k * rowH;
      const lbl = it.label.length > 22 ? it.label.slice(0, 21) + '…' : it.label;
      const t = text(s, m.l - 8, y + rowH / 2 + 4, lbl, { 'text-anchor': 'end' });
      el('title', {}, t).textContent = it.label;
      el('rect', { x: m.l, y: y + 4, width: iw, height: rowH - 8, rx: 5, fill: 'var(--surface-3)' }, s);
      el('rect', { x: m.l, y: y + 4, width: Math.max(2, (Math.abs(it.value) / mx) * iw), height: rowH - 8, rx: 5, fill: it.color || 'var(--primary)' }, s);
      text(s, Wd - m.r + 6, y + rowH / 2 + 4, opts.fmt ? opts.fmt(it.value) : U.d2s(it.value, 3));
    });
    return s;
  };

  // ----------------------------------------------------- confusion
  C.confusion = function (matrix, labels) {
    const k = labels.length;
    let mx = 0; for (const r of matrix) for (const v of r) if (v > mx) mx = v;
    const prim = App.cssVar('--primary') || '#4f46e5', danger = App.cssVar('--danger') || '#dc2626';
    const tbl = h('table.cm');
    const head = h('tr', h('th'), h('th', { colspan: k, style: { textAlign: 'center' } }, L`Πρόβλεψη →`));
    tbl.appendChild(head);
    tbl.appendChild(h('tr', h('th.rowh', L`Πραγματική ↓`), labels.map((l) => h('th', { title: l }, l))));
    matrix.forEach((row, i) => {
      const tot = U.sum(row);
      tbl.appendChild(h('tr', h('th.rowh', { title: labels[i] }, labels[i]), row.map((v, j) => {
        const a = mx > 0 ? v / mx : 0;
        const base = i === j ? prim : danger;
        const bg = v === 0 ? 'var(--surface-2)' : `color-mix(in srgb, ${base} ${Math.round(12 + a * 78)}%, transparent)`;
        const fg = a > 0.55 && v ? '#fff' : 'var(--text)';
        return h('td.cell', { style: { background: bg, color: fg }, title: L`${labels[i]} → ${labels[j]}: ${U.d2s(v, 2)} (${tot ? U.d2s((100 * v) / tot, 1) : 0}% της γραμμής)` }, U.d2s(v, 2));
      })));
    });
    return h('div.table-wrap', tbl);
  };

  // ------------------------------------------------------------ ROC
  C.roc = function (curves, opts = {}) {
    // curves: [{label, color, points:[{fpr,tpr}], auc}]
    const Wd = opts.width || 460, Ht = opts.height || 400, m = { l: 48, r: 14, t: 14, b: 42 };
    const s = svg(Wd, Ht), iw = Wd - m.l - m.r, ih = Ht - m.t - m.b;
    const X = (v) => m.l + v * iw, Y = (v) => m.t + ih - v * ih;
    const g = el('g', { class: 'grid' }, s);
    for (const t of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
      el('line', { x1: X(t), x2: X(t), y1: m.t, y2: m.t + ih }, g); el('line', { x1: m.l, x2: m.l + iw, y1: Y(t), y2: Y(t) }, g);
      text(s, X(t), Ht - m.b + 16, String(t), { 'text-anchor': 'middle' }); text(s, m.l - 8, Y(t) + 4, String(t), { 'text-anchor': 'end' });
    }
    el('line', { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: 'var(--border-strong)', 'stroke-dasharray': '4 4' }, s);
    text(s, m.l + iw / 2, Ht - 6, opts.xLabel || 'False positive rate', { 'text-anchor': 'middle' });
    const yl = text(s, 14, m.t + ih / 2, opts.yLabel || 'True positive rate', { 'text-anchor': 'middle', transform: `rotate(-90 14 ${m.t + ih / 2})` });
    for (const c of curves) {
      const xs = opts.pr ? 'recall' : 'fpr', ys = opts.pr ? 'precision' : 'tpr';
      const d = c.points.map((p, i) => (i ? 'L' : 'M') + X(p[xs]).toFixed(1) + ',' + Y(p[ys]).toFixed(1)).join('');
      el('path', { d, fill: 'none', stroke: c.color, 'stroke-width': 2.2, 'stroke-linejoin': 'round' }, s);
    }
    return s;
  };

  // --------------------------------------------------------- scatter
  /**
   * Scatter σε canvas με άξονες, jitter, tooltip και zoom με σύρσιμο.
   * cfg: { xs, ys, colors(array of css), xLabel, yLabel, xCats?, yCats?, info(i)->html, size, jitter, height }
   */
  C.scatter = function (cfg) {
    const wrap = h('div.scatter-wrap');
    const canvas = h('canvas');
    wrap.appendChild(canvas);
    const height = cfg.height || 460;
    let view = null; // zoom window
    let drag = null;
    const n = cfg.xs.length;
    const rand = new W.JavaRandom(7);
    const jx = new Float64Array(n), jy = new Float64Array(n);
    for (let i = 0; i < n; i++) { jx[i] = rand.nextDouble() - 0.5; jy[i] = rand.nextDouble() - 0.5; }
    const ext = (arr, cats) => {
      if (cats) return [-0.5, cats.length - 0.5];
      let mn = Infinity, mx = -Infinity;
      for (const v of arr) if (v === v) { if (v < mn) mn = v; if (v > mx) mx = v; }
      if (!isFinite(mn)) return [0, 1];
      if (mn === mx) { mn -= 1; mx += 1; }
      const pad = (mx - mn) * 0.04;
      return [mn - pad, mx + pad];
    };
    const full = { x: ext(cfg.xs, cfg.xCats), y: ext(cfg.ys, cfg.yCats) };
    let pts = [];
    const m = { l: 56, r: 14, t: 12, b: 44 };
    function draw() {
      const dpr = window.devicePixelRatio || 1;
      const Wd = wrap.clientWidth || 600;
      canvas.width = Wd * dpr; canvas.height = height * dpr;
      canvas.style.height = height + 'px';
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, Wd, height);
      const v = view || full;
      const iw = Wd - m.l - m.r, ih = height - m.t - m.b;
      const X = (x) => m.l + ((x - v.x[0]) / (v.x[1] - v.x[0])) * iw;
      const Y = (y) => m.t + ih - ((y - v.y[0]) / (v.y[1] - v.y[0])) * ih;
      const textC = App.cssVar('--text-2'), gridC = App.cssVar('--border'), axisC = App.cssVar('--border-strong');
      ctx.font = '11px ' + App.cssVar('--font');
      ctx.fillStyle = App.cssVar('--surface-2'); ctx.fillRect(m.l, m.t, iw, ih);
      ctx.strokeStyle = gridC; ctx.lineWidth = 1; ctx.fillStyle = textC;
      // grid + ticks
      const xt = cfg.xCats ? cfg.xCats.map((_, i) => i).filter((i) => i >= v.x[0] && i <= v.x[1]) : ticks(v.x[0], v.x[1], Math.max(2, Math.floor(iw / 90)));
      const yt = cfg.yCats ? cfg.yCats.map((_, i) => i).filter((i) => i >= v.y[0] && i <= v.y[1]) : ticks(v.y[0], v.y[1], Math.max(2, Math.floor(ih / 60)));
      ctx.textAlign = 'center';
      for (const t of xt) { const x = X(t); ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ih); ctx.stroke(); const lbl = cfg.xCats ? cfg.xCats[t] : shortNum(t); ctx.fillText(lbl.length > 14 ? lbl.slice(0, 13) + '…' : lbl, x, m.t + ih + 15); }
      ctx.textAlign = 'right';
      for (const t of yt) { const y = Y(t); ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + iw, y); ctx.stroke(); const lbl = cfg.yCats ? cfg.yCats[t] : shortNum(t); ctx.fillText(lbl.length > 9 ? lbl.slice(0, 8) + '…' : lbl, m.l - 6, y + 4); }
      ctx.strokeStyle = axisC; ctx.strokeRect(m.l + 0.5, m.t + 0.5, iw, ih);
      ctx.textAlign = 'center'; ctx.fillStyle = textC;
      ctx.fillText(cfg.xLabel || '', m.l + iw / 2, height - 8);
      ctx.save(); ctx.translate(13, m.t + ih / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(cfg.yLabel || '', 0, 0); ctx.restore();
      // points
      const jit = cfg.jitter || 0;
      const size = cfg.size || 3;
      const jxs = cfg.xCats ? 0.35 : (v.x[1] - v.x[0]) * 0.03, jys = cfg.yCats ? 0.35 : (v.y[1] - v.y[0]) * 0.03;
      pts = [];
      ctx.save();
      ctx.beginPath(); ctx.rect(m.l, m.t, iw, ih); ctx.clip();
      ctx.globalAlpha = n > 3000 ? 0.55 : 0.82;
      for (let i = 0; i < n; i++) {
        const xv = cfg.xs[i], yv = cfg.ys[i];
        if (xv !== xv || yv !== yv) continue;
        if (cfg.hidden && cfg.hidden(i)) continue;
        const px = X(xv + jx[i] * jit * jxs), py = Y(yv + jy[i] * jit * jys);
        if (px < m.l - 5 || px > m.l + iw + 5 || py < m.t - 5 || py > m.t + ih + 5) continue;
        ctx.fillStyle = cfg.colors[i];
        ctx.beginPath();
        if (cfg.marks && cfg.marks[i]) { ctx.rect(px - size, py - size, size * 2, size * 2); }
        else ctx.arc(px, py, size, 0, Math.PI * 2);
        ctx.fill();
        pts.push([px, py, i]);
      }
      ctx.restore();
      if (drag && drag.x1 !== undefined) {
        ctx.fillStyle = 'rgba(99,102,241,.15)'; ctx.strokeStyle = App.cssVar('--primary');
        const x = Math.min(drag.x0, drag.x1), y = Math.min(drag.y0, drag.y1), w = Math.abs(drag.x1 - drag.x0), hh = Math.abs(drag.y1 - drag.y0);
        ctx.fillRect(x, y, w, hh); ctx.strokeRect(x, y, w, hh);
      }
      wrap._inv = { X: (px) => v.x[0] + ((px - m.l) / iw) * (v.x[1] - v.x[0]), Y: (py) => v.y[0] + ((m.t + ih - py) / ih) * (v.y[1] - v.y[0]) };
    }
    const pos = (e) => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    canvas.addEventListener('mousedown', (e) => { const [x, y] = pos(e); drag = { x0: x, y0: y }; });
    window.addEventListener('mouseup', function up(e) {
      if (!document.body.contains(canvas)) { window.removeEventListener('mouseup', up); return; }
      if (!drag) return;
      const d0 = drag; drag = null;
      if (d0.x1 !== undefined && Math.abs(d0.x1 - d0.x0) > 6 && Math.abs(d0.y1 - d0.y0) > 6) {
        const inv = wrap._inv;
        const xa = [inv.X(Math.min(d0.x0, d0.x1)), inv.X(Math.max(d0.x0, d0.x1))], ya = [inv.Y(Math.max(d0.y0, d0.y1)), inv.Y(Math.min(d0.y0, d0.y1))];
        view = { x: xa, y: ya };
        resetBtn.classList.remove('hidden');
      }
      draw();
    });
    canvas.addEventListener('mousemove', (e) => {
      const [x, y] = pos(e);
      if (drag) { drag.x1 = x; drag.y1 = y; draw(); C.tip(null); return; }
      let best = null, bd = 64;
      for (const p of pts) { const dd = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (dd < bd) { bd = dd; best = p; } }
      if (best && cfg.info) C.tip(cfg.info(best[2]), e.clientX, e.clientY); else C.tip(null);
      canvas.style.cursor = best && cfg.onClick ? 'pointer' : 'crosshair';
    });
    canvas.addEventListener('mouseleave', () => C.tip(null));
    canvas.addEventListener('click', (e) => {
      if (!cfg.onClick) return;
      const [x, y] = pos(e);
      let best = null, bd = 64;
      for (const p of pts) { const dd = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (dd < bd) { bd = dd; best = p; } }
      if (best) cfg.onClick(best[2]);
    });
    const resetBtn = h('button.btn.sm.hidden', { style: { position: 'absolute', right: '20px', top: '18px' }, on: { click: () => { view = null; resetBtn.classList.add('hidden'); draw(); } } }, L`Επαναφορά zoom`);
    wrap.appendChild(resetBtn);
    let lastW = -1;
    const ro = new ResizeObserver(() => requestAnimationFrame(() => { const w = wrap.clientWidth; if (w !== lastW) { lastW = w; draw(); } }));
    ro.observe(wrap);
    wrap.redraw = (newCfg) => { if (newCfg) Object.assign(cfg, newCfg); draw(); };
    App.on('theme', () => { if (document.body.contains(wrap)) draw(); });
    setTimeout(draw, 0);
    return wrap;
  };

  /** Μικρό scatter σε canvas (για scatter matrix), χωρίς άξονες. */
  C.miniScatter = function (canvas, xs, ys, colors, xCat, yCat) {
    const dpr = window.devicePixelRatio || 1;
    const Wd = canvas.clientWidth || 100, Ht = canvas.clientHeight || 100;
    canvas.width = Wd * dpr; canvas.height = Ht * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const ext = (arr, cat) => { if (cat) return [-0.6, cat - 0.4]; let mn = Infinity, mx = -Infinity; for (const v of arr) if (v === v) { if (v < mn) mn = v; if (v > mx) mx = v; } if (!isFinite(mn)) return [0, 1]; if (mn === mx) { mn -= 1; mx += 1; } return [mn, mx]; };
    const ex = ext(xs, xCat), ey = ext(ys, yCat), p = 4;
    const rnd = new W.JavaRandom(3);
    ctx.globalAlpha = xs.length > 2000 ? 0.45 : 0.75;
    for (let i = 0; i < xs.length; i++) {
      const x = xs[i], y = ys[i];
      if (x !== x || y !== y) { rnd.nextDouble(); rnd.nextDouble(); continue; }
      const jx = xCat ? (rnd.nextDouble() - 0.5) * 0.5 : 0, jy = yCat ? (rnd.nextDouble() - 0.5) * 0.5 : 0;
      ctx.fillStyle = colors[i];
      ctx.fillRect(p + ((x + jx - ex[0]) / (ex[1] - ex[0])) * (Wd - 2 * p) - 1.2, Ht - p - ((y + jy - ey[0]) / (ey[1] - ey[0])) * (Ht - 2 * p) - 1.2, 2.4, 2.4);
    }
  };

  // ------------------------------------------------------------ tree
  /** Σχεδίαση δέντρου (J48/RandomTree/REPTree): graph = {leaf,label,children:[{edge,node}]} */
  C.tree = function (graph, classColors) {
    const nodeW = 132, nodeH = 40, gapX = 18, gapY = 74;
    // layout: Reingold–Tilford απλοποιημένο (θέσεις φύλλων διαδοχικά)
    let nextX = 0, maxDepth = 0;
    const layout = (n, depth) => {
      n._d = depth; maxDepth = Math.max(maxDepth, depth);
      if (n.leaf || !n.children || !n.children.length) { n._x = nextX; nextX += nodeW + gapX; return; }
      n.children.forEach((c) => layout(c.node, depth + 1));
      n._x = (n.children[0].node._x + n.children[n.children.length - 1].node._x) / 2;
    };
    layout(graph, 0);
    const Wd = Math.max(nextX, nodeW + 20) + 20, Ht = (maxDepth + 1) * (nodeH + gapY) + 20;
    const s = svg(Wd, Ht);
    s.setAttribute('width', Wd); s.setAttribute('height', Ht); s.style.width = Wd + 'px'; s.style.maxWidth = 'none';
    s.style.minWidth = Math.min(Wd, 3000) + 'px';
    const gE = el('g', {}, s), gN = el('g', {}, s);
    const cx = (n) => n._x + nodeW / 2 + 10, cy = (n) => n._d * (nodeH + gapY) + 10;
    const draw = (n) => {
      const x = cx(n), y = cy(n);
      if (n.children && !n.leaf) {
        n.children.forEach((c) => {
          const ch = c.node, x2 = cx(ch), y2 = cy(ch);
          const midY = y + nodeH + (gapY / 2);
          el('path', { d: `M${x},${y + nodeH} C${x},${midY} ${x2},${midY} ${x2},${y2}`, fill: 'none', stroke: 'var(--border-strong)', 'stroke-width': 1.5 }, gE);
          const lx = (x + x2) / 2, ly = midY;
          const lbl = c.edge.length > 22 ? c.edge.slice(0, 21) + '…' : c.edge;
          const tw = Math.min(150, lbl.length * 6.3 + 12);
          el('rect', { x: lx - tw / 2, y: ly - 10, width: tw, height: 19, rx: 9, fill: 'var(--surface)', stroke: 'var(--border)' }, gE);
          const t = text(gE, lx, ly + 4, lbl, { 'text-anchor': 'middle', style: 'font-size:10.5px;fill:var(--text-2)' });
          el('title', {}, t).textContent = c.edge;
          draw(ch);
        });
      }
      const g = el('g', {}, gN);
      if (n.leaf || !n.children) {
        const col = classColors ? classColors(n.cls) : 'var(--primary)';
        el('rect', { x: x - nodeW / 2, y, width: nodeW, height: nodeH, rx: 10, fill: `color-mix(in srgb, ${col} 18%, var(--surface))`, stroke: col, 'stroke-width': 1.5 }, g);
        const lbl = n.label.length > 22 ? n.label.slice(0, 21) + '…' : n.label;
        text(g, x, y + nodeH / 2 + 4, lbl, { 'text-anchor': 'middle', style: 'font-size:11px;font-weight:600;fill:var(--text)' });
      } else {
        el('ellipse', { cx: x, cy: y + nodeH / 2, rx: nodeW / 2, ry: nodeH / 2, fill: 'var(--surface)', stroke: 'var(--primary)', 'stroke-width': 1.6 }, g);
        const lbl = n.label.length > 20 ? n.label.slice(0, 19) + '…' : n.label;
        text(g, x, y + nodeH / 2 + 4, lbl, { 'text-anchor': 'middle', style: 'font-size:11.5px;font-weight:650;fill:var(--text)' });
      }
      el('title', {}, g).textContent = n.label;
    };
    draw(graph);
    return s;
  };

  // ------------------------------------------------------ dendrogram
  C.dendrogram = function (root, k, labelFn, colorOf) {
    // συλλογή φύλλων
    const leaves = [];
    const walk = (n) => { if (n.leaf) leaves.push(n); else { walk(n.left); walk(n.right); } };
    walk(root);
    const n = leaves.length;
    const Wd = Math.max(600, Math.min(2400, n * 9 + 80)), Ht = 380, m = { l: 50, r: 16, t: 14, b: n <= 60 ? 60 : 20 };
    const s = svg(Wd, Ht);
    s.setAttribute('width', Wd); s.setAttribute('height', Ht); s.style.width = Wd + 'px'; s.style.maxWidth = 'none';
    const iw = Wd - m.l - m.r, ih = Ht - m.t - m.b;
    const maxH = root.height || 1;
    leaves.forEach((l, i) => { l._x = m.l + (n === 1 ? iw / 2 : (i / (n - 1)) * iw); });
    const Y = (hh) => m.t + ih - (hh / maxH) * ih;
    for (const t of ticks(0, maxH, 5)) { el('line', { x1: m.l, x2: m.l + iw, y1: Y(t), y2: Y(t), stroke: 'var(--border)' }, s); text(s, m.l - 6, Y(t) + 4, shortNum(t), { 'text-anchor': 'end' }); }
    // κατώφλι τομής για k ομάδες
    const heights = [];
    const collectH = (nd) => { if (!nd.leaf) { heights.push(nd.height); collectH(nd.left); collectH(nd.right); } };
    collectH(root);
    heights.sort((a, b) => b - a);
    const cut = k > 1 && heights.length >= k - 1 ? (heights[k - 2] + (heights[k - 1] || 0)) / 2 : null;
    const pos = (nd) => {
      if (nd.leaf) { nd._y = Y(0); return nd._x; }
      const a = pos(nd.left), b = pos(nd.right);
      nd._x = (a + b) / 2; nd._y = Y(nd.height);
      const col = colorOf(nd);
      el('path', { d: `M${a},${nd.left._y}V${nd._y}H${b}V${nd.right._y}`, fill: 'none', stroke: col, 'stroke-width': 1.4 }, s);
      return nd._x;
    };
    pos(root);
    if (cut !== null) { el('line', { x1: m.l, x2: m.l + iw, y1: Y(cut), y2: Y(cut), stroke: 'var(--danger)', 'stroke-dasharray': '5 4' }, s); text(s, m.l + iw, Y(cut) - 5, L`τομή για ` + k + L` ομάδες`, { 'text-anchor': 'end', style: 'fill:var(--danger);font-size:10.5px' }); }
    if (n <= 60) leaves.forEach((l) => { const t = text(s, l._x, Ht - m.b + 10, labelFn(l), { 'text-anchor': 'end', transform: `rotate(-60 ${l._x} ${Ht - m.b + 10})`, style: 'font-size:9.5px' }); });
    return s;
  };
})();
