/* Weka Web — "Python / Colab" page: run Colab notebooks (OR-Tools CP-SAT, Pyro, matplotlib) in the browser,
 * plan tasks visually, and convert algorithms between Weka Web and Python / Google Colab in both directions. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, App = window.App, h = App.h;
  const ps = { tab: 'notebook', cells: [], fileName: null, interp: null, counter: 0,
    planner: { tasks: [{ name: 'Design', d: 3, pred: '', sd: 0.5 }, { name: 'Build', d: 5, pred: 'Design', sd: 1 }, { name: 'Docs', d: 2, pred: 'Design', sd: 0.3 }, { name: 'Test', d: 2, pred: 'Build', sd: 0.4 }], workers: 1, deadline: 14, out: null },
    exportSel: null, pyIn: '', pyDetect: null };

  // ------------------------------------------------------------------ examples (written for Weka Web)
  const md = (s) => ({ type: 'markdown', src: s });
  const code = (s) => ({ type: 'code', src: s });
  const EXAMPLES = [
    { id: 'precedence', title: L`Precedence: two parts, then assembly`, cells: [
      md('# Precedence constraints\nTwo parts can be made in parallel; the assembly can start only when both are ready. CP-SAT finds the earliest finish.'),
      code("from ortools.sat.python import cp_model\n\nmodel = cp_model.CpModel()\ntime_needed = {'frame': 4, 'wheels': 3, 'assembly': 2}\n\nstart = {job: model.NewIntVar(0, 30, f'start_{job}') for job in time_needed}\nend = {job: model.NewIntVar(0, 30, f'end_{job}') for job in time_needed}\nfor job, d in time_needed.items():\n    model.Add(end[job] == start[job] + d)\n\n# the assembly waits for both parts\nmodel.Add(end['frame'] <= start['assembly'])\nmodel.Add(end['wheels'] <= start['assembly'])\n\nmodel.Minimize(end['assembly'])\nsolver = cp_model.CpSolver()\nstatus = solver.Solve(model)\nprint(solver.StatusName(status))\nfor job in time_needed:\n    print(f'{job:>9}: {solver.Value(start[job])} → {solver.Value(end[job])}')\nprint('Bike ready at t =', solver.Value(end['assembly']))") ] },
    { id: 'one-worker', title: L`One worker: no overlapping tasks`, cells: [
      md('# A single worker\nInterval variables plus `AddNoOverlap` forbid two tasks at the same time, so the parts are made one after the other.'),
      code("from ortools.sat.python import cp_model\n\nmodel = cp_model.CpModel()\ntime_needed = {'frame': 4, 'wheels': 3, 'assembly': 2}\nstart = {j: model.NewIntVar(0, 30, f'start_{j}') for j in time_needed}\nwork = {j: model.NewIntervalVar(start[j], d, start[j] + d, f'interval_{j}') for j, d in time_needed.items()}\n\nmodel.Add(start['frame'] + time_needed['frame'] <= start['assembly'])\nmodel.Add(start['wheels'] + time_needed['wheels'] <= start['assembly'])\nmodel.AddNoOverlap(work.values())\n\nfinish = model.NewIntVar(0, 30, 'finish')\nmodel.Add(finish == start['assembly'] + time_needed['assembly'])\nmodel.Minimize(finish)\n\nsolver = cp_model.CpSolver()\nif solver.Solve(model) == cp_model.OPTIMAL:\n    for j in sorted(time_needed, key=lambda j: solver.Value(start[j])):\n        print(f'{j}: {solver.Value(start[j])}–{solver.Value(start[j]) + time_needed[j]}')\n    print('finished at', solver.Value(finish))") ] },
    { id: 'project', title: L`Software release: 6 tasks, one developer`, cells: [
      md('# A small project\nSix tasks, one developer, some tasks depend on others. The makespan is the end of the last task (`AddMaxEquality`).'),
      code("from ortools.sat.python import cp_model\n\ntasks = {'spec': 2, 'backend': 5, 'frontend': 4, 'tests': 3, 'docs': 1, 'release': 1}\nafter = [('spec', 'backend'), ('spec', 'frontend'), ('backend', 'tests'), ('frontend', 'tests'), ('tests', 'release'), ('docs', 'release')]\n\nm = cp_model.CpModel()\nH = sum(tasks.values())\ns = {t: m.NewIntVar(0, H, f'start_{t}') for t in tasks}\niv = {t: m.NewIntervalVar(s[t], d, s[t] + d, f'interval_{t}') for t, d in tasks.items()}\nfor a, b in after:\n    m.Add(s[a] + tasks[a] <= s[b])\nm.AddNoOverlap(list(iv.values()))\nmakespan = m.NewIntVar(0, H, 'makespan')\nm.AddMaxEquality(makespan, [s[t] + d for t, d in tasks.items()])\nm.Minimize(makespan)\n\nsolver = cp_model.CpSolver()\nstatus = solver.Solve(m)\nprint('status:', solver.StatusName(status), '| makespan:', solver.Value(makespan))\nfor t in sorted(tasks, key=lambda t: solver.Value(s[t])):\n    print(f'  {t:<9} {solver.Value(s[t]):>2} → {solver.Value(s[t]) + tasks[t]:>2}')") ] },
    { id: 'two-workers', title: L`Two workers (AddCumulative)`, cells: [
      md('# Two workers\n`AddCumulative` allows at most two tasks at the same time — compare the makespan with the one-worker plan.'),
      code("from ortools.sat.python import cp_model\n\ntasks = {'spec': 2, 'backend': 5, 'frontend': 4, 'tests': 3, 'docs': 1, 'release': 1}\nafter = [('spec', 'backend'), ('spec', 'frontend'), ('backend', 'tests'), ('frontend', 'tests'), ('tests', 'release'), ('docs', 'release')]\n\nfor workers in (1, 2):\n    m = cp_model.CpModel()\n    H = sum(tasks.values())\n    s = {t: m.NewIntVar(0, H, f'start_{t}') for t in tasks}\n    iv = [m.NewIntervalVar(s[t], d, s[t] + d, f'interval_{t}') for t, d in tasks.items()]\n    for a, b in after:\n        m.Add(s[a] + tasks[a] <= s[b])\n    m.AddCumulative(iv, [1] * len(iv), workers)\n    end = m.NewIntVar(0, H, 'end')\n    m.AddMaxEquality(end, [s[t] + d for t, d in tasks.items()])\n    m.Minimize(end)\n    solver = cp_model.CpSolver()\n    solver.Solve(m)\n    print(f'{workers} worker(s): makespan = {solver.Value(end)}')") ] },
    { id: 'uncertain', title: L`Uncertain durations: which order? (Pyro)`, cells: [
      md('# Planning under uncertainty\nA delivery must be packed and then shipped; a report can be written before or after. Durations are random, and a late delivery is penalised. Monte Carlo sampling with Pyro estimates the expected cost of each order.'),
      code("import pyro\nimport pyro.distributions as dist\nimport torch\n\npyro.set_rng_seed(7)\n\ndef durations():\n    pack = pyro.sample('pack', dist.Normal(2.0, 0.4))\n    ship = pyro.sample('ship', dist.Normal(3.0, 0.8))\n    report = pyro.sample('report', dist.Uniform(1.0, 2.0))\n    return pack, ship, report\n\ndef cost(order, n=3000, deadline=5.5):\n    total = []\n    for _ in range(n):\n        pack, ship, report = durations()\n        if order == 'report first':\n            delivered = report + pack + ship\n        else:\n            delivered = pack + ship\n        late = torch.clamp(delivered - deadline, min=0.0)\n        total.append(late)\n    return torch.stack(total).mean().item()\n\nfor order in ['report first', 'report last']:\n    print(f'{order:<13} expected lateness = {cost(order):.3f} h')") ] },
    { id: 'gantt', title: L`Gantt chart with matplotlib`, cells: [
      md('# Drawing the plan\nmatplotlib cells run in Weka Web too: `barh` with `left=` draws a Gantt chart.'),
      code("from ortools.sat.python import cp_model\nimport matplotlib.pyplot as plt\n\njobs = {'cut': 3, 'drill': 2, 'weld': 4, 'paint': 2}\nm = cp_model.CpModel()\ns = {j: m.NewIntVar(0, 20, f'start_{j}') for j in jobs}\nm.AddNoOverlap([m.NewIntervalVar(s[j], d, s[j] + d, f'interval_{j}') for j, d in jobs.items()])\nm.Add(s['cut'] + jobs['cut'] <= s['weld'])\nm.Add(s['weld'] + jobs['weld'] <= s['paint'])\nend = m.NewIntVar(0, 20, 'end')\nm.AddMaxEquality(end, [s[j] + d for j, d in jobs.items()])\nm.Minimize(end)\nsolver = cp_model.CpSolver()\nsolver.Solve(m)\n\norder = sorted(jobs, key=lambda j: solver.Value(s[j]))\nplt.barh(order, [jobs[j] for j in order], left=[solver.Value(s[j]) for j in order])\nplt.gca().invert_yaxis()\nplt.xlabel('hours')\nplt.title(f'Workshop plan (makespan {solver.Value(end)} h)')\nplt.show()") ] },
    { id: 'sklearn', title: L`scikit-learn code → Weka Web`, cells: [
      md('# Python → Weka Web\nCells that use scikit-learn are not executed in the browser; Weka Web recognises the models and offers to run the equivalent Weka algorithm on the same dataset.'),
      code("from sklearn.datasets import load_iris\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.model_selection import cross_val_score\n\nX, y = load_iris(return_X_y=True)\nfor model in [DecisionTreeClassifier(criterion='entropy'), RandomForestClassifier(n_estimators=100), KNeighborsClassifier(n_neighbors=3)]:\n    print(type(model).__name__, cross_val_score(model, X, y, cv=10).mean())") ] },
  ];

  // ------------------------------------------------------------------ running cells
  function interp() { if (!ps.interp) ps.interp = W.Py.create({}); return ps.interp; }
  const isML = (src) => /^\s*(from|import)\s+(sklearn|pandas|seaborn|weka|tensorflow|keras|xgboost)\b/m.test(src);
  function runCell(c) {
    if (c.type !== 'code') return;
    if (isML(c.src)) { c.out = { text: '', events: [], error: null, ml: W.PyBridge.detect(c.src) }; c.n = ++ps.counter; return; }
    const r = W.Py.runCell(interp(), c.src);
    c.out = r; c.n = ++ps.counter;
  }
  function runAll() { ps.interp = null; ps.counter = 0; for (const c of ps.cells) runCell(c); App.go('python'); }

  // ------------------------------------------------------------------ charts
  const SVGNS = 'http://www.w3.org/2000/svg';
  const svg = (tag, attrs = {}, ...kids) => { const e = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); kids.forEach((k) => k && e.appendChild(typeof k === 'string' ? document.createTextNode(k) : k)); return e; };
  function niceTicks(max, n = 8) { const step = Math.max(1, Math.ceil(max / n)); const t = []; for (let x = 0; x <= max; x += step) t.push(x); return t; }
  function gantt(rows, title) {
    const labelW = Math.min(150, 14 + 7.2 * Math.max(...rows.map((r) => r.name.length))), rowH = 28, top = title ? 28 : 8, Wd = 720, plotW = Wd - labelW - 20;
    const tmax = Math.max(1, ...rows.map((r) => r.end));
    const x = (t) => labelW + (t / tmax) * plotW;
    const Hh = top + rows.length * rowH + 30;
    const s = svg('svg', { viewBox: `0 0 ${Wd} ${Hh}`, class: 'py-chart', role: 'img' });
    if (title) s.appendChild(svg('text', { x: Wd / 2, y: 18, 'text-anchor': 'middle', class: 'py-title' }, title));
    for (const t of niceTicks(tmax)) { s.appendChild(svg('line', { x1: x(t), x2: x(t), y1: top - 4, y2: top + rows.length * rowH, class: 'py-grid' })); s.appendChild(svg('text', { x: x(t), y: top + rows.length * rowH + 16, 'text-anchor': 'middle', class: 'py-tick' }, String(t))); }
    rows.forEach((r, i) => {
      const y = top + i * rowH;
      s.appendChild(svg('text', { x: labelW - 8, y: y + rowH / 2 + 4, 'text-anchor': 'end', class: 'py-label' }, r.name));
      const bw = Math.max(2, x(r.end) - x(r.start));
      s.appendChild(svg('rect', { x: x(r.start), y: y + 4, width: bw, height: rowH - 8, rx: 5, fill: App.color(i) }, svg('title', {}, `${r.name}: ${r.start} → ${r.end}`)));
      if (bw > 34) s.appendChild(svg('text', { x: x(r.start) + bw / 2, y: y + rowH / 2 + 4, 'text-anchor': 'middle', class: 'py-bar-label' }, `${r.start}–${r.end}`));
    });
    return s;
  }
  function histogram(values, title, color) {
    const Wd = 340, Hh = 170, pad = 28, n = values.length;
    const mn = Math.min(...values), mx = Math.max(...values), bins = 30, bw = (mx - mn) / bins || 1;
    const counts = new Array(bins).fill(0);
    for (const v of values) counts[Math.min(bins - 1, Math.floor((v - mn) / bw))]++;
    const cmax = Math.max(...counts), mean = values.reduce((a, b) => a + b, 0) / n;
    const s = svg('svg', { viewBox: `0 0 ${Wd} ${Hh}`, class: 'py-chart small' });
    s.appendChild(svg('text', { x: Wd / 2, y: 14, 'text-anchor': 'middle', class: 'py-title' }, title));
    counts.forEach((c, i) => s.appendChild(svg('rect', { x: pad + (i * (Wd - 2 * pad)) / bins, y: Hh - pad - (c / cmax) * (Hh - 2 * pad - 10), width: (Wd - 2 * pad) / bins - 1, height: (c / cmax) * (Hh - 2 * pad - 10), fill: color || 'var(--primary)', opacity: 0.85 })));
    const mxp = pad + ((mean - mn) / (mx - mn || 1)) * (Wd - 2 * pad);
    s.appendChild(svg('line', { x1: mxp, x2: mxp, y1: 20, y2: Hh - pad, class: 'py-mean' }));
    s.appendChild(svg('text', { x: pad, y: Hh - 8, class: 'py-tick' }, U.d2s(mn, 2)));
    s.appendChild(svg('text', { x: Wd - pad, y: Hh - 8, 'text-anchor': 'end', class: 'py-tick' }, U.d2s(mx, 2)));
    s.appendChild(svg('text', { x: Math.min(Wd - 60, mxp + 4), y: 30, class: 'py-tick' }, 'mean ' + U.d2s(mean, 3)));
    return s;
  }
  function figure(fig) {
    const ser = fig.series;
    if (ser.length === 1 && ser[0].kind === 'barh' && ser[0].left) {
      const rows = ser[0].y.map((name, i) => ({ name: String(name), start: +ser[0].left[i], end: +ser[0].left[i] + +ser[0].w[i] }));
      return gantt(fig.invertY ? rows : rows.slice().reverse(), fig.title);
    }
    if (ser.every((x) => x.kind === 'hist')) return histogram(ser[0].data, fig.title || 'histogram');
    // generic x/y plot (line, scatter, bar, barh without left)
    const Wd = 560, Hh = 260, pad = 40;
    const xs = [], ys = [];
    ser.forEach((x) => { if (x.kind === 'line' || x.kind === 'scatter') { xs.push(...x.x.map(Number)); ys.push(...x.y.map(Number)); } else if (x.kind === 'bar') { xs.push(0, x.x.length); ys.push(0, ...x.h.map(Number)); } else if (x.kind === 'barh') { xs.push(0, ...x.w.map(Number)); ys.push(0, x.y.length); } });
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(0, ...ys), y1 = Math.max(...ys);
    const X = (v) => pad + ((v - x0) / (x1 - x0 || 1)) * (Wd - 2 * pad), Y = (v) => Hh - pad - ((v - y0) / (y1 - y0 || 1)) * (Hh - 2 * pad);
    const s = svg('svg', { viewBox: `0 0 ${Wd} ${Hh}`, class: 'py-chart' });
    if (fig.title) s.appendChild(svg('text', { x: Wd / 2, y: 16, 'text-anchor': 'middle', class: 'py-title' }, fig.title));
    s.appendChild(svg('line', { x1: pad, x2: Wd - pad, y1: Hh - pad, y2: Hh - pad, class: 'py-axis' }));
    s.appendChild(svg('line', { x1: pad, x2: pad, y1: pad / 2, y2: Hh - pad, class: 'py-axis' }));
    ser.forEach((x, k) => {
      const col = typeof x.color === 'string' ? x.color : App.color(k);
      if (x.kind === 'line') s.appendChild(svg('polyline', { points: x.x.map((v, i) => X(+v) + ',' + Y(+x.y[i])).join(' '), fill: 'none', stroke: col, 'stroke-width': 2 }));
      if (x.kind === 'scatter') x.x.forEach((v, i) => s.appendChild(svg('circle', { cx: X(+v), cy: Y(+x.y[i]), r: 3.5, fill: col })));
      if (x.kind === 'bar') x.h.forEach((v, i) => { const bw = (Wd - 2 * pad) / x.x.length * 0.7; s.appendChild(svg('rect', { x: X(i + 0.15), y: Y(+v), width: bw, height: Y(0) - Y(+v), fill: col })); s.appendChild(svg('text', { x: X(i + 0.5), y: Hh - pad + 14, 'text-anchor': 'middle', class: 'py-tick' }, String(x.x[i]))); });
    });
    if (fig.xlabel) s.appendChild(svg('text', { x: Wd / 2, y: Hh - 6, 'text-anchor': 'middle', class: 'py-tick' }, fig.xlabel));
    return s;
  }

  // ------------------------------------------------------------------ outputs
  function outputView(c) {
    const o = c.out;
    const box = h('div.py-out');
    if (!o) return c.saved ? h('div.py-saved', h('div.py-saved-h', L`Saved output (from the notebook file)`), h('pre', c.saved)) : null;
    if (o.ml) {
      const d = o.ml;
      box.appendChild(h('div.callout', App.icon('info'), h('div',
        h('div', L`This cell uses scikit-learn / pandas, which run in Colab. Weka Web recognised:`),
        d.algorithms.length ? h('div.py-ml', d.algorithms.map((a) => h('button.btn.sm', { on: { click: () => App.py.runInWeka(a, d.dataset) } }, App.icon('play'), `${a.source} → ${a.name}` + (Object.keys(a.opts).length ? ' ' + W.optionString(W.findClassifier(a.name) || W.findClusterer(a.name), Object.assign(W.defaultOptions(W.findClassifier(a.name) || W.findClusterer(a.name)), a.opts)) : '')))) : h('div.muted.small', L`no known algorithms`),
        d.dataset ? h('div.small', { style: { marginTop: '6px' } }, L`Dataset: ${d.dataset} (loaded automatically when you run it).`) : null)));
    }
    if (o.text) box.appendChild(h('pre.py-stdout', o.text));
    if (o.error) box.appendChild(h('pre.py-err', `${o.error.line ? L`Line ${o.error.line}: ` : ''}${o.error.type}: ${o.error.message}`));
    for (const e of o.events || []) {
      if (e.type === 'cpsat') {
        const rows = W.Py.ganttRows(e);
        const head = h('div.py-ev-head', h('span.badge.' + (e.status === 'OPTIMAL' ? 'success' : e.status === 'FEASIBLE' ? 'primary' : 'danger'), 'CP-SAT · ' + e.status),
          e.solver.res && e.solver.res.objective !== undefined && e.model.objective ? h('span.muted.small', L`objective = ${e.solver.res.objective}`) : null,
          h('span.muted.small', `${e.solver.res.nodes} ${L`search nodes`} · ${(e.solver.res.wall * 1000).toFixed(0)} ms`),
          W.Py.tasksFromModel(e.model) ? h('button.btn.ghost.sm', { on: { click: () => { const t = W.Py.tasksFromModel(e.model); ps.planner.tasks = t.tasks.map((x) => ({ name: x.name, d: x.d, pred: x.pred.join(', '), sd: 0 })); ps.planner.workers = t.oneWorker ? 1 : 0; ps.planner.out = null; ps.tab = 'planner'; App.go('python'); } } }, App.icon('edit'), L`Edit in planner`) : null);
        box.appendChild(h('div.py-ev', head, rows.length ? gantt(rows) : null));
      } else if (e.type === 'samples') {
        const wrap = h('div.py-hists');
        let k = 0;
        for (const [name, vals] of e.sites) wrap.appendChild(histogram(vals, `${name} · ${vals.length} ${L`samples`}`, App.color(k++)));
        box.appendChild(h('div.py-ev', h('div.py-ev-head', h('span.badge.primary', 'Pyro · Monte Carlo')), wrap));
      } else if (e.type === 'figure') box.appendChild(h('div.py-ev', h('div.py-ev-head', h('span.badge', 'matplotlib')), figure(e.fig)));
    }
    if (c.saved) box.appendChild(h('details.py-saved', h('summary', L`Compare with the saved Colab output`), h('pre', c.saved)));
    return box;
  }

  // ------------------------------------------------------------------ notebook tab
  function cellView(c, i) {
    if (c.type === 'markdown') {
      const div = h('div.py-md');
      for (const line of c.src.split('\n')) {
        const m = line.match(/^(#{1,3})\s+(.*)/);
        if (m) div.appendChild(h('h' + (m[1].length + 2), m[2]));
        else if (line.trim()) div.appendChild(h('p', ...line.split(/(`[^`]+`)/).map((part) => (/^`.*`$/.test(part) ? h('code', part.slice(1, -1)) : part))));
      }
      return h('div.py-cell.md', div, h('div.py-cell-tools', h('button.btn.ghost.icon.sm', { title: L`Delete`, on: { click: () => { ps.cells.splice(i, 1); App.go('python'); } } }, App.icon('trash'))));
    }
    const ta = h('textarea.py-code', { spellcheck: false, rows: Math.max(2, c.src.split('\n').length), on: { input: (e) => { c.src = e.target.value; e.target.rows = Math.max(2, c.src.split('\n').length); } } });
    ta.value = c.src;
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') { e.preventDefault(); const s = ta.selectionStart; ta.value = ta.value.slice(0, s) + '    ' + ta.value.slice(ta.selectionEnd); ta.selectionStart = ta.selectionEnd = s + 4; c.src = ta.value; }
      if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); runCell(c); App.go('python'); }
    });
    return h('div.py-cell',
      h('div.py-cell-head', h('button.btn.primary.icon.sm', { title: L`Run cell (Shift+Enter)`, on: { click: () => { runCell(c); App.go('python'); } } }, App.icon('play')), h('span.py-n', c.n ? `[${c.n}]` : '[ ]'), h('div.spacer'),
        h('button.btn.ghost.icon.sm', { title: L`Delete`, on: { click: () => { ps.cells.splice(i, 1); App.go('python'); } } }, App.icon('trash'))),
      ta, outputView(c));
  }
  function loadCells(cells, fileName) { ps.cells = cells.map((c) => Object.assign({}, c)); ps.fileName = fileName; ps.interp = null; ps.counter = 0; ps.tab = 'notebook'; App.state.forceTop = true; App.go('python'); }
  async function importFile(f) {
    if (!f) f = await App.openFileDialog('.ipynb,.py,.txt');
    if (!f) return;
    try {
      const cells = W.PyBridge.parseNotebook(await f.text(), f.name);
      loadCells(cells, f.name);
      const det = W.PyBridge.detect(cells.map((c) => c.src).join('\n'));
      App.toast(L`Imported ${f.name}: ${cells.length} cells.` + (det.planning ? ' ' + L`Press “Run all” to execute it in the browser.` : ''), 'success', 6000);
    } catch (e) { App.error(e); }
  }
  function exportNotebook(fmt) {
    const cells = ps.cells.map((c) => (c.type === 'markdown' ? { md: c.src } : { code: c.src }));
    const base = (ps.fileName || 'weka-web-notebook').replace(/\.(ipynb|py)$/i, '');
    if (fmt === 'py') App.download(base + '.py', W.PyBridge.script(cells), 'text/x-python');
    else App.download(base + '.ipynb', W.PyBridge.notebook(cells, base + '.ipynb'), 'application/json');
  }
  function notebookTab() {
    const exSel = h('select.select', { style: { width: 'auto', minWidth: '200px' }, on: { change: (e) => { const ex = EXAMPLES.find((x) => x.id === e.target.value); if (ex) { loadCells(ex.cells, ex.id + '.ipynb'); runAll(); } } } },
      h('option', { value: '' }, L`Examples…`), EXAMPLES.map((x) => h('option', { value: x.id }, x.title)));
    const bar = h('div.row.py-bar',
      h('button.btn', { on: { click: () => importFile() } }, App.icon('upload'), L`Import .ipynb / .py`), exSel,
      h('button.btn.primary', { disabled: !ps.cells.length, on: { click: runAll } }, App.icon('play'), L`Run all`),
      h('button.btn', { on: { click: () => { ps.cells.push({ type: 'code', src: '' }); App.go('python'); } } }, App.icon('plus'), L`Cell`),
      h('div.spacer'),
      h('button.btn', { disabled: !ps.cells.length, on: { click: () => exportNotebook('ipynb') } }, App.icon('download'), '.ipynb'),
      h('button.btn', { disabled: !ps.cells.length, on: { click: () => exportNotebook('py') } }, App.icon('download'), '.py'));
    const body = h('div');
    if (!ps.cells.length) body.appendChild(h('div.empty', App.icon('code'), h('h3', L`Run Colab notebooks in the browser`),
      h('p', L`Import a .ipynb from Google Colab or Jupyter, or open an example. OR-Tools CP-SAT, Pyro, torch, numpy, math, random and matplotlib code runs here unchanged; pip installs are skipped.`)));
    else {
      if (ps.fileName) body.appendChild(h('div.muted.small', { style: { margin: '0 0 10px' } }, App.icon('text'), ' ' + ps.fileName));
      ps.cells.forEach((c, i) => body.appendChild(cellView(c, i)));
    }
    // drop a notebook onto the page
    body.addEventListener('dragover', (e) => e.preventDefault());
    body.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) { e.preventDefault(); importFile(f); } });
    return h('div', bar, body);
  }

  // ------------------------------------------------------------------ planner tab
  function plannerTasks() {
    const names = new Set();
    return ps.planner.tasks.filter((t) => t.name.trim()).map((t) => {
      const name = t.name.trim();
      if (names.has(name)) throw new Error(L`Duplicate task name: ${name}`);
      names.add(name);
      const d = Math.round(+t.d);
      if (!(d >= 0)) throw new Error(L`Task “${name}” needs a non-negative whole duration.`);
      return { name, d, pred: String(t.pred || '').split(/[,;]/).map((x) => x.trim()).filter(Boolean), sd: Math.max(0, +t.sd || 0) };
    }).map((t, _, all) => { t.pred.forEach((p) => { if (!all.some((x) => x.name === p)) throw new Error(L`Unknown predecessor “${p}” of “${t.name}”.`); }); return t; });
  }
  function solvePlanner() {
    try {
      const tasks = plannerTasks();
      const g = W.PyBridge.plannerCode({ tasks, workers: +ps.planner.workers, deadline: ps.planner.deadline ? +ps.planner.deadline : null });
      const it = W.Py.create({});
      ps.planner.out = g.cells.filter((c) => c.code && !/^\s*!/.test(c.code)).map((c) => Object.assign({ type: 'code', src: c.code }, { out: W.Py.runCell(it, c.code) }));
      ps.planner.gen = g;
    } catch (e) { App.error(e); }
    App.go('python');
  }
  App.py = App.py || {};
  /** Used by the assistant: fill the planner and solve. */
  App.py.plan = (tasks, workers, deadline) => { ps.planner.tasks = tasks.map((t) => ({ name: t.name, d: t.d, pred: (t.pred || []).join(', '), sd: t.sd || 0 })); if (workers !== undefined) ps.planner.workers = workers; if (deadline !== undefined) ps.planner.deadline = deadline; ps.tab = 'planner'; App.state.forceTop = true; solvePlanner(); return ps.planner.out; };
  function plannerTab() {
    const P_ = ps.planner;
    const row = (t, i) => h('tr',
      h('td', h('input.input', { value: t.name, on: { input: (e) => { t.name = e.target.value; } } })),
      h('td', h('input.input', { type: 'number', min: 0, value: t.d, style: { width: '80px' }, on: { input: (e) => { t.d = e.target.value; } } })),
      h('td', h('input.input', { value: t.pred, placeholder: L`e.g. Design, Build`, on: { input: (e) => { t.pred = e.target.value; } } })),
      h('td', h('input.input', { type: 'number', min: 0, step: 0.1, value: t.sd, style: { width: '80px' }, on: { input: (e) => { t.sd = e.target.value; } } })),
      h('td', h('button.btn.ghost.icon.sm', { title: L`Delete`, on: { click: () => { P_.tasks.splice(i, 1); App.go('python'); } } }, App.icon('trash'))));
    const tbl = h('div.table-wrap', h('table.tbl.py-tasks', h('thead', h('tr', h('th', L`Task`), h('th', L`Duration`), h('th', L`Must start after`), h('th', L`Uncertainty (sd)`), h('th', ''))), h('tbody', P_.tasks.map(row))));
    const workers = h('select.select', { style: { width: 'auto' }, on: { change: (e) => { P_.workers = +e.target.value; } } },
      [[1, L`One worker (no overlap)`], [2, L`Two workers`], [3, L`Three workers`], [0, L`Unlimited (only precedences)`]].map(([v, l]) => h('option', { value: v, selected: +P_.workers === v }, l)));
    const left = h('div.card', App.cardHead('list', L`Tasks`), h('div.card-body', tbl,
      h('div.row', { style: { marginTop: '10px', flexWrap: 'wrap', gap: '10px' } },
        h('button.btn.sm', { on: { click: () => { P_.tasks.push({ name: 'Task' + (P_.tasks.length + 1), d: 1, pred: '', sd: 0 }); App.go('python'); } } }, App.icon('plus'), L`Add task`),
        h('label.field-label', L`Resources`), workers,
        h('label.field-label', L`Deadline`), h('input.input', { type: 'number', value: P_.deadline ?? '', style: { width: '90px' }, on: { input: (e) => { P_.deadline = e.target.value; } } })),
      h('div.row', { style: { marginTop: '14px', gap: '8px', flexWrap: 'wrap' } },
        h('button.btn.primary', { on: { click: solvePlanner } }, App.icon('play'), L`Solve (OR-Tools CP-SAT)`),
        h('button.btn', { on: { click: () => { try { const g = W.PyBridge.plannerCode({ tasks: plannerTasks(), workers: +P_.workers, deadline: P_.deadline ? +P_.deadline : null }); App.download('automated-planning.ipynb', g.ipynb, 'application/json'); } catch (e) { App.error(e); } } } }, App.icon('download'), L`Colab notebook (.ipynb)`),
        h('button.btn', { on: { click: () => { try { const g = W.PyBridge.plannerCode({ tasks: plannerTasks(), workers: +P_.workers, deadline: P_.deadline ? +P_.deadline : null }); App.download('automated-planning.py', g.py, 'text/x-python'); } catch (e) { App.error(e); } } } }, App.icon('download'), '.py'),
        h('button.btn.ghost', { on: { click: () => { try { const g = W.PyBridge.plannerCode({ tasks: plannerTasks(), workers: +P_.workers, deadline: P_.deadline ? +P_.deadline : null }); loadCells(g.cells.map((c) => (c.md ? md(c.md) : code(c.code))), 'automated-planning.ipynb'); runAll(); } catch (e) { App.error(e); } } } }, App.icon('text'), L`Open as notebook`)),
      h('div.callout', { style: { marginTop: '14px' } }, App.icon('info'), h('div', L`The planner writes real OR-Tools code (and Pyro code when a task has uncertainty). The same code runs here and in Google Colab.`))));
    const out = h('div');
    if (P_.out) P_.out.forEach((c) => { const v = outputView(c); if (v) out.appendChild(h('div.card', h('div.card-body', v))); });
    return h('div', left, out);
  }

  // ------------------------------------------------------------------ convert tab
  App.py.runInWeka = async function (a, dataset) {
    try {
      if (dataset && (!App.state.data || App.state.data.relation.toLowerCase() !== dataset.toLowerCase())) await App.openSample(dataset);
      if (!App.state.data) { App.toast(L`Load a dataset first.`, 'error'); return; }
      const spec = a.kind === 'clusterer' ? W.findClusterer(a.name) : W.findClassifier(a.name);
      const opts = Object.assign(W.defaultOptions(spec), a.opts);
      App.store.set(a.kind === 'clusterer' ? 'clusterer' : 'classifier', { name: spec.name, opts });
      App.state.forceTop = true;
      App.go(a.kind === 'clusterer' ? 'cluster' : 'classify');
      await new Promise((r) => setTimeout(r, 60));
      const btn = document.querySelector('#main button.btn.primary.lg');
      if (btn) btn.click();
    } catch (e) { App.error(e); }
  };
  function sources() {
    const out = [];
    App.state.results.classify.forEach((r, i) => out.push({ id: 'c' + i, label: `${r.name} · ${r.relation} · ${r.res.label}`, src: { spec: r.res.spec, options: r.res.options, data: r.data, kind: 'classifier', folds: r.res.folds, seed: 1 } }));
    App.state.results.cluster.forEach((r, i) => out.push({ id: 'k' + i, label: `${r.name} · ${r.relation}`, src: { spec: r.res.spec, options: r.res.options, data: r.data, kind: 'clusterer' } }));
    const d = App.state.data;
    if (d) {
      const sc = App.store.get('classifier', null), spec = W.findClassifier((sc && sc.name) || 'J48');
      out.push({ id: 'cur', label: L`Current Classify settings: ${spec.name} on ${d.relation}`, src: { spec, options: Object.assign(W.defaultOptions(spec), sc && sc.name === spec.name ? sc.opts : {}), data: d, kind: 'classifier' } });
    }
    return out;
  }
  App.py.exportResult = function (which = 'last', fmt = 'ipynb') {
    const s = sources();
    let pick = which === 'last' ? s.filter((x) => x.id !== 'cur').pop() || s.find((x) => x.id === 'cur') : s.find((x) => x.id === which);
    if (!pick) throw new Error(L`Run a classifier or load a dataset first.`);
    const nb = W.PyBridge.toNotebook(pick.src);
    App.download(nb.title + (fmt === 'py' ? '.py' : '.ipynb'), fmt === 'py' ? nb.py : nb.ipynb, fmt === 'py' ? 'text/x-python' : 'application/json');
    return nb;
  };
  function convertTab() {
    const s = sources();
    if (!ps.exportSel || !s.some((x) => x.id === ps.exportSel)) ps.exportSel = (s.filter((x) => x.id !== 'cur').pop() || s[0] || {}).id;
    const sel = h('select.select', { on: { change: (e) => { ps.exportSel = e.target.value; App.go('python'); } } }, s.map((x) => h('option', { value: x.id, selected: x.id === ps.exportSel }, x.label)));
    const pick = s.find((x) => x.id === ps.exportSel);
    let preview = null;
    if (pick) { try { const nb = W.PyBridge.toNotebook(pick.src); preview = h('pre.py-preview', nb.py.length > 9000 ? nb.py.slice(0, 9000).replace(/CSV = """[\s\S]*?"""/, 'CSV = """…data…"""') + '\n…' : nb.py.replace(/(CSV|ARFF) = """[\s\S]*?"""/g, '$1 = """…data…"""')); } catch (e) { preview = h('div.callout.warn', App.icon('info'), h('div', e.message)); } }
    const left = h('div.card', App.cardHead('download', L`Weka Web → Python / Colab`), h('div.card-body',
      s.length ? h('div', h('div.field', h('label', L`Algorithm and data to export`), sel),
        h('div.row', { style: { gap: '8px', flexWrap: 'wrap', marginBottom: '12px' } },
          h('button.btn.primary', { on: { click: () => { try { App.py.exportResult(ps.exportSel, 'ipynb'); } catch (e) { App.error(e); } } } }, App.icon('download'), L`Colab notebook (.ipynb)`),
          h('button.btn', { on: { click: () => { try { App.py.exportResult(ps.exportSel, 'py'); } catch (e) { App.error(e); } } } }, App.icon('download'), '.py'),
          h('a.btn.ghost', { href: 'https://colab.research.google.com/', target: '_blank', rel: 'noopener' }, App.icon('external'), L`Open Google Colab`)),
        h('p.muted.small', L`Part 1 uses the closest scikit-learn model; part 2 runs the real Weka algorithm with python-weka-wrapper3, so its numbers match Weka Web exactly. In Colab: File → Upload notebook.`), preview)
        : h('div.muted', L`Load a dataset and run a classifier or clusterer first.`)));
    const ta = h('textarea.py-code', { rows: 9, spellcheck: false, placeholder: L`Paste Python code (scikit-learn or python-weka-wrapper3)…`, on: { input: (e) => { ps.pyIn = e.target.value; } } });
    ta.value = ps.pyIn;
    const det = ps.pyDetect;
    const right = h('div.card', App.cardHead('upload', L`Python / Colab → Weka Web`), h('div.card-body', ta,
      h('div.row', { style: { margin: '10px 0', gap: '8px' } }, h('button.btn.primary', { on: { click: () => { ps.pyDetect = W.PyBridge.detect(ps.pyIn); App.go('python'); } } }, App.icon('search'), L`Find algorithms`),
        h('button.btn', { on: { click: async () => { const f = await App.openFileDialog('.ipynb,.py'); if (!f) return; const t = await f.text(); ps.pyIn = W.PyBridge.parseNotebook(t, f.name).filter((c) => c.type === 'code').map((c) => c.src).join('\n\n'); ps.pyDetect = W.PyBridge.detect(ps.pyIn); App.go('python'); } } }, App.icon('upload'), L`From file`)),
      det ? h('div', det.algorithms.length ? h('div.py-ml', det.algorithms.map((a) => h('button.btn.sm', { on: { click: () => App.py.runInWeka(a, det.dataset) } }, App.icon('play'), `${a.source} → ${a.name}` + (a.exact ? ' (exact)' : '')))) : h('div.muted.small', L`No scikit-learn or Weka algorithms found.`),
        det.dataset ? h('div.small', { style: { marginTop: '8px' } }, L`Dataset: ${det.dataset}`) : null,
        det.planning ? h('div.small', { style: { marginTop: '8px' } }, L`This code also contains planning (OR-Tools / Pyro) — open it on the Notebook tab to run it.`, ' ', h('button.btn.ghost.sm', { on: { click: () => { loadCells([code(ps.pyIn)], 'pasted.py'); runAll(); } } }, L`Run as notebook`)) : null) : null,
      h('div.muted.small', { style: { marginTop: '10px' } }, L`Recognised: DecisionTree, RandomForest, GaussianNB, CategoricalNB, KNeighbors, LogisticRegression, SVC, SGD, Perceptron, MLP, AdaBoost, GradientBoosting, Bagging, Dummy, KMeans, GaussianMixture, Agglomerative, and python-weka-wrapper3 classifiers.`)));
    return h('div.grid.cols-2.py-convert', left, right);
  }

  // ------------------------------------------------------------------ page
  App.pages.python = {
    render() {
      const page = h('div.page', App.pageHead(L`Python / Colab`, L`Automated planning with OR-Tools and Pyro in the browser, and two-way conversion between Weka Web and Python / Google Colab.`));
      const tabs = h('div.tabs', [['notebook', L`Notebook`, 'code'], ['planner', L`Planner`, 'list'], ['convert', L`Weka ⇄ Python`, 'link']].map(([k, l, ic]) =>
        h('button' + (ps.tab === k ? '.active' : ''), { on: { click: () => { ps.tab = k; App.go('python'); } } }, App.icon(ic), l)));
      page.appendChild(h('div.card', { style: { padding: '0' } }, tabs));
      page.appendChild(ps.tab === 'planner' ? plannerTab() : ps.tab === 'convert' ? convertTab() : notebookTab());
      return page;
    },
  };
  Object.assign(App.py, { importFile, runAll, loadExample: (id) => { const ex = EXAMPLES.find((x) => x.id === id) || EXAMPLES[0]; loadCells(ex.cells, ex.id + '.ipynb'); runAll(); return ex; }, examples: EXAMPLES, state: ps, solvePlanner, setTab: (t) => { ps.tab = t; App.go('python'); } });
})();
