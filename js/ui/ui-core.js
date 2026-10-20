/* Weka Web — πυρήνας διεπαφής: κατάσταση, βοηθητικά DOM, εικονίδια, φόρμες, επιλογείς. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils;
  const App = (window.App = {
    state: { data: null, undo: [], redo: [], fileName: null, page: 'home', results: { classify: [], cluster: [], associate: [], select: [] }, testSet: null },
    pages: {},
    listeners: {},
  });

  // ------------------------------------------------------------ events
  App.on = (ev, fn) => { (App.listeners[ev] = App.listeners[ev] || []).push(fn); };
  App.emit = (ev, ...a) => { for (const fn of App.listeners[ev] || []) { try { fn(...a); } catch (e) { console.error(e); } } };

  // ------------------------------------------------------------- DOM helper
  /** h('div.card#id', {attrs, on:{click}}, children...) */
  function h(sel, attrs, ...kids) {
    if (attrs == null || typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs)) { kids.unshift(attrs); attrs = {}; }
    const m = sel.match(/^([a-z0-9-]+)?((?:[.#][\w-]+)*)$/i);
    const el = document.createElement((m && m[1]) || 'div');
    if (m && m[2]) for (const p of m[2].match(/[.#][\w-]+/g)) { if (p[0] === '.') el.classList.add(p.slice(1)); else el.id = p.slice(1); }
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === 'on') for (const [e, f] of Object.entries(v)) el.addEventListener(e, f);
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k === 'class') el.className += ' ' + v;
      else if (k === 'html') el.innerHTML = v;
      else if (k in el && typeof v !== 'string') el[k] = v;
      else el.setAttribute(k, v === true ? '' : v);
    }
    append(el, kids);
    return el;
  }
  function append(el, kids) {
    for (const k of kids.flat(Infinity)) {
      if (k === null || k === undefined || k === false) continue;
      el.appendChild(k instanceof Node ? k : document.createTextNode(String(k)));
    }
    return el;
  }
  App.h = h;
  App.clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
  App.esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

  // --------------------------------------------------------------- icons
  const ICONS = {
    home: '<path d="M3 11.5 12 4l9 7.5"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/>',
    data: '<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5v13c0 1.7 3.6 3 8 3s8-1.3 8-3v-13"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    filter: '<path d="M3 5h18l-7 8.5V20l-4-2v-4.5z"/>',
    classify: '<path d="M12 3v6"/><circle cx="12" cy="11" r="2"/><path d="M12 13v2M6 21v-3a3 3 0 0 1 3-3h6a3 3 0 0 1 3 3v3"/><circle cx="6" cy="21" r="0.5"/><path d="M4 21h4M16 21h4"/>',
    tree: '<rect x="9" y="2.5" width="6" height="5" rx="1.2"/><rect x="2.5" y="16.5" width="6" height="5" rx="1.2"/><rect x="15.5" y="16.5" width="6" height="5" rx="1.2"/><path d="M12 7.5v4.5M5.5 16.5V12h13v4.5"/>',
    cluster: '<circle cx="7" cy="8" r="2.4"/><circle cx="5.5" cy="14.5" r="2"/><circle cx="11" cy="12" r="1.7"/><circle cx="17" cy="15" r="2.4"/><circle cx="18.5" cy="8.5" r="1.8"/><circle cx="14" cy="18.5" r="1.6"/>',
    link: '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
    select: '<path d="M4 6h10M4 12h7M4 18h4"/><path d="m15 15 2.5 2.5L22 13"/>',
    chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    scatter: '<path d="M3 3v18h18"/><circle cx="8" cy="15" r="1.4"/><circle cx="12" cy="10" r="1.4"/><circle cx="16" cy="12" r="1.4"/><circle cx="18" cy="6" r="1.4"/><circle cx="10" cy="17" r="1.4"/>',
    table: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9.5h18M3 14.5h18M9 4v16"/>',
    upload: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    download: '<path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/>',
    redo: '<path d="m15 14 5-5-5-5"/><path d="M20 9H10a6 6 0 0 0 0 12h3"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
    play: '<path d="M7 4.5v15l12-7.5z"/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    x: '<path d="M6 6l12 12M18 6 6 18"/>',
    check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
    chevron: '<path d="m6 9 6 6 6-6"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
    text: '<path d="M4 6h16M4 10h16M4 14h10M4 18h7"/>',
    target: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="0.8"/>',
    sparkles: '<path d="M12 3l1.8 4.7L18.5 9.5 13.8 11.3 12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
    layers: '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
    roc: '<path d="M3 21V3M3 21h18"/><path d="M3 21C6 9 11 5 21 3" /><path d="M3 21 21 3" stroke-dasharray="2 3"/>',
    list: '<path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/>',
    grid: '<rect x="3" y="3" width="7.5" height="7.5" rx="1.5"/><rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5"/><rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5"/><rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5"/>',
    edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13.5 6.5 4 4"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
    report: '<path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5"/><path d="M9 13h7M9 17h7M9 9h3"/>',
    video: '<rect x="3" y="5" width="14" height="14" rx="2"/><path d="m17 10 4-2.5v9L17 14"/>',
    gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3z"/>',
    shield: '<path d="M12 3 4.5 6v5.5c0 4.6 3.2 8.3 7.5 9.5 4.3-1.2 7.5-4.9 7.5-9.5V6z"/><path d="m9 12 2 2 4-4"/>',
    chat: '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3h11A2.5 2.5 0 0 1 20 5.5v8a2.5 2.5 0 0 1-2.5 2.5H10l-4.5 4v-4h0A2.5 2.5 0 0 1 4 13.5z"/><path d="M8.5 9.5h.01M12 9.5h.01M15.5 9.5h.01"/>',
    mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7"/>',
    image: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><circle cx="9" cy="10" r="1.8"/><path d="m4 18 5.5-5 4 3.5L16 14l4 4"/>',
    send: '<path d="M4 12 20 4l-6 16-2.5-6.5z"/><path d="m11.5 13.5 3-3"/>',
    external: '<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
    wand: '<path d="m4 20 12-12M14 4l1 2 2 1-2 1-1 2-1-2-2-1 2-1zM19 11l.6 1.4L21 13l-1.4.6L19 15l-.6-1.4L17 13l1.4-.6z"/>',
  };
  App.icon = (name, cls) => {
    const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('width', '18'); s.setAttribute('height', '18'); s.setAttribute('fill', 'none'); s.setAttribute('stroke', 'currentColor');
    s.setAttribute('stroke-width', '1.8'); s.setAttribute('stroke-linecap', 'round'); s.setAttribute('stroke-linejoin', 'round');
    if (cls) s.setAttribute('class', cls);
    s.innerHTML = ICONS[name] || '';
    return s;
  };

  // ------------------------------------------------------------- toasts
  App.toast = function (msg, type = 'info', ms = 4200) {
    let box = document.querySelector('.toasts');
    if (!box) { box = h('div.toasts'); document.body.appendChild(box); }
    const t = h('div.toast.' + type, h('div', msg), h('button.x', { on: { click: () => t.remove() }, 'aria-label': L`Κλείσιμο` }, '×'));
    box.appendChild(t);
    if (ms) setTimeout(() => t.remove(), ms);
  };
  App.error = (e) => { console.error(e); App.toast(e && e.message ? e.message : String(e), 'error', 7000); };

  // ------------------------------------------------------------- modal
  App.modal = function (title, body, buttons = [], wide = false) {
    const back = h('div.modal-back');
    const close = () => back.remove();
    const foot = h('div.modal-foot', buttons.map((b) => h('button.btn' + (b.primary ? '.primary' : ''), { on: { click: () => { if (b.onClick && b.onClick() === false) return; close(); } } }, b.label)));
    const m = h('div.modal' + (wide ? '.wide' : ''), h('div.modal-head', h('h3', title), h('div', { style: { flex: 1 } }), h('button.btn.ghost.icon.sm', { on: { click: close }, 'aria-label': L`Κλείσιμο` }, App.icon('x'))), h('div.modal-body', body), buttons.length ? foot : null);
    back.appendChild(m);
    back.addEventListener('mousedown', (e) => { if (e.target === back) close(); });
    document.body.appendChild(back);
    return { close, el: m };
  };

  // ---------------------------------------------------------- storage
  App.store = {
    get(k, d) { try { const v = localStorage.getItem('wekaweb.' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('wekaweb.' + k, JSON.stringify(v)); } catch (e) { /* ignore */ } },
  };

  // ----------------------------------------------------------- theme
  App.setTheme = (t) => {
    document.documentElement.setAttribute('data-theme', t);
    App.store.set('theme', t);
    App.emit('theme');
  };
  App.toggleTheme = () => App.setTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  App.cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  // ---------------------------------------------------------- formatting
  App.fmt = (v, d = 4) => (v === null || v === undefined || v !== v ? '—' : U.d2s(v, d));
  App.pct = (v, d = 2) => (v !== v ? '—' : U.d2s(v, d) + '%');
  App.typeBadge = (a) => {
    const map = { numeric: ['num', L`Αριθμητικό`], nominal: ['nom', 'Nominal'], string: ['str', 'String'], date: ['date', L`Ημερομηνία`] };
    const [c, l] = map[a.type] || ['', a.type];
    return h('span.badge.' + c, l);
  };
  App.PALETTE = ['#4f46e5', '#f59e0b', '#10b981', '#ef4444', '#06b6d4', '#a855f7', '#84cc16', '#ec4899', '#64748b', '#f97316', '#14b8a6', '#8b5cf6', '#eab308', '#0ea5e9', '#d946ef', '#22c55e', '#fb7185', '#a3a3a3', '#2563eb', '#b45309'];
  App.color = (i) => (i < App.PALETTE.length ? App.PALETTE[i] : `hsl(${(i * 137.5) % 360} 65% 55%)`);

  // ---------------------------------------------------------- data state
  App.setData = function (d, label, fileName) {
    const s = App.state;
    if (s.data) { s.undo.push({ data: s.data, label: s.lastLabel || '' }); if (s.undo.length > 20) s.undo.shift(); }
    s.redo = [];
    s.data = d;
    s.lastLabel = label;
    if (fileName !== undefined) s.fileName = fileName;
    App.emit('data', label);
  };
  App.loadNew = function (d, fileName) {
    const s = App.state;
    s.undo = []; s.redo = []; s.data = d; s.fileName = fileName; s.source = null; s.lastLabel = L`Φόρτωση ` + fileName; s.testSet = null; s.forceTop = true;
    s.ops = [{ label: L`Φόρτωση ` + fileName, time: new Date() }];
    App.emit('data', 'load');
    App.toast(L`Φορτώθηκε το "${d.relation}": ${d.numInstances} εγγραφές, ${d.numAttributes} χαρακτηριστικά.`, 'success');
  };
  App.applyChange = function (d, label) {
    App.setData(d, label);
    App.state.ops = (App.state.ops || []).concat([{ label, time: new Date() }]);
  };
  App.undo = function () {
    const s = App.state;
    if (!s.undo.length) return;
    s.redo.push({ data: s.data, label: s.lastLabel });
    const prev = s.undo.pop();
    s.data = prev.data; s.lastLabel = prev.label;
    if (s.ops && s.ops.length > 1) s.ops = s.ops.slice(0, -1);
    App.emit('data', 'undo');
    App.toast(L`Αναίρεση τελευταίας αλλαγής.`);
  };
  App.redoFn = function () {
    const s = App.state;
    if (!s.redo.length) return;
    s.undo.push({ data: s.data, label: s.lastLabel });
    const nx = s.redo.pop();
    s.data = nx.data; s.lastLabel = nx.label;
    App.emit('data', 'redo');
  };
  App.setClass = function (idx) {
    const d = App.state.data;
    if (!d) return;
    d.classIndex = idx;
    App.emit('class');
  };

  // ------------------------------------------------------------ files
  App.parseFile = async function (file) {
    if (file.size > 60 * 1024 * 1024) throw new Error(L`Το αρχείο είναι πολύ μεγάλο για επεξεργασία στον browser (>60 MB).`);
    const text = await file.text();
    const name = file.name.toLowerCase();
    if (name.endsWith('.csv') || name.endsWith('.tsv') || name.endsWith('.txt') && !/@relation/i.test(text.slice(0, 2000))) return W.parseCSV(text, file.name);
    if (name.endsWith('.json')) return parseJSON(text, file.name);
    return W.parseARFF(text);
  };
  function parseJSON(text, name) {
    const arr = JSON.parse(text);
    if (!Array.isArray(arr) || !arr.length || typeof arr[0] !== 'object') throw new Error(L`Το JSON πρέπει να είναι πίνακας αντικειμένων.`);
    const keys = Object.keys(arr[0]);
    const csv = [keys.join(',')].concat(arr.map((o) => keys.map((k) => { const v = o[k]; const s = v === null || v === undefined ? '?' : String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }).join(','))).join('\n');
    return W.parseCSV(csv, name);
  }
  App.openFileDialog = function (accept = '.arff,.csv,.tsv,.txt,.json') {
    return new Promise((resolve) => {
      const inp = h('input', { type: 'file', accept, style: { display: 'none' } });
      inp.addEventListener('change', () => { resolve(inp.files[0] || null); inp.remove(); });
      document.body.appendChild(inp);
      inp.click();
    });
  };
  App.openData = async function () {
    const f = await App.openFileDialog();
    if (!f) return;
    try { const d = await App.parseFile(f); App.loadNew(d, f.name); App.go('preprocess'); } catch (e) { App.error(e); }
  };
  App.download = function (name, text, type = 'text/plain') {
    const blob = new Blob([text], { type: type + ';charset=utf-8' });
    const a = h('a', { href: URL.createObjectURL(blob), download: name });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  };
  App.copy = async function (text) {
    try { await navigator.clipboard.writeText(text); App.toast(L`Αντιγράφηκε στο πρόχειρο.`, 'success', 2000); }
    catch (e) { App.toast(L`Δεν ήταν δυνατή η αντιγραφή.`, 'error'); }
  };
  /** Φόρτωση ενσωματωμένου συνόλου (datasets/<id>.js) με script injection — λειτουργεί και από file://. */
  App.loadSample = function (id) {
    return new Promise((resolve, reject) => {
      if (W.datasets[id]) { resolve(W.parseSample(id)); return; }
      const s = h('script', { src: 'datasets/' + id + '.js' });
      s.onload = () => { try { resolve(W.parseSample(id)); } catch (e) { reject(e); } };
      s.onerror = () => reject(new Error(L`Δεν βρέθηκε το σύνολο ` + id));
      document.head.appendChild(s);
    });
  };
  App.openSample = async function (id) {
    try { const d = await App.loadSample(id); App.loadNew(d, id + ((W.datasetFormats || {})[id] === 'csv' ? '.csv' : '.arff')); if (App.state.page === 'home') App.go('preprocess'); } catch (e) { App.error(e); }
  };

  // ------------------------------------------------------- options form
  /** Δημιουργεί φόρμα επιλογών για spec.options. Επιστρέφει {el, get(), set()} */
  App.optionsForm = function (spec, values, onChange) {
    const vals = Object.assign(W.defaultOptions(spec), values || {});
    const el = h('div.opts');
    const fire = () => onChange && onChange(Object.assign({}, vals));
    for (const o of spec.options || []) {
      if (o.hidden) continue;
      if (o.type === 'bool') {
        const inp = h('input', { type: 'checkbox', checked: !!vals[o.key], on: { change: () => { vals[o.key] = inp.checked; fire(); } } });
        el.appendChild(h('label.opt-bool', h('span', o.label), h('span.switch', inp, h('span'))));
        continue;
      }
      let control;
      if (o.type === 'select') {
        const choices = typeof o.choices === 'function' ? o.choices() : o.choices;
        control = h('select.select', { on: { change: () => { vals[o.key] = o.numeric ? Number(control.value) : control.value; fire(); } } },
          Object.entries(choices).map(([k, v]) => h('option', { value: k, selected: String(vals[o.key]) === String(k) }, v)));
      } else {
        const isNum = o.type === 'int' || o.type === 'number';
        control = h('input.input', { type: isNum ? 'number' : 'text', value: vals[o.key] === -1.7976931348623157e308 ? '' : vals[o.key], placeholder: o.placeholder || '', step: o.step || (o.type === 'int' ? 1 : 'any'), min: o.min, max: o.max,
          on: { input: () => {
            if (!isNum) { vals[o.key] = control.value; fire(); return; }
            const v = control.value === '' ? o.default : Number(control.value);
            if (!isNaN(v)) { vals[o.key] = o.type === 'int' ? Math.trunc(v) : v; fire(); }
          } } });
      }
      el.appendChild(h('div.field', h('label', o.label, o.flag ? h('span.muted.small', '  -' + o.flag) : null), control));
    }
    if (!(spec.options || []).length) el.appendChild(h('div.muted.small', L`Ο αλγόριθμος δεν έχει ρυθμίσεις.`));
    return { el, get: () => Object.assign({}, vals) };
  };

  // --------------------------------------------------- algorithm picker
  /**
   * Ευρετήριο αλγορίθμων με αναζήτηση. items: specs, groupLabels: {group: label}
   * onSelect(spec). Επιστρέφει {el, set(name)}
   */
  App.algoPicker = function (items, current, onSelect, groupOf, groupLabels = {}) {
    const wrap = h('div.algo-picker');
    let menu = null;
    const btnTitle = h('div.t'), btnSub = h('div.g');
    const btn = h('button.algo-btn', { type: 'button', on: { click: () => (menu ? closeMenu() : openMenu()) } }, h('div', btnTitle, btnSub), App.icon('chevron'));
    wrap.appendChild(btn);
    let cur = current;
    const setCur = (spec) => { cur = spec; btnTitle.textContent = spec.name; btnSub.textContent = spec.path; };
    setCur(current);
    function closeMenu() { if (menu) { menu.remove(); menu = null; document.removeEventListener('mousedown', outside); } }
    function outside(e) { if (!wrap.contains(e.target)) closeMenu(); }
    function openMenu() {
      const search = h('input.input', { placeholder: L`Αναζήτηση αλγορίθμου…`, on: { input: () => render() } });
      const list = h('div');
      menu = h('div.algo-menu', search, list);
      wrap.appendChild(menu);
      document.addEventListener('mousedown', outside);
      const render = () => {
        App.clear(list);
        const q = search.value.trim().toLowerCase();
        const groups = {};
        for (const s of items) {
          if (q && !(s.name.toLowerCase().includes(q) || (s.desc || '').toLowerCase().includes(q) || s.path.toLowerCase().includes(q))) continue;
          const g = groupOf ? groupOf(s) : '';
          (groups[g] = groups[g] || []).push(s);
        }
        for (const [g, arr] of Object.entries(groups)) {
          if (g) list.appendChild(h('div.algo-group', groupLabels[g] || g));
          for (const s of arr) list.appendChild(h('div.algo-opt' + (s === cur ? '.active' : ''), { on: { click: () => { setCur(s); closeMenu(); onSelect(s); } } }, h('div.n', s.name), h('div.d', s.desc || s.path)));
        }
        if (!list.firstChild) list.appendChild(h('div.muted.small', { style: { padding: '10px' } }, L`Δεν βρέθηκε αλγόριθμος.`));
      };
      render();
      setTimeout(() => search.focus(), 0);
    }
    return { el: wrap, set: setCur };
  };

  /** Κάρτα ρύθμισης αλγορίθμου: picker + περιγραφή + επιλογές + γραμμή εντολών. */
  App.algoConfig = function ({ title, items, groupOf, groupLabels, storeKey, defaultName, onChange }) {
    const saved = App.store.get(storeKey, null);
    let spec = items.find((s) => s.name === (saved && saved.name)) || items.find((s) => s.name === defaultName) || items[0];
    let opts = Object.assign(W.defaultOptions(spec), saved && saved.name === spec.name ? saved.opts : {});
    const desc = h('div.algo-desc');
    const optBox = h('div');
    const cmd = h('div.cmdline');
    const updateCmd = () => { const s = W.optionString(spec, opts); cmd.textContent = spec.path + (s ? ' ' + s : ''); App.store.set(storeKey, { name: spec.name, opts }); onChange && onChange(spec, opts); };
    const renderOpts = () => {
      desc.textContent = spec.desc || '';
      const f = App.optionsForm(spec, opts, (v) => { opts = v; updateCmd(); });
      App.clear(optBox).appendChild(f.el);
      updateCmd();
    };
    const picker = App.algoPicker(items, spec, (s) => { spec = s; opts = W.defaultOptions(s); renderOpts(); }, groupOf, groupLabels);
    const resetBtn = h('button.btn.ghost.sm', { type: 'button', on: { click: () => { opts = W.defaultOptions(spec); renderOpts(); } } }, L`Προεπιλογές`);
    const el = h('div',
      title ? h('div.field-label', { style: { marginBottom: '6px' } }, title) : null,
      picker.el, desc,
      h('div.row', { style: { justifyContent: 'space-between', marginBottom: '6px' } }, h('div.field-label', L`Ρυθμίσεις`), resetBtn),
      optBox,
      h('div.field', { style: { marginTop: '6px', marginBottom: 0 } }, h('label', L`Γραμμή εντολών Weka`), cmd));
    renderOpts();
    return { el, get: () => ({ spec, opts: Object.assign({}, opts) }) };
  };

  // ----------------------------------------------------- class selector
  App.classSelect = function (onChange, opts = {}) {
    const d = App.state.data;
    const sel = h('select.select', { on: { change: () => { App.setClass(Number(sel.value)); onChange && onChange(); } } });
    if (opts.allowNone) sel.appendChild(h('option', { value: -1 }, L`(καμία)`));
    if (d) d.attributes.forEach((a, j) => sel.appendChild(h('option', { value: j, selected: j === d.classIndex }, `(${a.type === 'nominal' ? 'Nom' : a.type === 'numeric' ? 'Num' : a.type}) ${a.name}`)));
    if (d && d.classIndex < 0 && opts.allowNone) sel.value = -1;
    return sel;
  };

  // ------------------------------------------------------------ progress
  App.runner = function () {
    const bar = h('div.progress', h('div'));
    const status = h('div.status-line');
    const el = h('div', bar, status);
    let cancel = false;
    return {
      el,
      start() { cancel = false; bar.classList.add('indeterminate'); bar.firstChild.style.width = ''; status.textContent = L`Εκτέλεση…`; },
      progress(p, msg) { if (cancel) throw new Error(L`Η εκτέλεση διακόπηκε.`); bar.classList.remove('indeterminate'); bar.firstChild.style.width = Math.round(p * 100) + '%'; if (msg) status.textContent = msg; },
      done(msg) { bar.classList.remove('indeterminate'); bar.firstChild.style.width = '100%'; status.textContent = msg || L`Ολοκληρώθηκε`; setTimeout(() => { if (!bar.classList.contains('indeterminate')) bar.firstChild.style.width = '0'; }, 1200); },
      fail(msg) { bar.classList.remove('indeterminate'); bar.firstChild.style.width = '0'; status.textContent = msg || ''; },
      cancel() { cancel = true; },
    };
  };

  // ---------------------------------------------------------- navigation
  App.go = function (page) {
    if (!App.pages[page]) page = 'home';
    const same = App.state.page === page && !App.state.forceTop;
    App.state.forceTop = false;
    App.state.page = page;
    document.querySelectorAll('[data-page]').forEach((b) => b.classList.toggle('active', b.dataset.page === page));
    const main = document.getElementById('main');
    const keep = same ? main.scrollTop : 0;
    App.clear(main);
    const p = App.pages[page];
    try { main.appendChild(p.render()); } catch (e) { App.error(e); }
    main.scrollTop = keep;
    if (same) requestAnimationFrame(() => { main.scrollTop = keep; });
    if (location.hash !== '#' + page) history.replaceState(null, '', '#' + page);
  };
  App.needData = function () {
    return h('div.card', h('div.empty', App.icon('data'), h('h3', L`Δεν έχουν φορτωθεί δεδομένα`), h('p', L`Άνοιξε ένα αρχείο ARFF/CSV ή διάλεξε ένα από τα ενσωματωμένα σύνολα του Weka.`),
      h('div.btn-group', { style: { justifyContent: 'center', marginTop: '10px' } }, h('button.btn.primary', { on: { click: App.openData } }, App.icon('upload'), L`Άνοιγμα αρχείου`), h('button.btn', { on: { click: () => App.go('home') } }, L`Δείγματα δεδομένων`))));
  };
  App.revealResult = () => requestAnimationFrame(() => {
    const el = document.querySelector('.result-view');
    if (!el) return;
    const top = el.getBoundingClientRect().top;
    if (window.innerWidth < 1150 || top < 50 || top > window.innerHeight * 0.6) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  App.pageHead = (title, sub, ...actions) => h('div.page-head', h('div', h('h1', title), sub ? h('p', sub) : null), actions.length ? h('div.btn-group', actions) : null);
  App.cardHead = (icon, title, ...extra) => h('div.card-head', icon ? App.icon(icon) : null, h('h3', title), h('div.spacer'), ...extra);
  App.timeStr = (d) => d.toLocaleTimeString(I18N.locale(), { hour: '2-digit', minute: '2-digit', second: '2-digit' });
})();
