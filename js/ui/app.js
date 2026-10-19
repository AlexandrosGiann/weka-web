/* Weka Web — κέλυφος εφαρμογής: πλοήγηση, topbar, συντομεύσεις, εκκίνηση. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, App = window.App, h = App.h;

  const NAV = [
    { id: 'home', label: L`Αρχική`, icon: 'home', short: L`Αρχική` },
    { section: L`Δεδομένα` },
    { id: 'preprocess', label: L`Προεπεξεργασία`, icon: 'filter', short: L`Δεδομένα`, needs: true },
    { id: 'table', label: L`Πίνακας δεδομένων`, icon: 'table', short: L`Πίνακας`, needs: true },
    { id: 'visualize', label: L`Οπτικοποίηση`, icon: 'scatter', short: L`Γραφήματα`, needs: true },
    { section: L`Ανάλυση` },
    { id: 'classify', label: L`Ταξινόμηση`, icon: 'tree', short: L`Ταξινόμ.`, needs: true, results: 'classify' },
    { id: 'cluster', label: L`Ομαδοποίηση`, icon: 'cluster', short: L`Ομάδες`, needs: true, results: 'cluster' },
    { id: 'associate', label: L`Συσχετίσεις`, icon: 'link', short: L`Κανόνες`, needs: true, results: 'associate' },
    { id: 'select', label: L`Επιλογή χαρακτηριστικών`, icon: 'select', short: L`Επιλογή`, needs: true, results: 'select' },
    { section: L`Share & learn` },
    { id: 'report', label: L`Report (PDF)`, icon: 'report', short: L`Report`, needs: true },
    { id: 'tutorials', label: L`Video tutorials`, icon: 'video', short: L`Videos` },
  ];

  function buildShell() {
    const sidebar = h('aside.sidebar',
      h('div.brand', h('div.brand-mark', 'W'), h('div', h('div.brand-name', 'Weka Web'), h('div.brand-sub', L`Explorer στον browser`))));
    for (const n of NAV) {
      if (n.section) { sidebar.appendChild(h('div.nav-label', n.section)); continue; }
      const badge = n.results ? h('span.badge', { 'data-count': n.results }) : null;
      sidebar.appendChild(h('button.nav-item', { 'data-page': n.id, on: { click: () => App.go(n.id) } }, App.icon(n.icon), h('span', n.label), badge));
    }
    sidebar.appendChild(h('div.sidebar-foot', L`Βασισμένο στο Weka 3.8.7 (University of Waikato, GPL). Όλοι οι υπολογισμοί γίνονται τοπικά στον browser — τα δεδομένα δεν φεύγουν από τον υπολογιστή σου.`));

    const dsName = h('span.ds-name');
    const dsMeta = h('span.muted.small.hide-mobile');
    const themeBtn = h('button.btn.ghost.icon', { title: L`Θέμα (φωτεινό/σκοτεινό)`, on: { click: () => { App.toggleTheme(); updateThemeIcon(); } } });
    const undoBtn = h('button.btn.ghost.icon', { title: L`Αναίρεση (Ctrl+Z)`, on: { click: App.undo } }, App.icon('undo'));
    const redoBtn = h('button.btn.ghost.icon.hide-mobile', { title: L`Επανάληψη (Ctrl+Y)`, on: { click: App.redoFn } }, App.icon('redo'));
    const openBtn = h('button.btn', { on: { click: App.openData } }, App.icon('upload'), h('span.hide-mobile', L`Άνοιγμα`));
    const sampleSel = h('select.select.hide-mobile', { style: { width: '170px' }, on: { change: (e) => { if (e.target.value) App.openSample(e.target.value); e.target.value = ''; } } },
      h('option', { value: '' }, L`Δείγματα Weka…`), (W.DATASET_INDEX || []).map((x) => h('option', { value: x.id }, x.id)));
    const langBtn = h('button.btn.ghost.lang-btn', { title: L`Language`, on: { click: () => App.switchLang(I18N.lang === 'en' ? 'el' : 'en') } }, I18N.lang === 'en' ? 'ΕΛ' : 'EN');
    const topbar = h('header.topbar',
      h('div.ds-info', App.icon('data'), dsName, dsMeta), h('div.spacer'), sampleSel, openBtn, undoBtn, redoBtn, langBtn, themeBtn);
    function updateThemeIcon() { App.clear(themeBtn).appendChild(App.icon(document.documentElement.getAttribute('data-theme') === 'dark' ? 'sun' : 'moon')); }
    updateThemeIcon();
    const main = h('main.main#main');
    const mobile = h('nav.mobile-nav', NAV.filter((n) => !n.section).map((n) => h('button', { 'data-page': n.id, on: { click: () => App.go(n.id) } }, App.icon(n.icon), n.short)));
    const app = h('div.app', sidebar, topbar, main);
    document.body.append(app, mobile);

    App.updateChrome = function () {
      const d = App.state.data;
      dsName.textContent = d ? d.relation : L`Καμία φόρτωση δεδομένων`;
      dsMeta.textContent = d ? L`${d.numInstances.toLocaleString(I18N.locale())} εγγραφές · ${d.numAttributes} χαρακτηριστικά${d.classAttr ? L` · κλάση: ` + d.classAttr.name : ''}` : '';
      undoBtn.disabled = !App.state.undo.length;
      redoBtn.disabled = !App.state.redo.length;
      document.querySelectorAll('.nav-item[data-page], .mobile-nav button[data-page]').forEach((b) => {
        const n = NAV.find((x) => x.id === b.dataset.page);
        b.disabled = !!(n && n.needs && !d);
      });
      document.querySelectorAll('[data-count]').forEach((b) => { const c = App.state.results[b.dataset.count].length; b.textContent = c; b.style.display = c ? '' : 'none'; });
    };
  }

  /** Αλλαγή γλώσσας: τα κείμενα δημιουργούνται κατά τη φόρτωση, οπότε κρατάμε τα δεδομένα και κάνουμε reload. */
  App.switchLang = function (l) {
    const d = App.state.data;
    try {
      if (d) sessionStorage.setItem('wekaweb.restore', JSON.stringify({ arff: W.toARFF(d), cls: d.classIndex, file: App.state.fileName, page: App.state.page }));
    } catch (e) { /* πολύ μεγάλο για το sessionStorage */ }
    I18N.setLang(l);
    location.reload();
  };
  function restoreSession() {
    let r = null;
    try { r = JSON.parse(sessionStorage.getItem('wekaweb.restore')); sessionStorage.removeItem('wekaweb.restore'); } catch (e) { r = null; }
    if (!r) return null;
    try {
      const d = W.parseARFF(r.arff); d.classIndex = r.cls;
      App.state.data = d; App.state.fileName = r.file; App.state.ops = [{ label: L`Φόρτωση ` + r.file, time: new Date() }];
      return r.page;
    } catch (e) { return null; }
  }

  function init() {
    const theme = App.store.get('theme', window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
    buildShell();
    App.on('data', () => { App.updateChrome(); App.go(App.state.page === 'home' ? 'preprocess' : App.state.page); });
    App.on('class', () => App.updateChrome());
    const origGo = App.go;
    App.go = function (p) { origGo(p); App.updateChrome(); };
    // συντομεύσεις
    document.addEventListener('keydown', (e) => {
      if (e.target.matches('input, textarea, select')) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !e.shiftKey) { e.preventDefault(); App.undo(); }
      else if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'y' || (e.key.toLowerCase() === 'z' && e.shiftKey))) { e.preventDefault(); App.redoFn(); }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'o') { e.preventDefault(); App.openData(); }
    });
    // drag & drop οπουδήποτε
    document.addEventListener('dragover', (e) => { if (e.dataTransfer && [...e.dataTransfer.types].includes('Files')) e.preventDefault(); });
    document.addEventListener('drop', async (e) => {
      if (e.defaultPrevented || !e.dataTransfer || !e.dataTransfer.files.length) return;
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      try { const d = await App.parseFile(f); App.loadNew(d, f.name); } catch (err) { App.error(err); }
    });
    window.addEventListener('hashchange', () => { const p = location.hash.slice(1); if (p && p !== App.state.page && App.pages[p]) App.go(p); });
    const restoredPage = restoreSession();
    if (restoredPage) { App.go(restoredPage); return; }
    const start = location.hash.slice(1);
    App.go(App.pages[start] && start !== '' ? (start === 'home' || !NAV.find((n) => n.id === start && n.needs) ? start : 'home') : 'home');
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
