/* Weka Web — σελίδες: Αρχική, Προεπεξεργασία, Πίνακας δεδομένων. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, App = window.App, h = App.h, C = App.charts;

  const SAMPLE_INFO = {
    'bank': [L`Bank marketing`, L`Will a client subscribe to a term deposit? 4,521 phone-campaign records (UCI).`, [L`classification`, L`clustering`]],
    'iris': [L`Λουλούδια Ίριδας`, L`Ο κλασικός πίνακας του Fisher: 3 είδη ίριδας από 4 μετρήσεις.`, [L`ταξινόμηση`, L`ομαδοποίηση`]],
    'iris.2D': [L`Ίριδα (2D)`, L`Η ίριδα με μόνο δύο χαρακτηριστικά — ιδανική για οπτικοποίηση.`, [L`ταξινόμηση`]],
    'weather.nominal': [L`Καιρός (nominal)`, L`Μικρό εκπαιδευτικό σύνολο: παίζουμε τένις;`, [L`συσχετίσεις`, L`ταξινόμηση`]],
    'weather.numeric': [L`Καιρός (αριθμητικός)`, L`Ο καιρός με αριθμητική θερμοκρασία/υγρασία.`, [L`ταξινόμηση`]],
    'diabetes': [L`Διαβήτης Pima`, L`Διάγνωση διαβήτη σε γυναίκες Pima από ιατρικές μετρήσεις.`, [L`ταξινόμηση`]],
    'breast-cancer': [L`Καρκίνος μαστού`, L`Υποτροπή καρκίνου από nominal κλινικά χαρακτηριστικά.`, [L`ταξινόμηση`]],
    'contact-lenses': [L`Φακοί επαφής`, L`Ποιος τύπος φακών ταιριάζει σε κάθε ασθενή.`, [L`κανόνες`, L`συσχετίσεις`]],
    'cpu': [L`Απόδοση CPU`, L`Πρόβλεψη σχετικής απόδοσης επεξεργαστών (παλινδρόμηση).`, [L`παλινδρόμηση`]],
    'cpu.with.vendor': [L`CPU με κατασκευαστή`, L`Όπως το cpu, με nominal χαρακτηριστικό κατασκευαστή.`, [L`παλινδρόμηση`]],
    'credit-g': [L`Γερμανική πίστη`, L`Πιστοληπτική αξιολόγηση: καλός ή κακός πελάτης.`, [L`ταξινόμηση`]],
    'glass': [L`Τύποι γυαλιού`, L`Αναγνώριση τύπου γυαλιού από χημική σύσταση (6 κλάσεις).`, [L`ταξινόμηση`]],
    'hypothyroid': [L`Υποθυρεοειδισμός`, L`Μεγαλύτερο ιατρικό σύνολο με πολλές ελλιπείς τιμές.`, [L`ταξινόμηση`]],
    'ionosphere': [L`Ιονόσφαιρα`, L`Σήματα ραντάρ: "καλά" ή "κακά" (34 αριθμητικά).`, [L`ταξινόμηση`]],
    'labor': [L`Εργασιακές συμβάσεις`, L`Αποδεκτές ή όχι συμβάσεις — πολλές ελλιπείς τιμές.`, [L`ταξινόμηση`]],
    'segment-challenge': [L`Τμηματοποίηση εικόνας`, L`Κατηγορία περιοχής εικόνας από 19 χαρακτηριστικά.`, [L`ταξινόμηση`]],
    'segment-test': [L`Τμηματοποίηση (test)`, L`Σύνολο ελέγχου για το segment-challenge.`, ['test set']],
    'soybean': [L`Ασθένειες σόγιας`, L`19 ασθένειες σόγιας από 35 nominal χαρακτηριστικά.`, [L`ταξινόμηση`]],
    'supermarket': [L`Σούπερ μάρκετ`, L`Καλάθια αγορών (4.627 συναλλαγές) για κανόνες συσχέτισης.`, [L`συσχετίσεις`]],
    'unbalanced': [L`Μη ισορροπημένο`, L`Πολύ άνιση κατανομή κλάσεων — για εξισορρόπηση.`, [L`ταξινόμηση`]],
    'vote': [L`Ψηφοφορίες Κογκρέσου`, L`Δημοκρατικοί ή Ρεπουμπλικάνοι από 16 ψηφοφορίες.`, [L`ταξινόμηση`]],
    'airline': [L`Επιβάτες αεροπορικών`, L`Μηνιαίοι επιβάτες — χρονοσειρά με ημερομηνίες.`, [L`παλινδρόμηση`]],
  };

  // ================================================================ HOME
  App.pages.home = {
    render() {
      const drop = h('div.dropzone', App.icon('upload'), h('h3', { style: { margin: '10px 0 4px' } }, L`Σύρε εδώ ένα αρχείο δεδομένων`), h('div.muted', L`ARFF, CSV, TSV ή JSON — η επεξεργασία γίνεται τοπικά στον browser σου.`),
        h('div', { style: { marginTop: '14px' } }, h('button.btn.primary', { on: { click: App.openData } }, App.icon('upload'), L`Επιλογή αρχείου`)));
      wireDrop(drop);
      const samples = h('div.samples');
      const idx = (W.DATASET_INDEX || []).slice().sort((a, b) => (a.id === 'bank' ? -1 : b.id === 'bank' ? 1 : 0) || (SAMPLE_INFO[a.id] ? 0 : 1) - (SAMPLE_INFO[b.id] ? 0 : 1));
      for (const s of idx) {
        const info = SAMPLE_INFO[s.id] || [s.id, '', []];
        samples.appendChild(h('button.sample', { on: { click: () => App.openSample(s.id) } },
          h('div.n', info[0]), h('div.d', info[1]),
          h('div.tags', h('span.badge', s.id + (s.format === 'csv' ? '.csv' : '.arff')), ...info[2].map((t) => h('span.badge.primary', t)), h('span.badge', U.d2s(s.size / 1024, 0) + ' KB'))));
      }
      const d = App.state.data;
      const features = [
        ['filter', L`Προεπεξεργασία`, L`19 φίλτρα, ιστογράμματα, αναίρεση αλλαγών`],
        ['tree', L`Ταξινόμηση & παλινδρόμηση`, L`25 αλγόριθμοι, cross-validation, ROC, σύγκριση κόστους, γραφικά δέντρα`],
        ['cluster', L`Ομαδοποίηση`, L`k-means, EM, ιεραρχική με δενδρόγραμμα`],
        ['link', L`Κανόνες συσχέτισης`, 'Apriori & FPGrowth'],
        ['select', L`Επιλογή χαρακτηριστικών`, 'InfoGain, CFS, ReliefF, PCA, Wrapper'],
        ['scatter', L`Οπτικοποίηση`, L`Scatter matrix με zoom και tooltips`],
        ['globe', L`Web data`, L`Tables from web pages, CSV/JSON and Wikipedia — robots.txt and rate limits respected`],
        ['chat', L`AI assistant`, L`Ask in English or Greek; it runs algorithms for you. Local models for voice and images`],
        ['code', L`Python / Colab`, L`Run Colab notebooks (OR-Tools, Pyro, matplotlib) in the browser; export any model to Colab`],
        ['report', L`Report (PDF)`, L`Export the whole session — data, steps, results, charts — as PDF`],
        ['video', L`Video tutorials`, L`Step-by-step videos in English and Greek`],
        ['sparkles', L`Same results as Weka`, L`Algorithms checked against Weka 3.8.7 — most match it exactly`],
      ];
      return h('div.page',
        h('div.hero',
          h('div.hero-card',
            h('div.small', { style: { opacity: 0.85, letterSpacing: '.08em', textTransform: 'uppercase', marginBottom: '10px' } }, L`Weka 3.8 · στον browser`),
            h('h1', L`Εξόρυξη δεδομένων, χωρίς εγκατάσταση.`),
            h('p', L`Όλες οι βασικές λειτουργίες του Weka Explorer — προεπεξεργασία, ταξινόμηση, ομαδοποίηση, συσχετίσεις, επιλογή χαρακτηριστικών και οπτικοποίηση — γραμμένες σε καθαρό HTML, CSS και JavaScript. Χωρίς Java, χωρίς server.`),
            h('div.btn-group',
              h('button.btn.white', { on: { click: () => App.openSample('iris') } }, App.icon('sparkles'), L`Δοκίμασε με το Iris`),
              d ? h('button.btn', { on: { click: () => App.go('preprocess') } }, L`Συνέχεια με ` + d.relation) : h('button.btn', { on: { click: App.openData } }, App.icon('upload'), L`Άνοιγμα αρχείου`)),
            (() => { const s = App.icon('tree', 'hero-art'); return s; })()),
          drop),
        h('div.card', App.cardHead('layers', L`Τι μπορείς να κάνεις`), h('div.card-body', h('div.grid.cols-3', features.map(([ic, t, s]) => h('div.row', { style: { alignItems: 'flex-start', flexWrap: 'nowrap' } },
          h('div', { style: { width: '36px', height: '36px', borderRadius: '10px', background: 'var(--primary-soft)', color: 'var(--primary)', display: 'grid', placeItems: 'center', flex: 'none' } }, App.icon(ic)),
          h('div', h('div', { style: { fontWeight: 650 } }, t), h('div.muted.small', s))))))),
        h('div.card', App.cardHead('data', L`Ενσωματωμένα σύνολα δεδομένων του Weka`, h('span.muted.small', (W.DATASET_INDEX || []).length + L` σύνολα`)), h('div.card-body', samples)));
    },
  };

  function wireDrop(zone) {
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
    zone.addEventListener('drop', async (e) => {
      e.preventDefault(); zone.classList.remove('drag');
      const f = e.dataTransfer.files[0];
      if (!f) return;
      try { const d = await App.parseFile(f); App.loadNew(d, f.name); App.go('preprocess'); } catch (err) { App.error(err); }
    });
  }
  App.wireDrop = wireDrop;

  // ========================================================== PREPROCESS
  const pre = { sel: 0, checked: new Set(), search: '', colorBy: null, showAll: false };
  App.pages.preprocess = {
    render() {
      const d = App.state.data;
      if (!d) return h('div.page', App.pageHead(L`Προεπεξεργασία`, L`Φόρτωσε δεδομένα για να ξεκινήσεις.`), App.needData());
      if (pre.sel >= d.numAttributes) pre.sel = 0;
      for (const c of [...pre.checked]) if (c >= d.numAttributes) pre.checked.delete(c);
      const page = h('div.page');
      const st = App.state;
      const saveMenu = h('div.btn-group',
        h('button.btn', { on: { click: () => App.download((d.relation || 'data').replace(/[^\w.-]+/g, '_') + '.arff', W.toARFF(d)) } }, App.icon('download'), 'ARFF'),
        h('button.btn', { on: { click: () => App.download((d.relation || 'data').replace(/[^\w.-]+/g, '_') + '.csv', W.toCSV(d), 'text/csv') } }, App.icon('download'), 'CSV'));
      page.appendChild(App.pageHead(L`Προεπεξεργασία`, L`Εξέτασε τα χαρακτηριστικά, εφάρμοσε φίλτρα και αποθήκευσε το αποτέλεσμα.`,
        h('button.btn', { on: { click: App.openData } }, App.icon('upload'), L`Άνοιγμα`),
        h('button.btn', { disabled: !st.undo.length, on: { click: App.undo }, title: L`Αναίρεση (Ctrl+Z)` }, App.icon('undo'), L`Αναίρεση`),
        saveMenu));

      // ------------------------------------------------ stats
      let missing = 0;
      for (const r of d.rows) for (const v of r) if (v !== v) missing++;
      const total = d.numInstances * d.numAttributes;
      const ca = d.classAttr;
      page.appendChild(h('div.stats',
        stat(L`Σχέση (relation)`, d.relation, st.fileName || '', true),
        stat(L`Εγγραφές`, d.numInstances.toLocaleString(I18N.locale()), L`Άθροισμα βαρών ` + U.d2s(d.sumOfWeights(), 2)),
        stat(L`Χαρακτηριστικά`, d.numAttributes, countTypes(d)),
        stat(L`Ελλιπείς τιμές`, total ? U.d2s((100 * missing) / total, 1) + '%' : '0%', missing.toLocaleString(I18N.locale()) + L` κελιά`),
        stat(L`Κλάση (στόχος)`, ca ? ca.name : '—', ca ? (ca.type === 'nominal' ? ca.values.length + L` τιμές` : L`αριθμητική`) : L`καμία`)));

      // ------------------------------------------------ filter card
      page.appendChild(filterCard());

      // ------------------------------------------------ attributes + details
      const grid = h('div.split');
      grid.appendChild(attributesCard(d));
      const det = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '18px', minWidth: 0 } });
      det.appendChild(attributeDetail(d, pre.sel));
      grid.appendChild(det);
      page.appendChild(grid);
      page.appendChild(allHistogramsCard(d));
      return page;
    },
  };
  function stat(label, value, sub, primary) {
    return h('div.stat' + (primary ? '.primary' : ''), h('div.label', label), h('div.value', { title: String(value), style: { whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' } }, value), h('div.sub', sub));
  }
  function countTypes(d) {
    const c = {};
    for (const a of d.attributes) c[a.type] = (c[a.type] || 0) + 1;
    const names = { numeric: L`αριθμ.`, nominal: 'nominal', string: 'string', date: L`ημερ.` };
    return Object.entries(c).map(([k, v]) => v + ' ' + (names[k] || k)).join(' · ');
  }

  function attributesCard(d) {
    const list = h('div.attr-list');
    const countLbl = h('span.muted.small');
    const removeBtn = h('button.btn.danger.sm', { on: { click: removeChecked } }, App.icon('trash'), L`Αφαίρεση`);
    const search = h('input.input', { placeholder: L`Αναζήτηση χαρακτηριστικού…`, value: pre.search, style: { height: '32px' }, on: { input: () => { pre.search = search.value; renderList(); } } });
    function renderList() {
      App.clear(list);
      const q = pre.search.trim().toLowerCase();
      d.attributes.forEach((a, j) => {
        if (q && !a.name.toLowerCase().includes(q)) return;
        const st = d.attributeStats(j);
        const cb = h('input', { type: 'checkbox', checked: pre.checked.has(j), on: { click: (e) => { e.stopPropagation(); if (cb.checked) pre.checked.add(j); else pre.checked.delete(j); upd(); } } });
        const row = h('div.attr-row' + (j === pre.sel ? '.active' : ''), { on: { click: () => { pre.sel = j; App.go('preprocess'); } } },
          cb, h('span.idx', j + 1), h('span.nm', { title: a.name }, a.name, j === d.classIndex ? h('span.cls-mark', L`ΚΛΑΣΗ`) : null), App.typeBadge(a),
          h('span.miss', { title: L`Ελλιπείς τιμές` }, st.missing ? U.d2s((100 * st.missing) / st.total, 0) + '%' : '—'));
        list.appendChild(row);
      });
      upd();
    }
    function upd() { countLbl.textContent = pre.checked.size ? pre.checked.size + L` επιλεγμένα` : ''; removeBtn.disabled = pre.checked.size === 0; }
    function removeChecked() {
      if (!pre.checked.size) return;
      if (pre.checked.size >= d.numAttributes) { App.toast(L`Δεν μπορείς να αφαιρέσεις όλα τα χαρακτηριστικά.`, 'error'); return; }
      const idx = [...pre.checked];
      const f = new W.Remove({ indices: idx }); f.setInput(d);
      const nd = f.apply(d);
      if (idx.includes(d.classIndex)) nd.classIndex = -1;
      pre.checked.clear(); pre.sel = 0;
      App.applyChange(nd, L`Remove attributes: ${idx.map((j) => d.attributes[j].name).join(", ")}`);
    }
    const selAll = (fn) => { d.attributes.forEach((_, j) => { if (fn(j)) pre.checked.add(j); else pre.checked.delete(j); }); renderList(); };
    const pattern = () => {
      const inp = h('input.input', { placeholder: L`π.χ. ^sepal|width$` });
      App.modal(L`Επιλογή με μοτίβο (regex)`, h('div', h('div.field', h('label', L`Κανονική έκφραση για τα ονόματα`), inp)), [
        { label: L`Άκυρο` },
        { label: L`Επιλογή`, primary: true, onClick: () => { try { const re = new RegExp(inp.value, 'i'); selAll((j) => re.test(d.attributes[j].name)); } catch (e) { App.error(new Error(L`Μη έγκυρη έκφραση`)); return false; } } },
      ]);
      setTimeout(() => inp.focus(), 50);
    };
    const card = h('div.card',
      App.cardHead('list', L`Χαρακτηριστικά`, countLbl),
      h('div.attr-tools', h('div.btn-group',
        h('button.btn.sm', { on: { click: () => selAll(() => true) } }, L`Όλα`),
        h('button.btn.sm', { on: { click: () => selAll(() => false) } }, L`Κανένα`),
        h('button.btn.sm', { on: { click: () => selAll((j) => !pre.checked.has(j)) } }, L`Αντιστροφή`),
        h('button.btn.sm', { on: { click: pattern } }, L`Μοτίβο`)), h('div', { style: { flex: 1 } }), removeBtn),
      h('div', { style: { padding: '10px 14px', borderBottom: '1px solid var(--border)' } }, search),
      list,
      h('div', { style: { padding: '12px 14px', borderTop: '1px solid var(--border)' } }, h('div.field', { style: { margin: 0 } }, h('label', L`Χαρακτηριστικό κλάσης (στόχος)`), App.classSelect(() => App.go('preprocess'), { allowNone: true }))));
    renderList();
    return card;
  }

  function attributeDetail(d, j) {
    const a = d.attributes[j];
    const st = d.attributeStats(j);
    const head = App.cardHead(null, a.name, App.typeBadge(a), j === d.classIndex ? h('span.badge.primary', L`Κλάση`) : h('button.btn.sm', { on: { click: () => { App.setClass(j); App.go('preprocess'); } } }, App.icon('target'), L`Ορισμός ως κλάση`));
    const kv = (k, v) => h('div', h('div.muted.small', k), h('div', { style: { fontWeight: 650, fontSize: '15px' } }, v));
    const summary = h('div.grid', { style: { gridTemplateColumns: 'repeat(auto-fit,minmax(110px,1fr))', gap: '12px', marginBottom: '14px' } },
      kv(L`Ελλιπείς`, `${st.missing} (${st.total ? U.d2s((100 * st.missing) / st.total, 0) : 0}%)`), kv(L`Διακριτές τιμές`, st.distinct), kv(L`Μοναδικές`, `${st.unique} (${st.total ? U.d2s((100 * st.unique) / st.total, 0) : 0}%)`));
    let table;
    if (a.type === 'nominal' || a.type === 'string') {
      const rows = (st.counts || []).map((c, k) => ({ k, label: a.values[k], c }));
      const mx = Math.max(1, ...rows.map((r) => r.c));
      const wsum = U.zeros(a.values.length);
      d.rows.forEach((r, i) => { const v = r[j]; if (v === v) wsum[v] += d.weights[i]; });
      table = h('div.table-wrap', { style: { maxHeight: '260px' } }, h('table.tbl.compact', h('thead', h('tr', h('th', L`Αρ.`), h('th', L`Τιμή`), h('th.num', L`Πλήθος`), h('th.num', L`Βάρος`), h('th', { style: { width: '30%' } }, ''))),
        h('tbody', rows.slice(0, 500).map((r) => h('tr', h('td.muted', r.k + 1), h('td', r.label), h('td.num', r.c), h('td.num', U.d2s(wsum[r.k], 2)), h('td', h('div.bar', { style: { width: (100 * r.c) / mx + '%', background: d.classIndex === j && a.type === 'nominal' ? App.color(r.k) : undefined } })))))));
    } else {
      const f = (v) => (a.type === 'date' && v === v ? W.formatDate(v, a.dateFormat) : App.fmt(v, 3));
      table = h('table.tbl.compact', h('thead', h('tr', h('th', L`Στατιστικό`), h('th.num', L`Τιμή`))), h('tbody',
        h('tr', h('td', L`Ελάχιστο`), h('td.num', f(st.min))), h('tr', h('td', L`Μέγιστο`), h('td.num', f(st.max))), h('tr', h('td', L`Μέσος όρος`), h('td.num', App.fmt(st.mean, 3))), h('tr', h('td', L`Τυπική απόκλιση`), h('td.num', App.fmt(st.stdDev, 3)))));
    }
    // histogram
    if (pre.colorBy === null || pre.colorBy >= d.numAttributes) pre.colorBy = d.classIndex;
    const colorSel = h('select.select', { style: { width: 'auto', height: '30px' }, on: { change: () => { pre.colorBy = Number(colorSel.value); App.go('preprocess'); } } },
      h('option', { value: -1 }, L`Χωρίς χρωματισμό`), d.attributes.map((x, k) => (x.type === 'nominal' ? h('option', { value: k, selected: k === pre.colorBy }, L`Χρώμα: ` + x.name) : null)));
    if (pre.colorBy < 0) colorSel.value = -1;
    const hist = C.histogram(d, j, pre.colorBy, { width: 640, height: 240 });
    const cAtt = pre.colorBy >= 0 ? d.attributes[pre.colorBy] : null;
    return h('div.card', head, h('div.card-body', summary, h('div.grid.cols-2', { style: { alignItems: 'start' } },
      h('div', table),
      h('div', h('div.row', { style: { justifyContent: 'space-between', marginBottom: '6px' } }, h('div.field-label', L`Κατανομή`), colorSel), hist, cAtt && cAtt.type === 'nominal' ? C.legend(cAtt.values) : null))));
  }

  function allHistogramsCard(d) {
    const body = h('div.card-body');
    const toggle = h('button.btn.sm', { on: { click: () => { pre.showAll = !pre.showAll; App.go('preprocess'); } } }, pre.showAll ? L`Απόκρυψη` : L`Εμφάνιση`);
    if (pre.showAll) {
      const g = h('div.grid', { style: { gridTemplateColumns: 'repeat(auto-fill,minmax(260px,1fr))', gap: '12px' } });
      d.attributes.slice(0, 80).forEach((a, j) => {
        g.appendChild(h('div', { style: { border: '1px solid var(--border)', borderRadius: '10px', padding: '8px 10px', cursor: 'pointer' }, on: { click: () => { pre.sel = j; App.go('preprocess'); } } },
          h('div.row', { style: { justifyContent: 'space-between' } }, h('div', { style: { fontWeight: 600, fontSize: '13px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '170px' } }, a.name), App.typeBadge(a)),
          C.histogram(d, j, pre.colorBy, { width: 300, height: 140, bins: 20 })));
      });
      body.appendChild(g);
      const cAtt = pre.colorBy >= 0 ? d.attributes[pre.colorBy] : null;
      if (cAtt && cAtt.type === 'nominal') body.appendChild(h('div', { style: { marginTop: '12px' } }, C.legend(cAtt.values)));
    } else body.appendChild(h('div.muted.small', L`Ιστογράμματα όλων των χαρακτηριστικών (όπως το "Visualize All" του Weka).`));
    return h('div.card', App.cardHead('grid', L`Όλα τα χαρακτηριστικά`, toggle), body);
  }

  // ------------------------------------------------------- filter card
  const KIND_LABELS = { 'unsup-attribute': 'Unsupervised · attribute', 'unsup-instance': 'Unsupervised · instance', 'sup-attribute': 'Supervised · attribute', 'sup-instance': 'Supervised · instance' };
  function filterCard() {
    const runner = App.runner();
    const cfg = App.algoConfig({
      items: W.filters, storeKey: 'filter', defaultName: 'Normalize',
      groupOf: (s) => (s.supervised ? 'sup-' : 'unsup-') + s.kind, groupLabels: KIND_LABELS,
    });
    const apply = async () => {
      const d = App.state.data;
      const { spec, opts } = cfg.get();
      runner.start();
      await W.nextFrame();
      try {
        const t0 = performance.now();
        const nd = W.applyFilter(spec, opts, d);
        const o = W.optionString(spec, opts);
        App.applyChange(nd, spec.name + (o ? ' ' + o : ''));
        App.toast(L`${spec.name}: ${d.numInstances} → ${nd.numInstances} εγγραφές, ${d.numAttributes} → ${nd.numAttributes} χαρακτηριστικά (${U.d2s(performance.now() - t0, 0)} ms).`, 'success');
      } catch (e) { runner.fail(''); App.error(e); }
    };
    const ops = App.state.ops || [];
    const hist = h('div.history', ops.slice(-6).reverse().map((o) => h('div.h', h('span.dot'), h('span', { style: { flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, o.label), h('span.muted.small', App.timeStr(o.time)))));
    return h('div.card', App.cardHead('filter', L`Φίλτρο`),
      h('div.card-body', h('div.grid', { style: { gridTemplateColumns: 'minmax(0,1.4fr) minmax(0,1fr)', gap: '22px' } },
        h('div', cfg.el),
        h('div', { style: { display: 'flex', flexDirection: 'column', gap: '12px' } },
          h('button.btn.primary.lg.block', { on: { click: apply } }, App.icon('wand'), L`Εφαρμογή φίλτρου`), runner.el,
          h('div.callout', App.icon('info'), h('div', L`Τα φίλτρα αλλάζουν το τρέχον σύνολο. Μπορείς πάντα να κάνεις `, h('b', L`Αναίρεση`), ' (Ctrl+Z).')),
          h('div', h('div.field-label', { style: { marginBottom: '8px' } }, L`Ιστορικό αλλαγών`), hist)))));
  }

  // ============================================================== TABLE
  const tv = { page: 0, size: 100, sort: null, dir: 1, q: '' };
  App.pages.table = {
    render() {
      const d = App.state.data;
      if (!d) return h('div.page', App.pageHead(L`Πίνακας δεδομένων`), App.needData());
      const page = h('div.page');
      page.appendChild(App.pageHead(L`Πίνακας δεδομένων`, L`Προβολή και επεξεργασία εγγραφών. Διπλό κλικ σε κελί για αλλαγή τιμής.`,
        h('button.btn', { on: { click: addRow } }, App.icon('plus'), L`Νέα εγγραφή`),
        h('button.btn', { disabled: !App.state.undo.length, on: { click: App.undo } }, App.icon('undo'), L`Αναίρεση`)));
      const search = h('input.input', { placeholder: L`Φιλτράρισμα εγγραφών (κείμενο σε οποιοδήποτε πεδίο)…`, value: tv.q, on: { input: () => { tv.q = search.value; tv.page = 0; renderTable(); } } });
      const tableBox = h('div.table-wrap', { style: { maxHeight: '64vh' } });
      const pager = h('div.row', { style: { padding: '12px 16px', borderTop: '1px solid var(--border)' } });
      page.appendChild(h('div.card', h('div', { style: { padding: '12px 16px', borderBottom: '1px solid var(--border)' } }, search), tableBox, pager));
      function order() {
        let idx = d.rows.map((_, i) => i);
        if (tv.q.trim()) {
          const q = tv.q.trim().toLowerCase();
          idx = idx.filter((i) => d.rows[i].some((v, j) => d.stringValue(d.rows[i], j).toLowerCase().includes(q)));
        }
        if (tv.sort !== null && tv.sort < d.numAttributes) {
          const j = tv.sort, a = d.attributes[j];
          const key = (i) => { const v = d.rows[i][j]; if (v !== v) return null; return a.type === 'nominal' || a.type === 'string' ? a.values[v] : v; };
          idx.sort((x, y) => { const p = key(x), q = key(y); if (p === null) return 1; if (q === null) return -1; return (p < q ? -1 : p > q ? 1 : 0) * tv.dir; });
        }
        return idx;
      }
      function renderTable() {
        const idx = order();
        const pages = Math.max(1, Math.ceil(idx.length / tv.size));
        if (tv.page >= pages) tv.page = pages - 1;
        const slice = idx.slice(tv.page * tv.size, (tv.page + 1) * tv.size);
        const thead = h('tr', h('th.num', '#'), d.attributes.map((a, j) => h('th.sortable', { on: { click: () => { if (tv.sort === j) tv.dir = -tv.dir; else { tv.sort = j; tv.dir = 1; } renderTable(); } } },
          a.name, ' ', h('span.muted', a.type === 'nominal' ? 'N' : a.type === 'numeric' ? '#' : a.type[0].toUpperCase()), tv.sort === j ? (tv.dir > 0 ? ' ▲' : ' ▼') : '', j === d.classIndex ? h('span.cls-mark', { style: { color: 'var(--accent)' } }, ' ●') : '')), h('th.num', L`Βάρος`), h('th'));
        const tbody = h('tbody', slice.map((i) => {
          const r = d.rows[i];
          return h('tr', h('td.num.muted', i + 1), r.map((v, j) => {
            const td = h('td' + (v !== v ? '.missing' : '') + (d.attributes[j].type === 'numeric' ? '.num' : ''), v !== v ? '?' : d.stringValue(r, j));
            td.addEventListener('dblclick', () => editCell(i, j));
            return td;
          }), h('td.num.muted', U.d2s(d.weights[i], 3)), h('td', h('button.btn.ghost.sm.icon', { title: L`Διαγραφή εγγραφής`, on: { click: () => deleteRow(i) } }, App.icon('trash'))));
        }));
        App.clear(tableBox).appendChild(h('table.tbl', h('thead', thead), tbody));
        App.clear(pager).append(
          h('span.muted.small', L`${idx.length.toLocaleString(I18N.locale())} εγγραφές${idx.length !== d.numInstances ? L` (φιλτραρισμένες από ` + d.numInstances + ')' : ''}`),
          h('div', { style: { flex: 1 } }),
          h('button.btn.sm', { disabled: tv.page === 0, on: { click: () => { tv.page--; renderTable(); } } }, L`‹ Προηγ.`),
          h('span.small', L` Σελίδα ${tv.page + 1} / ${pages} `),
          h('button.btn.sm', { disabled: tv.page >= pages - 1, on: { click: () => { tv.page++; renderTable(); } } }, L`Επόμ. ›`));
      }
      function editCell(i, j) {
        const a = d.attributes[j], cur = d.rows[i][j];
        let input;
        if (a.type === 'nominal') input = h('select.select', h('option', { value: '' }, L`? (ελλιπής)`), a.values.map((v, k) => h('option', { value: k, selected: cur === k }, v)));
        else input = h('input.input', { value: cur !== cur ? '' : a.type === 'date' ? W.formatDate(cur, a.dateFormat) : a.type === 'string' ? a.values[cur] : String(cur), placeholder: L`κενό = ελλιπής` });
        App.modal(L`Εγγραφή ${i + 1} · ${a.name}`, h('div.field', h('label', L`Νέα τιμή`), input), [
          { label: L`Άκυρο` },
          { label: L`Αποθήκευση`, primary: true, onClick: () => {
            let v;
            const s = input.value.trim();
            if (s === '' || s === '?') v = NaN;
            else if (a.type === 'nominal') v = Number(s);
            else if (a.type === 'numeric') { v = Number(s); if (isNaN(v)) { App.toast(L`Μη αριθμητική τιμή.`, 'error'); return false; } }
            else if (a.type === 'date') { v = Date.parse(s); if (isNaN(v)) { App.toast(L`Μη έγκυρη ημερομηνία.`, 'error'); return false; } }
            else { v = a.values.indexOf(s); if (v < 0) { v = a.values.length; a.values.push(s); } }
            const nd = d.copy(); nd.rows[i] = d.rows[i].slice(); nd.rows[i][j] = v;
            App.applyChange(nd, L`Αλλαγή τιμής (εγγραφή ${i + 1}, ${a.name})`);
          } },
        ]);
        setTimeout(() => input.focus(), 50);
      }
      function deleteRow(i) {
        const nd = d.copy(); nd.rows.splice(i, 1); nd.weights.splice(i, 1);
        App.applyChange(nd, L`Διαγραφή εγγραφής ${i + 1}`);
      }
      function addRow() {
        const nd = d.copy(); nd.rows.push(d.attributes.map(() => NaN)); nd.weights.push(1);
        tv.page = Math.floor((nd.numInstances - 1) / tv.size); tv.q = '';
        App.applyChange(nd, L`Νέα εγγραφή`);
      }
      renderTable();
      return page;
    },
  };
})();
