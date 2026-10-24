/* Weka Web — σελίδα Ταξινόμησης / Παλινδρόμησης. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, App = window.App, h = App.h, C = App.charts;
  const GROUPS = { bayes: 'Bayes', functions: 'Functions', lazy: 'Lazy', meta: 'Meta', rules: 'Rules', trees: 'Trees' };
  const ORDER = ['bayes', 'functions', 'lazy', 'meta', 'rules', 'trees'];
  const st = { mode: App.store.get('cls.mode', 'cv'), folds: 10, pct: 66, seed: 1, preserve: false, active: -1, tab: 'summary', rocClass: 0, errX: null, errY: null, onlyErr: false };
  let running = null;

  App.pages.classify = {
    render() {
      const d = App.state.data;
      if (!d) return h('div.page', App.pageHead(L`Ταξινόμηση`, L`Εκπαίδευση και αξιολόγηση ταξινομητών & μοντέλων παλινδρόμησης.`), App.needData());
      const page = h('div.page');
      page.appendChild(App.pageHead(L`Ταξινόμηση`, L`Εκπαίδευση και αξιολόγηση ταξινομητών & μοντέλων παλινδρόμησης.`));
      const left = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '18px' } });
      const right = h('div.result-view', { style: { minWidth: 0 } });
      page.appendChild(h('div.split', left, right));

      const items = ORDER.flatMap((g) => W.classifiers.filter((s) => s.group === g));
      const cfg = App.algoConfig({ items, storeKey: 'classifier', defaultName: 'J48', groupOf: (s) => s.group, groupLabels: GROUPS });
      left.appendChild(h('div.card', App.cardHead('tree', L`Ταξινομητής`), h('div.card-body', cfg.el)));

      // ---------------------------------------------------- test options
      const modes = [['cv', 'Cross-validation'], ['split', L`Ποσοστό`], ['train', L`Εκπαίδευσης`], ['test', 'Test set']];
      const seg = h('div.seg', modes.map(([k, l]) => h('button' + (st.mode === k ? '.active' : ''), { type: 'button', on: { click: () => { st.mode = k; App.store.set('cls.mode', k); App.go('classify'); } } }, l)));
      const num = (key, label, min, max) => h('div.field', h('label', label), h('input.input', { type: 'number', value: st[key], min, max, on: { input: (e) => { const v = Number(e.target.value); if (!isNaN(v)) st[key] = v; } } }));
      const modeBox = h('div', { style: { marginTop: '12px' } });
      if (st.mode === 'cv') modeBox.append(h('div.grid.cols-2', { style: { gap: '10px' } }, num('folds', 'Folds', 2, 1000), num('seed', 'Seed', 0)));
      else if (st.mode === 'split') modeBox.append(h('div.grid.cols-2', { style: { gap: '10px' } }, num('pct', L`Εκπαίδευση (%)`, 1, 99), num('seed', 'Seed', 0)),
        h('label.check', h('input', { type: 'checkbox', checked: st.preserve, on: { change: (e) => { st.preserve = e.target.checked; } } }), L`Διατήρηση σειράς (χωρίς ανακάτεμα)`));
      else if (st.mode === 'train') modeBox.append(h('div.callout.warn', App.icon('info'), h('div', L`Η αξιολόγηση στα ίδια δεδομένα εκπαίδευσης δίνει υπεραισιόδοξη εκτίμηση.`)));
      else {
        const ts = App.state.testSet;
        modeBox.append(h('div.row', h('button.btn.sm', { on: { click: async () => {
          const f = await App.openFileDialog(); if (!f) return;
          try { const t = await App.parseFile(f); App.state.testSet = t; App.state.testSetName = f.name; App.toast(L`Φορτώθηκε σύνολο ελέγχου: ` + t.numInstances + L` εγγραφές.`, 'success'); App.go('classify'); } catch (e) { App.error(e); }
        } } }, App.icon('upload'), L`Φόρτωση αρχείου`), h('select.select', { style: { flex: 1, height: '30px' }, on: { change: async (e) => { if (!e.target.value) return; try { const t = await App.loadSample(e.target.value); App.state.testSet = t; App.state.testSetName = e.target.value + '.arff'; App.go('classify'); } catch (err) { App.error(err); } } } },
          h('option', { value: '' }, L`ή δείγμα…`), (W.DATASET_INDEX || []).map((x) => h('option', { value: x.id }, x.id)))),
          h('div.small', { style: { marginTop: '8px' } }, ts ? h('span.badge.success', App.state.testSetName + ' · ' + ts.numInstances + L` εγγραφές`) : h('span.muted', L`Δεν έχει φορτωθεί σύνολο ελέγχου.`)));
      }
      const runner = App.runner();
      const startBtn = h('button.btn.primary.lg', { style: { flex: 1 }, on: { click: start } }, App.icon('play'), L`Εκτέλεση`);
      const stopBtn = h('button.btn.lg', { disabled: true, on: { click: () => running && running.cancel() } }, App.icon('stop'));
      left.appendChild(h('div.card', App.cardHead('target', L`Αξιολόγηση`), h('div.card-body',
        h('div.field', h('label', L`Χαρακτηριστικό κλάσης`), App.classSelect(() => App.go('classify'))),
        h('div.field-label', { style: { marginBottom: '6px' } }, L`Τρόπος ελέγχου`), seg, modeBox,
        h('div.row', { style: { marginTop: '14px', flexWrap: 'nowrap' } }, startBtn, stopBtn), h('div', { style: { marginTop: '10px' } }, runner.el))));

      // ------------------------------------------------------ result list
      const resCard = h('div.card', App.cardHead('list', L`Αποτελέσματα`, App.state.results.classify.length ? h('button.btn.ghost.sm', { on: { click: () => { App.state.results.classify = []; st.active = -1; App.go('classify'); } } }, L`Καθαρισμός`) : null));
      const list = h('div.results', { style: { padding: '12px' } });
      const results = App.state.results.classify;
      if (!results.length) list.appendChild(h('div.muted.small', { style: { padding: '6px' } }, L`Τα αποτελέσματα κάθε εκτέλεσης εμφανίζονται εδώ για σύγκριση.`));
      results.forEach((r, i) => {
        const e = r.res.evaluation;
        const score = e.nominal ? U.d2s(e.pctCorrect(), 2) + '%' : 'r=' + U.d2s(e.correlationCoefficient(), 3);
        list.appendChild(h('div.result-item' + (i === st.active ? '.active' : ''), { on: { click: () => { st.active = i; App.go('classify'); } } },
          h('div.meta', h('div.t', r.name), h('div.s', App.timeStr(r.time) + ' · ' + r.res.label + ' · ' + r.relation)),
          h('div.score', score),
          h('button.btn.ghost.sm.icon.x', { title: L`Διαγραφή`, on: { click: (ev) => { ev.stopPropagation(); results.splice(i, 1); if (st.active >= results.length) st.active = results.length - 1; App.go('classify'); } } }, App.icon('x'))));
      });
      resCard.appendChild(list);
      left.appendChild(resCard);

      // ---------------------------------------------------------- view
      const active = results[st.active];
      right.appendChild(active ? resultView(active) : h('div.card', h('div.empty', App.icon('classify'), h('h3', L`Έτοιμο για εκτέλεση`), h('p', L`Διάλεξε αλγόριθμο και τρόπο αξιολόγησης και πάτησε «Εκτέλεση».`))));
      const cmp = App.analysis.compareCard(results, () => App.go('classify'));
      if (cmp) page.appendChild(cmp);

      async function start() {
        const { spec, opts } = cfg.get();
        const data = App.state.data;
        startBtn.disabled = true; stopBtn.disabled = false;
        runner.start();
        running = runner;
        try {
          const res = await W.runClassifier({ data, spec, options: opts, mode: st.mode, folds: st.folds, seed: st.seed, percentage: st.pct, preserveOrder: st.preserve, testData: App.state.testSet, onProgress: (p, m) => runner.progress(p, m) });
          const entry = { name: spec.name, time: new Date(), res, relation: data.relation, data };
          App.state.results.classify.push(entry);
          st.active = App.state.results.classify.length - 1;
          runner.done(L`Ολοκληρώθηκε σε ${U.d2s(res.time, 2)} s`);
          App.go('classify');
          App.revealResult();
        } catch (e) { runner.fail(''); App.error(e); }
        finally { startBtn.disabled = false; stopBtn.disabled = true; running = null; }
      }
      return page;
    },
  };

  // ======================================================== result view
  function resultView(entry) {
    const r = entry.res, e = r.evaluation;
    const tabsDef = [['summary', L`Σύνοψη`, 'chart'], ['model', L`Μοντέλο`, 'tree'], ['preds', L`Προβλέψεις`, 'table']];
    if (e.nominal) tabsDef.push(['roc', 'ROC / PR', 'roc']);
    tabsDef.push(['text', L`Πλήρης έξοδος`, 'text']);
    if (!tabsDef.some((t) => t[0] === st.tab)) st.tab = 'summary';
    const tabs = h('div.tabs', tabsDef.map(([k, l, ic]) => h('button' + (st.tab === k ? '.active' : ''), { on: { click: () => { st.tab = k; App.go('classify'); } } }, App.icon(ic), l)));
    const body = h('div');
    const head = h('div.card-head', h('div', h('h3', entry.name), h('div.muted.small', r.spec.path + (r.optStr ? ' ' + r.optStr : ''))), h('div.spacer'),
      h('button.btn.sm', { title: L`Export as a Google Colab notebook (scikit-learn + exact Weka)`, on: { click: () => { try { App.py.exportResult('c' + App.state.results.classify.indexOf(entry), 'ipynb'); } catch (e) { App.error(e); } } } }, App.icon('code'), 'Colab'),
      h('button.btn.sm', { on: { click: () => App.download(entry.name + '-' + entry.relation + '.txt', r.report) } }, App.icon('download'), L`Αναφορά`));
    const card = h('div.card', head, tabs, body);
    if (st.tab === 'summary') body.appendChild(summaryTab(entry));
    else if (st.tab === 'model') body.appendChild(modelTab(entry));
    else if (st.tab === 'preds') body.appendChild(predsTab(entry));
    else if (st.tab === 'roc') body.appendChild(rocTab(entry));
    else body.appendChild(h('div', h('div.row', { style: { padding: '10px 16px', borderBottom: '1px solid var(--border)' } }, h('div', { style: { flex: 1 } }), h('button.btn.sm', { on: { click: () => App.copy(r.report) } }, App.icon('copy'), L`Αντιγραφή`)), h('pre.output', r.report)));
    return card;
  }

  function metric(label, value, sub, primary) {
    return h('div.stat' + (primary ? '.primary' : ''), h('div.label', label), h('div.value', value), sub ? h('div.sub', sub) : null);
  }

  function summaryTab(entry) {
    const r = entry.res, e = r.evaluation;
    const box = h('div.card-body', { style: { display: 'flex', flexDirection: 'column', gap: '18px' } });
    box.appendChild(h('div.row', h('span.badge.primary', r.label), h('span.badge', L`Εγγραφές αξιολόγησης: ` + U.d2s(e.withClass, 0)), h('span.badge', L`Χρόνος: ` + U.d2s(r.time, 2) + ' s')));
    if (e.nominal) {
      box.appendChild(h('div.stats',
        metric(L`Ακρίβεια`, U.d2s(e.pctCorrect(), 2) + '%', U.d2s(e.correct, 0) + L` σωστές από ` + U.d2s(e.withClass, 0), true),
        metric('Kappa', U.d2s(e.kappa(), 4), L`συμφωνία πέρα από τύχη`),
        metric('MAE', U.d2s(e.meanAbsoluteError(), 4), L`μέσο απόλυτο σφάλμα`),
        metric('RMSE', U.d2s(e.rootMeanSquaredError(), 4), L`ρίζα μέσου τετρ. σφάλματος`),
        metric(L`ROC AUC (σταθμ.)`, U.d2s(e.weighted(e.rocArea), 3), L`περιοχή κάτω από ROC`),
        metric(L`F-Measure (σταθμ.)`, U.d2s(e.weighted(e.fMeasure), 3), 'precision/recall')));
      const labels = e.header.classAttr.values;
      const det = h('table.tbl.compact', h('thead', h('tr', h('th', L`Κλάση`), h('th.num', 'TP Rate'), h('th.num', 'FP Rate'), h('th.num', 'Precision'), h('th.num', 'Recall'), h('th.num', 'F-Measure'), h('th.num', 'MCC'), h('th.num', 'ROC'), h('th.num', 'PRC'))),
        h('tbody', labels.map((l, c) => h('tr', h('td', h('span', { style: { display: 'inline-block', width: '9px', height: '9px', borderRadius: '3px', background: App.color(c), marginRight: '7px' } }), l),
          [e.truePositiveRate(c), e.falsePositiveRate(c), e.precision(c), e.recall(c), e.fMeasure(c), e.mcc(c), e.rocArea(c), e.prcArea(c)].map((v) => h('td.num', App.fmt(v, 3))))),
          h('tr', h('td', h('b', L`Σταθμ. μέσος`)), [e.truePositiveRate, e.falsePositiveRate, e.precision, e.recall, e.fMeasure, e.mcc, e.rocArea, e.prcArea].map((f) => h('td.num', h('b', App.fmt(e.weighted(f), 3)))))));
      box.appendChild(h('div.grid', { style: { gridTemplateColumns: 'minmax(0,1fr)', gap: '18px' } },
        h('div', h('div.field-label', { style: { marginBottom: '8px' } }, L`Πίνακας σύγχυσης`), C.confusion(e.confusion, labels)),
        h('div', h('div.field-label', { style: { marginBottom: '8px' } }, L`Ακρίβεια ανά κλάση`), h('div.table-wrap', det))));
    } else {
      box.appendChild(h('div.stats',
        metric(L`Συσχέτιση`, U.d2s(e.correlationCoefficient(), 4), L`πραγματικές vs προβλέψεις`, true),
        metric('MAE', U.d2s(e.meanAbsoluteError(), 4), L`μέσο απόλυτο σφάλμα`),
        metric('RMSE', U.d2s(e.rootMeanSquaredError(), 4), L`ρίζα μέσου τετρ. σφάλματος`),
        metric('RAE', U.d2s(e.relativeAbsoluteError(), 2) + '%', L`σχετικό απόλυτο σφάλμα`),
        metric('RRSE', U.d2s(e.rootRelativeSquaredError(), 2) + '%', L`σχετ. τετρ. σφάλμα`)));
      const preds = e.predictions.filter((p) => p.predicted === p.predicted);
      const sc = C.scatter({ xs: preds.map((p) => p.actual), ys: preds.map((p) => p.predicted), colors: preds.map((p) => App.color(0)), xLabel: L`Πραγματική τιμή`, yLabel: L`Πρόβλεψη`, height: 380,
        info: (i) => L`Πραγματική: <b>${U.d2s(preds[i].actual, 4)}</b><br>Πρόβλεψη: <b>${U.d2s(preds[i].predicted, 4)}</b><br>Σφάλμα: ${U.d2s(preds[i].predicted - preds[i].actual, 4)}` });
      box.appendChild(h('div', h('div.field-label', { style: { marginBottom: '8px' } }, L`Πραγματικές τιμές έναντι προβλέψεων`), sc));
    }
    return box;
  }

  function modelTab(entry) {
    const r = entry.res;
    const box = h('div');
    const m = r.model;
    if (m.graph && r.data.classIsNominal()) {
      try {
        const g = m.graph();
        box.appendChild(h('div.row', { style: { padding: '10px 16px', borderBottom: '1px solid var(--border)' } }, h('span.field-label', L`Γραφική απεικόνιση δέντρου`), h('div', { style: { flex: 1 } }), C.legend(r.data.classAttr.values)));
        const tw = h('div.tree-wrap', { style: { borderRadius: 0 } }, C.tree(g, (c) => App.color(c)));
        box.appendChild(tw);
        requestAnimationFrame(() => { tw.scrollLeft = (tw.scrollWidth - tw.clientWidth) / 2; });
      } catch (e) { console.warn(e); }
    } else if (m.graph) {
      try { box.appendChild(h('div.tree-wrap', { style: { borderRadius: 0 } }, C.tree(m.graph(), () => App.color(0)))); } catch (e) { console.warn(e); }
    }
    box.appendChild(h('div.row', { style: { padding: '10px 16px', borderBottom: '1px solid var(--border)', borderTop: m.graph ? '1px solid var(--border)' : null } }, h('span.field-label', L`Μοντέλο (κείμενο)`), h('div', { style: { flex: 1 } }),
      h('button.btn.sm', { on: { click: () => App.copy(m.toString()) } }, App.icon('copy'), L`Αντιγραφή`)));
    box.appendChild(h('pre.output', m.toString()));
    return box;
  }

  function predsTab(entry) {
    const r = entry.res, e = r.evaluation, d = entry.data;
    const box = h('div');
    const nominal = e.nominal;
    const ca = d.classAttr;
    const rowIndex = new Map(d.rows.map((row, i) => [row, i]));
    const preds = e.predictions;
    const nErr = nominal ? preds.filter((p) => p.predicted !== p.actual).length : 0;
    const tools = h('div.row', { style: { padding: '10px 16px', borderBottom: '1px solid var(--border)' } });
    if (nominal) tools.append(h('label.check', h('input', { type: 'checkbox', checked: st.onlyErr, on: { change: (ev) => { st.onlyErr = ev.target.checked; App.go('classify'); } } }), L`Μόνο λάθη (${nErr})`));
    tools.append(h('div', { style: { flex: 1 } }), h('button.btn.sm', { on: { click: () => App.download('predictions.csv', predCSV()) } }, App.icon('download'), 'CSV'));
    box.appendChild(tools);
    const shown = (nominal && st.onlyErr ? preds.filter((p) => p.predicted !== p.actual) : preds).slice(0, 3000);
    const tbl = h('table.tbl.compact', h('thead', h('tr', h('th.num', '#'), h('th.num', L`Εγγραφή`), h('th', L`Πραγματική`), h('th', L`Πρόβλεψη`), nominal ? h('th', L`Σωστό;`) : h('th.num', L`Σφάλμα`), nominal ? h('th.num', L`Πιθανότητα`) : null, nominal ? h('th', L`Κατανομή`) : null)),
      h('tbody', shown.map((p, k) => {
        const ok = nominal ? p.actual === p.predicted : null;
        const idx = rowIndex.get(p.row);
        return h('tr', h('td.num.muted', k + 1), h('td.num', idx === undefined ? '—' : idx + 1),
          h('td', nominal ? ca.values[p.actual] : U.d2s(p.actual, 4)),
          h('td' + (nominal && !ok ? '.err' : ''), nominal ? (p.predicted >= 0 ? ca.values[p.predicted] : '?') : U.d2s(p.predicted, 4)),
          nominal ? h('td', ok ? h('span.badge.success', '✓') : h('span.badge.danger', '✗')) : h('td.num', U.d2s(p.predicted - p.actual, 4)),
          nominal ? h('td.num', U.d2s(p.dist[p.predicted] || 0, 3)) : null,
          nominal ? h('td', h('div', { style: { display: 'flex', height: '9px', width: '140px', borderRadius: '5px', overflow: 'hidden', background: 'var(--surface-3)' }, title: p.dist.map((v, c) => ca.values[c] + ': ' + U.d2s(v, 3)).join('\n') },
            p.dist.map((v, c) => h('div', { style: { width: v * 100 + '%', background: App.color(c) } })))) : null);
      })));
    function predCSV() {
      const lines = [['inst#', 'row', 'actual', 'predicted', nominal ? 'error' : 'diff', ...(nominal ? ca.values.map((v) => 'p(' + v + ')') : [])].join(',')];
      preds.forEach((p, k) => {
        const idx = rowIndex.get(p.row);
        lines.push([k + 1, idx === undefined ? '' : idx + 1, nominal ? ca.values[p.actual] : p.actual, nominal ? (p.predicted >= 0 ? ca.values[p.predicted] : '?') : p.predicted, nominal ? (p.actual !== p.predicted ? '+' : '') : p.predicted - p.actual, ...(nominal ? p.dist.map((v) => U.d2s(v, 4)) : [])].join(','));
      });
      return lines.join('\n') + '\n';
    }
    // οπτικοποίηση λαθών (όπως "Visualize classifier errors")
    if (nominal) {
      const numAtts = d.attributes.map((a, j) => j).filter((j) => j !== d.classIndex && d.attributes[j].type !== 'string');
      if (numAtts.length) {
        if (st.errX === null || st.errX >= d.numAttributes || st.errX === d.classIndex) st.errX = numAtts[0];
        if (st.errY === null || st.errY >= d.numAttributes || st.errY === d.classIndex) st.errY = numAtts[Math.min(1, numAtts.length - 1)];
        const sel = (key) => h('select.select', { style: { width: 'auto', height: '30px' }, on: { change: (ev) => { st[key] = Number(ev.target.value); App.go('classify'); } } }, numAtts.map((j) => h('option', { value: j, selected: j === st[key] }, d.attributes[j].name)));
        const valid = preds.filter((p) => p.row);
        const xs = valid.map((p) => p.row[st.errX]), ys = valid.map((p) => p.row[st.errY]);
        const ax = d.attributes[st.errX], ay = d.attributes[st.errY];
        const sc = C.scatter({ xs, ys, colors: valid.map((p) => App.color(Math.max(0, p.predicted))), marks: valid.map((p) => p.predicted !== p.actual), xLabel: ax.name, yLabel: ay.name,
          xCats: ax.type === 'nominal' ? ax.values : null, yCats: ay.type === 'nominal' ? ay.values : null, jitter: ax.type === 'nominal' || ay.type === 'nominal' ? 1 : 0, height: 360, size: 3.4,
          info: (i) => { const p = valid[i]; return L`<b>${p.predicted === p.actual ? L`✓ Σωστό` : L`✗ Λάθος`}</b><br>Πραγματική: ${App.esc(ca.values[p.actual])}<br>Πρόβλεψη: ${App.esc(ca.values[p.predicted] || '?')}<br>${App.esc(ax.name)}: ${App.esc(d.stringValue(p.row, st.errX))}<br>${App.esc(ay.name)}: ${App.esc(d.stringValue(p.row, st.errY))}`; } });
        box.appendChild(h('div.card-body', { style: { borderBottom: '1px solid var(--border)' } },
          h('div.row', { style: { marginBottom: '10px' } }, h('span.field-label', L`Οπτικοποίηση λαθών`), h('div', { style: { flex: 1 } }), h('span.small.muted', 'X'), sel('errX'), h('span.small.muted', 'Y'), sel('errY')),
          sc, h('div.row', { style: { marginTop: '8px', justifyContent: 'space-between' } }, C.legend(ca.values), h('span.muted.small', L`● σωστή πρόβλεψη · ■ λάθος (χρώμα = πρόβλεψη) · σύρε για zoom`))));
      }
    }
    box.appendChild(h('div.table-wrap', { style: { maxHeight: '520px' } }, tbl));
    if (shown.length >= 3000) box.appendChild(h('div.muted.small', { style: { padding: '10px 16px' } }, L`Εμφανίζονται οι πρώτες 3.000 προβλέψεις. Κατέβασε το CSV για όλες.`));
    return box;
  }

  function rocTab(entry) {
    const e = entry.res.evaluation, labels = e.header.classAttr.values;
    if (st.rocClass >= labels.length) st.rocClass = 0;
    const c = st.rocClass;
    const tc = e.thresholdCurve(c);
    const sel = h('select.select', { style: { width: 'auto' }, on: { change: (ev) => { st.rocClass = Number(ev.target.value); App.go('classify'); } } }, labels.map((l, k) => h('option', { value: k, selected: k === c }, l)));
    const allCurves = labels.map((l, k) => ({ label: l, color: App.color(k), points: e.thresholdCurve(k).points }));
    return h('div.card-body',
      h('div.row', { style: { marginBottom: '14px' } }, h('span.field-label', L`Θετική κλάση`), sel, h('span.badge.primary', 'ROC AUC = ' + App.fmt(e.rocArea(c), 4)), h('span.badge', 'PRC AUC = ' + App.fmt(e.prcArea(c), 4)), h('span.badge', L`Θετικά: ${U.d2s(tc.P, 0)} · Αρνητικά: ${U.d2s(tc.N, 0)}`)),
      h('div.grid.cols-2',
        h('div', h('div.field-label', L`Καμπύλη ROC`), C.roc([{ color: App.color(c), points: tc.points }])),
        h('div', h('div.field-label', L`Καμπύλη Precision–Recall`), C.roc([{ color: App.color(c), points: tc.points.slice(1) }], { pr: true, xLabel: 'Recall', yLabel: 'Precision' }))),
      h('div', { style: { marginTop: '16px' } }, h('div.field-label', L`Όλες οι κλάσεις (one-vs-rest)`), h('div.grid.cols-2', h('div', C.roc(allCurves)), h('div', { style: { alignSelf: 'center' } }, C.legend(labels.map((l, k) => `${l} (AUC ${App.fmt(e.rocArea(k), 3)})`))))));
  }
})();
