/* Weka Web — σελίδες: Ομαδοποίηση, Συσχετίσεις, Επιλογή χαρακτηριστικών. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, App = window.App, h = App.h, C = App.charts;

  function resultsCard(key, st, scoreFn, page) {
    const results = App.state.results[key];
    const card = h('div.card', App.cardHead('list', L`Αποτελέσματα`, results.length ? h('button.btn.ghost.sm', { on: { click: () => { App.state.results[key] = []; st.active = -1; App.go(page); } } }, L`Καθαρισμός`) : null));
    const list = h('div.results', { style: { padding: '12px' } });
    if (!results.length) list.appendChild(h('div.muted.small', { style: { padding: '6px' } }, L`Τα αποτελέσματα κάθε εκτέλεσης εμφανίζονται εδώ.`));
    results.forEach((r, i) => list.appendChild(h('div.result-item' + (i === st.active ? '.active' : ''), { on: { click: () => { st.active = i; App.go(page); } } },
      h('div.meta', h('div.t', r.name), h('div.s', App.timeStr(r.time) + ' · ' + r.relation)), h('div.score', scoreFn(r)),
      h('button.btn.ghost.sm.icon.x', { on: { click: (ev) => { ev.stopPropagation(); results.splice(i, 1); if (st.active >= results.length) st.active = results.length - 1; App.go(page); } } }, App.icon('x')))));
    card.appendChild(list);
    return card;
  }
  function textBlock(text, name) {
    return h('div', h('div.row', { style: { padding: '10px 16px', borderBottom: '1px solid var(--border)' } }, h('div', { style: { flex: 1 } }),
      h('button.btn.sm', { on: { click: () => App.copy(text) } }, App.icon('copy'), L`Αντιγραφή`), h('button.btn.sm', { on: { click: () => App.download(name, text) } }, App.icon('download'), L`Λήψη`)), h('pre.output', text));
  }
  function tabsBar(defs, st, page) {
    if (!defs.some((t) => t[0] === st.tab)) st.tab = defs[0][0];
    return h('div.tabs', defs.map(([k, l, ic]) => h('button' + (st.tab === k ? '.active' : ''), { on: { click: () => { st.tab = k; App.go(page); } } }, App.icon(ic), l)));
  }

  // ================================================================ CLUSTER
  const cs = { mode: 'train', pct: 66, seed: 1, ignore: new Set(), active: -1, tab: 'summary', x: null, y: null, color: 'cluster', xtab: null };
  App.pages.cluster = {
    render() {
      const d = App.state.data;
      if (!d) return h('div.page', App.pageHead(L`Ομαδοποίηση`, L`Ανακάλυψη ομάδων σε δεδομένα χωρίς ετικέτες.`), App.needData());
      for (const j of [...cs.ignore]) if (j >= d.numAttributes) cs.ignore.delete(j);
      const page = h('div.page', App.pageHead(L`Ομαδοποίηση`, L`Ανακάλυψη ομάδων σε δεδομένα χωρίς ετικέτες.`));
      const left = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '18px' } }), right = h('div.result-view', { style: { minWidth: 0 } });
      page.appendChild(h('div.split', left, right));
      const cfg = App.algoConfig({ items: W.clusterers, storeKey: 'clusterer', defaultName: 'SimpleKMeans' });
      left.appendChild(h('div.card', App.cardHead('cluster', L`Αλγόριθμος`), h('div.card-body', cfg.el)));
      const modes = [['train', L`Εκπαίδευσης`], ['classes', 'Classes to clusters'], ['split', L`Ποσοστό`], ['test', 'Test set']];
      const seg = h('div.seg', modes.map(([k, l]) => h('button' + (cs.mode === k ? '.active' : ''), { on: { click: () => { cs.mode = k; App.go('cluster'); } } }, l)));
      const extra = h('div', { style: { marginTop: '12px' } });
      if (cs.mode === 'classes') extra.append(h('div.field', h('label', L`Κλάση για αξιολόγηση (αγνοείται στην ομαδοποίηση)`), App.classSelect(() => App.go('cluster'))));
      if (cs.mode === 'split') extra.append(h('div.grid.cols-2', { style: { gap: '10px' } },
        h('div.field', h('label', L`Εκπαίδευση (%)`), h('input.input', { type: 'number', value: cs.pct, on: { input: (e) => { cs.pct = Number(e.target.value) || 66; } } })),
        h('div.field', h('label', 'Seed'), h('input.input', { type: 'number', value: cs.seed, on: { input: (e) => { cs.seed = Number(e.target.value) || 0; } } }))));
      if (cs.mode === 'test') extra.append(h('div.row', h('button.btn.sm', { on: { click: async () => { const f = await App.openFileDialog(); if (!f) return; try { App.state.testSet = await App.parseFile(f); App.state.testSetName = f.name; App.go('cluster'); } catch (e) { App.error(e); } } } }, App.icon('upload'), L`Φόρτωση test set`),
        App.state.testSet ? h('span.badge.success', App.state.testSetName) : h('span.muted.small', L`κανένα`)));
      // ignore attributes
      const ign = h('div.pill-list', { style: { maxHeight: '150px', overflow: 'auto' } }, d.attributes.map((a, j) => {
        const forced = cs.mode === 'classes' && j === d.classIndex;
        const on = cs.ignore.has(j) || forced;
        return h('button.pill', { style: { border: '1px solid ' + (on ? 'var(--danger)' : 'transparent'), cursor: forced ? 'default' : 'pointer', textDecoration: on ? 'line-through' : 'none', opacity: on ? 0.7 : 1 },
          on: { click: () => { if (forced) return; if (cs.ignore.has(j)) cs.ignore.delete(j); else cs.ignore.add(j); App.go('cluster'); } } }, a.name);
      }));
      const runner = App.runner();
      const startBtn = h('button.btn.primary.lg.block', { on: { click: start } }, App.icon('play'), L`Εκτέλεση`);
      left.appendChild(h('div.card', App.cardHead('target', L`Τρόπος αξιολόγησης`), h('div.card-body', seg, extra,
        h('div.field', { style: { marginTop: '12px' } }, h('label', L`Αγνόηση χαρακτηριστικών (κλικ για εναλλαγή)`), ign),
        startBtn, h('div', { style: { marginTop: '10px' } }, runner.el))));
      left.appendChild(resultsCard('cluster', cs, (r) => r.res.k + L` ομάδες`, 'cluster'));
      const active = App.state.results.cluster[cs.active];
      right.appendChild(active ? clusterView(active) : h('div.card', h('div.empty', App.icon('cluster'), h('h3', L`Έτοιμο για ομαδοποίηση`), h('p', L`Διάλεξε αλγόριθμο και πάτησε «Εκτέλεση».`))));
      async function start() {
        const { spec, opts } = cfg.get();
        startBtn.disabled = true; runner.start();
        try {
          const res = await W.runClusterer({ data: App.state.data, spec, options: opts, mode: cs.mode, ignore: [...cs.ignore], percentage: cs.pct, seed: cs.seed, testData: App.state.testSet, onProgress: (p, m) => runner.progress(p, m) });
          App.state.results.cluster.push({ name: spec.name, time: new Date(), res, relation: App.state.data.relation, data: App.state.data });
          cs.active = App.state.results.cluster.length - 1;
          runner.done(L`Ολοκληρώθηκε σε ` + U.d2s(res.buildTime, 2) + ' s');
          App.go('cluster');
          App.revealResult();
        } catch (e) { runner.fail(''); App.error(e); } finally { startBtn.disabled = false; }
      }
      return page;
    },
  };

  function clusterView(entry) {
    const r = entry.res, d = entry.data;
    const defs = [['summary', L`Σύνοψη`, 'chart'], ['viz', L`Οπτικοποίηση`, 'scatter']];
    if (r.model instanceof Object && r.model.centroids) defs.push(['centroids', L`Κέντρα`, 'table']);
    if (r.spec.dendrogram) defs.push(['dendro', L`Δενδρόγραμμα`, 'tree']);
    defs.push(['text', L`Πλήρης έξοδος`, 'text']);
    const tabs = tabsBar(defs, cs, 'cluster');
    const body = h('div');
    const card = h('div.card', h('div.card-head', h('div', h('h3', entry.name), h('div.muted.small', r.spec.path + (r.optStr ? ' ' + r.optStr : ''))), h('div.spacer'),
      r.evalRows ? h('button.btn.sm', { title: L`Προσθήκη χαρακτηριστικού cluster στα δεδομένα (AddCluster)`, on: { click: addClusterAttr } }, App.icon('plus'), L`Προσθήκη ως χαρακτηριστικό`) : null), tabs, body);
    function addClusterAttr() {
      const cur = App.state.data;
      if (cur !== d) { App.toast(L`Τα δεδομένα έχουν αλλάξει από την εκτέλεση· ξανατρέξε την ομαδοποίηση.`, 'error'); return; }
      const attrs = d.attributes.concat([{ name: 'cluster', type: 'nominal', values: Array.from({ length: r.k }, (_, i) => 'cluster' + (i + 1)) }]);
      const nd = new W.Dataset(d.relation + '_clustered', attrs, d.rows.map((row, i) => row.concat([r.assignments[i]])), d.weights.slice(), d.classIndex);
      App.applyChange(nd, 'AddCluster (' + entry.name + ')');
      App.toast(L`Προστέθηκε το χαρακτηριστικό "cluster".`, 'success');
    }
    if (cs.tab === 'summary') {
      const tot = U.sum(r.counts);
      const b = h('div.card-body', { style: { display: 'flex', flexDirection: 'column', gap: '18px' } });
      const stats = h('div.stats', h('div.stat.primary', h('div.label', L`Ομάδες`), h('div.value', r.k), h('div.sub', L`Χρόνος: ` + U.d2s(r.buildTime, 2) + ' s')));
      if (r.model.sse) stats.appendChild(h('div.stat', h('div.label', L`SSE (εντός ομάδων)`), h('div.value', U.d2s(U.sum(r.model.sse), 3)), h('div.sub', r.model.iterations + L` επαναλήψεις`)));
      if (r.model.loglik !== undefined) stats.appendChild(h('div.stat', h('div.label', 'Log likelihood'), h('div.value', U.d2s(r.model.loglik, 4)), h('div.sub', r.model.iterations + L` επαναλήψεις`)));
      if (r.classes) stats.appendChild(h('div.stat', h('div.label', L`Λάθος ομαδοποιημένες`), h('div.value', U.d2s((100 * r.classes.errors) / U.sum(r.classes.matrix.flat()), 2) + '%'), h('div.sub', U.d2s(r.classes.errors, 0) + L` εγγραφές`)));
      b.appendChild(stats);
      if (r.evalRows) {
        const noms = d.attributes.map((a, j) => j).filter((j) => d.attributes[j].type === 'nominal');
        if (noms.length) {
          if (cs.xtab === null || !noms.includes(cs.xtab)) cs.xtab = d.classIndex >= 0 && d.classIsNominal() ? d.classIndex : noms[noms.length - 1];
          const ct = App.analysis.crosstab(d, r.assignments, r.k, cs.xtab);
          const counts = ct.labels.map((_, j) => U.sum(ct.m.map((row) => row[j])));
          const posIdx = App.analysis.positiveIndex(ct.labels, counts, App.state.cost.pos);
          const sel = h('select.select', { style: { width: 'auto', height: '30px' }, on: { change: (e) => { cs.xtab = Number(e.target.value); App.go('cluster'); } } }, noms.map((j) => h('option', { value: j, selected: j === cs.xtab }, d.attributes[j].name)));
          b.appendChild(h('div', h('div.row', { style: { marginBottom: '8px' } }, h('div.field-label', L`Segments vs. attribute (e.g. conversion rate per cluster)`), h('div', { style: { flex: 1 } }), sel), App.analysis.crosstabTable(ct, posIdx)));
        }
      }
      b.appendChild(h('div', h('div.field-label', { style: { marginBottom: '8px' } }, L`Μέγεθος ομάδων`), C.bars(r.counts.map((c, i) => ({ label: L`Ομάδα ` + i, value: c, color: App.color(i) })), { fmt: (v) => U.d2s(v, 0) + ' (' + U.d2s((100 * v) / tot, 0) + '%)', labelW: 90 })));
      if (r.classes) {
        const cl = r.classes;
        const mt = cl.matrix[0].map((_, j) => cl.matrix.map((row) => row[j])); // κλάσεις × ομάδες
        b.appendChild(h('div', h('div.field-label', { style: { marginBottom: '8px' } }, L`Classes to clusters (γραμμές: κλάσεις, στήλες: ομάδες)`),
          h('div.table-wrap', h('table.tbl.compact', h('thead', h('tr', h('th', L`Κλάση`), Array.from({ length: r.k }, (_, c) => h('th.num', L`Ομάδα ` + c)))),
            h('tbody', mt.map((row, j) => h('tr', h('td', cl.labels[j]), row.map((v, c) => h('td.num', { style: cl.map[c] === j ? { fontWeight: 700, color: 'var(--primary)' } : {} }, U.d2s(v, 0))))),
              h('tr', h('td.muted', L`Αντιστοίχιση`), cl.map.map((m) => h('td.num.muted', m < 0 ? '—' : cl.labels[m]))))))));
      }
      body.appendChild(b);
    } else if (cs.tab === 'viz') {
      if (!r.evalRows) { body.appendChild(h('div.card-body', h('div.muted', L`Η οπτικοποίηση είναι διαθέσιμη όταν η αξιολόγηση γίνεται στα δεδομένα εκπαίδευσης.`))); return card; }
      const atts = d.attributes.map((a, j) => j).filter((j) => d.attributes[j].type !== 'string');
      if (cs.x === null || cs.x >= d.numAttributes) cs.x = atts[0];
      if (cs.y === null || cs.y >= d.numAttributes) cs.y = atts[Math.min(1, atts.length - 1)];
      const sel = (k) => h('select.select', { style: { width: 'auto', height: '30px' }, on: { change: (e) => { cs[k] = Number(e.target.value); App.go('cluster'); } } }, atts.map((j) => h('option', { value: j, selected: j === cs[k] }, d.attributes[j].name)));
      const ax = d.attributes[cs.x], ay = d.attributes[cs.y];
      const nomAtts = d.attributes.map((a, j) => j).filter((j) => d.attributes[j].type === 'nominal');
      const colorAtt = cs.color === 'cluster' || !nomAtts.includes(Number(cs.color)) ? null : Number(cs.color);
      const colorSel = h('select.select', { style: { width: 'auto', height: '30px' }, on: { change: (e) => { cs.color = e.target.value; App.go('cluster'); } } },
        h('option', { value: 'cluster', selected: colorAtt === null }, L`Colour: cluster`), nomAtts.map((j) => h('option', { value: j, selected: j === colorAtt }, L`Colour: ${d.attributes[j].name}`)));
      const ptColors = colorAtt === null ? r.assignments.map((c) => App.color(c)) : d.rows.map((row) => (row[colorAtt] === row[colorAtt] ? App.color(row[colorAtt]) : '#9ca3af'));
      const sc = C.scatter({ xs: d.rows.map((row) => row[cs.x]), ys: d.rows.map((row) => row[cs.y]), colors: ptColors, xLabel: ax.name, yLabel: ay.name,
        xCats: ax.type === 'nominal' ? ax.values : null, yCats: ay.type === 'nominal' ? ay.values : null, jitter: ax.type === 'nominal' || ay.type === 'nominal' ? 1 : 0, height: 440,
        marks: r.classes ? d.rows.map((row, i) => { const m = r.classes.map[r.assignments[i]]; return row[d.classIndex] === row[d.classIndex] && m !== row[d.classIndex]; }) : null,
        info: (i) => L`<b>Εγγραφή ${i + 1}</b> · Ομάδα ${r.assignments[i]}<br>` + d.attributes.slice(0, 12).map((a, j) => `${App.esc(a.name)}: ${App.esc(d.stringValue(d.rows[i], j))}`).join('<br>') });
      body.appendChild(h('div.card-body', h('div.row', { style: { marginBottom: '10px' } }, h('span.small.muted', 'X'), sel('x'), h('span.small.muted', 'Y'), sel('y'), colorSel), sc,
        h('div.row', { style: { marginTop: '8px', justifyContent: 'space-between' } }, colorAtt === null ? C.legend(Array.from({ length: r.k }, (_, c) => L`Ομάδα ` + c)) : C.legend(d.attributes[colorAtt].values), r.classes ? h('span.muted.small', L`■ = λάθος ομαδοποίηση ως προς την κλάση`) : null)));
    } else if (cs.tab === 'centroids') {
      const m = r.model, hd = m.header;
      const rows = hd.attributes.map((a, j) => {
        const cell = (v) => (v !== v ? '?' : a.type === 'nominal' ? a.values[v] : U.d2s(v, 4));
        return h('tr', h('td', a.name), h('td.num', cell(m.full[j])), m.centroids.map((c, ci) => h('td.num', { style: { color: App.color(ci) } }, cell(c[j]))));
      });
      body.appendChild(h('div.table-wrap', h('table.tbl', h('thead', h('tr', h('th', L`Χαρακτηριστικό`), h('th.num', L`Όλα (` + U.d2s(m.fullSize, 0) + ')'), m.centroids.map((_, ci) => h('th.num', L`Ομάδα ` + ci + ' (' + U.d2s(m.sizes[ci], 0) + ')')))), h('tbody', rows))));
    } else if (cs.tab === 'dendro') {
      const m = r.model;
      const leafCluster = new Map();
      m.clusters.forEach((cn, ci) => { const walk = (n) => { if (n.leaf) leafCluster.set(n, ci); else { walk(n.left); walk(n.right); } }; walk(cn); });
      const colorOf = (node) => { const idx = m.clusters.indexOf(node); if (idx >= 0) return App.color(idx); let found = null; const walk = (n) => { if (found !== null) return; if (n.leaf) found = leafCluster.get(n); else { walk(n.left); walk(n.right); } }; walk(node); return m.clusters.some((c) => isInside(c, node)) ? App.color(found) : 'var(--text-3)'; };
      const isInside = (c, node) => { if (c === node) return true; if (c.leaf) return false; return isInside(c.left, node) || isInside(c.right, node); };
      body.appendChild(h('div.tree-wrap', { style: { padding: '10px' } }, C.dendrogram(m.root, m.k, (l) => String(l.id + 1), colorOf)));
      body.appendChild(h('div.card-body', C.legend(Array.from({ length: m.k }, (_, c) => L`Ομάδα ` + c))));
    } else body.appendChild(textBlock(r.report, 'clusterer.txt'));
    return card;
  }

  // ============================================================ ASSOCIATE
  const as = { active: -1, tab: 'rules', sort: 'conf' };
  App.pages.associate = {
    render() {
      const d = App.state.data;
      if (!d) return h('div.page', App.pageHead(L`Κανόνες συσχέτισης`, L`Εύρεση κανόνων της μορφής «αν Χ τότε Υ».`), App.needData());
      const page = h('div.page', App.pageHead(L`Κανόνες συσχέτισης`, L`Εύρεση κανόνων της μορφής «αν Χ τότε Υ».`));
      const left = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '18px' } }), right = h('div.result-view', { style: { minWidth: 0 } });
      page.appendChild(h('div.split', left, right));
      const cfg = App.algoConfig({ items: W.associators, storeKey: 'associator', defaultName: 'Apriori' });
      const runner = App.runner();
      const nonNominal = d.attributes.filter((a) => a.type !== 'nominal').length;
      const startBtn = h('button.btn.primary.lg.block', { on: { click: start } }, App.icon('play'), L`Εκτέλεση`);
      left.appendChild(h('div.card', App.cardHead('link', 'Associator'), h('div.card-body', cfg.el,
        nonNominal ? h('div.callout.warn', { style: { marginTop: '12px' } }, App.icon('info'), h('div', L`Υπάρχουν ${nonNominal} μη-nominal χαρακτηριστικά. Εφάρμοσε πρώτα Discretize/NumericToNominal ή αφαίρεσέ τα.`,
          h('div', { style: { marginTop: '8px' } }, h('button.btn.sm', { on: { click: quickDiscretize } }, App.icon('wand'), L`Γρήγορη διακριτοποίηση (10 bins)`)))) : null,
        h('div', { style: { marginTop: '14px' } }, startBtn), h('div', { style: { marginTop: '10px' } }, runner.el))));
      left.appendChild(resultsCard('associate', as, (r) => r.res.rules.length + L` κανόνες`, 'associate'));
      const active = App.state.results.associate[as.active];
      right.appendChild(active ? assocView(active) : h('div.card', h('div.empty', App.icon('link'), h('h3', L`Έτοιμο για εξόρυξη κανόνων`), h('p', L`Δοκίμασε το Apriori με τα δεδομένα weather.nominal ή supermarket.`))));
      function quickDiscretize() {
        const cur = App.state.data;
        const disc = W.findFilter('Discretize').create({ B: 10, ignoreClass: true }); disc.setInput(cur);
        let nd = disc.apply(cur);
        const s2n = W.findFilter('StringToNominal').create({}); s2n.setInput(nd); nd = s2n.apply(nd);
        App.applyChange(nd, L`Discretize -B 10 (για συσχετίσεις)`);
        App.go('associate');
      }
      async function start() {
        const { spec, opts } = cfg.get();
        startBtn.disabled = true; runner.start();
        try {
          const res = await W.runAssociator({ data: App.state.data, spec, options: opts });
          App.state.results.associate.push({ name: spec.name, time: new Date(), res, relation: App.state.data.relation });
          as.active = App.state.results.associate.length - 1;
          runner.done(L`${res.rules.length} κανόνες σε ${U.d2s(res.time, 2)} s`);
          App.go('associate');
          App.revealResult();
        } catch (e) { runner.fail(''); App.error(e); } finally { startBtn.disabled = false; }
      }
      return page;
    },
  };
  function assocView(entry) {
    const r = entry.res;
    const tabs = tabsBar([['rules', L`Κανόνες`, 'list'], ['text', L`Πλήρης έξοδος`, 'text']], as, 'associate');
    const body = h('div');
    if (as.tab === 'rules') {
      const rules = r.rules.slice();
      const keys = { conf: 'Confidence', lift: 'Lift', lev: 'Leverage', conv: 'Conviction', support: 'Support' };
      if (as.sort !== 'rank') rules.sort((a, b) => b[as.sort] - a[as.sort]);
      const mx = Math.max(...r.rules.map((x) => x.support), 1);
      const th = (k, l) => h('th.num.sortable', { on: { click: () => { as.sort = k; App.go('associate'); } } }, l, as.sort === k ? ' ▼' : '');
      body.appendChild(h('div.table-wrap', { style: { maxHeight: '640px' } }, h('table.tbl', h('thead', h('tr', h('th.num.sortable', { on: { click: () => { as.sort = 'rank'; App.go('associate'); } } }, '#'), h('th', L`Αν (premise)`), h('th', ''), h('th', L`Τότε (consequence)`), th('support', 'Support'), th('conf', 'Conf.'), th('lift', 'Lift'), th('lev', 'Lev.'), th('conv', 'Conv.'))),
        h('tbody', rules.map((x) => h('tr', h('td.num.muted', r.rules.indexOf(x) + 1),
          h('td', { style: { whiteSpace: 'normal', minWidth: '200px' } }, x.premise.split(/ (?=[^ =]+=)|, /).map((p) => h('span.pill', { style: { margin: '2px' } }, p))), h('td.muted', '⇒'),
          h('td', { style: { whiteSpace: 'normal', minWidth: '160px' } }, x.consequence.split(/ (?=[^ =]+=)|, /).map((p) => h('span.pill', { style: { margin: '2px', background: 'var(--primary-soft)', color: 'var(--primary)' } }, p))),
          h('td.num', h('div', U.d2s(x.support, 0)), h('div.bar', { style: { width: (60 * x.support) / mx + 'px', marginLeft: 'auto' } })), h('td.num', h('b', U.d2s(x.conf, 2))), h('td.num', U.d2s(x.lift, 2)), h('td.num', U.d2s(x.lev, 3)), h('td.num', U.d2s(x.conv, 2))))))));
      if (!rules.length) body.appendChild(h('div.empty', h('h3', L`Δεν βρέθηκαν κανόνες`), h('p', L`Μείωσε την ελάχιστη τιμή μετρικής ή το κάτω όριο support.`)));
    } else body.appendChild(textBlock(r.report, 'associations.txt'));
    return h('div.card', h('div.card-head', h('div', h('h3', entry.name), h('div.muted.small', r.spec.path + (r.optStr ? ' ' + r.optStr : '')))), tabs, body);
  }

  // ========================================================= SELECT ATTRS
  const ss = { mode: 'full', folds: 10, seed: 1, active: -1, tab: 'summary' };
  App.pages.select = {
    render() {
      const d = App.state.data;
      if (!d) return h('div.page', App.pageHead(L`Επιλογή χαρακτηριστικών`, L`Βρες τα πιο χρήσιμα χαρακτηριστικά για την πρόβλεψη της κλάσης.`), App.needData());
      const page = h('div.page', App.pageHead(L`Επιλογή χαρακτηριστικών`, L`Βρες τα πιο χρήσιμα χαρακτηριστικά για την πρόβλεψη της κλάσης.`));
      const left = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '18px' } }), right = h('div.result-view', { style: { minWidth: 0 } });
      page.appendChild(h('div.split', left, right));
      const groups = { single: L`Αξιολόγηση μεμονωμένων χαρακτηριστικών`, subset: L`Αξιολόγηση υποσυνόλων`, transform: L`Μετασχηματισμός` };
      const ev = App.algoConfig({ items: W.evaluators, storeKey: 'evaluator', defaultName: 'InfoGainAttributeEval', groupOf: (s) => s.kind, groupLabels: groups, title: 'Attribute Evaluator' });
      const se = App.algoConfig({ items: W.searches, storeKey: 'search', defaultName: 'Ranker', title: 'Search Method' });
      const runner = App.runner();
      const startBtn = h('button.btn.primary.lg.block', { on: { click: start } }, App.icon('play'), L`Εκτέλεση`);
      const seg = h('div.seg', [['full', L`Όλα τα δεδομένα`], ['cv', 'Cross-validation']].map(([k, l]) => h('button' + (ss.mode === k ? '.active' : ''), { on: { click: () => { ss.mode = k; App.go('select'); } } }, l)));
      left.appendChild(h('div.card', App.cardHead('select', L`Ρυθμίσεις`), h('div.card-body', ev.el, h('div.divider'), se.el, h('div.divider'),
        h('div.callout', { style: { marginBottom: '12px' } }, App.icon('info'), h('div', L`Ranker για μεμονωμένους evaluators (InfoGain, ReliefF, PCA…)· BestFirst/GreedyStepwise για CFS και Wrapper.`)),
        h('div.field', h('label', L`Χαρακτηριστικό κλάσης`), App.classSelect(() => App.go('select'), { allowNone: true })),
        h('div.field-label', { style: { marginBottom: '6px' } }, L`Τρόπος`), seg,
        ss.mode === 'cv' ? h('div.grid.cols-2', { style: { gap: '10px', marginTop: '10px' } }, h('div.field', h('label', 'Folds'), h('input.input', { type: 'number', value: ss.folds, on: { input: (e) => { ss.folds = Number(e.target.value) || 10; } } })), h('div.field', h('label', 'Seed'), h('input.input', { type: 'number', value: ss.seed, on: { input: (e) => { ss.seed = Number(e.target.value) || 0; } } }))) : null,
        h('div', { style: { marginTop: '14px' } }, startBtn), h('div', { style: { marginTop: '10px' } }, runner.el))));
      left.appendChild(resultsCard('select', ss, (r) => r.res.selected.length + L` επιλεγμένα`, 'select'));
      const active = App.state.results.select[ss.active];
      right.appendChild(active ? selectView(active) : h('div.card', h('div.empty', App.icon('select'), h('h3', L`Έτοιμο`), h('p', L`Διάλεξε evaluator και μέθοδο αναζήτησης και πάτησε «Εκτέλεση».`))));
      async function start() {
        const a = ev.get(), b = se.get();
        startBtn.disabled = true; runner.start();
        try {
          const res = await W.runAttributeSelection({ data: App.state.data, evalSpec: a.spec, evalOptions: a.opts, searchSpec: b.spec, searchOptions: b.opts, mode: ss.mode, folds: ss.folds, seed: ss.seed, onProgress: (p, m) => runner.progress(p, m) });
          App.state.results.select.push({ name: a.spec.name + ' + ' + b.spec.name, time: new Date(), res, relation: App.state.data.relation, data: App.state.data });
          ss.active = App.state.results.select.length - 1;
          runner.done(L`Ολοκληρώθηκε σε ` + U.d2s(res.time, 2) + ' s');
          App.go('select');
          App.revealResult();
        } catch (e) { runner.fail(''); App.error(e); } finally { startBtn.disabled = false; }
      }
      return page;
    },
  };
  function selectView(entry) {
    const r = entry.res;
    const tabs = tabsBar([['summary', L`Σύνοψη`, 'chart'], ['text', L`Πλήρης έξοδος`, 'text']], ss, 'select');
    const body = h('div');
    const applyBtn = h('button.btn.primary.sm', { on: { click: () => {
      if (App.state.data !== entry.data) App.toast(L`Προσοχή: τα τρέχοντα δεδομένα έχουν αλλάξει από την εκτέλεση· εφαρμόζεται το αποτέλεσμα της εκτέλεσης.`, 'info');
      App.applyChange(r.reduced, L`Επιλογή χαρακτηριστικών (` + entry.name + ')');
      App.toast(L`Το σύνολο έχει πλέον ${r.reduced.numAttributes} χαρακτηριστικά.`, 'success');
    } } }, App.icon('check'), r.evalSpec.kind === 'transform' ? L`Χρήση μετασχηματισμένων δεδομένων` : L`Κράτα μόνο τα επιλεγμένα`);
    const head = h('div.card-head', h('div', h('h3', entry.name), h('div.muted.small', r.selected.length + L` επιλεγμένα χαρακτηριστικά`)), h('div.spacer'),
      h('button.btn.sm', { on: { click: () => App.download(r.reduced.relation.replace(/[^\w.-]+/g, '_') + '.arff', W.toARFF(r.reduced)) } }, App.icon('download'), 'ARFF'), applyBtn);
    if (ss.tab === 'summary') {
      const b = h('div.card-body', { style: { display: 'flex', flexDirection: 'column', gap: '18px' } });
      if (r.ranking) {
        const selSet = new Set(r.selected);
        b.appendChild(h('div', h('div.field-label', { style: { marginBottom: '8px' } }, L`Κατάταξη χαρακτηριστικών (merit)`),
          C.bars(r.ranking.map((x) => ({ label: (x.index + 1) + '. ' + x.name, value: x.merit, color: selSet.has(x.index) ? 'var(--primary)' : 'var(--border-strong)' })), { labelW: 220, width: 640, fmt: (v) => U.d2s(v, 4) })));
      }
      if (r.merit !== undefined && !r.ranking) b.appendChild(h('div.stats', h('div.stat.primary', h('div.label', L`Merit βέλτιστου υποσυνόλου`), h('div.value', U.d2s(r.merit, 4)), h('div.sub', r.selected.length + L` χαρακτηριστικά`))));
      if (r.evalSpec.kind !== 'transform') b.appendChild(h('div', h('div.field-label', { style: { marginBottom: '8px' } }, L`Επιλεγμένα χαρακτηριστικά`), h('div.pill-list', r.selected.map((j) => h('span.pill', { style: { background: 'var(--primary-soft)', color: 'var(--primary)' } }, (j + 1) + '. ' + r.names[j])))));
      if (r.cvInfo && r.cvInfo.folds) b.appendChild(h('div', h('div.field-label', { style: { marginBottom: '8px' } }, L`Συχνότητα επιλογής σε ${r.cvInfo.numFolds} folds`),
        C.bars(r.cvInfo.folds.map((x) => ({ label: (x.j + 1) + '. ' + r.names[x.j], value: x.pct })), { labelW: 220, width: 640, fmt: (v) => U.d2s(v, 0) + '%' })));
      if (r.cvInfo && r.cvInfo.ranks) b.appendChild(h('div.table-wrap', h('table.tbl.compact', h('thead', h('tr', h('th', L`Χαρακτηριστικό`), h('th.num', L`Μέσο merit`), h('th.num', L`± τ.α.`), h('th.num', L`Μέση θέση`), h('th.num', L`± τ.α.`))),
        h('tbody', r.cvInfo.ranks.map((x) => h('tr', h('td', (x.j + 1) + '. ' + r.names[x.j]), h('td.num', U.d2s(x.am, 4)), h('td.num', U.d2s(x.sm, 4)), h('td.num', U.d2s(x.ar, 1)), h('td.num', U.d2s(x.sr, 2))))))));
      body.appendChild(b);
    } else body.appendChild(textBlock(r.report, 'attribute-selection.txt'));
    return h('div.card', head, tabs, body);
  }
})();
