/* Weka Web — Report page: collects the whole session (data, preprocessing, results, charts, notes) into a printable document → PDF. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, App = window.App, h = App.h, C = App.charts;
  const R = (App.report = {});
  const rs = App.store.get('report', { title: '', author: '', notes: '', sections: { data: true, attrs: true, prep: true, classify: true, select: true, cluster: true, associate: true, appendix: false } });
  rs.excluded = new Set();
  R.settings = rs;
  const save = () => App.store.set('report', { title: rs.title, author: rs.author, notes: rs.notes, sections: rs.sections });
  const keyOf = (kind, r) => kind + ':' + r.time.getTime();

  // --------------------------------------------------------- document
  R.build = function () {
    const d = App.state.data, res = App.state.results;
    const paper = h('div.paper');
    const title = rs.title || L`Data mining report — ${d ? d.relation : ''}`;
    paper.appendChild(h('header.paper-head',
      h('div.paper-kicker', 'Weka Web · ' + new Date().toLocaleDateString(I18N.locale(), { year: 'numeric', month: 'long', day: 'numeric' })),
      h('h1', title),
      h('div.paper-meta', rs.author ? h('span', rs.author) : null, App.state.fileName ? h('span', L`Source: ${App.state.fileName}`) : null, d ? h('span', L`${d.numInstances} instances · ${d.numAttributes} attributes`) : null)));
    let sec = 0;
    const section = (t) => { sec++; const s = h('section.rsec', h('h2', sec + '. ' + t)); paper.appendChild(s); return s; };
    if (rs.notes.trim()) {
      const s = section(L`Summary & conclusions`);
      rs.notes.trim().split(/\n{2,}/).forEach((p) => s.appendChild(h('p', p)));
    }
    if (d && rs.sections.data) {
      const s = section(L`Dataset`);
      if ((App.state.ops || []).length > 1) s.appendChild(h('p.small.muted', L`The dataset as it is after the preprocessing steps listed below.`));
      let missing = 0; for (const r of d.rows) for (const v of r) if (v !== v) missing++;
      s.appendChild(h('div.kpis', kpi(L`Relation`, d.relation), kpi(L`Instances`, d.numInstances.toLocaleString(I18N.locale())), kpi(L`Attributes`, d.numAttributes),
        kpi(L`Missing values`, U.d2s((100 * missing) / Math.max(1, d.numInstances * d.numAttributes), 2) + '%'), kpi(L`Class`, d.classAttr ? d.classAttr.name : '—')));
      if (d.classAttr && d.classAttr.type === 'nominal') {
        const st = d.attributeStats(d.classIndex);
        s.appendChild(h('div.fig', h('div.cap', L`Class distribution of “${d.classAttr.name}”`),
          C.bars(d.classAttr.values.map((v, k) => ({ label: v, value: st.counts[k], color: App.color(k) })), { width: 620, labelW: 160, fmt: (x) => U.d2s(x, 0) + ' (' + U.d2s((100 * x) / Math.max(1, st.total), 1) + '%)' })));
      }
      if (rs.sections.attrs) {
        s.appendChild(h('table.rtbl', h('thead', h('tr', h('th', '#'), h('th', L`Attribute`), h('th', L`Type`), h('th.num', L`Missing`), h('th.num', L`Distinct`), h('th', L`Statistics / most frequent values`))),
          h('tbody', d.attributes.map((a, j) => {
            const st = d.attributeStats(j);
            let desc;
            if (a.type === 'nominal' || a.type === 'string') {
              const top = (st.counts || []).map((c, k) => [c, k]).sort((x, y) => y[0] - x[0]).slice(0, 4);
              desc = top.map(([c, k]) => a.values[k] + ' (' + c + ')').join(', ') + (a.values.length > 4 ? ' …' : '');
            } else desc = `min ${U.d2s(st.min, 2)} · max ${U.d2s(st.max, 2)} · ${L`mean`} ${U.d2s(st.mean, 2)} · sd ${U.d2s(st.stdDev, 2)}`;
            return h('tr', h('td', j + 1), h('td', h('b', a.name), j === d.classIndex ? h('span.rtag', L`class`) : null), h('td', a.type), h('td.num', st.missing), h('td.num', st.distinct), h('td.small', desc));
          }))));
      }
    }
    const ops = App.state.ops || [];
    if (rs.sections.prep && ops.length) {
      const s = section(L`Preprocessing steps`);
      s.appendChild(h('ol.steps', ops.map((o) => h('li', h('span', o.label), h('span.t', App.timeStr(o.time))))));
    }
    const incl = (kind) => res[kind].filter((r) => !rs.excluded.has(keyOf(kind, r)));
    const cls = incl('classify');
    if (rs.sections.classify && cls.length) {
      const s = section(L`Classification`);
      const nominal = cls.filter((r) => r.res.evaluation.nominal);
      if (nominal.length) {
        const labels = nominal[nominal.length - 1].res.evaluation.header.classAttr.values;
        const counts = nominal[nominal.length - 1].res.evaluation.confusion.map((r) => U.sum(r));
        const pos = labels[App.analysis.positiveIndex(labels, counts, App.state.cost.pos)];
        s.appendChild(h('h3', L`Model comparison`));
        s.appendChild(h('p.small', L`Positive class “${pos}”. Cost model: a missed “${pos}” (false negative) costs €${App.state.cost.fn}, a false alarm (false positive) costs €${App.state.cost.fp}.`));
        s.appendChild(App.analysis.compareTable(App.analysis.compareRows(nominal, pos), pos, { compact: false }));
      }
      cls.forEach((r, i) => s.appendChild(classifyBlock(r, i + 1)));
    }
    const sel = incl('select');
    if (rs.sections.select && sel.length) {
      const s = section(L`Attribute selection`);
      sel.forEach((r) => {
        const x = r.res, b = h('div.rblock', h('h3', r.name), h('div.small.muted', x.evalSpec.path + ' ' + x.eo + ' · ' + x.searchSpec.path + ' ' + x.so + ' · ' + r.relation));
        if (x.ranking) {
          const selSet = new Set(x.selected);
          b.appendChild(C.bars(x.ranking.slice(0, 25).map((q) => ({ label: (q.index + 1) + '. ' + q.name, value: q.merit, color: selSet.has(q.index) ? 'var(--primary)' : '#b8bfd3' })), { width: 640, labelW: 210, fmt: (v) => U.d2s(v, 4) }));
        } else b.appendChild(h('p', L`Merit of best subset: `, h('b', U.d2s(x.merit, 4))));
        b.appendChild(h('p.small', L`Selected (${x.selected.length}): `, x.selected.map((j) => x.names[j]).join(', ')));
        s.appendChild(b);
      });
    }
    const clu = incl('cluster');
    if (rs.sections.cluster && clu.length) {
      const s = section(L`Clustering`);
      clu.forEach((r) => s.appendChild(clusterBlock(r)));
    }
    const ass = incl('associate');
    if (rs.sections.associate && ass.length) {
      const s = section(L`Association rules`);
      ass.forEach((r) => {
        const b = h('div.rblock', h('h3', r.name), h('div.small.muted', r.res.spec.path + ' ' + r.res.optStr + ' · ' + r.relation));
        b.appendChild(h('table.rtbl', h('thead', h('tr', h('th', '#'), h('th', L`If`), h('th', L`Then`), h('th.num', L`Support`), h('th.num', L`Conf.`), h('th.num', L`Lift`))),
          h('tbody', r.res.rules.slice(0, 20).map((x, i) => h('tr', h('td', i + 1), h('td.small', x.premise), h('td.small', x.consequence), h('td.num', x.support), h('td.num', U.d2s(x.conf, 2)), h('td.num', U.d2s(x.lift, 2)))))));
        s.appendChild(b);
      });
    }
    if (rs.sections.appendix) {
      const all = [...cls.map((r) => ['Classify', r]), ...sel.map((r) => ['Select attributes', r]), ...clu.map((r) => ['Cluster', r]), ...ass.map((r) => ['Associate', r])];
      if (all.length) {
        const s = section(L`Appendix: full Weka output`);
        all.forEach(([k, r]) => s.appendChild(h('div.rblock', h('h3', k + ' — ' + r.name), h('pre.rpre', r.res.report))));
      }
    }
    if (!d) paper.appendChild(h('p', L`Load a dataset and run some analyses to fill the report.`));
    paper.appendChild(h('footer.paper-foot', L`Generated with Weka Web (a browser port of Weka 3.8.7, University of Waikato). All computations ran locally.`));
    return paper;
  };
  function kpi(l, v) { return h('div.kpi', h('div.l', l), h('div.v', String(v))); }

  function classifyBlock(r, idx) {
    const x = r.res, e = x.evaluation;
    const b = h('div.rblock');
    const head = h('div.rhead', h('h3', idx + '. ' + r.name + ' ', h('span.rtag', x.label)), h('div.small.muted', x.spec.path + (x.optStr ? ' ' + x.optStr : '') + ' · ' + r.relation + ' (' + r.data.numAttributes + L` attributes)`));
    b.appendChild(head);
    if (e.nominal) {
      const labels = e.header.classAttr.values;
      head.appendChild(h('div.kpis', kpi(L`Accuracy`, U.d2s(e.pctCorrect(), 2) + '%'), kpi('Kappa', U.d2s(e.kappa(), 3)), kpi('MAE', U.d2s(e.meanAbsoluteError(), 4)), kpi('RMSE', U.d2s(e.rootMeanSquaredError(), 4)), kpi(L`ROC AUC (weighted)`, U.d2s(e.weighted(e.rocArea), 3))));
      const counts = e.confusion.map((row) => U.sum(row));
      const p = App.analysis.positiveIndex(labels, counts, App.state.cost.pos);
      const tc = e.thresholdCurve(p);
      b.appendChild(h('div.two',
        h('div.fig', h('div.cap', L`Confusion matrix (rows = actual)`), C.confusion(e.confusion, labels)),
        h('div.fig', h('div.cap', L`ROC curve for “${labels[p]}” (AUC ${U.d2s(e.rocArea(p), 3)})`), C.roc([{ color: App.color(p), points: tc.points }], { width: 360, height: 300 }))));
      b.appendChild(h('table.rtbl', h('thead', h('tr', h('th', L`Class`), ['TP rate', 'FP rate', 'Precision', 'Recall', 'F-Measure', 'MCC', 'ROC', 'PRC'].map((t) => h('th.num', t)))),
        h('tbody', labels.map((l, c) => h('tr', h('td', l), [e.truePositiveRate(c), e.falsePositiveRate(c), e.precision(c), e.recall(c), e.fMeasure(c), e.mcc(c), e.rocArea(c), e.prcArea(c)].map((v) => h('td.num', App.fmt(v, 3))))))));
    } else {
      head.appendChild(h('div.kpis', kpi(L`Correlation`, U.d2s(e.correlationCoefficient(), 4)), kpi('MAE', U.d2s(e.meanAbsoluteError(), 4)), kpi('RMSE', U.d2s(e.rootMeanSquaredError(), 4)), kpi('RAE', U.d2s(e.relativeAbsoluteError(), 2) + '%'), kpi('RRSE', U.d2s(e.rootRelativeSquaredError(), 2) + '%')));
    }
    const m = x.model;
    let drew = false;
    if (m.graph && x.data.classIsNominal()) {
      try {
        const n = m.root && m.root.numNodes ? m.root.numNodes() : 0;
        if (n && n <= 31) { b.appendChild(h('div.fig', h('div.cap', L`Decision tree`), h('div.treebox', C.tree(m.graph(), (c) => App.color(c))))); drew = true; }
      } catch (err) { /* text fallback */ }
    }
    const txt = m.toString().split('\n');
    b.appendChild(h('div.fig', h('div.cap', drew ? L`Model (text)` : L`Model`), h('pre.rpre', txt.slice(0, 45).join('\n') + (txt.length > 45 ? '\n' + L`… (${txt.length - 45} more lines — see the appendix)` : ''))));
    return b;
  }

  function clusterBlock(r) {
    const x = r.res, d = r.data, m = x.model;
    const b = h('div.rblock', h('h3', r.name), h('div.small.muted', x.spec.path + (x.optStr ? ' ' + x.optStr : '') + ' · ' + r.relation));
    const ign = d.attributes.map((a, j) => j).filter((j) => !x.keep.includes(j));
    if (ign.length) b.appendChild(h('p.small', L`Ignored attributes: `, ign.map((j) => d.attributes[j].name).join(', ')));
    const tot = U.sum(x.counts);
    b.appendChild(h('div.fig', h('div.cap', L`Cluster sizes`), C.bars(x.counts.map((c, i) => ({ label: L`Cluster ${i}`, value: c, color: App.color(i) })), { width: 620, labelW: 110, fmt: (v) => U.d2s(v, 0) + ' (' + U.d2s((100 * v) / tot, 0) + '%)' })));
    if (m.centroids) {
      const hd = m.header;
      b.appendChild(h('div.fig', h('div.cap', L`Cluster centroids`), h('table.rtbl', h('thead', h('tr', h('th', L`Attribute`), h('th.num', L`Full data`), m.centroids.map((_, i) => h('th.num', L`Cluster ${i}`)))),
        h('tbody', hd.attributes.map((a, j) => { const f = (v) => (v !== v ? '?' : a.type === 'nominal' ? a.values[v] : U.d2s(v, 2)); return h('tr', h('td', a.name), h('td.num', f(m.full[j])), m.centroids.map((c) => h('td.num', f(c[j])))); })))));
    }
    if (x.evalRows) {
      const att = d.classIndex >= 0 && d.classIsNominal() ? d.classIndex : -1;
      if (att >= 0) {
        const ct = App.analysis.crosstab(d, x.assignments, x.k, att);
        const counts = ct.labels.map((_, j) => U.sum(ct.m.map((row) => row[j])));
        b.appendChild(h('div.fig', h('div.cap', L`Segments vs. “${ct.name}”`), App.analysis.crosstabTable(ct, App.analysis.positiveIndex(ct.labels, counts, App.state.cost.pos))));
      }
    }
    if (x.classes) b.appendChild(h('p.small', L`Classes-to-clusters: ${U.d2s((100 * x.classes.errors) / U.sum(x.classes.matrix.flat()), 2)}% incorrectly clustered.`));
    return b;
  }

  /** Puts a copy of the report in a print-only root; used by the PDF button and by automation. */
  R.preparePrint = function () {
    let root = document.getElementById('print-root');
    if (root) root.remove();
    root = h('div#print-root', R.build());
    document.body.appendChild(root);
    document.body.classList.add('printing');
    return root;
  };
  R.cleanupPrint = function () { const r = document.getElementById('print-root'); if (r) r.remove(); document.body.classList.remove('printing'); };
  R.exportPDF = function () {
    const old = document.title;
    document.title = (rs.title || 'weka-report-' + (App.state.data ? App.state.data.relation : '')).replace(/[^\w\-. ]+/g, '_');
    R.preparePrint();
    setTimeout(() => { window.print(); setTimeout(() => { R.cleanupPrint(); document.title = old; }, 300); }, 50);
  };
  R.exportHTML = function () {
    const css = [...document.styleSheets].map((s) => { try { return [...s.cssRules].map((r) => r.cssText).join('\n'); } catch (e) { return ''; } }).join('\n');
    const html = `<!doctype html><html lang="${I18N.lang}"><head><meta charset="utf-8"><title>${App.esc(rs.title || 'Weka report')}</title><style>${css}</style></head><body class="printing standalone">${R.build().outerHTML}</body></html>`;
    App.download((rs.title || 'weka-report').replace(/[^\w\-. ]+/g, '_') + '.html', html, 'text/html');
  };
  window.addEventListener('afterprint', () => R.cleanupPrint());

  // ------------------------------------------------------------- page
  App.pages.report = {
    render() {
      const d = App.state.data;
      if (!d) return h('div.page', App.pageHead(L`Report`, L`Turn your whole session into a PDF.`), App.needData());
      const page = h('div.page', App.pageHead(L`Report`, L`Everything you did in this session — data, preprocessing, models, charts and your conclusions — as one document.`,
        h('button.btn', { on: { click: R.exportHTML } }, App.icon('download'), 'HTML'),
        h('button.btn.primary', { on: { click: R.exportPDF } }, App.icon('download'), L`Export PDF`)));
      const preview = h('div.paper-wrap');
      const refresh = () => { App.clear(preview).appendChild(R.build()); };
      const input = (k, ph, area) => {
        const el = h(area ? 'textarea.input' : 'input.input', { rows: area ? 8 : undefined, placeholder: ph, value: rs[k], on: { input: (e) => { rs[k] = e.target.value; save(); clearTimeout(input.t); input.t = setTimeout(refresh, 250); } } });
        if (area) el.value = rs[k];
        return el;
      };
      const toggle = (k, label) => h('label.opt-bool', h('span', label), h('span.switch', h('input', { type: 'checkbox', checked: rs.sections[k], on: { change: (e) => { rs.sections[k] = e.target.checked; save(); refresh(); } } }), h('span')));
      const resList = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '4px' } });
      const kinds = [['classify', L`Classify`], ['select', L`Select attributes`], ['cluster', L`Cluster`], ['associate', L`Associate`]];
      let any = false;
      for (const [k, label] of kinds) for (const r of App.state.results[k]) {
        any = true;
        const key = keyOf(k, r);
        resList.appendChild(h('label.check', h('input', { type: 'checkbox', checked: !rs.excluded.has(key), on: { change: (e) => { if (e.target.checked) rs.excluded.delete(key); else rs.excluded.add(key); refresh(); } } }), h('span', h('b', r.name), h('span.muted.small', ' · ' + label + ' · ' + r.relation))));
      }
      if (!any) resList.appendChild(h('div.muted.small', L`No results yet — run classifiers, clusterers, etc. and they will appear here.`));
      const left = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '18px' } },
        h('div.card', App.cardHead('edit', L`Document`), h('div.card-body',
          h('div.field', h('label', L`Title`), input('title', L`Data mining report — ${d.relation}`)),
          h('div.field', h('label', L`Author`), input('author', L`Your name`)),
          h('div.field', h('label', L`Summary & conclusions`), input('notes', L`Write your interpretation here. Separate paragraphs with a blank line.`, true)))),
        h('div.card', App.cardHead('list', L`Contents`), h('div.card-body',
          toggle('data', L`Dataset overview`), toggle('attrs', L`Attribute table`), toggle('prep', L`Preprocessing steps`), toggle('classify', L`Classification`), toggle('select', L`Attribute selection`), toggle('cluster', L`Clustering`), toggle('associate', L`Association rules`), toggle('appendix', L`Appendix with full Weka output`),
          h('div.divider'), h('div.field-label', { style: { marginBottom: '8px' } }, L`Results to include`), resList)),
        h('div.callout', App.icon('info'), h('div', L`“Export PDF” opens the print dialog — choose “Save as PDF” as the destination.`)));
      page.appendChild(h('div.split', left, h('div.result-view', { style: { minWidth: 0 } }, preview)));
      refresh();
      return page;
    },
  };
})();
