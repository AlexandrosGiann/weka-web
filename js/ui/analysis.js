/* Weka Web — shared analysis widgets: model comparison with business costs, cluster × attribute cross-tab. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, App = window.App, h = App.h;
  const A = (App.analysis = {});
  App.state.cost = App.store.get('cost', { fn: 50, fp: 2 });

  /** Index of the positive class: the stored choice, else "yes", else the minority class. */
  A.positiveIndex = function (labels, counts, pref) {
    if (pref && labels.includes(pref)) return labels.indexOf(pref);
    const y = labels.findIndex((l) => /^(yes|y|true|1|t)$/i.test(l));
    if (y >= 0) return y;
    return counts ? U.minIndex(counts) : 0;
  };

  /** Rows for comparing classification results (nominal class only). */
  A.compareRows = function (results, posLabel, cost = App.state.cost) {
    return results.filter((r) => r.res.evaluation.nominal).map((r) => {
      const e = r.res.evaluation, labels = e.header.classAttr.values;
      const p = labels.indexOf(posLabel);
      const row = { name: r.name, opts: r.res.optStr, label: r.res.label, relation: r.relation, atts: r.data.numAttributes, acc: e.pctCorrect(), kappa: e.kappa(), aucW: e.weighted(e.rocArea) };
      if (p >= 0) {
        row.prec = e.precision(p); row.rec = e.recall(p); row.f = e.fMeasure(p); row.auc = e.rocArea(p);
        row.fn = e.numFalseNegatives(p); row.fp = e.numFalsePositives(p); row.tp = e.numTruePositives(p);
        row.cost = row.fn * cost.fn + row.fp * cost.fp;
      }
      return row;
    });
  };

  A.compareTable = function (rows, posLabel, opts = {}) {
    const best = (k, max = true) => { const v = rows.map((r) => r[k]).filter((x) => x === x && x !== undefined); return v.length ? (max ? Math.max(...v) : Math.min(...v)) : NaN; };
    const bAcc = best('acc'), bF = best('f'), bAuc = best('auc'), bCost = best('cost', false), bRec = best('rec');
    const cell = (v, d, b, fmt) => h('td.num', { style: v === b && rows.length > 1 ? { fontWeight: 700, color: 'var(--success)' } : {} }, v === undefined || v !== v ? '—' : fmt ? fmt(v) : U.d2s(v, d));
    return h('div.table-wrap', h('table.tbl.compact',
      h('thead', h('tr', h('th', '#'), h('th', L`Model`), h('th', L`Data`), h('th.num', L`Accuracy`), h('th.num', 'Kappa'),
        h('th.num', L`Precision (${posLabel})`), h('th.num', L`Recall (${posLabel})`), h('th.num', L`F (${posLabel})`), h('th.num', L`ROC AUC`),
        h('th.num', 'FN'), h('th.num', 'FP'), h('th.num', L`Cost (€)`))),
      h('tbody', rows.map((r, i) => h('tr', h('td.muted', i + 1), h('td', h('b', r.name), opts.compact ? null : h('div.muted.small', String(r.label).replace('cross-validation', 'CV').replace(/user supplied test set: (\d+) instances/, 'test set ($1)'))),
        h('td.small', r.relation + ' · ' + r.atts + L` att.`),
        cell(r.acc, 2, bAcc, (v) => U.d2s(v, 2) + '%'), cell(r.kappa, 3), cell(r.prec, 3), cell(r.rec, 3, bRec), cell(r.f, 3, bF), cell(r.auc, 3, bAuc),
        cell(r.fn, 0), cell(r.fp, 0), cell(r.cost, 0, bCost, (v) => '€' + Math.round(v).toLocaleString(I18N.locale())))))));
  };

  /** Interactive comparison card for the Classify page. */
  A.compareCard = function (results, rerender) {
    const nominal = results.filter((r) => r.res.evaluation.nominal);
    if (!nominal.length) return null;
    const labels = nominal[nominal.length - 1].res.evaluation.header.classAttr.values;
    const counts = nominal[nominal.length - 1].res.evaluation.confusion.map((r) => U.sum(r));
    const pos = labels[A.positiveIndex(labels, counts, App.state.cost.pos)];
    const posSel = h('select.select', { style: { width: 'auto', height: '30px' }, on: { change: (e) => { App.state.cost.pos = e.target.value; App.store.set('cost', App.state.cost); rerender(); } } }, labels.map((l) => h('option', { value: l, selected: l === pos }, l)));
    const num = (k) => h('input.input', { type: 'number', min: 0, value: App.state.cost[k], style: { width: '90px', height: '30px' }, on: { change: (e) => { App.state.cost[k] = Math.max(0, Number(e.target.value) || 0); App.store.set('cost', App.state.cost); rerender(); } } });
    const rows = A.compareRows(nominal, pos);
    const best = rows.filter((r) => r.cost === r.cost).sort((a, b) => a.cost - b.cost)[0];
    return h('div.card', App.cardHead('chart', L`Model comparison & business cost`),
      h('div.card-body',
        h('div.row', { style: { marginBottom: '12px' } },
          h('span.field-label', L`Positive class`), posSel,
          h('span.field-label', { style: { marginLeft: '10px' } }, L`Cost of a missed positive (FN) €`), num('fn'),
          h('span.field-label', L`Cost of a false alarm (FP) €`), num('fp')),
        A.compareTable(rows, pos),
        best ? h('div.callout', { style: { marginTop: '12px' } }, App.icon('info'),
          h('div', L`Lowest expected cost: `, h('b', best.name), L` (${best.relation}) — €${Math.round(best.cost).toLocaleString(I18N.locale())} from ${U.d2s(best.fn, 0)} missed “${pos}” and ${U.d2s(best.fp, 0)} false alarms. Accuracy alone can hide this when classes are imbalanced.`)) : null));
  };

  /** Cross-tab of cluster assignments against a nominal attribute (e.g. the target). */
  A.crosstab = function (d, assignments, k, att) {
    const a = d.attributes[att];
    const m = U.matrix(k, a.values.length);
    d.rows.forEach((r, i) => { const v = r[att]; if (v === v && assignments[i] >= 0) m[assignments[i]][v] += d.weights[i]; });
    return { m, labels: a.values, name: a.name };
  };
  A.crosstabTable = function (ct, posIdx) {
    const tot = ct.m.map((r) => U.sum(r));
    const rates = ct.m.map((r, c) => (tot[c] ? r[posIdx] / tot[c] : 0));
    const overall = U.sum(ct.m.map((r) => r[posIdx])) / Math.max(1e-12, U.sum(tot));
    const mx = Math.max(...rates, 1e-12);
    return h('div.table-wrap', h('table.tbl.compact',
      h('thead', h('tr', h('th', L`Cluster`), h('th.num', L`Size`), ct.labels.map((l) => h('th.num', ct.name + '=' + l)), h('th', L`% ${ct.name}=${ct.labels[posIdx]}`))),
      h('tbody', ct.m.map((r, c) => h('tr', h('td', h('span', { style: { display: 'inline-block', width: '9px', height: '9px', borderRadius: '3px', background: App.color(c), marginRight: '7px' } }), L`Cluster ${c}`),
        h('td.num', U.d2s(tot[c], 0)), r.map((v) => h('td.num', U.d2s(v, 0))),
        h('td', h('div.row', { style: { flexWrap: 'nowrap', gap: '8px' } }, h('div.bar', { style: { width: Math.max(2, (90 * rates[c]) / mx) + 'px', background: rates[c] > overall ? 'var(--success)' : 'var(--border-strong)' } }), h('b', U.d2s(100 * rates[c], 1) + '%'))))),
        h('tr', h('td.muted', L`All`), h('td.num.muted', U.d2s(U.sum(tot), 0)), ct.labels.map((_, j) => h('td.num.muted', U.d2s(U.sum(ct.m.map((r) => r[j])), 0))), h('td.muted', U.d2s(100 * overall, 1) + '%')))));
  };
})();
