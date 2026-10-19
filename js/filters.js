/* Weka Web — φίλτρα προεπεξεργασίας (unsupervised/supervised, attribute/instance). */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, S = W.Stats;
  W.filters = [];
  const reg = (spec) => { W.filters.push(spec); return spec; };

  /**
   * Βάση για attribute filters: setInput(data) υπολογίζει παραμέτρους και την έξοδο,
   * convertRow(row) μετατρέπει μία εγγραφή, apply(data) εφαρμόζει σε σύνολο με ίδια δομή.
   */
  class AttrFilter {
    setInput(d) { this.inHeader = d; this.compute(d); return this; }
    compute() {}
    makeOut(d, attrs, classIndex) { this.out = new W.Dataset(d.relation, attrs, [], [], classIndex); }
    apply(d) {
      const rows = new Array(d.rows.length);
      for (let i = 0; i < d.rows.length; i++) rows[i] = this.convertRow(d.rows[i]);
      return new W.Dataset(this.out.relation, this.out.attributes, rows, d.weights.slice(), this.out.classIndex);
    }
  }
  W.AttrFilter = AttrFilter;

  // ---------------------------------------------------------- ReplaceMissingValues
  class ReplaceMissingValues extends AttrFilter {
    constructor(o = {}) { super(); this.ignoreClass = !!o.ignoreClass; }
    compute(d) {
      this.fill = d.attributes.map((a, j) => {
        if (j === d.classIndex && !this.ignoreClass) return NaN;
        if (a.type === 'nominal' || a.type === 'string') {
          const c = U.zeros(a.values.length);
          for (let i = 0; i < d.rows.length; i++) { const v = d.rows[i][j]; if (v === v) c[v] += d.weights[i]; }
          return c.length ? U.maxIndex(c) : NaN;
        }
        let s = 0, w = 0;
        for (let i = 0; i < d.rows.length; i++) { const v = d.rows[i][j]; if (v === v) { s += v * d.weights[i]; w += d.weights[i]; } }
        return w > 0 ? s / w : NaN;
      });
      this.makeOut(d, d.attributes, d.classIndex);
    }
    convertRow(r) {
      const o = r.slice();
      for (let j = 0; j < o.length; j++) if (o[j] !== o[j] && this.fill[j] === this.fill[j]) o[j] = this.fill[j];
      return o;
    }
  }
  W.ReplaceMissingValues = ReplaceMissingValues;
  reg({ name: 'ReplaceMissingValues', kind: 'attribute', supervised: false, path: 'weka.filters.unsupervised.attribute.ReplaceMissingValues',
    desc: L`Αντικαθιστά τις ελλιπείς τιμές με τον μέσο όρο (αριθμητικά) ή την επικρατούσα τιμή (nominal).`,
    options: [{ key: 'ignoreClass', flag: 'unset-class-temporarily', label: L`Να επηρεάζεται και η κλάση`, type: 'bool', default: false }],
    create: (o) => new ReplaceMissingValues(o) });

  // ---------------------------------------------------------------- Normalize
  class Normalize extends AttrFilter {
    constructor(o = {}) { super(); this.scale = o.S ?? 1; this.trans = o.T ?? 0; this.ignoreClass = !!o.ignoreClass; }
    compute(d) {
      this.min = []; this.max = [];
      d.attributes.forEach((a, j) => {
        let mn = NaN, mx = NaN;
        if (W.isNumericAttr(a) && (j !== d.classIndex || this.ignoreClass)) for (const r of d.rows) { const v = r[j]; if (v === v) { if (!(v >= mn)) mn = v; if (!(v <= mx)) mx = v; } }
        this.min.push(mn); this.max.push(mx);
      });
      this.makeOut(d, d.attributes.map((a, j) => (this.min[j] === this.min[j] ? Object.assign({}, a, { type: 'numeric' }) : a)), d.classIndex);
    }
    convertRow(r) {
      const o = r.slice();
      for (let j = 0; j < o.length; j++) {
        if (this.min[j] !== this.min[j] || o[j] !== o[j]) continue;
        const w = this.max[j] - this.min[j];
        o[j] = w === 0 ? this.trans : ((o[j] - this.min[j]) / w) * this.scale + this.trans;
      }
      return o;
    }
  }
  W.Normalize = Normalize;
  reg({ name: 'Normalize', kind: 'attribute', supervised: false, path: 'weka.filters.unsupervised.attribute.Normalize',
    desc: L`Κανονικοποιεί όλα τα αριθμητικά χαρακτηριστικά στο διάστημα [0,1] (ή [T, T+S]).`,
    options: [
      { key: 'S', flag: 'S', label: L`Κλίμακα (scale)`, type: 'number', default: 1 },
      { key: 'T', flag: 'T', label: L`Μετατόπιση (translation)`, type: 'number', default: 0 },
      { key: 'ignoreClass', flag: 'unset-class-temporarily', label: L`Να επηρεάζεται και η κλάση`, type: 'bool', default: false },
    ], create: (o) => new Normalize(o) });

  // -------------------------------------------------------------- Standardize
  class Standardize extends AttrFilter {
    constructor(o = {}) { super(); this.ignoreClass = !!o.ignoreClass; }
    compute(d) {
      this.mean = []; this.sd = [];
      d.attributes.forEach((a, j) => {
        if (!W.isNumericAttr(a) || (j === d.classIndex && !this.ignoreClass)) { this.mean.push(NaN); this.sd.push(NaN); return; }
        const st = d.attributeStats(j);
        this.mean.push(st.mean); this.sd.push(st.stdDev);
      });
      this.makeOut(d, d.attributes, d.classIndex);
    }
    convertRow(r) {
      const o = r.slice();
      for (let j = 0; j < o.length; j++) {
        if (this.mean[j] !== this.mean[j] || o[j] !== o[j]) continue;
        o[j] = this.sd[j] > 0 ? (o[j] - this.mean[j]) / this.sd[j] : o[j] - this.mean[j];
      }
      return o;
    }
  }
  reg({ name: 'Standardize', kind: 'attribute', supervised: false, path: 'weka.filters.unsupervised.attribute.Standardize',
    desc: L`Μετασχηματίζει τα αριθμητικά χαρακτηριστικά σε μέσο 0 και τυπική απόκλιση 1.`,
    options: [{ key: 'ignoreClass', flag: 'unset-class-temporarily', label: L`Να επηρεάζεται και η κλάση`, type: 'bool', default: false }],
    create: (o) => new Standardize(o) });

  // -------------------------------------------------------------------- Remove
  class Remove extends AttrFilter {
    constructor(o = {}) { super(); this.R = o.R ?? ''; this.V = !!o.V; this.indices = o.indices || null; }
    compute(d) {
      let sel = this.indices ? this.indices.slice() : this.R ? U.parseRange(this.R, d.numAttributes) : [];
      const set = new Set(sel);
      this.keep = [];
      for (let j = 0; j < d.numAttributes; j++) if (set.has(j) === this.V) this.keep.push(j);
      const ci = this.keep.indexOf(d.classIndex);
      this.makeOut(d, this.keep.map((j) => d.attributes[j]), ci);
    }
    convertRow(r) { return this.keep.map((j) => r[j]); }
  }
  W.Remove = Remove;
  reg({ name: 'Remove', kind: 'attribute', supervised: false, path: 'weka.filters.unsupervised.attribute.Remove',
    desc: L`Αφαιρεί τα χαρακτηριστικά του εύρους (π.χ. "1,3-5,last").`,
    options: [
      { key: 'R', flag: 'R', label: L`Εύρος χαρακτηριστικών`, type: 'text', default: '', placeholder: L`π.χ. 1,3-5` },
      { key: 'V', flag: 'V', label: L`Αντιστροφή (κράτα μόνο αυτά)`, type: 'bool', default: false },
    ], create: (o) => new Remove(o) });

  // --------------------------------------------------------- Discretize (unsup.)
  function binLabels(cuts) {
    if (!cuts || !cuts.length) return ["'All'"].map(() => 'All');
    const f = (v) => U.d2s(v, 6);
    const out = ['(-inf-' + f(cuts[0]) + ']'];
    for (let i = 1; i < cuts.length; i++) out.push('(' + f(cuts[i - 1]) + '-' + f(cuts[i]) + ']');
    out.push('(' + f(cuts[cuts.length - 1]) + '-inf)');
    return out;
  }
  class DiscretizeBase extends AttrFilter {
    finish(d) {
      this.makeOut(d, d.attributes.map((a, j) => (this.cuts[j] !== undefined ? { name: a.name, type: 'nominal', values: binLabels(this.cuts[j]) } : a)), d.classIndex);
    }
    convertRow(r) {
      const o = r.slice();
      for (let j = 0; j < o.length; j++) {
        if (this.cuts[j] === undefined || o[j] !== o[j]) continue;
        const c = this.cuts[j];
        if (!c || !c.length) { o[j] = 0; continue; }
        let k = 0; while (k < c.length && o[j] > c[k]) k++;
        o[j] = k;
      }
      return o;
    }
  }
  class Discretize extends DiscretizeBase {
    constructor(o = {}) { super(); this.B = o.B ?? 10; this.F = !!o.F; this.R = o.R || 'first-last'; this.ignoreClass = !!o.ignoreClass; }
    compute(d) {
      const sel = new Set(U.parseRange(this.R, d.numAttributes));
      this.cuts = [];
      d.attributes.forEach((a, j) => {
        if (!W.isNumericAttr(a) || !sel.has(j) || (j === d.classIndex && !this.ignoreClass)) { this.cuts[j] = undefined; return; }
        this.cuts[j] = this.F ? eqFreq(d, j, this.B) : eqWidth(d, j, this.B);
      });
      this.finish(d);
    }
  }
  function eqWidth(d, j, bins) {
    let mn = Infinity, mx = -Infinity;
    for (const r of d.rows) { const v = r[j]; if (v === v) { if (v < mn) mn = v; if (v > mx) mx = v; } }
    const w = (mx - mn) / bins;
    if (bins > 1 && w > 0) { const c = []; for (let i = 1; i < bins; i++) c.push(mn + w * i); return c; }
    return null;
  }
  function eqFreq(d, j, bins) {
    const s = d.copy(); s.sortBy(j);
    let sumW = 0;
    for (let i = 0; i < s.rows.length; i++) if (s.rows[i][j] === s.rows[i][j]) sumW += s.weights[i];
    const cuts = new Array(bins - 1);
    let freq = sumW / bins, cp = 0, counter = 0, last = 0, lastIndex = -1;
    for (let i = 0; i < s.rows.length - 1; i++) {
      const vn = s.rows[i + 1][j];
      if (vn !== vn) break;
      counter += s.weights[i]; sumW -= s.weights[i];
      if (s.rows[i][j] < vn) {
        if (counter >= freq) {
          if (freq - last < counter - freq && lastIndex !== -1) {
            cuts[cp] = (s.rows[lastIndex][j] + s.rows[lastIndex + 1][j]) / 2;
            counter -= last; last = counter; lastIndex = i;
          } else {
            cuts[cp] = (s.rows[i][j] + vn) / 2;
            counter = 0; last = 0; lastIndex = -1;
          }
          cp++;
          freq = (sumW + counter) / (cuts.length + 1 - cp);
        } else { lastIndex = i; last = counter; }
      }
    }
    if (cp < cuts.length && lastIndex !== -1) { cuts[cp] = (s.rows[lastIndex][j] + s.rows[lastIndex + 1][j]) / 2; cp++; }
    return cp === 0 ? null : cuts.slice(0, cp);
  }
  reg({ name: 'Discretize', kind: 'attribute', supervised: false, path: 'weka.filters.unsupervised.attribute.Discretize',
    desc: L`Μετατρέπει αριθμητικά χαρακτηριστικά σε nominal διαστήματα ίσου πλάτους ή ίσης συχνότητας.`,
    options: [
      { key: 'B', flag: 'B', label: L`Πλήθος διαστημάτων (bins)`, type: 'int', default: 10, min: 1 },
      { key: 'F', flag: 'F', label: L`Ίση συχνότητα (αντί για ίσο πλάτος)`, type: 'bool', default: false },
      { key: 'R', flag: 'R', label: L`Εύρος χαρακτηριστικών`, type: 'text', default: 'first-last' },
      { key: 'ignoreClass', flag: 'unset-class-temporarily', label: L`Να επηρεάζεται και η κλάση`, type: 'bool', default: false },
    ], create: (o) => new Discretize(o) });

  // ------------------------------------------------------- Discretize (supervised MDL)
  function log2Binomial(a, b) { return (lnGamma(a + 1) - lnGamma(b + 1) - lnGamma(a - b + 1)) / Math.LN2; }
  function log2Multinomial(a, bs) { let s = lnGamma(a + 1); for (const b of bs) s -= lnGamma(b + 1); return s / Math.LN2; }
  function lnGamma(x) {
    // Lanczos
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
    x -= 1; let a = c[0]; const t = x + g + 0.5;
    for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }
  W.lnGamma = lnGamma;
  /** Σημεία τομής MDL (Fayyad & Irani) για ταξινομημένο υποσύνολο [first, last). */
  function mdlCutPoints(d, att, first, last, betterEncoding = false, kononenko = false) {
    if (last - first < 2) return null;
    const nc = d.numClasses, ci = d.classIndex;
    const counts = [U.zeros(nc), U.zeros(nc)];
    let numInst = 0;
    for (let i = first; i < last; i++) { numInst += d.weights[i]; counts[1][d.rows[i][ci]] += d.weights[i]; }
    const prior = counts[1].slice();
    const priorEnt = S.entropy(prior);
    let bestEnt = priorEnt, bestCut = -1, bestIndex = -1, numCuts = 0;
    const bestCounts = [U.zeros(nc), U.zeros(nc)];
    for (let i = first; i < last - 1; i++) {
      const c = d.rows[i][ci], w = d.weights[i];
      counts[0][c] += w; counts[1][c] -= w;
      if (d.rows[i][att] < d.rows[i + 1][att]) {
        const cut = (d.rows[i][att] + d.rows[i + 1][att]) / 2;
        const ent = S.entropyConditionedOnRows(counts);
        if (ent < bestEnt) { bestCut = cut; bestEnt = ent; bestIndex = i; bestCounts[0] = counts[0].slice(); bestCounts[1] = counts[1].slice(); }
        numCuts++;
      }
    }
    if (!betterEncoding) numCuts = last - first - 1;
    const gain = priorEnt - bestEnt;
    if (gain <= 0) return null;
    let accept;
    if (kononenko) {
      let k = 0; for (const p of prior) if (p > 0) k++;
      const before = log2Multinomial(numInst, prior) + log2Binomial(numInst + k - 1, k - 1);
      let after = U.log2(numCuts);
      for (const bc of bestCounts) { const s = U.sum(bc); after += log2Binomial(s + k - 1, k - 1) + log2Multinomial(s, bc); }
      accept = before > after;
    } else {
      let k = 0, kl = 0, kr = 0;
      for (const p of prior) if (p > 0) k++;
      for (const p of bestCounts[0]) if (p > 0) kl++;
      for (const p of bestCounts[1]) if (p > 0) kr++;
      const el = S.entropy(bestCounts[0]), er = S.entropy(bestCounts[1]);
      const delta = U.log2(Math.pow(3, k) - 2) - (k * priorEnt - kr * er - kl * el);
      accept = gain > (U.log2(numCuts) + delta) / numInst;
    }
    if (!accept) return null;
    const left = mdlCutPoints(d, att, first, bestIndex + 1, betterEncoding, kononenko);
    const right = mdlCutPoints(d, att, bestIndex + 1, last, betterEncoding, kononenko);
    return [...(left || []), bestCut, ...(right || [])];
  }
  W.mdlCutPoints = mdlCutPoints;
  class SupervisedDiscretize extends DiscretizeBase {
    constructor(o = {}) { super(); this.R = o.R || 'first-last'; this.E = !!o.E; this.K = !!o.K; }
    compute(d) {
      if (!d.classIsNominal()) throw new Error(L`Η εποπτευόμενη διακριτοποίηση απαιτεί nominal κλάση.`);
      const sel = new Set(U.parseRange(this.R, d.numAttributes));
      const dd = d.withoutMissingClass();
      this.cuts = [];
      d.attributes.forEach((a, j) => {
        if (!W.isNumericAttr(a) || !sel.has(j) || j === d.classIndex) { this.cuts[j] = undefined; return; }
        const s = dd.copy(); s.sortBy(j);
        let firstMissing = s.rows.length;
        for (let i = 0; i < s.rows.length; i++) if (s.rows[i][j] !== s.rows[i][j]) { firstMissing = i; break; }
        this.cuts[j] = mdlCutPoints(s, j, 0, firstMissing, this.E, this.K);
      });
      this.finish(d);
    }
  }
  reg({ name: 'Discretize (supervised)', kind: 'attribute', supervised: true, path: 'weka.filters.supervised.attribute.Discretize',
    desc: L`Διακριτοποίηση με βάση την κλάση (MDL, Fayyad & Irani). Απαιτεί nominal κλάση.`,
    options: [
      { key: 'R', flag: 'R', label: L`Εύρος χαρακτηριστικών`, type: 'text', default: 'first-last' },
      { key: 'E', flag: 'E', label: 'Better encoding', type: 'bool', default: false },
      { key: 'K', flag: 'K', label: L`Κριτήριο Kononenko`, type: 'bool', default: false },
    ], create: (o) => new SupervisedDiscretize(o) });

  // ---------------------------------------------------------- NominalToBinary (unsup.)
  class NominalToBinary extends AttrFilter {
    constructor(o = {}) { super(); this.A = !!o.A; this.R = o.R || 'first-last'; this.N = !!o.N; }
    compute(d) {
      const sel = new Set(U.parseRange(this.R, d.numAttributes));
      const attrs = [], map = [];
      let ci = -1;
      d.attributes.forEach((a, j) => {
        if (j === d.classIndex) ci = attrs.length;
        if (a.type !== 'nominal' || j === d.classIndex || !sel.has(j)) { attrs.push(a); map.push({ j, kind: 'copy' }); return; }
        if (a.values.length <= 2 && !this.A) {
          if (this.N) attrs.push({ name: a.name, type: 'numeric' }); else attrs.push(a);
          map.push({ j, kind: 'copy' });
          return;
        }
        a.values.forEach((v, k) => {
          attrs.push(this.N ? { name: a.name + '=' + v, type: 'numeric' } : { name: a.name + '=' + v, type: 'nominal', values: ['f', 't'] });
          map.push({ j, kind: 'ind', k });
        });
      });
      this.map = map;
      this.makeOut(d, attrs, ci);
    }
    convertRow(r) {
      return this.map.map((m) => {
        const v = r[m.j];
        if (m.kind === 'copy') return v;
        return v !== v ? NaN : v === m.k ? 1 : 0;
      });
    }
  }
  W.NominalToBinary = NominalToBinary;
  reg({ name: 'NominalToBinary', kind: 'attribute', supervised: false, path: 'weka.filters.unsupervised.attribute.NominalToBinary',
    desc: L`Μετατρέπει κάθε nominal χαρακτηριστικό k τιμών σε k δυαδικά (one-hot).`,
    options: [
      { key: 'N', flag: 'N', label: L`Αριθμητικά (0/1) αντί για nominal`, type: 'bool', default: false },
      { key: 'A', flag: 'A', label: L`Μετατροπή και των δυαδικών`, type: 'bool', default: false },
      { key: 'R', flag: 'R', label: L`Εύρος χαρακτηριστικών`, type: 'text', default: 'first-last' },
    ], create: (o) => new NominalToBinary(o) });

  // ------------------------------------------------------ NominalToBinary (supervised)
  class SupervisedNominalToBinary extends AttrFilter {
    constructor(o = {}) { super(); this.N = o.N ?? true; }
    compute(d) {
      if (d.classIsNominal()) { this.inner = new NominalToBinary({ N: this.N }).setInput(d); this.out = this.inner.out; return; }
      const ci = d.classIndex, attrs = [], map = [];
      let nci = -1;
      d.attributes.forEach((a, j) => {
        if (j === ci) nci = attrs.length;
        if (a.type !== 'nominal' || j === ci) { attrs.push(a); map.push({ j, kind: 'copy' }); return; }
        const nv = a.values.length, avg = U.zeros(nv), cnt = U.zeros(nv);
        for (let i = 0; i < d.rows.length; i++) { const r = d.rows[i]; if (r[ci] === r[ci] && r[j] === r[j]) { cnt[r[j]] += d.weights[i]; avg[r[j]] += d.weights[i] * r[ci]; } }
        const sum = U.sum(avg), tc = U.sum(cnt);
        if (U.gr(tc, 0)) for (let k = 0; k < nv; k++) avg[k] = U.gr(cnt[k], 0) ? avg[k] / cnt[k] : sum / tc;
        const order = U.wekaSort(avg);
        const rank = new Array(nv); order.forEach((v, r) => { rank[v] = r; });
        for (let k = 1; k < nv; k++) {
          const nm = a.name + '=' + order.slice(k).map((v) => a.values[v]).join(',');
          attrs.push(this.N ? { name: nm, type: 'numeric' } : { name: nm, type: 'nominal', values: ['f', 't'] });
          map.push({ j, kind: 'rank', k, rank });
        }
      });
      this.map = map;
      this.makeOut(d, attrs, nci);
    }
    convertRow(r) {
      if (this.inner) return this.inner.convertRow(r);
      return this.map.map((m) => {
        const v = r[m.j];
        if (m.kind === 'copy') return v;
        if (v !== v) return NaN;
        return m.rank[v] >= m.k ? 1 : 0;
      });
    }
  }
  W.SupervisedNominalToBinary = SupervisedNominalToBinary;
  reg({ name: 'NominalToBinary (supervised)', kind: 'attribute', supervised: true, path: 'weka.filters.supervised.attribute.NominalToBinary',
    desc: L`Με αριθμητική κλάση: k−1 δυαδικά χαρακτηριστικά ταξινομημένα κατά τη μέση τιμή της κλάσης.`,
    options: [{ key: 'N', flag: 'N', label: L`Αριθμητικά (0/1) αντί για nominal`, type: 'bool', default: false }],
    create: (o) => new SupervisedNominalToBinary(o) });

  // ------------------------------------------------------------- RemoveUseless
  class RemoveUseless extends AttrFilter {
    constructor(o = {}) { super(); this.M = o.M ?? 99; }
    compute(d) {
      const keep = [];
      for (let j = 0; j < d.numAttributes; j++) {
        if (j === d.classIndex) { keep.push(j); continue; }
        const st = d.attributeStats(j);
        if (st.distinct < 2) continue;
        if (d.attributes[j].type === 'nominal') {
          const pct = (st.distinct / (st.total - st.missing)) * 100;
          if (pct > this.M) continue;
        }
        keep.push(j);
      }
      this.keep = keep;
      this.makeOut(d, keep.map((j) => d.attributes[j]), keep.indexOf(d.classIndex));
    }
    convertRow(r) { return this.keep.map((j) => r[j]); }
  }
  W.RemoveUseless = RemoveUseless;
  reg({ name: 'RemoveUseless', kind: 'attribute', supervised: false, path: 'weka.filters.unsupervised.attribute.RemoveUseless',
    desc: L`Αφαιρεί χαρακτηριστικά που δεν μεταβάλλονται ή μεταβάλλονται υπερβολικά (π.χ. αναγνωριστικά).`,
    options: [{ key: 'M', flag: 'M', label: L`Μέγιστο ποσοστό διακύμανσης (%)`, type: 'number', default: 99, min: 0, max: 100 }],
    create: (o) => new RemoveUseless(o) });

  // ---------------------------------------------------------- NumericToNominal
  class NumericToNominal extends AttrFilter {
    constructor(o = {}) { super(); this.R = o.R || 'first-last'; }
    compute(d) {
      const sel = new Set(U.parseRange(this.R, d.numAttributes));
      this.maps = [];
      const attrs = d.attributes.map((a, j) => {
        if (!W.isNumericAttr(a) || !sel.has(j)) return a;
        const vals = [...new Set(d.rows.map((r) => r[j]).filter((v) => v === v))].sort((x, y) => x - y);
        this.maps[j] = new Map(vals.map((v, k) => [v, k]));
        return { name: a.name, type: 'nominal', values: vals.map((v) => U.d2s(v, 6)) };
      });
      this.makeOut(d, attrs, d.classIndex);
    }
    convertRow(r) {
      return r.map((v, j) => {
        if (!this.maps[j] || v !== v) return v;
        const k = this.maps[j].get(v);
        return k === undefined ? NaN : k;
      });
    }
  }
  reg({ name: 'NumericToNominal', kind: 'attribute', supervised: false, path: 'weka.filters.unsupervised.attribute.NumericToNominal',
    desc: L`Μετατρέπει αριθμητικά χαρακτηριστικά σε nominal, με κάθε διακριτή τιμή ως κατηγορία.`,
    options: [{ key: 'R', flag: 'R', label: L`Εύρος χαρακτηριστικών`, type: 'text', default: 'first-last' }],
    create: (o) => new NumericToNominal(o) });

  // ----------------------------------------------------------- StringToNominal
  class StringToNominal extends AttrFilter {
    constructor(o = {}) { super(); this.R = o.R || 'first-last'; }
    compute(d) {
      const sel = new Set(U.parseRange(this.R, d.numAttributes));
      this.makeOut(d, d.attributes.map((a, j) => (a.type === 'string' && sel.has(j) ? { name: a.name, type: 'nominal', values: a.values.slice() } : a)), d.classIndex);
    }
    convertRow(r) { return r.slice(); }
  }
  reg({ name: 'StringToNominal', kind: 'attribute', supervised: false, path: 'weka.filters.unsupervised.attribute.StringToNominal',
    desc: L`Μετατρέπει χαρακτηριστικά τύπου string σε nominal.`,
    options: [{ key: 'R', flag: 'R', label: L`Εύρος χαρακτηριστικών`, type: 'text', default: 'first-last' }],
    create: (o) => new StringToNominal(o) });

  // ------------------------------------------------------------- AddExpression
  class AddExpression extends AttrFilter {
    constructor(o = {}) { super(); this.E = o.E || 'a1'; this.N = o.N || 'expression'; }
    compute(d) {
      // Ασφαλής μεταγλώττιση: μόνο αριθμοί, a1..aN, τελεστές και επιτρεπτές συναρτήσεις
      const allowed = /^(\s|\d|\.|a\d+|[-+*/^()]|,|e(?=[-+]?\d)|abs|sqrt|log|exp|sin|cos|tan|floor|ceil|rint|pow|max|min)*$/i;
      if (!allowed.test(this.E)) throw new Error(L`Μη έγκυρη έκφραση. Χρησιμοποίησε a1, a2…, αριθμούς, + - * / ^ ( ) και abs, sqrt, log, exp, sin, cos, tan, floor, ceil, rint, pow, max, min.`);
      const js = this.E.replace(/\^/g, '**').replace(/a(\d+)/gi, (_, n) => { const k = +n - 1; if (k < 0 || k >= d.numAttributes) throw new Error(L`Δεν υπάρχει χαρακτηριστικό a` + n); return `r[${k}]`; })
        .replace(/\b(abs|sqrt|log|exp|sin|cos|tan|floor|ceil|pow|max|min)\b/gi, (m) => 'Math.' + m.toLowerCase()).replace(/\brint\b/gi, 'Math.round');
      // eslint-disable-next-line no-new-func
      this.fn = new Function('r', '"use strict"; return (' + js + ');');
      this.makeOut(d, d.attributes.concat([{ name: this.N, type: 'numeric' }]), d.classIndex);
    }
    convertRow(r) {
      let v;
      try { v = this.fn(r.map((x, j) => (this.inHeader.attributes[j].type === 'numeric' || this.inHeader.attributes[j].type === 'date' ? x : NaN))); } catch (e) { v = NaN; }
      return r.concat([typeof v === 'number' && isFinite(v) ? v : NaN]);
    }
  }
  reg({ name: 'AddExpression', kind: 'attribute', supervised: false, path: 'weka.filters.unsupervised.attribute.AddExpression',
    desc: L`Προσθέτει νέο χαρακτηριστικό υπολογισμένο από έκφραση (π.χ. a1*a2, sqrt(a3)).`,
    options: [
      { key: 'E', flag: 'E', label: L`Έκφραση`, type: 'text', default: 'a1^2', placeholder: L`π.χ. a1/a2` },
      { key: 'N', flag: 'N', label: L`Όνομα νέου χαρακτηριστικού`, type: 'text', default: 'expression' },
    ], create: (o) => new AddExpression(o) });

  // ======================================================== instance filters
  class InstanceFilter { setInput(d) { this.inHeader = d; return this; } }
  class Randomize extends InstanceFilter {
    constructor(o = {}) { super(); this.S = o.S ?? 42; }
    apply(d) { const c = d.copy(); c.randomize(new W.JavaRandom(this.S)); return c; }
  }
  reg({ name: 'Randomize', kind: 'instance', supervised: false, path: 'weka.filters.unsupervised.instance.Randomize',
    desc: L`Ανακατεύει τυχαία τη σειρά των εγγραφών.`,
    options: [{ key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 42 }], create: (o) => new Randomize(o) });

  class Resample extends InstanceFilter {
    constructor(o = {}) { super(); this.Z = o.Z ?? 100; this.S = o.S ?? 1; this.noRepl = !!o.noReplacement; }
    apply(d) {
      const rand = new W.JavaRandom(this.S), n = d.numInstances;
      const size = Math.floor(n * this.Z / 100);
      const idx = [];
      if (this.noRepl) {
        if (size > n) throw new Error(L`Χωρίς επανάθεση το ποσοστό δεν μπορεί να ξεπερνά το 100%.`);
        const all = [...Array(n).keys()];
        for (let i = n - 1; i > 0; i--) { const k = rand.nextInt(i + 1); const t = all[i]; all[i] = all[k]; all[k] = t; }
        idx.push(...all.slice(0, size).sort((a, b) => a - b));
      } else for (let i = 0; i < size; i++) idx.push(rand.nextInt(n));
      return d.subset(idx);
    }
  }
  reg({ name: 'Resample', kind: 'instance', supervised: false, path: 'weka.filters.unsupervised.instance.Resample',
    desc: L`Παράγει τυχαίο δείγμα (με ή χωρίς επανάθεση).`,
    options: [
      { key: 'Z', flag: 'Z', label: L`Μέγεθος δείγματος (% του αρχικού)`, type: 'number', default: 100, min: 0 },
      { key: 'noReplacement', flag: 'no-replacement', label: L`Χωρίς επανάθεση`, type: 'bool', default: false },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 1 },
    ], create: (o) => new Resample(o) });

  class ClassBalancer extends InstanceFilter {
    apply(d) {
      if (!d.classIsNominal()) throw new Error(L`Το ClassBalancer απαιτεί nominal κλάση.`);
      const ci = d.classIndex, nc = d.numClasses, sums = U.zeros(nc);
      let tot = 0, present = 0;
      for (let i = 0; i < d.numInstances; i++) { const c = d.rows[i][ci]; if (c === c) { sums[c] += d.weights[i]; tot += d.weights[i]; } }
      for (const s of sums) if (s > 0) present++;
      const c = d.copy();
      c.weights = c.weights.map((w, i) => { const k = c.rows[i][ci]; return k === k && sums[k] > 0 ? w * tot / (present * sums[k]) : w; });
      return c;
    }
  }
  reg({ name: 'ClassBalancer', kind: 'instance', supervised: true, path: 'weka.filters.supervised.instance.ClassBalancer',
    desc: L`Αναπροσαρμόζει τα βάρη ώστε κάθε κλάση να έχει το ίδιο συνολικό βάρος.`,
    options: [], create: () => new ClassBalancer() });

  class SupervisedResample extends InstanceFilter {
    constructor(o = {}) { super(); this.Z = o.Z ?? 100; this.B = o.B ?? 0; this.S = o.S ?? 1; }
    apply(d) {
      if (!d.classIsNominal()) throw new Error(L`Απαιτείται nominal κλάση.`);
      const ci = d.classIndex, nc = d.numClasses, rand = new W.JavaRandom(this.S);
      const byClass = Array.from({ length: nc }, () => []);
      d.rows.forEach((r, i) => { if (r[ci] === r[ci]) byClass[r[ci]].push(i); });
      const n = d.numInstances, size = Math.floor(n * this.Z / 100), present = byClass.filter((x) => x.length).length;
      const idx = [];
      byClass.forEach((list) => {
        if (!list.length) return;
        const natural = list.length / n, uniform = 1 / present;
        const share = (1 - this.B) * natural + this.B * uniform;
        const k = Math.round(size * share);
        for (let i = 0; i < k; i++) idx.push(list[rand.nextInt(list.length)]);
      });
      return d.subset(idx);
    }
  }
  reg({ name: 'Resample (supervised)', kind: 'instance', supervised: true, path: 'weka.filters.supervised.instance.Resample',
    desc: L`Δείγμα με επανάθεση που μπορεί να εξισορροπήσει τις κλάσεις (bias προς ομοιόμορφη κατανομή).`,
    options: [
      { key: 'Z', flag: 'Z', label: L`Μέγεθος δείγματος (%)`, type: 'number', default: 100, min: 0 },
      { key: 'B', flag: 'B', label: L`Bias προς ομοιόμορφη κατανομή (0–1)`, type: 'number', default: 0, min: 0, max: 1, step: 0.1 },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 1 },
    ], create: (o) => new SupervisedResample(o) });

  class RemovePercentage extends InstanceFilter {
    constructor(o = {}) { super(); this.P = o.P ?? 50; this.V = !!o.V; }
    apply(d) {
      const n = d.numInstances, cut = Math.round(n * this.P / 100);
      const idx = [];
      for (let i = 0; i < n; i++) if ((i < cut) === this.V) idx.push(i);
      return d.subset(idx);
    }
  }
  reg({ name: 'RemovePercentage', kind: 'instance', supervised: false, path: 'weka.filters.unsupervised.instance.RemovePercentage',
    desc: L`Αφαιρεί το πρώτο Ρ% των εγγραφών (ή κρατά μόνο αυτό με αντιστροφή).`,
    options: [
      { key: 'P', flag: 'P', label: L`Ποσοστό (%)`, type: 'number', default: 50, min: 0, max: 100 },
      { key: 'V', flag: 'V', label: L`Αντιστροφή επιλογής`, type: 'bool', default: false },
    ], create: (o) => new RemovePercentage(o) });

  class RemoveWithValues extends InstanceFilter {
    constructor(o = {}) { super(); this.C = o.C || 'last'; this.S = o.S ?? 0; this.L = o.L || ''; this.V = !!o.V; this.M = !!o.M; }
    apply(d) {
      const att = U.parseRange(this.C, d.numAttributes)[0];
      const a = d.attributes[att];
      let test;
      if (a.type === 'nominal' || a.type === 'string') {
        const vals = new Set(this.L ? U.parseRange(this.L, a.values.length) : []);
        test = (v) => vals.has(v);
      } else test = (v) => v < this.S;
      const idx = [];
      d.rows.forEach((r, i) => {
        const v = r[att];
        let match = v !== v ? this.M : test(v);
        if (match === this.V) idx.push(i);
      });
      return d.subset(idx);
    }
  }
  reg({ name: 'RemoveWithValues', kind: 'instance', supervised: false, path: 'weka.filters.unsupervised.instance.RemoveWithValues',
    desc: L`Αφαιρεί εγγραφές βάσει τιμής: nominal στο σύνολο τιμών ή αριθμητικό μικρότερο από όριο.`,
    options: [
      { key: 'C', flag: 'C', label: L`Χαρακτηριστικό (δείκτης)`, type: 'text', default: 'last' },
      { key: 'S', flag: 'S', label: L`Όριο (για αριθμητικά: αφαίρεση αν < όριο)`, type: 'number', default: 0 },
      { key: 'L', flag: 'L', label: L`Δείκτες nominal τιμών (π.χ. 1,3)`, type: 'text', default: '' },
      { key: 'M', flag: 'M', label: L`Οι ελλιπείς θεωρούνται ταίριασμα`, type: 'bool', default: false },
      { key: 'V', flag: 'V', label: L`Αντιστροφή επιλογής`, type: 'bool', default: false },
    ], create: (o) => new RemoveWithValues(o) });

  class RemoveDuplicates extends InstanceFilter {
    apply(d) {
      const seen = new Set(), idx = [];
      d.rows.forEach((r, i) => { const k = r.map((v) => (v !== v ? '?' : v)).join('\u0001'); if (!seen.has(k)) { seen.add(k); idx.push(i); } });
      return d.subset(idx);
    }
  }
  reg({ name: 'RemoveDuplicates', kind: 'instance', supervised: false, path: 'weka.filters.unsupervised.instance.RemoveDuplicates',
    desc: L`Αφαιρεί τις διπλότυπες εγγραφές.`, options: [], create: () => new RemoveDuplicates() });

  // -------------------------------------------------------------- helpers
  W.isNumericAttr = (a) => a.type === 'numeric' || a.type === 'date';
  W.findFilter = (name) => W.filters.find((f) => f.name === name);
  /** Εφαρμογή φίλτρου σε σύνολο δεδομένων (attribute ή instance). */
  W.applyFilter = function (spec, options, data) {
    const f = spec.create(Object.assign({}, options));
    if (spec.supervised && data.classIndex < 0) throw new Error(L`Το εποπτευόμενο φίλτρο απαιτεί χαρακτηριστικό κλάσης.`);
    if (spec.kind === 'instance') return f.setInput(data).apply(data);
    f.setInput(data);
    return f.apply(data);
  };
  W.binLabels = binLabels;
})();
