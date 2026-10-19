/* Weka Web — επιλογή χαρακτηριστικών: evaluators και μέθοδοι αναζήτησης. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, S = W.Stats;
  W.evaluators = [];
  W.searches = [];
  const regE = (s) => { W.evaluators.push(s); return s; };
  const regS = (s) => { W.searches.push(s); return s; };
  W.findEvaluator = (n) => W.evaluators.find((s) => s.name === n);
  W.findSearch = (n) => W.searches.find((s) => s.name === n);

  // ------------------------------------------------------------ βοηθητικά
  /** Διακριτοποίηση MDL με better encoding (όπως χρησιμοποιούν InfoGain/GainRatio/CFS του Weka). */
  function discretize(d) {
    if (!d.attributes.some((a, j) => j !== d.classIndex && W.isNumericAttr(a))) return d;
    const f = W.findFilter('Discretize (supervised)').create({ E: true });
    f.setInput(d);
    return f.apply(d);
  }
  /** Πίνακας συνάφειας attribute × class με διαμοιρασμό ελλιπών (missing merge), όπως το Weka. */
  function contingency(d, att, mergeMissing = true) {
    const nv = d.attributes[att].values.length, nc = d.numClasses, ci = d.classIndex;
    const counts = U.matrix(nv + 1, nc + 1);
    for (let i = 0; i < d.numInstances; i++) {
      const r = d.rows[i], w = d.weights[i];
      const c = r[ci] !== r[ci] ? nc : r[ci];
      const v = r[att] !== r[att] ? nv : r[att];
      counts[v][c] += w;
    }
    if (!mergeMissing) return counts.map((row) => row.slice(0, nc)).filter((_, i) => i < nv || U.sum(counts[nv]) > 0);
    const rowSums = U.zeros(nv), colSums = U.zeros(nc);
    let sum = 0;
    for (let i = 0; i < nv; i++) { for (let j = 0; j < nc; j++) { rowSums[i] += counts[i][j]; colSums[j] += counts[i][j]; } sum += rowSums[i]; }
    if (!U.gr(sum, 0)) return counts.slice(0, nv).map((r) => r.slice(0, nc));
    const add = U.matrix(nv, nc);
    for (let i = 0; i < nv; i++) for (let j = 0; j < nc; j++) add[i][j] = (rowSums[i] / sum) * counts[nv][j];
    for (let j = 0; j < nc; j++) for (let i = 0; i < nv; i++) add[i][j] += (colSums[j] / sum) * counts[i][nc];
    for (let j = 0; j < nc; j++) for (let i = 0; i < nv; i++) add[i][j] += (counts[i][j] / sum) * counts[nv][nc];
    return counts.slice(0, nv).map((r, i) => r.slice(0, nc).map((v, j) => v + add[i][j]));
  }
  function gainRatioOf(m) {
    let pre = 0, post = 0, split = 0, total = 0;
    const nc = m[0].length;
    for (let j = 0; j < nc; j++) { let s = 0; for (const r of m) s += r[j]; pre += U.lnFunc(s); total += s; }
    pre -= U.lnFunc(total);
    for (const r of m) { let s = 0; for (const v of r) { post += U.lnFunc(v); s += v; } split += U.lnFunc(s); }
    post -= split; split -= U.lnFunc(total);
    const ig = pre - post;
    if (U.eq(split, 0)) return 0;
    return ig / split;
  }
  function symmUncert(m) {
    const hc = S.entropyOverColumns(m), hr = S.entropyOverRows(m), hcr = S.entropyConditionedOnRows(m);
    const ig = hc - hcr;
    if (hc + hr <= 0) return 0;
    return (2 * ig) / (hc + hr);
  }
  function pearson(x, y) {
    const n = x.length;
    if (n <= 1) return 1;
    let a1 = 0, a2 = 0;
    for (let i = 0; i < n; i++) { a1 += x[i]; a2 += y[i]; }
    a1 /= n; a2 /= n;
    let s1 = 0, s2 = 0, sxy = 0;
    for (let i = 0; i < n; i++) { const d1 = x[i] - a1, d2 = y[i] - a2; s1 += d1 * d1; s2 += d2 * d2; sxy += d1 * d2; }
    if (s1 * s2 > 0) return (sxy / (n - 1)) / (Math.sqrt(s1 / (n - 1)) * Math.sqrt(s2 / (n - 1)));
    return 1;
  }
  const variance = (x) => { const n = x.length; if (n < 2) return 0; const m = U.sum(x) / n; let s = 0; for (const v of x) s += (v - m) * (v - m); return s / (n - 1); };

  // ======================================================= single-attribute
  class InfoGainEval {
    constructor(o = {}) { this.M = !!o.M; }
    build(d) {
      if (!d.classIsNominal()) throw new Error(L`Το InfoGain απαιτεί nominal κλάση.`);
      d = discretize(d.withoutMissingClass());
      this.d = d;
      this.scores = d.attributes.map((a, j) => {
        if (j === d.classIndex) return NaN;
        if (a.type !== 'nominal') return 0;
        const m = contingency(d, j, !this.M);
        return S.entropyOverColumns(m) - S.entropyConditionedOnRows(m);
      });
      return this;
    }
    evaluate(j) { return this.scores[j]; }
    describe() { return 'Information Gain Ranking Filter' + (this.M ? '\n\tMissing values treated as separate' : ''); }
  }
  regE({ name: 'InfoGainAttributeEval', kind: 'single', path: 'weka.attributeSelection.InfoGainAttributeEval', supervised: true,
    desc: L`Κέρδος πληροφορίας ως προς την κλάση: H(C) − H(C|A). Τα αριθμητικά διακριτοποιούνται με MDL.`,
    options: [{ key: 'M', flag: 'M', label: L`Ελλιπείς τιμές ως ξεχωριστή τιμή`, type: 'bool', default: false }],
    create: (o) => new InfoGainEval(o) });

  class GainRatioEval extends InfoGainEval {
    build(d) {
      if (!d.classIsNominal()) throw new Error(L`Το GainRatio απαιτεί nominal κλάση.`);
      d = discretize(d.withoutMissingClass());
      this.scores = d.attributes.map((a, j) => (j === d.classIndex ? NaN : a.type !== 'nominal' ? 0 : gainRatioOf(contingency(d, j, !this.M))));
      return this;
    }
    describe() { return 'Gain Ratio feature evaluator'; }
  }
  regE({ name: 'GainRatioAttributeEval', kind: 'single', path: 'weka.attributeSelection.GainRatioAttributeEval', supervised: true,
    desc: L`Λόγος κέρδους: Information Gain / H(A). Τιμωρεί χαρακτηριστικά με πολλές τιμές.`,
    options: [{ key: 'M', flag: 'M', label: L`Ελλιπείς τιμές ως ξεχωριστή τιμή`, type: 'bool', default: false }],
    create: (o) => new GainRatioEval(o) });

  class SymmUncertEval extends InfoGainEval {
    build(d) {
      if (!d.classIsNominal()) throw new Error(L`Απαιτείται nominal κλάση.`);
      d = discretize(d.withoutMissingClass());
      this.scores = d.attributes.map((a, j) => (j === d.classIndex ? NaN : a.type !== 'nominal' ? 0 : symmUncert(contingency(d, j, !this.M))));
      return this;
    }
    describe() { return 'Symmetrical Uncertainty Ranking Filter'; }
  }
  regE({ name: 'SymmetricalUncertAttributeEval', kind: 'single', path: 'weka.attributeSelection.SymmetricalUncertAttributeEval', supervised: true,
    desc: L`Συμμετρική αβεβαιότητα: 2·IG / (H(A)+H(C)).`,
    options: [{ key: 'M', flag: 'M', label: L`Ελλιπείς τιμές ως ξεχωριστή τιμή`, type: 'bool', default: false }],
    create: (o) => new SymmUncertEval(o) });

  class CorrelationEval {
    build(d) {
      d = d.withoutMissingClass();
      const rmv = new W.ReplaceMissingValues(); rmv.setInput(d); d = rmv.apply(d);
      const ci = d.classIndex, n = d.numInstances;
      const col = (j) => d.rows.map((r) => r[j]);
      const ind = (j, v) => d.rows.map((r) => (r[j] === v ? 1 : 0));
      this.scores = d.attributes.map((a, j) => {
        if (j === ci) return NaN;
        if (!d.classIsNominal()) {
          const cv = col(ci);
          if (a.type !== 'nominal') { const x = col(j); let c = pearson(x, cv); if (c === 1 && variance(x) === 0) c = 0; return c; }
          let sum = 0, sc = 0;
          a.values.forEach((_, v) => { const x = ind(j, v), sv = U.sum(x); let c = pearson(x, cv); if (sv === n || sv === 0) c = 0; sc += sv * Math.abs(c); sum += sv; });
          return sum > 0 ? sc / sum : 0;
        }
        const nc = d.numClasses, bc = [], cc = [];
        for (let k = 0; k < nc; k++) { bc.push(ind(ci, k)); cc.push(U.sum(bc[k])); }
        const sumClass = U.sum(cc);
        if (a.type !== 'nominal') {
          const x = col(j); let sc = 0;
          for (let k = 0; k < nc; k++) { let c = Math.abs(pearson(x, bc[k])); if (c === 1 && variance(x) === 0) c = 0; sc += cc[k] * c; }
          return sc / sumClass;
        }
        let sumAtt = 0, corrAtt = 0;
        a.values.forEach((_, v) => {
          const x = ind(j, v), sv = U.sum(x); let sc = 0;
          sumAtt += sv;
          for (let k = 0; k < nc; k++) { let c = pearson(x, bc[k]); if (sv === n || sv === 0) c = 0; sc += cc[k] * Math.abs(c); }
          corrAtt += sv * (sc / sumClass);
        });
        return sumAtt > 0 ? corrAtt / sumAtt : 0;
      });
      return this;
    }
    evaluate(j) { return this.scores[j]; }
    describe() { return 'Correlation Ranking Filter'; }
  }
  regE({ name: 'CorrelationAttributeEval', kind: 'single', path: 'weka.attributeSelection.CorrelationAttributeEval', supervised: true,
    desc: L`Συσχέτιση Pearson με την κλάση (για nominal: σταθμισμένος μέσος των δυαδικών δεικτών).`,
    options: [], create: () => new CorrelationEval() });

  class ReliefFEval {
    constructor(o = {}) { this.M = o.M ?? -1; this.K = o.K ?? 10; this.seed = o.D ?? 1; this.W = !!o.W; this.A = o.A ?? 2; }
    build(d) {
      this.d = d;
      const ci = d.classIndex, na = d.numAttributes, n = d.numInstances, numeric = !d.classIsNominal();
      const nc = numeric ? 1 : d.numClasses;
      const mn = new Array(na).fill(NaN), mx = new Array(na).fill(NaN);
      for (const r of d.rows) for (let j = 0; j < na; j++) { const v = r[j]; if (v !== v || d.attributes[j].type === 'nominal') continue; if (mn[j] !== mn[j]) { mn[j] = mx[j] = v; } else if (v < mn[j]) mn[j] = v; else if (v > mx[j]) mx[j] = v; }
      const norm = (x, j) => (mn[j] !== mn[j] || U.eq(mx[j], mn[j]) ? 0 : (x - mn[j]) / (mx[j] - mn[j]));
      const diff = (j, a, b) => {
        const at = d.attributes[j];
        if (at.type === 'nominal') { if (a !== a || b !== b) return 1 - 1 / at.values.length; return a !== b ? 1 : 0; }
        if (a !== a || b !== b) { if (a !== a && b !== b) return 1; let x = b !== b ? norm(a, j) : norm(b, j); return x < 0.5 ? 1 - x : x; }
        return Math.abs(norm(a, j) - norm(b, j));
      };
      const dist = (x, y) => { let s = 0; for (let j = 0; j < na; j++) if (j !== ci) s += diff(j, x[j], y[j]); return s; };
      const probs = U.zeros(nc);
      if (!numeric) { for (const r of d.rows) if (r[ci] === r[ci]) probs[r[ci]]++; for (let c = 0; c < nc; c++) probs[c] /= n; }
      const rankW = []; for (let i = 0; i < this.K; i++) rankW.push(Math.exp(-((i / this.A) * (i / this.A))));
      const w = U.zeros(na), nda = U.zeros(na), ndcda = U.zeros(na);
      let ndc = 0;
      const total = this.M > n || this.M < 0 ? n : this.M;
      const rand = new W.JavaRandom(this.seed);
      let cRange = 1;
      if (numeric) { cRange = mx[ci] - mn[ci] || 1; }
      for (let s = 0; s < total; s++) {
        let z = total === n ? s : rand.nextInt() % n;
        if (z < 0) z = -z;
        const inst = d.rows[z];
        if (inst[ci] !== inst[ci]) continue;
        // k πλησιέστεροι ανά κλάση (με αντικατάσταση του χειρότερου όπως στο Weka)
        const near = Array.from({ length: nc }, () => []);
        for (let i = 0; i < n; i++) {
          if (i === z) continue;
          const r = d.rows[i];
          if (r[ci] !== r[ci]) continue;
          const c = numeric ? 0 : r[ci];
          const dd = dist(inst, r);
          const L = near[c];
          if (L.length < this.K) L.push([dd, i]);
          else { let wi = 0; for (let q = 1; q < L.length; q++) if (L[q][0] > L[wi][0]) wi = q; if (dd < L[wi][0]) L[wi] = [dd, i]; }
        }
        for (const L of near) L.sort((a, b) => a[0] - b[0]);
        if (numeric) {
          const L = near[0];
          const wsum = this.W ? rankW.slice(0, L.length).reduce((a, b) => a + b, 0) : L.length;
          L.forEach(([, i], rank) => {
            const r = d.rows[i], wt = (this.W ? rankW[rank] : 1) / wsum;
            const dc = diff(ci, inst[ci], r[ci]);
            ndc += dc * wt;
            for (let j = 0; j < na; j++) { if (j === ci) continue; const da = diff(j, inst[j], r[j]); nda[j] += da * wt; ndcda[j] += dc * da * wt; }
          });
        } else {
          const cl = inst[ci];
          const hits = near[cl];
          const hw = this.W ? rankW.slice(0, hits.length).reduce((a, b) => a + b, 0) : hits.length;
          hits.forEach(([, i], rank) => { const r = d.rows[i]; for (let j = 0; j < na; j++) if (j !== ci) w[j] -= diff(j, inst[j], r[j]) * (this.W ? rankW[rank] : 1) / hw; });
          const wnorm = 1 - probs[cl];
          for (let k = 0; k < nc; k++) {
            if (k === cl) continue;
            const miss = near[k];
            const mw = this.W ? rankW.slice(0, miss.length).reduce((a, b) => a + b, 0) : miss.length;
            miss.forEach(([, i], rank) => { const r = d.rows[i]; for (let j = 0; j < na; j++) if (j !== ci) w[j] += (probs[k] / wnorm) * diff(j, inst[j], r[j]) * (this.W ? rankW[rank] : 1) / mw; });
          }
        }
      }
      this.scores = w.map((v, j) => {
        if (j === ci) return NaN;
        if (numeric) return ndcda[j] / ndc - (nda[j] - ndcda[j]) / (total - ndc);
        return v / total;
      });
      return this;
    }
    evaluate(j) { return this.scores[j]; }
    describe() { return 'ReliefF Ranking Filter\n\tInstances sampled: ' + (this.M < 0 ? 'all' : this.M) + '\n\tNumber of nearest neighbours (k): ' + this.K + '\n\tEqual influence nearest neighbours'; }
  }
  regE({ name: 'ReliefFAttributeEval', kind: 'single', path: 'weka.attributeSelection.ReliefFAttributeEval', supervised: true,
    desc: L`ReliefF: αξιολογεί πόσο καλά ένα χαρακτηριστικό ξεχωρίζει γειτονικές εγγραφές διαφορετικών κλάσεων.`,
    options: [
      { key: 'M', flag: 'M', label: L`Δείγματα (−1 = όλα)`, type: 'int', default: -1, min: -1 },
      { key: 'K', flag: 'K', label: L`Πλησιέστεροι γείτονες (k)`, type: 'int', default: 10, min: 1 },
      { key: 'W', flag: 'W', label: L`Στάθμιση γειτόνων με απόσταση`, type: 'bool', default: false },
      { key: 'A', flag: 'A', label: L`Sigma (για στάθμιση)`, type: 'int', default: 2, min: 1 },
      { key: 'D', flag: 'D', label: 'Seed', type: 'int', default: 1 },
    ], create: (o) => new ReliefFEval(o) });

  class OneRAttrEval {
    constructor(o = {}) { this.F = o.F ?? 10; this.seed = o.S ?? 1; this.B = o.B ?? 6; this.D = !!o.D; }
    build(d) {
      if (!d.classIsNominal()) throw new Error(L`Απαιτείται nominal κλάση.`);
      d = d.withoutMissingClass();
      const ci = d.classIndex;
      this.scores = d.attributes.map((a, j) => {
        if (j === ci) return NaN;
        const sub = new W.Remove({ indices: [j, ci], V: true }); sub.setInput(d); const dj = sub.apply(d);
        const spec = W.findClassifier('OneR');
        if (this.D) { const m = spec.create({ B: this.B }).build(dj); const e = new W.Evaluation(dj); e.evaluateModel(m, dj); return e.pctCorrect(); }
        const folds = Math.min(this.F, dj.numInstances);
        const c = dj.copy(); const rand = new W.JavaRandom(this.seed); c.randomize(rand); c.stratify(folds);
        const e = new W.Evaluation(c);
        for (let f = 0; f < folds; f++) { const tr = c.trainCV(folds, f, rand); e.setPriors(tr); const m = spec.create({ B: this.B }).build(tr); e.evaluateModel(m, c.testCV(folds, f)); }
        return e.pctCorrect();
      });
      return this;
    }
    evaluate(j) { return this.scores[j]; }
    describe() { return 'OneR feature evaluator.\n\n\tUsing ' + (this.D ? 'training data for evaluation of attributes.' : this.F + ' fold cross validation for evaluating attributes.') + '\n\tMinimum bucket size for OneR: ' + this.B; }
  }
  regE({ name: 'OneRAttributeEval', kind: 'single', path: 'weka.attributeSelection.OneRAttributeEval', supervised: true,
    desc: L`Αξιολογεί κάθε χαρακτηριστικό με την ακρίβεια (%) του ταξινομητή OneR.`,
    options: [
      { key: 'F', flag: 'F', label: 'Folds', type: 'int', default: 10, min: 2 },
      { key: 'D', flag: 'D', label: L`Αξιολόγηση στα δεδομένα εκπαίδευσης`, type: 'bool', default: false },
      { key: 'B', flag: 'B', label: L`Ελάχιστο μέγεθος κάδου`, type: 'int', default: 6, min: 1 },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 1 },
    ], create: (o) => new OneRAttrEval(o) });

  // ================================================================ subset
  class CfsEval {
    constructor(o = {}) { this.M = !!o.M; this.L = !!o.L; }
    build(d) {
      d = d.withoutMissingClass();
      this.nominalClass = d.classIsNominal();
      if (this.nominalClass) d = discretize(d);
      else { const rmv = new W.ReplaceMissingValues(); rmv.setInput(d); d = rmv.apply(d); }
      this.d = d; this.cache = new Map();
      return this;
    }
    corr(i, j) {
      const k = i < j ? i * 100000 + j : j * 100000 + i;
      if (this.cache.has(k)) return this.cache.get(k);
      const d = this.d;
      let v;
      const ai = d.attributes[i], aj = d.attributes[j];
      if (ai.type === 'nominal' && aj.type === 'nominal') {
        // πίνακας συνάφειας i × j με διαμοιρασμό ελλιπών
        // μεταφορά του CfsSubsetEval.symmUncertCorr (μη σταθμισμένες μετρήσεις, διαμοιρασμός ελλιπών)
        const ci = d.classIndex, flag = i === ci || j === ci;
        const NI = ai.values.length + 1, NJ = aj.values.length + 1, N = d.numInstances;
        const c = U.matrix(NI, NJ);
        for (let r = 0; r < N; r++) { const x = d.rows[r][i], y = d.rows[r][j]; c[x !== x ? NI - 1 : x][y !== y ? NJ - 1 : y]++; }
        const si = U.zeros(NI), sj = U.zeros(NJ); let sum = 0;
        for (let a = 0; a < NI; a++) for (let b = 0; b < NJ; b++) { si[a] += c[a][b]; sj[b] += c[a][b]; sum += c[a][b]; }
        if (!this.M && si[NI - 1] < N && sj[NJ - 1] < N) {
          const ic = si.slice(), jc = sj.slice(), cc = c.map((row) => row.slice());
          const totMiss = si[NI - 1] + sj[NJ - 1] - c[NI - 1][NJ - 1];
          if (si[NI - 1] > 0) for (let b = 0; b < NJ - 1; b++) if (c[NI - 1][b] > 0) { for (let a = 0; a < NI - 1; a++) { const t = (ic[a] / (sum - ic[NI - 1])) * c[NI - 1][b]; c[a][b] += t; si[a] += t; } c[NI - 1][b] = 0; }
          si[NI - 1] = 0;
          if (sj[NJ - 1] > 0) for (let a = 0; a < NI - 1; a++) if (c[a][NJ - 1] > 0) { for (let b = 0; b < NJ - 1; b++) { const t = (jc[b] / (sum - jc[NJ - 1])) * c[a][NJ - 1]; c[a][b] += t; sj[b] += t; } c[a][NJ - 1] = 0; }
          sj[NJ - 1] = 0;
          if (c[NI - 1][NJ - 1] > 0 && totMiss !== sum) {
            for (let a = 0; a < NI - 1; a++) for (let b = 0; b < NJ - 1; b++) { const t = (cc[a][b] / (sum - totMiss)) * cc[NI - 1][NJ - 1]; c[a][b] += t; si[a] += t; sj[b] += t; }
            c[NI - 1][NJ - 1] = 0;
          }
        }
        // ContingencyTables.symmetricalUncertainty
        let colE = 0, rowE = 0, condE = 0, tot = 0;
        for (let b = 0; b < NJ; b++) { let s = 0; for (let a = 0; a < NI; a++) s += c[a][b]; colE += U.lnFunc(s); tot += s; }
        colE -= U.lnFunc(tot);
        for (let a = 0; a < NI; a++) { let s = 0; for (let b = 0; b < NJ; b++) { s += c[a][b]; condE += U.lnFunc(c[a][b]); } rowE += U.lnFunc(s); }
        condE -= rowE; rowE -= U.lnFunc(tot);
        const su = U.eq(colE, 0) || U.eq(rowE, 0) ? 0 : 2 * ((colE - condE) / (colE + rowE));
        v = U.eq(su, 0) ? (flag ? 0 : 1) : su;
      } else {
        const toVec = (a, idx) => (a.type === 'nominal' ? null : d.rows.map((r) => r[idx]));
        const xi = toVec(ai, i), xj = toVec(aj, j);
        if (xi && xj) v = Math.abs(pearson(xi, xj));
        else {
          const [nomIdx, numVec] = xi ? [j, xi] : [i, xj];
          const na = d.attributes[nomIdx];
          let sc = 0, st = 0;
          na.values.forEach((_, q) => { const ind = d.rows.map((r) => (r[nomIdx] === q ? 1 : 0)); const sv = U.sum(ind); const c = sv === 0 || sv === d.numInstances ? 0 : Math.abs(pearson(ind, numVec)); sc += sv * c; st += sv; });
          v = st > 0 ? sc / st : 0;
        }
      }
      if (!isFinite(v)) v = 0;
      this.cache.set(k, v);
      return v;
    }
    evaluateSubset(sub) {
      const ci = this.d.classIndex;
      let num = 0, den = 0;
      for (const i of sub) { num += this.corr(i, ci); den += 1; }
      for (let a = 0; a < sub.length; a++) for (let b = a + 1; b < sub.length; b++) den += 2 * this.corr(sub[a], sub[b]);
      if (den < 0) den = -den;
      if (den === 0) return 0;
      let merit = num / Math.sqrt(den);
      return merit < 0 ? -merit : merit;
    }
    /** Προσθήκη "locally predictive" χαρακτηριστικών, όπως το CfsSubsetEval.postProcess του Weka. */
    postProcess(sel) {
      if (this.L) return sel;
      const ci = this.d.classIndex, na = this.d.numAttributes;
      const f = (v) => Math.fround(v);
      const best = new Set(sel), temp = new Set(sel);
      for (;;) {
        let tb = -1, j = 0;
        for (let i = 0; i < na; i++) if (!temp.has(i) && i !== ci) { const c = f(this.corr(i, ci)); if (c > tb) { tb = c; j = i; } }
        if (tb === -1) break;
        temp.add(j);
        let ok = true;
        for (const i of best) if (f(this.corr(i, j)) > tb) { ok = false; break; }
        if (ok) best.add(j);
      }
      return [...best].sort((a, b) => a - b);
    }
    describe() { return 'CFS Subset Evaluator' + (this.L ? '' : '\n\tIncluding locally predictive attributes'); }
  }
  regE({ name: 'CfsSubsetEval', kind: 'subset', path: 'weka.attributeSelection.CfsSubsetEval', supervised: true,
    desc: L`Correlation-based Feature Selection: προτιμά υποσύνολα που συσχετίζονται με την κλάση αλλά όχι μεταξύ τους.`,
    options: [
      { key: 'M', flag: 'M', label: L`Ελλιπείς τιμές ως ξεχωριστή τιμή`, type: 'bool', default: false },
      { key: 'L', flag: 'L', label: L`Χωρίς locally predictive χαρακτηριστικά`, type: 'bool', default: false },
    ],
    create: (o) => new CfsEval(o) });

  class WrapperEval {
    constructor(o = {}) { this.B = o.B || 'J48'; this.F = o.F ?? 5; this.seed = o.R ?? 1; }
    build(d) { this.d = d.withoutMissingClass(); this.spec = W.findClassifier(this.B); this.cache = new Map(); return this; }
    evaluateSubset(sub) {
      const k = sub.slice().sort((a, b) => a - b).join(',');
      if (this.cache.has(k)) return this.cache.get(k);
      const d = this.d, ci = d.classIndex;
      const rm = new W.Remove({ indices: sub.concat([ci]), V: true }); rm.setInput(d); const dd = rm.apply(d);
      const folds = Math.min(this.F, dd.numInstances);
      const c = dd.copy(); const rand = new W.JavaRandom(this.seed); c.randomize(rand); if (c.classIsNominal()) c.stratify(folds);
      const e = new W.Evaluation(c);
      for (let f = 0; f < folds; f++) { const tr = c.trainCV(folds, f, rand); e.setPriors(tr); const m = this.spec.create(W.defaultOptions(this.spec)).build(tr); e.evaluateModel(m, c.testCV(folds, f)); }
      const v = c.classIsNominal() ? e.pctCorrect() / 100 : -e.rootMeanSquaredError();
      this.cache.set(k, v);
      return v;
    }
    describe() { return 'Wrapper Subset Evaluator (' + this.spec.path + ')\n\tNumber of folds for accuracy estimation: ' + this.F; }
  }
  regE({ name: 'WrapperSubsetEval', kind: 'subset', path: 'weka.attributeSelection.WrapperSubsetEval', supervised: true,
    desc: L`Αξιολογεί υποσύνολα με την ακρίβεια cross-validation ενός ταξινομητή (αργό αλλά ακριβές).`,
    options: [
      { key: 'B', flag: 'B', label: L`Ταξινομητής`, type: 'select', default: 'J48', choices: () => Object.fromEntries(W.classifiers.filter((s) => !s.meta).map((s) => [s.name, s.name])) },
      { key: 'F', flag: 'F', label: 'Folds', type: 'int', default: 5, min: 2 },
      { key: 'R', flag: 'R', label: 'Seed', type: 'int', default: 1 },
    ], create: (o) => new WrapperEval(o) });

  // ===================================================== PrincipalComponents
  function jacobiEigen(A) {
    const n = A.length, a = A.map((r) => r.slice()), V = U.matrix(n, n);
    for (let i = 0; i < n; i++) V[i][i] = 1;
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    return { values: a.map((r, i) => r[i]), vectors: V };
  }
  W.jacobiEigen = jacobiEigen;
  class PCAEval {
    constructor(o = {}) { this.R = o.R ?? 0.95; this.A = o.A ?? 5; this.C = !!o.C; this.O = !!o.O; }
    build(d) {
      this.orig = d;
      this.pipe = new W.Pipeline([new W.ReplaceMissingValues(), new W.NominalToBinary({ N: true }), new W.RemoveUseless()]);
      const dd = this.pipe.fit(d);
      const ci = dd.classIndex;
      this.cols = []; for (let j = 0; j < dd.numAttributes; j++) if (j !== ci) this.cols.push(j);
      const p = this.cols.length, n = dd.numInstances;
      if (p === 0) throw new Error(L`Δεν υπάρχουν χαρακτηριστικά για PCA.`);
      this.mean = this.cols.map((j) => U.sum(dd.rows.map((r) => r[j])) / n);
      this.sd = this.cols.map((j, k) => Math.sqrt(dd.rows.reduce((s, r) => s + (r[j] - this.mean[k]) ** 2, 0) / (n - 1)) || 1);
      const X = dd.rows.map((r) => this.cols.map((j, k) => (this.C ? r[j] - this.mean[k] : (r[j] - this.mean[k]) / this.sd[k])));
      const M = U.matrix(p, p);
      for (const x of X) for (let a = 0; a < p; a++) for (let b = a; b < p; b++) M[a][b] += x[a] * x[b];
      for (let a = 0; a < p; a++) for (let b = a; b < p; b++) { M[a][b] /= n - 1; M[b][a] = M[a][b]; }
      this.matrix = M;
      const { values, vectors } = jacobiEigen(M);
      const order = values.map((v, i) => i).sort((a, b) => values[b] - values[a]);
      this.values = order.map((i) => values[i]);
      this.vectors = order.map((i) => vectors.map((row) => row[i]));
      // σταθερό πρόσημο: μεγαλύτερο |στοιχείο| θετικό
      this.vectors = this.vectors.map((v) => { let mi = 0; v.forEach((x, i) => { if (Math.abs(x) > Math.abs(v[mi])) mi = i; }); return v[mi] < 0 ? v.map((x) => -x) : v; });
      const total = U.sum(this.values.map((v) => Math.max(0, v)));
      let cum = 0;
      this.cum = this.values.map((v) => (cum += Math.max(0, v)) / total);
      this.numKeep = this.cum.findIndex((c) => c >= this.R) + 1 || this.values.length;
      this.header = dd;
      this.names = this.vectors.slice(0, this.numKeep).map((v) => this.compName(v));
      return this;
    }
    compName(v) {
      const idx = v.map((x, i) => i).sort((a, b) => Math.abs(v[b]) - Math.abs(v[a])).slice(0, this.A < 0 ? v.length : this.A);
      return idx.map((i, k) => { const c = U.d2s(v[i], 3); return (k > 0 && v[i] >= 0 ? '+' : '') + c + this.header.attributes[this.cols[i]].name; }).join('');
    }
    evaluate(j) { return 1 - this.cum[j]; }
    transformedHeaderNames() { return this.names; }
    transform(d) {
      const ci = d.classIndex;
      const attrs = this.names.map((nm) => ({ name: nm, type: 'numeric' }));
      if (ci >= 0) attrs.push(W.cloneAttr(d.attributes[ci]));
      const rows = d.rows.map((r) => {
        const rr = this.pipe.row(r);
        const x = this.cols.map((j, k) => (this.C ? rr[j] - this.mean[k] : (rr[j] - this.mean[k]) / this.sd[k]));
        const out = this.vectors.slice(0, this.numKeep).map((v) => v.reduce((s, c, i) => s + c * x[i], 0));
        if (ci >= 0) out.push(r[ci]);
        return out;
      });
      return new W.Dataset(d.relation + '_principal components', attrs, rows, d.weights.slice(), ci >= 0 ? attrs.length - 1 : -1);
    }
    describe() {
      const p = this.cols.length, t = ['Principal Components Attribute Transformer', '', this.C ? 'Covariance matrix' : 'Correlation matrix'];
      for (let a = 0; a < p; a++) t.push(' ' + this.matrix[a].map((v) => U.padR(U.d2s(v, 2), 6)).join(' '));
      t.push('', '', 'eigenvalue\tproportion\tcumulative');
      const total = U.sum(this.values.map((v) => Math.max(0, v)));
      for (let k = 0; k < this.numKeep; k++) t.push(U.padL(U.d2s(this.values[k], 5), 9) + '\t' + U.padL(U.d2s(this.values[k] / total, 5), 9) + '\t' + U.padL(U.d2s(this.cum[k], 5), 9) + '\t' + this.names[k]);
      t.push('', 'Eigenvectors');
      t.push(this.vectors.slice(0, this.numKeep).map((_, k) => ' V' + (k + 1)).join('\t'));
      for (let i = 0; i < p; i++) t.push(this.vectors.slice(0, this.numKeep).map((v) => U.padL(U.d2s(v[i], 4), 7)).join('\t') + '\t' + this.header.attributes[this.cols[i]].name);
      return t.join('\n');
    }
  }
  regE({ name: 'PrincipalComponents', kind: 'transform', path: 'weka.attributeSelection.PrincipalComponents', supervised: false,
    desc: L`Ανάλυση κύριων συνιστωσών: μετασχηματίζει τα δεδομένα σε ασυσχέτιστες συνιστώσες (μαζί με Ranker).`,
    options: [
      { key: 'R', flag: 'R', label: L`Ποσοστό διακύμανσης που καλύπτεται`, type: 'number', default: 0.95, min: 0, max: 1, step: 0.05 },
      { key: 'A', flag: 'A', label: L`Μέγιστα ονόματα χαρακτηριστικών ανά συνιστώσα`, type: 'int', default: 5, min: -1 },
      { key: 'C', flag: 'C', label: L`Covariance (χωρίς τυποποίηση)`, type: 'bool', default: false },
    ], create: (o) => new PCAEval(o) });

  // ================================================================ searches
  class Ranker {
    constructor(o = {}) { this.N = o.N ?? -1; this.T = o.T ?? -Number.MAX_VALUE; }
    search(ev, d) {
      let idx;
      if (ev instanceof PCAEval) idx = ev.values.slice(0, ev.numKeep).map((_, k) => k);
      else { idx = []; for (let j = 0; j < d.numAttributes; j++) if (j !== d.classIndex) idx.push(j); }
      // σειρά όπως το Weka: Utils.sort (quicksort) αύξουσα και αντιστροφή
      const merits = idx.map((j) => ev.evaluate(j));
      const order = U.wekaSort(merits).reverse();
      const ranked = order.map((k) => [idx[k], merits[k]]);
      let sel = ranked.filter((r) => r[1] > this.T);
      if (this.N >= 0) sel = sel.slice(0, this.N);
      return { ranked, selected: sel.map((r) => r[0]) };
    }
    describe() { return '\tAttribute ranking.' + (this.T > -Number.MAX_VALUE ? '\n\tThreshold for discarding attributes: ' + this.T : ''); }
  }
  regS({ name: 'Ranker', kind: 'ranker', path: 'weka.attributeSelection.Ranker', desc: L`Κατάταξη χαρακτηριστικών βάσει της αξιολόγησής τους (για single-attribute evaluators).`,
    options: [
      { key: 'N', flag: 'N', label: L`Πλήθος προς επιλογή (−1 = όλα)`, type: 'int', default: -1, min: -1 },
      { key: 'T', flag: 'T', label: L`Κατώφλι απόρριψης`, type: 'number', default: -1.7976931348623157e308 },
    ], create: (o) => new Ranker(o) });

  class BestFirst {
    constructor(o = {}) { this.D = o.D ?? 1; this.N = o.N ?? 5; }
    search(ev, d) {
      const ci = d.classIndex, all = []; for (let j = 0; j < d.numAttributes; j++) if (j !== ci) all.push(j);
      const key = (s) => s.slice().sort((a, b) => a - b).join(',');
      const start = this.D === 0 ? all.slice() : [];
      let best = start, bestMerit = ev.evaluateSubset(start);
      const open = [{ set: start, merit: bestMerit }];
      const seen = new Set([key(start)]);
      let stale = 0, evals = 1;
      while (open.length && stale < this.N) {
        open.sort((a, b) => b.merit - a.merit);
        const cur = open.shift();
        let improved = false;
        const children = [];
        if (this.D !== 0) for (const j of all) if (!cur.set.includes(j)) children.push(cur.set.concat([j]));
        if (this.D !== 1) for (const j of cur.set) children.push(cur.set.filter((x) => x !== j));
        for (const c of children) {
          const k = key(c);
          if (seen.has(k)) continue;
          seen.add(k);
          const m = ev.evaluateSubset(c); evals++;
          open.push({ set: c, merit: m });
          if (m - bestMerit > 1e-5) { best = c; bestMerit = m; improved = true; }
        }
        if (improved) stale = 0; else stale++;
      }
      this.evals = evals; this.merit = bestMerit;
      return { selected: best.slice().sort((a, b) => a - b), merit: bestMerit, evals };
    }
    describe() { return '\tBest first.\n\tStart set: no attributes\n\tSearch direction: ' + ['backward', 'forward', 'bi-directional'][this.D] + '\n\tStale search after ' + this.N + ' node expansions\n\tTotal number of subsets evaluated: ' + this.evals + '\n\tMerit of best subset found: ' + U.d2sw(this.merit, 7, 3).trim(); }
  }
  regS({ name: 'BestFirst', kind: 'subset', path: 'weka.attributeSelection.BestFirst', desc: L`Αναζήτηση best-first με backtracking (για subset evaluators).`,
    options: [
      { key: 'D', flag: 'D', label: L`Κατεύθυνση`, type: 'select', default: 1, numeric: true, choices: { 1: 'Forward', 0: 'Backward', 2: 'Bi-directional' } },
      { key: 'N', flag: 'N', label: L`Τερματισμός μετά από μη βελτιωτικές επεκτάσεις`, type: 'int', default: 5, min: 1 },
    ], create: (o) => new BestFirst(o) });

  class GreedyStepwise {
    constructor(o = {}) { this.B = !!o.B; }
    search(ev, d) {
      const ci = d.classIndex, all = []; for (let j = 0; j < d.numAttributes; j++) if (j !== ci) all.push(j);
      let cur = this.B ? all.slice() : [], merit = ev.evaluateSubset(cur), evals = 1;
      for (;;) {
        let best = null, bm = merit;
        const cands = this.B ? cur.map((j) => cur.filter((x) => x !== j)) : all.filter((j) => !cur.includes(j)).map((j) => cur.concat([j]));
        for (const c of cands) { const m = ev.evaluateSubset(c); evals++; if (m > bm) { bm = m; best = c; } }
        if (!best) break;
        cur = best; merit = bm;
      }
      this.evals = evals; this.merit = merit;
      return { selected: cur.slice().sort((a, b) => a - b), merit, evals };
    }
    describe() { return '\tGreedy Stepwise (' + (this.B ? 'backwards' : 'forwards') + ').\n\tStart set: ' + (this.B ? 'all attributes' : 'no attributes') + '\n\tMerit of best subset found: ' + U.d2sw(this.merit, 7, 3).trim() + '\n\tTotal number of subsets evaluated: ' + this.evals; }
  }
  regS({ name: 'GreedyStepwise', kind: 'subset', path: 'weka.attributeSelection.GreedyStepwise', desc: L`Άπληστη προσθήκη (ή αφαίρεση) χαρακτηριστικών όσο βελτιώνεται η αξιολόγηση.`,
    options: [{ key: 'B', flag: 'B', label: L`Αναζήτηση προς τα πίσω (backward)`, type: 'bool', default: false }], create: (o) => new GreedyStepwise(o) });

  // ============================================================== runner
  W.runAttributeSelection = async function (opts) {
    const { evalSpec, evalOptions, searchSpec, searchOptions } = opts;
    const data = opts.data;
    if (evalSpec.supervised && data.classIndex < 0) throw new Error(L`Ο evaluator απαιτεί χαρακτηριστικό κλάσης.`);
    const kind = evalSpec.kind;
    if ((kind === 'single' || kind === 'transform') && searchSpec.kind !== 'ranker') throw new Error(L`Ο ${evalSpec.name} αξιολογεί μεμονωμένα χαρακτηριστικά· χρησιμοποίησε τη μέθοδο αναζήτησης Ranker.`);
    if (kind === 'subset' && searchSpec.kind === 'ranker') throw new Error(L`Ο ${evalSpec.name} αξιολογεί υποσύνολα· χρησιμοποίησε BestFirst ή GreedyStepwise.`);
    for (let j = 0; j < data.numAttributes; j++) if (data.attributes[j].type === 'string') throw new Error(L`Το χαρακτηριστικό "${data.attributes[j].name}" είναι string· αφαίρεσέ το πρώτα.`);
    await W.nextFrame();
    const t0 = performance.now();
    const run = (d) => {
      const ev = evalSpec.create(Object.assign({}, evalOptions)).build(d);
      const se = searchSpec.create(Object.assign({}, searchOptions));
      const res = se.search(ev, d);
      if (ev.postProcess && !res.ranked) res.selected = ev.postProcess(res.selected);
      return { ev, se, res };
    };
    const full = run(data);
    const eo = W.optionString(evalSpec, evalOptions), so = W.optionString(searchSpec, searchOptions);
    const lines = ['=== Run information ===', '', 'Evaluator:    ' + evalSpec.path + (eo ? ' ' + eo : ''), 'Search:       ' + searchSpec.path + (so ? ' ' + so : ''),
      'Relation:     ' + data.relation, 'Instances:    ' + data.numInstances, 'Attributes:   ' + data.numAttributes, ...data.attributes.map((a) => '              ' + a.name),
      'Evaluation mode:    ' + (opts.mode === 'cv' ? (opts.folds || 10) + '-fold cross-validation' : 'evaluate on all training data'), '', '',
      '=== Attribute Selection on all input data ===', '', 'Search Method:', full.se.describe(), ''];
    const ca = data.classAttr;
    lines.push('Attribute ' + (kind === 'transform' ? 'Evaluator (unsupervised):' : `Evaluator (supervised, Class (${ca && ca.type === 'nominal' ? 'nominal' : 'numeric'}): ${data.classIndex + 1} ${ca ? ca.name : ''}):`));
    lines.push('\t' + full.ev.describe().split('\n').join('\n\t'), '');
    const names = kind === 'transform' ? full.ev.names : data.attributes.map((a) => a.name);
    let ranking = null;
    if (full.res.ranked) {
      lines.push('Ranked attributes:');
      ranking = full.res.ranked.map(([j, m]) => ({ index: j, name: names[j], merit: m }));
      for (const r of ranking) lines.push(W.dts(r.merit, 10, 4) + U.padL(r.index + 1, 4) + ' ' + r.name);
      lines.push('');
    }
    const sel = full.res.selected;
    lines.push('Selected attributes: ' + sel.map((j) => j + 1).join(',') + ' : ' + sel.length);
    if (kind !== 'transform') for (const j of sel) lines.push('                     ' + names[j]);
    let cvInfo = null;
    if (opts.mode === 'cv') {
      const folds = Math.min(opts.folds || 10, data.numInstances);
      const d = data.copy(); const rand = new W.JavaRandom(opts.seed ?? 1); d.randomize(rand);
      if (d.classIsNominal()) d.stratify(folds);
      const counts = U.zeros(data.numAttributes), merits = data.attributes.map(() => []), ranks = data.attributes.map(() => []);
      for (let f = 0; f < folds; f++) {
        if (opts.onProgress) opts.onProgress((f + 1) / folds, 'Fold ' + (f + 1));
        await W.nextFrame();
        const tr = d.trainCV(folds, f, rand);
        const r = run(tr);
        if (r.res.ranked) r.res.ranked.forEach(([j, m], k) => { merits[j].push(m); ranks[j].push(k + 1); });
        for (const j of r.res.selected) counts[j]++;
      }
      lines.push('', '', `=== Attribute selection ${folds} fold cross-validation ${d.classIsNominal() ? '(stratified), ' : ''}seed: ${opts.seed ?? 1} ===`, '');
      if (full.res.ranked && kind !== 'transform') {
        lines.push('average merit      average rank  attribute');
        const avg = (a) => U.sum(a) / a.length, sd = (a) => { const m = avg(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
        const rows = [];
        for (let j = 0; j < data.numAttributes; j++) if (merits[j].length) rows.push({ j, am: avg(merits[j]), sm: sd(merits[j]), ar: avg(ranks[j]), sr: sd(ranks[j]) });
        rows.sort((a, b) => a.ar - b.ar);
        for (const r of rows) lines.push(U.padL(U.d2s(r.am, 3), 7) + ' +- ' + U.padR(U.d2s(r.sm, 3), 7) + U.padL(U.d2s(r.ar, 1), 5) + ' +- ' + U.padR(U.d2s(r.sr, 2), 6) + U.padL(r.j + 1, 4) + ' ' + names[r.j]);
        cvInfo = { ranks: rows };
      } else {
        lines.push('number of folds (%)  attribute');
        const rows = [];
        for (let j = 0; j < data.numAttributes; j++) if (j !== data.classIndex) { rows.push({ j, count: counts[j], pct: (100 * counts[j]) / folds }); lines.push(U.padL(counts[j] + '(' + U.padL(U.d2s((100 * counts[j]) / folds, 0), 3) + ' %)', 13) + U.padL(j + 1, 8) + ' ' + names[j]); }
        cvInfo = { folds: rows, numFolds: folds };
      }
    }
    // μειωμένο σύνολο δεδομένων
    let reduced;
    if (kind === 'transform') reduced = full.ev.transform(data);
    else {
      const keep = sel.slice(); if (data.classIndex >= 0) keep.push(data.classIndex);
      const rm = new W.Remove({ indices: keep, V: true }); rm.setInput(data); reduced = rm.apply(data);
    }
    return { report: lines.join('\n') + '\n', ranking, selected: sel, names, reduced, cvInfo, merit: full.res.merit, time: (performance.now() - t0) / 1000, evalSpec, searchSpec, evalOptions, searchOptions, eo, so };
  };
})();
