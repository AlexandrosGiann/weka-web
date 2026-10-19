/* Weka Web — μητρώο ταξινομητών και βασικοί αλγόριθμοι (rules, bayes, lazy, stump). */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, S = W.Stats;

  W.classifiers = [];
  W.registerClassifier = (spec) => { W.classifiers.push(spec); return spec; };
  W.findClassifier = (name) => W.classifiers.find((s) => s.name === name);

  // -------------------------------------------- Cephes normal CDF (όπως weka.core.Statistics)
  function polevl(x, c, n) { let a = c[0]; for (let i = 1; i <= n; i++) a = a * x + c[i]; return a; }
  function p1evl(x, c, n) { let a = x + c[0]; for (let i = 1; i < n; i++) a = a * x + c[i]; return a; }
  const T = [9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3, 7.00332514112805075473E3, 5.55923013010394962768E4];
  const Uu = [3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3, 2.26290000613890934246E4, 4.92673942608635921086E4];
  const P = [2.46196981473530512524E-10, 5.64189564831068821977E-1, 7.46321056442269912687E0, 4.86371970985681366614E1, 1.96520832956077098242E2, 5.26445194995477358631E2, 9.34528527171957607540E2, 1.02755188689515710272E3, 5.57535335369399327526E2];
  const Q = [1.32281951154744992508E1, 8.67072140885989742329E1, 3.54937778887819891062E2, 9.75708501743205489753E2, 1.82390916687909736289E3, 2.24633760818710981792E3, 1.65666309194161350182E3, 5.57535340817727675546E2];
  const R = [5.64189583547755073984E-1, 1.27536670759978104416E0, 5.01905042251180477414E0, 6.16021097993053585195E0, 7.40974269950448939160E0, 2.97886665372100240670E0];
  const Ss = [2.26052863220117276590E0, 9.39603524938001434673E0, 1.20489539808096656605E1, 1.70814450747565897222E1, 9.60896809063285878198E0, 3.36907645100081516050E0];
  const MAXLOG = 7.09782712893383996732E2, SQRTH = 7.07106781186547524401E-1;
  function erf(x) {
    if (Math.abs(x) > 1) return 1 - erfc(x);
    const z = x * x;
    return x * polevl(z, T, 4) / p1evl(z, Uu, 5);
  }
  function erfc(a) {
    const x = Math.abs(a);
    if (x < 1) return 1 - erf(a);
    let z = -a * a;
    if (z < -MAXLOG) return a < 0 ? 2 : 0;
    z = Math.exp(z);
    let p, q;
    if (x < 8) { p = polevl(x, P, 8); q = p1evl(x, Q, 8); } else { p = polevl(x, R, 5); q = p1evl(x, Ss, 6); }
    let y = (z * p) / q;
    if (a < 0) y = 2 - y;
    if (y === 0) return a < 0 ? 2 : 0;
    return y;
  }
  S.normalProbability = function (a) {
    const x = a * SQRTH, z = Math.abs(x);
    let y;
    if (z < SQRTH) y = 0.5 + 0.5 * erf(x);
    else { y = 0.5 * erfc(z); if (x > 0) y = 1 - y; }
    return y;
  };
  S.erf = erf;
  /** Math.rint της Java (στρογγυλοποίηση στο πλησιέστερο άρτιο). */
  const rint = (x) => { const r = Math.round(x); return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r; };
  U.rint = rint;

  // ---------------------------------------------------------------- ZeroR
  class ZeroR {
    constructor() {}
    build(data) {
      this.header = data;
      const ci = data.classIndex;
      if (data.classIsNominal()) {
        const c = new Array(data.numClasses).fill(1);
        let tot = data.numClasses;
        for (let i = 0; i < data.numInstances; i++) { const v = data.rows[i][ci]; if (v === v) { c[v] += data.weights[i]; tot += data.weights[i]; } }
        this.counts = c.map((x) => x / tot);
        this.classValue = U.maxIndex(c);
      } else {
        let s = 0, w = 0;
        for (let i = 0; i < data.numInstances; i++) { const v = data.rows[i][ci]; if (v === v) { s += v * data.weights[i]; w += data.weights[i]; } }
        this.classValue = w > 0 ? s / w : 0;
      }
      return this;
    }
    distributionForRow() { return this.header.classIsNominal() ? this.counts.slice() : [this.classValue]; }
    toString() {
      const ca = this.header.classAttr;
      if (ca.type === 'nominal') return 'ZeroR predicts class value: ' + ca.values[this.classValue];
      return 'ZeroR predicts class value: ' + this.classValue;
    }
  }
  W.ZeroR = ZeroR;
  W.registerClassifier({
    name: 'ZeroR', group: 'rules', path: 'weka.classifiers.rules.ZeroR',
    desc: L`Προβλέπει πάντα την πλειοψηφούσα κλάση (ή τον μέσο όρο). Χρήσιμο ως baseline.`,
    options: [], create: () => new ZeroR(),
  });

  // ----------------------------------------------------------------- OneR
  class OneR {
    constructor(o = {}) { this.B = o.B ?? 6; }
    build(data) {
      data = data.withoutMissingClass();
      this.header = data;
      this.total = data.numInstances;
      let best = null;
      for (let j = 0; j < data.numAttributes; j++) {
        if (j === data.classIndex) continue;
        let r;
        try { r = this.newRule(j, data); } catch (e) { continue; }
        if (!best || r.correct > best.correct) best = r;
      }
      if (!best) { this.zeroR = new ZeroR().build(data); }
      this.rule = best;
      return this;
    }
    newRule(att, data) {
      const nc = data.numClasses, ci = data.classIndex;
      const missing = U.zeros(nc);
      let r;
      if (data.isNominal(att)) {
        const nv = data.attributes[att].values.length;
        const counts = U.matrix(nv, nc);
        for (const row of data.rows) {
          if (row[att] !== row[att]) missing[row[ci]]++; else counts[row[att]][row[ci]]++;
        }
        r = { att, nominal: true, classifications: [], correct: 0 };
        for (let v = 0; v < nv; v++) { const b = U.maxIndex(counts[v]); r.classifications[v] = b; r.correct += counts[v][b]; }
      } else {
        const d = data.copy(); d.sortBy(att);
        let last = d.numInstances;
        while (last > 0 && d.rows[last - 1][att] !== d.rows[last - 1][att]) { last--; missing[d.rows[last][ci]]++; }
        if (last === 0) throw new Error('only missing');
        let dists = [], vals = [], dist = null, lastValue = 0;
        for (let i = 0; i < last; i++) {
          const v = d.rows[i][att];
          if (i === 0 || v > lastValue) {
            if (i !== 0) vals.push((lastValue + v) / 2);
            lastValue = v;
            dist = U.zeros(nc); dists.push(dist);
          }
          dist[d.rows[i][ci]]++;
        }
        vals.push(Number.MAX_VALUE);
        // πρώτο πέρασμα: συγχώνευση με minBucketSize
        let nd = [dists[0]], nv = [vals[0]];
        for (let k = 1; k < dists.length; k++) {
          const old = nd[nd.length - 1], cur = dists[k];
          if (U.maxIndex(cur) === U.maxIndex(old) || old[U.maxIndex(old)] < this.B) {
            for (let j = 0; j < nc; j++) cur[j] += old[j];
            nd[nd.length - 1] = cur; nv[nv.length - 1] = vals[k];
          } else { nd.push(cur); nv.push(vals[k]); }
        }
        // δεύτερο πέρασμα: συγχώνευση γειτονικών με ίδια κλάση
        let correct = 0;
        const fd = [], fv = [];
        for (let k = 0; k < nd.length; k++) {
          const cur = nd[k];
          correct += cur[U.maxIndex(cur)];
          if (fd.length && U.maxIndex(cur) === U.maxIndex(fd[fd.length - 1])) {
            const old = fd[fd.length - 1];
            for (let j = 0; j < nc; j++) cur[j] += old[j];
            fd[fd.length - 1] = cur; fv[fv.length - 1] = nv[k];
          } else { fd.push(cur); fv.push(nv[k]); }
        }
        r = { att, nominal: false, classifications: fd.map((x) => U.maxIndex(x)), breakpoints: fv.slice(0, -1), correct };
      }
      r.missingClass = U.maxIndex(missing);
      if (missing[r.missingClass] === 0) r.missingClass = -1; else r.correct += missing[r.missingClass];
      return r;
    }
    distributionForRow(row) {
      if (this.zeroR) return this.zeroR.distributionForRow(row);
      const r = this.rule, nc = this.header.numClasses;
      const out = U.zeros(nc);
      const v = row[r.att];
      let c;
      if (v !== v) c = r.missingClass >= 0 ? r.missingClass : -1;
      else if (r.nominal) c = r.classifications[v];
      else { let k = 0; while (k < r.breakpoints.length && v >= r.breakpoints[k]) k++; c = r.classifications[k]; }
      if (c >= 0) out[c] = 1;
      return out;
    }
    toString() {
      if (this.zeroR) return this.zeroR.toString();
      const r = this.rule, h = this.header, a = h.attributes[r.att], cv = h.classAttr.values;
      const t = [a.name + ':'];
      for (let v = 0; v < r.classifications.length; v++) {
        let lhs;
        if (r.nominal) lhs = a.values[v];
        else if (v < r.breakpoints.length) lhs = '< ' + javaDouble(r.breakpoints[v]);
        else if (v > 0) lhs = '>= ' + javaDouble(r.breakpoints[v - 1]);
        else lhs = 'not ?';
        t.push('\t' + lhs + '\t-> ' + cv[r.classifications[v]]);
      }
      if (r.missingClass !== -1) t.push('\t?\t-> ' + cv[r.missingClass]);
      t.push('(' + r.correct + '/' + this.total + ' instances correct)');
      return t.join('\n') + '\n';
    }
  }
  /** Double.toString της Java (κατά προσέγγιση: ακέραιοι με ".0"). */
  function javaDouble(v) {
    if (Number.isInteger(v)) return v.toFixed(1);
    const s = String(v);
    return s.includes('e') ? s.replace('e', 'E').replace('E+', 'E') : s;
  }
  W.javaDouble = javaDouble;
  W.registerClassifier({
    name: 'OneR', group: 'rules', path: 'weka.classifiers.rules.OneR',
    desc: L`Κανόνας ενός χαρακτηριστικού: διαλέγει το χαρακτηριστικό με το μικρότερο σφάλμα.`,
    caps: { numericClass: false },
    options: [{ key: 'B', flag: 'B', label: L`Ελάχιστο μέγεθος κάδου`, type: 'int', default: 6, min: 1 }],
    create: (o) => new OneR(o),
  });

  // ------------------------------------------------------------ NaiveBayes
  class NormalEstimator {
    constructor(precision) { this.prec = precision; this.sw = 0; this.sv = 0; this.svv = 0; this.mean = 0; this.sd = precision / 6; }
    round(x) { return rint(x / this.prec) * this.prec; }
    add(x, w) {
      if (w === 0) return;
      x = this.round(x);
      this.sw += w; this.sv += x * w; this.svv += x * x * w;
      if (this.sw > 0) {
        this.mean = this.sv / this.sw;
        const sd = Math.sqrt(Math.abs(this.svv - this.mean * this.sv) / this.sw);
        if (sd > 1e-10) this.sd = Math.max(this.prec / 6, sd);
      }
    }
    prob(x) {
      x = this.round(x);
      const zl = (x - this.mean - this.prec / 2) / this.sd, zu = (x - this.mean + this.prec / 2) / this.sd;
      return (S.normalProbability(zu) - S.normalProbability(zl)) / this.prec;
    }
  }
  class KernelEstimator {
    // Απλοποιημένη έκδοση του weka KernelEstimator (Gaussian kernels με bandwidth 1/sqrt(n) του εύρους)
    constructor(precision) { this.prec = precision; this.vals = []; this.ws = []; this.sw = 0; }
    round(x) { return rint(x / this.prec) * this.prec; }
    add(x, w) { if (w === 0) return; this.vals.push(this.round(x)); this.ws.push(w); this.sw += w; }
    finish() {
      const n = this.vals.length;
      if (!n) { this.sd = this.prec / 6; return; }
      const mn = Math.min(...this.vals), mx = Math.max(...this.vals);
      this.sd = Math.max(this.prec / 6, (mx - mn) / Math.sqrt(this.sw));
    }
    prob(x) {
      if (!this.vals.length) return 0;
      if (this.sd === undefined) this.finish();
      x = this.round(x);
      let p = 0;
      for (let i = 0; i < this.vals.length; i++) {
        const zl = (x - this.vals[i] - this.prec / 2) / this.sd, zu = (x - this.vals[i] + this.prec / 2) / this.sd;
        p += this.ws[i] * (S.normalProbability(zu) - S.normalProbability(zl));
      }
      return p / this.sw / this.prec;
    }
  }
  class DiscreteEstimator {
    constructor(n, laplace = true) { this.counts = new Array(n).fill(laplace ? 1 : 0); this.sum = laplace ? n : 0; }
    add(v, w) { this.counts[v] += w; this.sum += w; }
    prob(v) { return this.sum === 0 ? 0 : this.counts[v] / this.sum; }
  }
  W.NormalEstimator = NormalEstimator; W.DiscreteEstimator = DiscreteEstimator;

  class NaiveBayes {
    constructor(o = {}) { this.K = !!o.K; this.D = !!o.D; }
    build(data) {
      data = data.withoutMissingClass();
      if (this.D) {
        this.disc = W.filters.find((f) => f.name === 'Discretize (supervised)').create({}).setInput(data);
        data = this.disc.apply(data);
      }
      this.header = data;
      const ci = data.classIndex, nc = data.numClasses;
      this.classEst = new DiscreteEstimator(nc, true);
      this.est = [];
      for (let j = 0; j < data.numAttributes; j++) {
        if (j === ci) { this.est.push(null); continue; }
        const a = data.attributes[j];
        if (a.type === 'nominal') { this.est.push(Array.from({ length: nc }, () => new DiscreteEstimator(a.values.length, true))); continue; }
        let prec = 0.01;
        const vals = [];
        for (const r of data.rows) if (r[j] === r[j]) vals.push(r[j]);
        vals.sort((x, y) => x - y);
        if (vals.length) {
          let last = vals[0], delta = 0, distinct = 0;
          for (let i = 1; i < vals.length; i++) if (vals[i] !== last) { delta += vals[i] - last; last = vals[i]; distinct++; }
          if (distinct > 0) prec = delta / distinct;
        }
        this.est.push(Array.from({ length: nc }, () => (this.K ? new KernelEstimator(prec) : new NormalEstimator(prec))));
      }
      for (let i = 0; i < data.numInstances; i++) {
        const r = data.rows[i], w = data.weights[i], c = r[ci];
        for (let j = 0; j < data.numAttributes; j++) {
          if (j === ci || r[j] !== r[j]) continue;
          this.est[j][c].add(r[j], w);
        }
        this.classEst.add(c, w);
      }
      return this;
    }
    distributionForRow(row) {
      if (this.disc) row = this.disc.convertRow(row);
      const h = this.header, nc = h.numClasses;
      const probs = [];
      for (let c = 0; c < nc; c++) probs.push(this.classEst.prob(c));
      for (let j = 0; j < h.numAttributes; j++) {
        if (j === h.classIndex || row[j] !== row[j]) continue;
        let max = 0;
        for (let c = 0; c < nc; c++) {
          probs[c] *= Math.max(1e-75, this.est[j][c].prob(row[j]));
          if (probs[c] > max) max = probs[c];
        }
        if (max > 0 && max < 1e-75) for (let c = 0; c < nc; c++) probs[c] *= 1e75;
      }
      return U.normalize(probs);
    }
    toString() {
      const h = this.header, nc = h.numClasses, cv = h.classAttr.values;
      const lines = ['Naive Bayes Classifier', ''];
      // πίνακας σε στήλες όπως στο Weka
      const colW = Math.max(...cv.map((s) => s.length), 6) + 2;
      const prior = cv.map((_, c) => U.d2s(this.classEst.prob(c), 2));
      let nameW = 'Attribute'.length;
      for (let j = 0; j < h.numAttributes; j++) if (j !== h.classIndex) {
        nameW = Math.max(nameW, h.attributes[j].name.length);
        if (h.attributes[j].type === 'nominal') for (const v of h.attributes[j].values) nameW = Math.max(nameW, v.length + 2);
      }
      nameW = Math.max(nameW, '  [total]'.length, '  precision'.length) + 1;
      const row = (label, cells) => U.padR(label, nameW) + cells.map((c) => U.padL(c, colW)).join('');
      lines.push(row('', cv.map(() => 'Class').slice(0, 0).concat(cv.map((s) => s))));
      lines.push(row('Attribute', prior.map((p) => '(' + p + ')')));
      lines.push('='.repeat(nameW + colW * nc));
      for (let j = 0; j < h.numAttributes; j++) {
        if (j === h.classIndex) continue;
        const a = h.attributes[j];
        lines.push(a.name);
        if (a.type === 'nominal') {
          a.values.forEach((v, k) => lines.push(row('  ' + v, this.est[j].map((e) => U.d2s(e.counts[k], 1)))));
          lines.push(row('  [total]', this.est[j].map((e) => U.d2s(e.sum, 1))));
        } else if (this.K) {
          lines.push(row('  [# kernels]', this.est[j].map((e) => String(e.vals.length))));
          lines.push(row('  [std. dev]', this.est[j].map((e) => { if (e.sd === undefined) e.finish(); return U.d2s(e.sd, 4); })));
          lines.push(row('  [precision]', this.est[j].map((e) => U.d2s(e.prec, 4))));
        } else {
          lines.push(row('  mean', this.est[j].map((e) => U.d2s(e.mean, 4))));
          lines.push(row('  std. dev.', this.est[j].map((e) => U.d2s(e.sd, 4))));
          lines.push(row('  weight sum', this.est[j].map((e) => U.d2s(e.sw, 4))));
          lines.push(row('  precision', this.est[j].map((e) => U.d2s(e.prec, 4))));
        }
        lines.push('');
      }
      return lines.join('\n');
    }
  }
  W.registerClassifier({
    name: 'NaiveBayes', group: 'bayes', path: 'weka.classifiers.bayes.NaiveBayes',
    desc: L`Πιθανοτικός ταξινομητής Bayes με υπόθεση ανεξαρτησίας. Κανονική κατανομή για αριθμητικά χαρακτηριστικά.`,
    caps: { numericClass: false },
    options: [
      { key: 'K', flag: 'K', label: 'Kernel density estimator', type: 'bool', default: false },
      { key: 'D', flag: 'D', label: 'Supervised discretization', type: 'bool', default: false },
    ],
    create: (o) => new NaiveBayes(o),
  });

  // -------------------------------------------------- NaiveBayesMultinomial
  class NaiveBayesMultinomial {
    build(data) {
      data = data.withoutMissingClass();
      this.header = data;
      const ci = data.classIndex, nc = data.numClasses, na = data.numAttributes;
      this.prob = U.matrix(nc, na, 1);
      const wordsPerClass = new Array(nc).fill(na - 1);
      this.prior = new Array(nc).fill(1);
      let priorSum = nc;
      for (let i = 0; i < data.numInstances; i++) {
        const r = data.rows[i], c = r[ci], w = data.weights[i];
        this.prior[c] += w; priorSum += w;
        for (let j = 0; j < na; j++) {
          if (j === ci || r[j] !== r[j]) continue;
          const nw = r[j] * w;
          if (nw < 0) throw new Error(L`Το NaiveBayesMultinomial απαιτεί μη αρνητικές τιμές (συχνότητες λέξεων).`);
          this.prob[c][j] += nw; wordsPerClass[c] += nw;
        }
      }
      for (let c = 0; c < nc; c++) {
        for (let j = 0; j < na; j++) if (j !== ci) this.prob[c][j] = Math.log(this.prob[c][j] / wordsPerClass[c]);
        this.prior[c] = Math.log(this.prior[c] / priorSum);
      }
      return this;
    }
    distributionForRow(row) {
      const h = this.header, nc = h.numClasses, ci = h.classIndex;
      const lp = this.prior.slice();
      for (let c = 0; c < nc; c++) for (let j = 0; j < h.numAttributes; j++) {
        if (j === ci || row[j] !== row[j] || row[j] === 0) continue;
        lp[c] += row[j] * this.prob[c][j];
      }
      const mx = Math.max(...lp);
      return U.normalize(lp.map((v) => Math.exp(v - mx)));
    }
    toString() {
      const h = this.header, cv = h.classAttr.values;
      const t = ['The independent probability of a class', '--------------------------------------'];
      cv.forEach((v, c) => t.push(v + '\t' + U.d2s(Math.exp(this.prior[c]), 4)));
      t.push('', 'The probability of a word given the class', '-----------------------------------------');
      t.push('\t' + cv.join('\t'));
      for (let j = 0; j < h.numAttributes; j++) if (j !== h.classIndex) t.push(h.attributes[j].name + '\t' + cv.map((_, c) => U.d2s(Math.exp(this.prob[c][j]), 4)).join('\t'));
      return t.join('\n');
    }
  }
  W.registerClassifier({
    name: 'NaiveBayesMultinomial', group: 'bayes', path: 'weka.classifiers.bayes.NaiveBayesMultinomial',
    desc: L`Πολυωνυμικό Naive Bayes για δεδομένα συχνοτήτων (π.χ. κείμενα).`,
    caps: { numericClass: false, nominalAttrs: false },
    options: [], create: () => new NaiveBayesMultinomial(),
  });

  // -------------------------------------------------------- DecisionStump
  class DecisionStump {
    build(data) {
      data = data.withoutMissingClass();
      this.header = data;
      const ci = data.classIndex, nominal = data.classIsNominal();
      const nc = nominal ? data.numClasses : 1;
      let best = null;
      // συνολική κατανομή
      const overall = this.dist(data, () => 0, 1)[0];
      for (let j = 0; j < data.numAttributes; j++) {
        if (j === ci) continue;
        const cands = [];
        if (data.isNominal(j)) {
          const nv = data.attributes[j].values.length;
          for (let v = 0; v < nv; v++) cands.push({ att: j, value: v, nominal: true });
        } else {
          const vals = [...new Set(data.rows.map((r) => r[j]).filter((x) => x === x))].sort((a, b) => a - b);
          for (let k = 1; k < vals.length; k++) cands.push({ att: j, split: (vals[k - 1] + vals[k]) / 2, nominal: false });
        }
        for (const c of cands) {
          const branch = (r) => {
            const v = r[c.att];
            if (v !== v) return 2;
            return c.nominal ? (v === c.value ? 0 : 1) : (v <= c.split ? 0 : 1);
          };
          const d = this.dist(data, branch, 3);
          const score = nominal ? S.entropyConditionedOnRows(d.map((x) => x.counts)) : this.variance(d);
          if (!best || score < best.score - 1e-12) best = Object.assign({ score, d }, c);
        }
      }
      if (!best) { this.zero = new ZeroR().build(data); return this; }
      this.best = best;
      // κατανομές ανά κλάδο (άδειος κλάδος → συνολική)
      this.out = best.d.map((x) => {
        if (nominal) {
          const tot = U.sum(x.counts);
          return tot > 0 ? x.counts.map((v) => v / tot) : overall.counts.map((v) => v / U.sum(overall.counts));
        }
        return x.w > 0 ? [x.sum / x.w] : [overall.sum / overall.w];
      });
      return this;
    }
    dist(data, branch, nb) {
      const ci = data.classIndex, nominal = data.classIsNominal(), nc = nominal ? data.numClasses : 1;
      const out = Array.from({ length: nb }, () => ({ counts: U.zeros(nc), sum: 0, sq: 0, w: 0 }));
      for (let i = 0; i < data.numInstances; i++) {
        const r = data.rows[i], b = out[branch(r)], w = data.weights[i], c = r[ci];
        if (nominal) b.counts[c] += w; else { b.sum += w * c; b.sq += w * c * c; }
        b.w += w;
      }
      return out;
    }
    variance(d) { let v = 0; for (const b of d) if (b.w > 0) v += b.sq - (b.sum * b.sum) / b.w; return v; }
    distributionForRow(row) {
      if (this.zero) return this.zero.distributionForRow(row);
      const b = this.best, v = row[b.att];
      const k = v !== v ? 2 : b.nominal ? (v === b.value ? 0 : 1) : v <= b.split ? 0 : 1;
      return this.out[k].slice();
    }
    toString() {
      if (this.zero) return this.zero.toString();
      const h = this.header, b = this.best, a = h.attributes[b.att];
      const nominal = h.classIsNominal();
      const lhs = [b.nominal ? `${a.name} = ${a.values[b.value]}` : `${a.name} <= ${U.d2s(b.split, 6)}`,
        b.nominal ? `${a.name} != ${a.values[b.value]}` : `${a.name} > ${U.d2s(b.split, 6)}`, `${a.name} is missing`];
      const pred = (k) => (nominal ? h.classAttr.values[U.maxIndex(this.out[k])] : U.d2s(this.out[k][0], 6));
      const t = ['Decision Stump', '', 'Classifications', ''];
      for (let k = 0; k < 3; k++) t.push(lhs[k] + ' : ' + pred(k));
      if (nominal) {
        t.push('', 'Class distributions', '');
        for (let k = 0; k < 3; k++) { t.push(lhs[k]); t.push(h.classAttr.values.join('\t')); t.push(this.out[k].map((p) => U.d2s(p, 6)).join('\t')); }
      }
      return t.join('\n');
    }
  }
  W.DecisionStump = DecisionStump;
  W.registerClassifier({
    name: 'DecisionStump', group: 'trees', path: 'weka.classifiers.trees.DecisionStump',
    desc: L`Δέντρο ενός επιπέδου (μία διάσπαση). Συνήθης βάση για boosting.`,
    options: [], create: () => new DecisionStump(),
  });

  // ----------------------------------------------------------------- IBk
  /** Ευκλείδεια απόσταση με κανονικοποίηση όπως weka.core.EuclideanDistance. */
  class EuclideanDistance {
    constructor(data, skip) {
      this.skip = skip; this.atts = data.attributes;
      const na = data.numAttributes;
      this.min = new Array(na).fill(NaN); this.max = new Array(na).fill(NaN);
      for (const r of data.rows) this.update(r);
    }
    update(r) {
      for (let j = 0; j < r.length; j++) {
        const v = r[j];
        if (v !== v || this.atts[j].type === 'nominal') continue;
        if (this.min[j] !== this.min[j]) { this.min[j] = this.max[j] = v; }
        else { if (v < this.min[j]) this.min[j] = v; if (v > this.max[j]) this.max[j] = v; }
      }
    }
    norm(j, v) {
      const mn = this.min[j], w = this.max[j] - mn;
      if (w === 0 || w !== w) return 0;
      return (v - mn) / w;
    }
    diff(j, a, b) {
      if (this.atts[j].type === 'nominal') return a !== a || b !== b || a !== b ? 1 : 0;
      const am = a !== a, bm = b !== b;
      if (am && bm) return 1;
      if (am || bm) { let d = am ? this.norm(j, b) : this.norm(j, a); d = d < 0.5 ? 1 - d : d; return d; }
      return this.norm(j, a) - this.norm(j, b);
    }
    distance(x, y) {
      let s = 0;
      for (let j = 0; j < x.length; j++) { if (j === this.skip) continue; const d = this.diff(j, x[j], y[j]); s += d * d; }
      return Math.sqrt(s);
    }
  }
  W.EuclideanDistance = EuclideanDistance;

  class IBk {
    constructor(o = {}) { this.K = o.K ?? 1; this.Wt = o.weighting || 'none'; this.X = !!o.X; this.window = 0; }
    build(data) {
      data = data.withoutMissingClass();
      this.train = data;
      this.dist = new EuclideanDistance(data, data.classIndex);
      this.k = this.K;
      if (this.X && this.K > 1) this.crossValidateK();
      return this;
    }
    neighbours(row, k, exclude = -1) {
      const t = this.train, ds = [];
      for (let i = 0; i < t.numInstances; i++) if (i !== exclude) ds.push([this.dist.distance(row, t.rows[i]), i]);
      ds.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      let n = Math.min(k, ds.length);
      while (n < ds.length && n > 0 && ds[n][0] === ds[n - 1][0]) n++; // ισοβαθμίες στην k-οστή θέση (όπως LinearNNSearch)
      return ds.slice(0, n);
    }
    makeDist(neigh) {
      const t = this.train, nominal = t.classIsNominal(), ci = t.classIndex;
      const nc = nominal ? t.numClasses : 1;
      const dist = U.zeros(nc);
      let total = 0;
      if (nominal) { for (let c = 0; c < nc; c++) dist[c] = 1 / Math.max(1, t.numInstances); total = nc / Math.max(1, t.numInstances); }
      for (const [d, i] of neigh) {
        let w = this.Wt === 'inverse' ? 1 / (d + 0.001) : this.Wt === 'similarity' ? 1 - d : 1;
        w *= t.weights[i];
        if (nominal) dist[t.rows[i][ci]] += w; else dist[0] += t.rows[i][ci] * w;
        total += w;
      }
      if (total > 0) for (let c = 0; c < nc; c++) dist[c] /= total;
      return dist;
    }
    crossValidateK() {
      const t = this.train, nominal = t.classIsNominal(), ci = t.classIndex;
      const err = U.zeros(this.K);
      for (let i = 0; i < t.numInstances; i++) {
        const nb = this.neighbours(t.rows[i], this.K, i);
        for (let k = 1; k <= this.K; k++) {
          let n = Math.min(k, nb.length);
          while (n < nb.length && n > 0 && nb[n][0] === nb[n - 1][0]) n++;
          const d = this.makeDist(nb.slice(0, n));
          if (nominal) { if (U.maxIndex(d) !== t.rows[i][ci]) err[k - 1] += t.weights[i]; }
          else { const e = d[0] - t.rows[i][ci]; err[k - 1] += e * e * t.weights[i]; }
        }
      }
      this.k = U.minIndex(err) + 1;
      this.cvErrors = err;
    }
    distributionForRow(row) {
      this.dist.update(row); // όπως το Weka ενημερώνει τα εύρη με το νέο στιγμιότυπο
      if (this.train.numInstances === 0) return U.zeros(this.train.classIsNominal() ? this.train.numClasses : 1);
      return this.makeDist(this.neighbours(row, this.k));
    }
    toString() {
      let s = 'IB1 instance-based classifier\nusing ' + this.k + ' nearest neighbour(s) for classification\n';
      if (this.k > 1 || this.Wt !== 'none') s = `IBk instance-based classifier\nusing ${this.k} ${this.Wt === 'inverse' ? 'inverse-distance-weighted ' : this.Wt === 'similarity' ? 'similarity-weighted ' : ''}nearest neighbour(s) for classification\n`;
      if (this.cvErrors) s += L`
Επιλογή k με leave-one-out: k = ` + this.k + '\n';
      return s;
    }
  }
  W.IBk = IBk;
  W.registerClassifier({
    name: 'IBk', group: 'lazy', path: 'weka.classifiers.lazy.IBk',
    desc: L`k-πλησιέστεροι γείτονες με κανονικοποιημένη Ευκλείδεια απόσταση.`,
    options: [
      { key: 'K', flag: 'K', label: L`Πλήθος γειτόνων (k)`, type: 'int', default: 1, min: 1 },
      { key: 'weighting', label: L`Στάθμιση απόστασης`, type: 'select', default: 'none', choices: { none: L`Καμία`, inverse: L`1/απόσταση (-I)`, similarity: L`1-απόσταση (-F)` }, cliFlags: { inverse: 'I', similarity: 'F' } },
      { key: 'X', flag: 'X', label: L`Επιλογή k με leave-one-out (έως K)`, type: 'bool', default: false },
    ],
    create: (o) => new IBk(o),
  });

  W.ZeroRClass = ZeroR;
})();
