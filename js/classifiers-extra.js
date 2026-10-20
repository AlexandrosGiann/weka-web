/* Weka Web — extra classifiers: BayesNet, KStar, LogitBoost, SGD, VotedPerceptron. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, dts = (v, w, d) => W.dts(v, w, d);

  // ================================================================ BayesNet
  /** Bayesian network: supervised MDL discretisation, K2 structure search (Bayes score), SimpleEstimator CPTs. */
  class BayesNet {
    constructor(o = {}) { this.P = o.P ?? 1; this.A = o.A ?? 0.5; this.N = o.N ?? true; }
    build(data) {
      data = data.withoutMissingClass();
      const fl = [];
      if (data.attributes.some((a, j) => j !== data.classIndex && a.type !== 'nominal')) fl.push(W.findFilter('Discretize (supervised)').create({}));
      fl.push(new W.ReplaceMissingValues());
      this.pipe = new W.Pipeline(fl);
      const d = this.pipe.fit(data);
      this.header = d.emptyCopy();
      const na = d.numAttributes, ci = d.classIndex, card = d.attributes.map((a) => a.values.length);
      this.card = card;
      // parent sets: class is a parent of every attribute when initialised as naive Bayes
      const parents = d.attributes.map((a, j) => (j !== ci && this.N ? [ci] : []));
      const alpha = this.A, lg = W.lnGamma;
      const nodeScore = (node, ps) => {
        let nCard = 1; for (const p of ps) nCard *= card[p];
        const nv = card[node], counts = new Int32Array(nCard * nv);
        for (const r of d.rows) {
          let k = 0; for (const p of ps) k = k * card[p] + r[p];
          counts[nv * k + r[node]]++;
        }
        let s = 0;
        for (let j = 0; j < nCard; j++) {
          let sum = 0;
          for (let v = 0; v < nv; v++) { const c = alpha + counts[j * nv + v]; if (c !== 0) { s += lg(c); sum += c; } }
          if (sum !== 0) s -= lg(sum);
          if (alpha !== 0) { s -= nv * lg(alpha); s += lg(nv * alpha); }
        }
        return s;
      };
      // K2: class first, then the attributes in their order
      const order = [ci]; for (let j = 0; j < na; j++) if (j !== ci) order.push(j);
      for (let io = 1; io < na; io++) {
        const att = order[io];
        let best = nodeScore(att, parents[att]);
        let progress = parents[att].length < this.P;
        while (progress) {
          let bestAtt = -1;
          for (let io2 = 0; io2 < io; io2++) {
            const a2 = order[io2];
            if (parents[att].includes(a2)) continue;
            const sc = nodeScore(att, parents[att].concat([a2]));
            if (sc > best) { best = sc; bestAtt = a2; }
          }
          if (bestAtt >= 0) { parents[att].push(bestAtt); progress = parents[att].length < this.P; } else progress = false;
        }
      }
      this.parents = parents;
      // CPTs (DiscreteEstimatorBayes: counts start at alpha)
      this.cpt = d.attributes.map((a, j) => {
        let nCard = 1; for (const p of parents[j]) nCard *= card[p];
        return Array.from({ length: nCard }, () => new Array(card[j]).fill(alpha));
      });
      for (let i = 0; i < d.rows.length; i++) {
        const r = d.rows[i], w = d.weights[i];
        for (let j = 0; j < na; j++) this.cpt[j][this.cfg(j, r, r[ci])][r[j]] += w;
      }
      this.cpt = this.cpt.map((t) => t.map((row) => { const s = U.sum(row); return row.map((c) => c / s); }));
      return this;
    }
    cfg(j, r, cls) {
      let k = 0; const ci = this.header.classIndex;
      for (const p of this.parents[j]) k = k * this.card[p] + (p === ci ? cls : r[p]);
      return k;
    }
    distributionForRow(row) {
      const r = this.pipe.row(row), h = this.header, ci = h.classIndex, nc = h.numClasses, lp = U.zeros(nc);
      for (let c = 0; c < nc; c++) {
        let s = 0;
        for (let j = 0; j < h.numAttributes; j++) s += Math.log(this.cpt[j][this.cfg(j, r, c)][j === ci ? c : r[j]]);
        lp[c] = s;
      }
      const mx = Math.max(...lp);
      return U.normalize(lp.map((v) => Math.exp(v - mx)));
    }
    toString() {
      const h = this.header, t = ['Bayes Network Classifier', 'not using ADTree', `#attributes=${h.numAttributes} #classindex=${h.classIndex}`, 'Network structure (nodes followed by parents)'];
      h.attributes.forEach((a, j) => t.push(`${a.name}(${a.values.length}): ${this.parents[j].map((p) => h.attributes[p].name).join(' ')} `));
      t.push('', L`Conditional probability tables`, '');
      h.attributes.forEach((a, j) => {
        const ps = this.parents[j];
        t.push(`${a.name}` + (ps.length ? ' | ' + ps.map((p) => h.attributes[p].name).join(', ') : ''));
        this.cpt[j].forEach((row, k) => {
          const lab = []; let rem = k;
          for (let q = ps.length - 1; q >= 0; q--) { const c = this.card[ps[q]]; lab.unshift(h.attributes[ps[q]].values[rem % c]); rem = Math.floor(rem / c); }
          t.push('  ' + (lab.length ? lab.join(', ') + ': ' : '') + row.map((p, v) => `${a.values[v]}=${U.d2s(p, 3)}`).join('  '));
        });
      });
      return t.join('\n') + '\n';
    }
  }
  W.registerClassifier({
    name: 'BayesNet', group: 'bayes', path: 'weka.classifiers.bayes.BayesNet',
    desc: L`Bayesian network: learns which attributes depend on each other (K2 search) and estimates conditional probability tables. Numeric attributes are discretised (MDL).`,
    caps: { numericClass: false },
    options: [
      { key: 'P', label: L`Maximum number of parents (K2)`, type: 'int', default: 1, min: 1, max: 10, cli: null },
      { key: 'A', label: L`Estimator alpha (prior counts)`, type: 'number', default: 0.5, min: 0, step: 0.1, cli: null },
      { key: 'N', label: L`Start from naive Bayes structure`, type: 'bool', default: true, cli: null },
    ],
    create: (o) => new BayesNet(o),
  });

  // ================================================================== KStar
  const KS_ACC = 0.01, KS_MAXIT = 40;
  /** K*: instance-based learner with an entropic (transformation-probability) distance. */
  class KStar {
    constructor(o = {}) { this.B = o.B ?? 20; this.M = o.M || 'a'; }
    build(data) {
      data = data.withoutMissingClass();
      this.train = data;
      this.header = data.emptyCopy();
      this.cache = data.attributes.map(() => new Map());
      this.dist = data.attributes.map((a, j) => {
        if (a.type !== 'nominal' || j === data.classIndex) return null;
        const c = new Array(a.values.length).fill(0); let tot = 0;
        for (const r of data.rows) { const v = r[j]; if (v === v) { c[v]++; tot++; } }
        return { c, tot };
      });
      return this;
    }
    missingProb(minP, avgP) { return this.M === 'd' ? 0 : this.M === 'n' ? 1 : this.M === 'm' ? minP : avgP; }
    nominalParams(j, tv) {
      const { c: distr, tot } = this.dist[j], nv = distr.length;
      const sphere = (stop) => {
        let tval = 0, t1 = 0, minp = 1;
        for (let i = 0; i < nv; i++) {
          const cnt = distr[i]; if (!cnt) continue;
          const tp = (i === tv ? stop + (1 - stop) / nv : (1 - stop) / nv) / tot;
          tval += tp * cnt; t1 += tp * tp * cnt;
          if (minp > tp * tot) minp = tp * tot;
        }
        return { sphere: t1 === 0 ? 0 : (tval * tval) / t1, avg: tval, min: minp };
      };
      const aim = (tot - distr[tv]) * this.B / 100 + distr[tv];
      let tstop = 1 - this.B / 100, lower = KS_ACC / 2, upper = 1 - KS_ACC / 2, vals;
      const up = sphere(upper); up.sphere -= aim;
      if (up.avg === 0) vals = sphere(tstop);
      else if (up.sphere > 0) { tstop = upper; vals = up; }
      else {
        for (let it = 1; ; it++) {
          vals = sphere(tstop); vals.sphere -= aim;
          if (Math.abs(vals.sphere) <= KS_ACC || it >= KS_MAXIT) break;
          if (vals.sphere > 0) { lower = tstop; tstop = (upper + lower) / 2; } else { upper = tstop; tstop = (upper + lower) / 2; }
        }
      }
      return { p: tstop, miss: this.missingProb(vals.min, vals.avg) };
    }
    numericParams(j, tv) {
      const rows = this.train.rows, n = rows.length, dist = new Float64Array(n);
      let lowest = -1, next = -1, lowCount = 0, actual = 0;
      for (let i = 0; i < n; i++) {
        const v = rows[i][j];
        if (v !== v) { dist[i] = -1; continue; }
        const dd = (dist[i] = Math.abs(v - tv));
        if (dd + 1e-5 < next || next === -1) {
          if (dd + 1e-5 < lowest || lowest === -1) { next = lowest; lowest = dd; lowCount = 1; }
          else if (Math.abs(dd - lowest) < 1e-5) lowCount++;
          else next = dd;
        }
        actual++;
      }
      if (next === -1 || lowest === -1) return { p: 1, miss: this.missingProb(1, 1) };
      const sphere = (scale) => {
        let minp = 1, s = 0, s2 = 0;
        for (let i = 0; i < n; i++) {
          if (dist[i] < 0) continue;
          const ps = scale * Math.exp(-2 * dist[i] * scale);
          if (minp > ps) minp = ps;
          const inc = ps / actual; s += inc; s2 += inc * inc;
        }
        return { sphere: s2 === 0 ? 0 : (s * s) / s2, avg: s, min: minp };
      };
      let root = 1 / (next - lowest);
      let aim = (actual - lowCount) * this.B / 100 + lowCount;
      if (this.B === 0) aim += 1;
      let bot = KS_ACC / 2, up = root * 16, pos, avg, min, best = 9e300;
      const bv = sphere(bot); bv.sphere -= aim;
      const uv = sphere(up); uv.sphere -= aim;
      if (bv.sphere < 0) { pos = bot; avg = bv.avg; min = bv.min; }
      else if (uv.sphere > 0) { pos = up; avg = uv.avg; min = uv.min; }
      else {
        for (let it = 0; ; ) {
          const v = sphere(root); v.sphere -= aim;
          if (Math.abs(v.sphere) < best) { best = Math.abs(v.sphere); pos = root; avg = v.avg; min = v.min; }
          if (Math.abs(v.sphere) <= KS_ACC) break;
          if (v.sphere > 0) { const b = (root + up) / 2; bot = root; root = b; } else { const b = (root + bot) / 2; up = root; root = b; }
          if (++it > KS_MAXIT) break;
        }
      }
      return { p: pos, miss: this.missingProb(min, avg) };
    }
    params(j, tv) {
      const c = this.cache[j];
      let e = c.get(tv);
      if (!e) { e = this.dist[j] ? this.nominalParams(j, tv) : this.numericParams(j, tv); c.set(tv, e); }
      return e;
    }
    transProb(test, tr) {
      const h = this.train, na = h.numAttributes, ci = h.classIndex;
      let p = 1, miss = 0;
      for (let j = 0; j < na; j++) {
        if (j === ci) continue;
        const tv = test[j];
        if (tv !== tv) { miss++; continue; }
        const e = this.params(j, tv), v = tr[j];
        if (v !== v) p *= e.miss;
        else if (this.dist[j]) p *= (1 - e.p) / this.dist[j].c.length + (v === tv ? e.p : 0);
        else p *= e.p * Math.exp(-2 * Math.abs(tv - v) * e.p);
        // Weka re-normalises for missing test values after every attribute (kept for fidelity)
        if (miss) p = Math.pow(p, na / (na - miss));
      }
      return p;
    }
    distributionForRow(row) {
      const h = this.train, ci = h.classIndex;
      if (h.classIsNominal()) {
        const nc = h.numClasses, cp = U.zeros(nc);
        for (const r of h.rows) cp[r[ci]] += this.transProb(row, r);
        const s = U.sum(cp);
        return s <= 0 ? cp.map(() => 1 / nc) : cp.map((v) => v / s);
      }
      let num = 0, den = 0;
      for (const r of h.rows) { const p = this.transProb(row, r); num += p * r[ci]; den += p; }
      return [den !== 0 ? num / den : 0];
    }
    toString() {
      return 'KStar Beta Verion (0.1b).\nCopyright (c) 1995-97 by Len Trigg (trigg@cs.waikato.ac.nz).\nJava port to Weka by Abdelaziz Mahoui (am14@cs.waikato.ac.nz).\n\nKStar options : -B ' + this.B + ' -M ' + this.M + '\n';
    }
  }
  W.registerClassifier({
    name: 'KStar', group: 'lazy', path: 'weka.classifiers.lazy.KStar',
    desc: L`K*: instance-based learner that compares instances with an entropy-based distance (probability of transforming one instance into another).`,
    options: [
      { key: 'B', flag: 'B', label: L`Global blend (%)`, type: 'int', default: 20, min: 0, max: 100 },
      { key: 'M', flag: 'M', label: L`Missing values`, type: 'select', default: 'a', choices: { a: L`Average column entropy curves`, d: L`Ignore instances with missing values`, m: L`Treat as maximally different`, n: L`Normalise over attributes` } },
    ],
    create: (o) => new KStar(o),
  });

  // ============================================================== LogitBoost
  const regressionBases = () => {
    const o = {};
    for (const s of W.classifiers) if (!s.meta && !(s.caps && s.caps.numericClass === false)) o[s.name] = s.name;
    return o;
  };
  /** Additive logistic regression (Friedman, Hastie & Tibshirani) with a regression base learner. */
  class LogitBoost {
    constructor(o = {}) { this.I = o.I ?? 10; this.H = o.H ?? 1; this.Z = o.Z ?? 3; this.base = o.W || 'DecisionStump'; }
    build(data) {
      data = data.withoutMissingClass();
      this.header = data.emptyCopy();
      const nc = data.numClasses, ci = data.classIndex, n = data.numInstances;
      this.nc = nc;
      if (data.numAttributes === 1) { this.zero = new W.ZeroR().build(data); return this; }
      const spec = W.findClassifier(this.base);
      if (!spec) throw new Error(L`Unknown base classifier: ${this.base}`);
      // working dataset with a numeric pseudo class
      const attrs = data.attributes.slice(); attrs[ci] = { name: 'pseudo class', type: 'numeric' };
      const origW = data.weights.slice(), origSum = data.sumOfWeights();
      const ys = data.rows.map((r) => { const y = U.zeros(nc); y[r[ci]] = 1; return y; });
      const fs = data.rows.map(() => U.zeros(nc));
      let probs = data.rows.map(() => new Array(nc).fill(1 / nc));
      this.models = [];
      for (let it = 0; it < this.I; it++) {
        const ms = [];
        for (let j = 0; j < nc; j++) {
          const rows = new Array(n), ws = new Array(n);
          for (let i = 0; i < n; i++) {
            const p = probs[i][j], actual = ys[i][j];
            let z;
            if (actual === 1) { z = 1 / p; if (z > this.Z) z = this.Z; } else { z = -1 / (1 - p); if (z < -this.Z) z = -this.Z; }
            const w = (actual - p) / z;
            const r = data.rows[i].slice(); r[ci] = z; rows[i] = r; ws[i] = origW[i] * w;
          }
          const s = U.sum(ws), f = origSum / s;
          const train = new W.Dataset(data.relation, attrs, rows, ws.map((w) => w * f), ci);
          ms.push(spec.create(W.defaultOptions(spec)).build(train));
          if (nc === 2) break;
        }
        this.models.push(ms);
        for (let i = 0; i < n; i++) this.addF(fs[i], ms, data.rows[i]);
        probs = fs.map((F) => this.probs(F));
      }
      return this;
    }
    addF(F, ms, row) {
      const nc = this.nc, pred = U.zeros(nc);
      let sum = 0;
      for (let j = 0; j < nc; j++) {
        const v = this.H * ms[j].distributionForRow(row)[0];
        pred[j] = v;
        if (nc === 2) { pred[1] = -v; break; }
        sum += v;
      }
      sum /= nc;
      for (let j = 0; j < nc; j++) F[j] += ((pred[j] - sum) * (nc - 1)) / nc;
    }
    probs(F) {
      let mx = -Infinity; for (const v of F) if (v > mx) mx = v;
      const p = F.map((v) => Math.exp(v - mx));
      return U.normalize(p);
    }
    distributionForRow(row) {
      if (this.zero) return this.zero.distributionForRow(row);
      const F = U.zeros(this.nc);
      for (const ms of this.models) this.addF(F, ms, row);
      return this.probs(F);
    }
    toString() {
      if (this.zero) return this.zero.toString();
      const cv = this.header.classAttr.values, t = ['LogitBoost: Base classifiers and their weights: ', ''];
      this.models.forEach((ms, i) => {
        t.push('Iteration ' + (i + 1));
        ms.forEach((m, j) => t.push('\tClass ' + (j + 1) + ' (' + cv[j] + ')', '', m.toString(), ''));
      });
      t.push('Number of performed iterations: ' + this.models.length);
      return t.join('\n') + '\n';
    }
  }
  W.registerClassifier({
    name: 'LogitBoost', group: 'meta', meta: true, path: 'weka.classifiers.meta.LogitBoost',
    desc: L`Boosting as additive logistic regression: each round fits a regression model (default DecisionStump) to the working responses. Gives well-calibrated probabilities.`,
    caps: { numericClass: false },
    options: [
      { key: 'W', flag: 'W', label: L`Base regressor`, type: 'select', default: 'DecisionStump', choices: regressionBases, cli: null },
      { key: 'I', flag: 'I', label: L`Iterations`, type: 'int', default: 10, min: 1 },
      { key: 'H', flag: 'H', label: L`Shrinkage`, type: 'number', default: 1, min: 0, step: 0.1 },
      { key: 'Z', flag: 'Z', label: L`Z max threshold`, type: 'number', default: 3, min: 0 },
    ],
    create: (o) => new LogitBoost(o),
  });

  // ===================================================================== SGD
  const LOSS = { 0: 'Hinge loss (SVM)', 1: 'Log loss (logistic regression)', 2: 'Squared loss (regression)' };
  /** Stochastic gradient descent for linear models (hinge, log or squared loss). */
  class SGD {
    constructor(o = {}) {
      this.F = +(o.F ?? 0); this.lr = o.L ?? 0.01; this.lambda = o.R ?? 1e-4; this.E = o.E ?? 500;
      this.normalize = !o.N; this.replaceMissing = !o.M; this.seed = o.S ?? 1;
    }
    build(data) {
      data = data.withoutMissingClass();
      const nominal = data.classIsNominal();
      if (this.F === 2 && nominal) throw new Error(L`Squared loss needs a numeric class. Choose hinge or log loss for a nominal class.`);
      if (this.F !== 2 && !nominal) throw new Error(L`Hinge and log loss need a nominal (two-valued) class. Choose squared loss for a numeric class.`);
      if (nominal && data.numClasses !== 2) throw new Error(L`SGD supports only two-valued nominal classes. Try SMO or Logistic for more classes.`);
      const fl = [];
      if (this.replaceMissing) fl.push(new W.ReplaceMissingValues());
      if (data.attributes.some((a, j) => j !== data.classIndex && a.type === 'nominal')) fl.push(new W.SupervisedNominalToBinary({}));
      if (this.normalize) fl.push(new W.Normalize());
      this.pipe = new W.Pipeline(fl);
      const d = this.pipe.fit(data).copy();
      this.header = d.emptyCopy();
      const ci = d.classIndex, na = d.numAttributes, n = d.numInstances;
      const w = (this.w = U.zeros(na + 1));
      if (!n) return this;
      d.randomize(new W.JavaRandom(this.seed));
      const mult = 1 - (this.lr * this.lambda) / n;
      for (let e = 0; e < this.E; e++) {
        for (let i = 0; i < n; i++) {
          const r = d.rows[i];
          const wx = this.dot(r);
          let y, z;
          if (nominal) { y = r[ci] === 0 ? -1 : 1; z = y * (wx + w[na]); } else { y = r[ci]; z = y - (wx + w[na]); y = 1; }
          for (let j = 0; j < na; j++) if (j !== ci) w[j] *= mult;
          if (this.F !== 0 || z < 1) {
            const f = this.lr * y * this.dloss(z);
            for (let j = 0; j < na; j++) if (j !== ci && r[j] === r[j]) w[j] += f * r[j];
            w[na] += f;
          }
        }
      }
      return this;
    }
    dloss(z) {
      if (this.F === 0) return z < 1 ? 1 : 0;
      if (this.F === 1) { if (z < 0) return 1 / (Math.exp(z) + 1); const t = Math.exp(-z); return t / (t + 1); }
      return z;
    }
    dot(r) {
      let s = 0; const ci = this.header.classIndex, w = this.w;
      for (let j = 0; j < r.length; j++) if (j !== ci && r[j] === r[j]) s += w[j] * r[j];
      return s;
    }
    distributionForRow(row) {
      const r = this.pipe.row(row), wx = this.dot(r) + this.w[this.w.length - 1];
      if (!this.header.classIsNominal()) return [wx];
      if (this.F === 0) return wx <= 0 ? [1, 0] : [0, 1];
      const p1 = 1 / (1 + Math.exp(-wx));
      return [1 - p1, p1];
    }
    toString() {
      const h = this.header, w = this.w, ci = h.classIndex;
      const t = ['Loss function: ' + LOSS[this.F], '', h.classAttr.name + ' = ', '', ''];
      let s = '', printed = 0;
      for (let j = 0; j < w.length - 1; j++) {
        if (j === ci) continue;
        s += (printed ? ' + ' : '   ') + dts(w[j], 12, 4) + ' ' + (this.normalize ? '(normalized) ' : '') + h.attributes[j].name + '\n';
        printed++;
      }
      const b = w[w.length - 1];
      s += b > 0 ? ' + ' + dts(b, 12, 4) : ' - ' + dts(-b, 12, 4);
      return t.join('\n') + s + '\n';
    }
  }
  W.registerClassifier({
    name: 'SGD', group: 'functions', path: 'weka.classifiers.functions.SGD',
    desc: L`Linear model trained with stochastic gradient descent: linear SVM (hinge loss), logistic regression (log loss) or linear regression (squared loss). Fast on large data.`,
    options: [
      { key: 'F', flag: 'F', label: L`Loss function`, type: 'select', default: 0, numeric: true, choices: { 0: L`Hinge loss (SVM)`, 1: L`Log loss (logistic regression)`, 2: L`Squared loss (regression)` } },
      { key: 'L', flag: 'L', label: 'Learning rate', type: 'number', default: 0.01, min: 0, step: 0.005 },
      { key: 'R', flag: 'R', label: L`Regularisation (lambda)`, type: 'number', default: 0.0001, min: 0, step: 0.0001 },
      { key: 'E', flag: 'E', label: L`Epochs`, type: 'int', default: 500, min: 1 },
      { key: 'N', flag: 'N', label: L`Do not normalise`, type: 'bool', default: false },
      { key: 'M', flag: 'M', label: L`Do not replace missing values`, type: 'bool', default: false },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 1 },
    ],
    create: (o) => new SGD(o),
  });

  // ========================================================= VotedPerceptron
  /** Voted perceptron (Freund & Schapire) with a polynomial kernel. Two-class problems only. */
  class VotedPerceptron {
    constructor(o = {}) { this.I = o.I ?? 1; this.E = o.E ?? 1; this.seed = o.S ?? 1; this.M = o.M ?? 10000; }
    build(data) {
      data = data.withoutMissingClass();
      if (data.numClasses !== 2) throw new Error(L`VotedPerceptron supports only two-valued nominal classes.`);
      this.pipe = new W.Pipeline([new W.ReplaceMissingValues(), new W.NominalToBinary({})]);
      const d = this.pipe.fit(data).copy();
      d.randomize(new W.JavaRandom(this.seed));
      this.header = d.emptyCopy();
      const ci = d.classIndex;
      this.X = d.rows;
      this.add = []; this.isAdd = []; this.wts = [0];
      let k = 0;
      out: for (let it = 0; it < this.I; it++) {
        for (let i = 0; i < d.rows.length; i++) {
          const r = d.rows[i], cls = r[ci];
          if (this.predict(k, r) === cls) this.wts[k]++;
          else { this.isAdd[k] = cls === 1; this.add[k] = i; k++; this.wts[k] = (this.wts[k] || 0) + 1; }
          if (k === this.M) break out;
        }
      }
      this.K = k;
      return this;
    }
    inner(a, b) {
      const ci = this.header.classIndex;
      let s = 0; for (let j = 0; j < a.length; j++) if (j !== ci) s += a[j] * b[j];
      s += 1;
      return this.E !== 1 ? Math.pow(s, this.E) : s;
    }
    predict(k, r) {
      let s = 0;
      for (let i = 0; i < k; i++) s += (this.isAdd[i] ? 1 : -1) * this.inner(this.X[this.add[i]], r);
      return s < 0 ? 0 : 1;
    }
    distributionForRow(row) {
      const r = this.pipe.row(row);
      let out = 0, sum = 0;
      if (this.K > 0) {
        for (let i = 0; i <= this.K; i++) {
          if (sum < 0) out -= this.wts[i]; else out += this.wts[i];
          if (i < this.K) sum += (this.isAdd[i] ? 1 : -1) * this.inner(this.X[this.add[i]], r);
        }
      }
      const p1 = 1 / (1 + Math.exp(-out));
      return [1 - p1, p1];
    }
    toString() { return 'VotedPerceptron: Number of perceptrons=' + this.K + '\n'; }
  }
  W.registerClassifier({
    name: 'VotedPerceptron', group: 'functions', path: 'weka.classifiers.functions.VotedPerceptron',
    desc: L`Voted perceptron (Freund & Schapire): keeps every perceptron it creates and lets them vote, weighted by how long each survived. Two-class problems.`,
    caps: { numericClass: false, binaryClass: true },
    options: [
      { key: 'I', flag: 'I', label: L`Iterations`, type: 'int', default: 1, min: 1 },
      { key: 'E', flag: 'E', label: L`Kernel exponent`, type: 'number', default: 1, min: 0.1 },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 1 },
      { key: 'M', flag: 'M', label: L`Maximum number of alterations`, type: 'int', default: 10000, min: 1 },
    ],
    create: (o) => new VotedPerceptron(o),
  });
})();
