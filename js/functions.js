/* Weka Web — functions: LinearRegression, SimpleLinearRegression, Logistic, SMO, MultilayerPerceptron. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, dts = (v, w, d) => W.dts(v, w, d);

  // ------------------------------------------------------------- linear algebra
  /** Λύση συμμετρικού θετικά ορισμένου συστήματος με Cholesky. Επιστρέφει null αν αποτύχει. */
  function choleskySolve(A, b) {
    const n = b.length, L = U.matrix(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) { if (s <= 0) return null; L[i][i] = Math.sqrt(s); }
        else L[i][j] = s / L[j][j];
      }
    }
    const y = new Array(n);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
    return x;
  }
  /** Γκαουσιανή απαλοιφή με μερική οδήγηση (εφεδρική λύση). */
  function gaussSolve(A, b) {
    const n = b.length, M = A.map((r, i) => r.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      if (Math.abs(M[c][c]) < 1e-300) continue;
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let k = i + 1; k < n; k++) s -= M[i][k] * x[k]; x[i] = Math.abs(M[i][i]) < 1e-300 ? 0 : s / M[i][i]; }
    return x;
  }
  W.linalg = { choleskySolve, gaussSolve };

  /** Αλυσίδα φίλτρων για αλγορίθμους που χρειάζονται αριθμητική είσοδο. */
  class Pipeline {
    constructor(filters) { this.filters = filters; }
    fit(d) { for (const f of this.filters) { f.setInput(d); d = f.apply(d); } this.out = d; return d; }
    row(r) { for (const f of this.filters) r = f.convertRow(r); return r; }
  }
  W.Pipeline = Pipeline;

  function weightedMeanVar(d, j) {
    let sw = 0, s = 0, ss = 0;
    for (let i = 0; i < d.rows.length; i++) { const v = d.rows[i][j]; if (v !== v) continue; const w = d.weights[i]; sw += w; s += w * v; ss += w * v * v; }
    const mean = sw > 0 ? s / sw : 0;
    let variance = sw > 1 ? (ss - (s * s) / sw) / (sw - 1) : 0;
    if (variance < 0) variance = 0;
    return { mean, variance, sw };
  }

  // ========================================================= LinearRegression
  class LinearRegression {
    constructor(o = {}) { this.S = o.S ?? 0; this.ridge = o.R ?? 1e-8; this.elimColinear = !o.C; this.decimals = o.decimals ?? 4; }
    build(data) {
      data = data.withoutMissingClass();
      this.origHeader = data;
      if (data.numInstances === 1) { this.zero = data.rows[0][data.classIndex]; return this; }
      this.pipe = new Pipeline([new W.SupervisedNominalToBinary({ N: true }), new W.ReplaceMissingValues()]);
      const d = this.pipe.fit(data).withoutMissingClass();
      this.d = d;
      const ci = d.classIndex, na = d.numAttributes;
      this.sel = new Array(na).fill(false); this.means = U.zeros(na); this.sds = U.zeros(na);
      for (let j = 0; j < na; j++) {
        if (j === ci) continue;
        const mv = weightedMeanVar(d, j);
        this.sel[j] = true; this.means[j] = mv.mean; this.sds[j] = Math.sqrt(mv.variance);
        if (this.sds[j] === 0) this.sel[j] = false;
      }
      const cmv = weightedMeanVar(d, ci);
      this.classSd = Math.sqrt(cmv.variance); this.classMean = cmv.mean;
      this.findBestModel();
      this.header = d.emptyCopy();
      this.d = null;
      return this;
    }
    doRegression(sel) {
      const d = this.d, ci = d.classIndex, cols = [];
      for (let j = 0; j < sel.length; j++) if (sel[j]) cols.push(j);
      const p = cols.length, coef = new Array(p + 1).fill(0);
      if (p > 0) {
        const ata = U.matrix(p, p), aty = U.zeros(p);
        const x = new Array(p);
        for (let i = 0; i < d.rows.length; i++) {
          const r = d.rows[i], sw = Math.sqrt(d.weights[i]);
          for (let k = 0; k < p; k++) x[k] = ((r[cols[k]] - this.means[cols[k]]) / this.sds[cols[k]]) * sw;
          const y = r[ci] * sw;
          for (let a = 0; a < p; a++) { aty[a] += x[a] * y; for (let b = 0; b <= a; b++) ata[a][b] += x[a] * x[b]; }
        }
        for (let a = 0; a < p; a++) { for (let b = 0; b < a; b++) ata[b][a] = ata[a][b]; ata[a][a] += this.ridge; }
        const sol = choleskySolve(ata, aty) || gaussSolve(ata, aty);
        for (let k = 0; k < p; k++) coef[k] = sol[k];
      }
      coef[p] = this.classMean;
      for (let k = 0; k < p; k++) { coef[k] /= this.sds[cols[k]]; coef[p] -= coef[k] * this.means[cols[k]]; }
      return coef;
    }
    predict(r, sel, coef) {
      let res = 0, c = 0;
      for (let j = 0; j < sel.length; j++) if (sel[j]) res += coef[c++] * r[j];
      return res + coef[c];
    }
    se(sel, coef) {
      const d = this.d, ci = d.classIndex;
      let m = 0;
      for (const r of d.rows) { const e = this.predict(r, sel, coef) - r[ci]; m += e * e; }
      return m;
    }
    deselectColinear(sel, coef) {
      let maxSC = 1.5, maxAttr = -1, c = 0;
      for (let j = 0; j < sel.length; j++) if (sel[j]) { const sc = Math.abs(coef[c] * this.sds[j] / this.classSd); if (sc > maxSC) { maxSC = sc; maxAttr = j; } c++; }
      if (maxAttr >= 0) { sel[maxAttr] = false; return true; }
      return false;
    }
    findBestModel() {
      const n = this.d.numInstances;
      do { this.coef = this.doRegression(this.sel); } while (this.elimColinear && this.deselectColinear(this.sel, this.coef));
      let numAttr = 1; for (const s of this.sel) if (s) numAttr++;
      const fullMSE = this.se(this.sel, this.coef);
      let akaike = n - numAttr + 2 * numAttr, improved, cur = numAttr;
      if (this.S === 2) {
        do {
          const cs = this.sel.slice(); improved = false; cur--;
          for (let j = 0; j < this.sel.length; j++) {
            if (!cs[j]) continue;
            cs[j] = false;
            const cc = this.doRegression(cs), mse = this.se(cs, cc);
            const ak = (mse / fullMSE) * (n - numAttr) + 2 * cur;
            if (ak < akaike) { improved = true; akaike = ak; this.sel = cs.slice(); this.coef = cc; }
            cs[j] = true;
          }
        } while (improved);
      } else if (this.S === 0) {
        do {
          improved = false; cur--;
          let minSC = 0, minAttr = -1, c = 0;
          for (let j = 0; j < this.sel.length; j++) if (this.sel[j]) { const sc = Math.abs(this.coef[c] * this.sds[j] / this.classSd); if (c === 0 || sc < minSC) { minSC = sc; minAttr = j; } c++; }
          if (minAttr >= 0) {
            this.sel[minAttr] = false;
            const cc = this.doRegression(this.sel), mse = this.se(this.sel, cc);
            const ak = (mse / fullMSE) * (n - numAttr) + 2 * cur;
            if (ak < akaike) { improved = true; akaike = ak; this.coef = cc; } else this.sel[minAttr] = true;
          }
        } while (improved);
      }
    }
    distributionForRow(r) {
      if (this.zero !== undefined) return [this.zero];
      return [this.predict(this.pipe.row(r), this.sel, this.coef)];
    }
    toString() {
      if (this.zero !== undefined) return 'Linear Regression: only one instance, predicting ' + this.zero;
      const h = this.header, t = ['', 'Linear Regression Model', '', h.classAttr.name + ' =', ''];
      let c = 0; const parts = [];
      for (let j = 0; j < this.sel.length; j++) if (this.sel[j]) parts.push(dts(this.coef[c++], 12, this.decimals) + ' * ' + h.attributes[j].name);
      parts.push(dts(this.coef[c], 12, this.decimals));
      return t.join('\n') + '\n' + parts.join(' +\n');
    }
  }
  W.registerClassifier({
    name: 'LinearRegression', group: 'functions', path: 'weka.classifiers.functions.LinearRegression',
    desc: L`Γραμμική παλινδρόμηση με επιλογή χαρακτηριστικών (μέθοδος M5 / κριτήριο Akaike) και ridge.`,
    caps: { nominalClass: false },
    options: [
      { key: 'S', flag: 'S', label: L`Επιλογή χαρακτηριστικών`, type: 'select', default: 0, choices: { 0: 'M5 method', 1: L`Καμία`, 2: 'Greedy' }, numeric: true },
      { key: 'C', flag: 'C', label: L`Χωρίς απαλοιφή συγγραμμικών`, type: 'bool', default: false },
      { key: 'R', flag: 'R', label: 'Ridge', type: 'number', default: 1e-8, min: 0 },
    ],
    create: (o) => new LinearRegression(o),
  });

  // =================================================== SimpleLinearRegression
  class SimpleLinearRegression {
    build(data) {
      data = data.withoutMissingClass();
      this.header = data;
      const ci = data.classIndex;
      const cm = weightedMeanVar(data, ci);
      this.attr = -1; this.slope = 0; this.intercept = cm.mean;
      let minErr = Infinity;
      for (let j = 0; j < data.numAttributes; j++) {
        if (j === ci || !W.isNumericAttr(data.attributes[j])) continue;
        let sw = 0, sx = 0, sy = 0;
        for (let i = 0; i < data.rows.length; i++) { const x = data.rows[i][j]; if (x !== x) continue; const w = data.weights[i]; sw += w; sx += w * x; sy += w * data.rows[i][ci]; }
        if (sw === 0) continue;
        const xm = sx / sw, ym = sy / sw;
        let sxy = 0, sxx = 0;
        for (let i = 0; i < data.rows.length; i++) { const x = data.rows[i][j]; if (x !== x) continue; const w = data.weights[i]; sxy += w * (x - xm) * (data.rows[i][ci] - ym); sxx += w * (x - xm) * (x - xm); }
        if (sxx <= 0) continue;
        const slope = sxy / sxx, inter = ym - slope * xm;
        let err = 0;
        for (let i = 0; i < data.rows.length; i++) { const x = data.rows[i][j]; if (x !== x) continue; const e = slope * x + inter - data.rows[i][ci]; err += data.weights[i] * e * e; }
        if (err < minErr) { minErr = err; this.attr = j; this.slope = slope; this.intercept = inter; }
      }
      return this;
    }
    distributionForRow(r) {
      if (this.attr < 0) return [this.intercept];
      const x = r[this.attr];
      return [x !== x ? this.intercept : this.slope * x + this.intercept];
    }
    toString() {
      if (this.attr < 0) return 'Predicting constant ' + this.intercept;
      return `Linear regression on ${this.header.attributes[this.attr].name}\n\n${U.d2s(this.slope, 2)} * ${this.header.attributes[this.attr].name} ${this.intercept >= 0 ? '+' : '-'} ${U.d2s(Math.abs(this.intercept), 2)}\n`;
    }
  }
  W.registerClassifier({
    name: 'SimpleLinearRegression', group: 'functions', path: 'weka.classifiers.functions.SimpleLinearRegression',
    desc: L`Παλινδρόμηση με ένα μόνο χαρακτηριστικό: εκείνο με το μικρότερο τετραγωνικό σφάλμα.`,
    caps: { nominalClass: false }, options: [], create: () => new SimpleLinearRegression(),
  });

  // ================================================================= L-BFGS
  /** Ελαχιστοποίηση με L-BFGS και line search Armijo/backtracking. f(x) → [value, grad]. */
  function lbfgs(f, x0, maxIter = 1000, tol = 1e-9) {
    const n = x0.length, m = 10;
    let x = x0.slice(), [fx, g] = f(x);
    const sList = [], yList = [], rho = [];
    for (let it = 0; it < maxIter; it++) {
      let gn = 0; for (const v of g) gn = Math.max(gn, Math.abs(v));
      if (gn < tol * Math.max(1, Math.abs(fx))) break;
      // two-loop recursion
      const q = g.slice(), alpha = [];
      for (let k = sList.length - 1; k >= 0; k--) { let a = 0; for (let i = 0; i < n; i++) a += sList[k][i] * q[i]; a *= rho[k]; alpha[k] = a; for (let i = 0; i < n; i++) q[i] -= a * yList[k][i]; }
      if (sList.length) { const k = sList.length - 1; let sy = 0, yy = 0; for (let i = 0; i < n; i++) { sy += sList[k][i] * yList[k][i]; yy += yList[k][i] * yList[k][i]; } const gamma = sy / yy; for (let i = 0; i < n; i++) q[i] *= gamma; }
      else { const s = 1 / Math.max(1, gn); for (let i = 0; i < n; i++) q[i] *= s; }
      for (let k = 0; k < sList.length; k++) { let b = 0; for (let i = 0; i < n; i++) b += yList[k][i] * q[i]; b *= rho[k]; for (let i = 0; i < n; i++) q[i] += sList[k][i] * (alpha[k] - b); }
      const dir = q.map((v) => -v);
      let dg = 0; for (let i = 0; i < n; i++) dg += dir[i] * g[i];
      if (dg >= 0) { for (let i = 0; i < n; i++) dir[i] = -g[i]; dg = 0; for (let i = 0; i < n; i++) dg -= g[i] * g[i]; sList.length = yList.length = rho.length = 0; }
      let step = 1, nx, nf, ng, ok = false;
      for (let ls = 0; ls < 60; ls++) {
        nx = x.map((v, i) => v + step * dir[i]);
        [nf, ng] = f(nx);
        if (nf <= fx + 1e-4 * step * dg && isFinite(nf)) { ok = true; break; }
        step *= 0.5;
      }
      if (!ok) break;
      const s = nx.map((v, i) => v - x[i]), y = ng.map((v, i) => v - g[i]);
      let sy = 0; for (let i = 0; i < n; i++) sy += s[i] * y[i];
      if (sy > 1e-12) { sList.push(s); yList.push(y); rho.push(1 / sy); if (sList.length > m) { sList.shift(); yList.shift(); rho.shift(); } }
      const df = fx - nf;
      x = nx; fx = nf; g = ng;
      if (Math.abs(df) < 1e-15 * Math.max(1, Math.abs(fx))) break;
    }
    return { x, fx };
  }
  W.lbfgs = lbfgs;

  // ================================================================= Logistic
  class Logistic {
    constructor(o = {}) { this.ridge = o.R ?? 1e-8; this.maxIts = o.M ?? -1; this.decimals = o.decimals ?? 4; }
    build(data) {
      data = data.withoutMissingClass();
      this.pipe = new Pipeline([new W.ReplaceMissingValues(), new W.RemoveUseless(), new W.NominalToBinary({ N: true })]);
      const d = this.pipe.fit(data);
      this.header = d.emptyCopy();
      const ci = d.classIndex, K = d.numClasses, nK = K - 1;
      const preds = []; for (let j = 0; j < d.numAttributes; j++) if (j !== ci) preds.push(j);
      this.preds = preds;
      const nR = preds.length, nC = d.numInstances;
      const X = new Array(nC), Y = new Int32Array(nC), w = new Float64Array(nC);
      const mean = U.zeros(nR + 1), sd = U.zeros(nR + 1), sY = U.zeros(K);
      let tot = 0;
      for (let i = 0; i < nC; i++) {
        const r = d.rows[i]; const x = new Float64Array(nR + 1); x[0] = 1;
        for (let k = 0; k < nR; k++) { x[k + 1] = r[preds[k]]; mean[k + 1] += d.weights[i] * x[k + 1]; sd[k + 1] += d.weights[i] * x[k + 1] * x[k + 1]; }
        X[i] = x; Y[i] = r[ci]; w[i] = d.weights[i]; tot += w[i]; sY[Y[i]]++;
      }
      mean[0] = 0; sd[0] = 1;
      for (let j = 1; j <= nR; j++) { mean[j] /= tot; sd[j] = tot > 1 ? Math.sqrt(Math.abs(sd[j] - tot * mean[j] * mean[j]) / (tot - 1)) : 0; }
      for (const x of X) for (let j = 1; j <= nR; j++) if (sd[j] !== 0) x[j] = (x[j] - mean[j]) / sd[j];
      const P = nR + 1, ridge = this.ridge;
      const x0 = new Array(P * nK).fill(0);
      for (let p = 0; p < nK; p++) x0[p * P] = Math.log(sY[p] + 1) - Math.log(sY[nK] + 1);
      const ex = new Float64Array(K);
      const fn = (par) => {
        let nll = 0; const grad = new Array(par.length).fill(0);
        for (let i = 0; i < nC; i++) {
          const x = X[i];
          let mx = 0;
          for (let k = 0; k < nK; k++) { let v = 0; const off = k * P; for (let j = 0; j < P; j++) v += par[off + j] * x[j]; ex[k] = v; if (v > mx) mx = v; }
          ex[nK] = 0;
          let den = 0; for (let k = 0; k <= nK; k++) den += Math.exp(ex[k] - mx);
          nll -= w[i] * (ex[Y[i]] - mx - Math.log(den));
          for (let k = 0; k < nK; k++) {
            const pk = Math.exp(ex[k] - mx) / den, e = w[i] * (pk - (Y[i] === k ? 1 : 0)), off = k * P;
            for (let j = 0; j < P; j++) grad[off + j] += e * x[j];
          }
        }
        for (let k = 0; k < nK; k++) for (let j = 1; j < P; j++) { const v = par[k * P + j]; nll += ridge * v * v; grad[k * P + j] += 2 * ridge * v; }
        return [nll, grad];
      };
      const res = lbfgs(fn, x0, this.maxIts > 0 ? this.maxIts : 5000, 1e-10);
      this.LL = -res.fx;
      this.par = U.matrix(P, nK);
      for (let k = 0; k < nK; k++) {
        this.par[0][k] = res.x[k * P];
        for (let j = 1; j <= nR; j++) {
          this.par[j][k] = res.x[k * P + j];
          if (sd[j] !== 0) { this.par[j][k] /= sd[j]; this.par[0][k] -= this.par[j][k] * mean[j]; }
        }
      }
      return this;
    }
    distributionForRow(row) {
      const r = this.pipe.row(row), nK = this.header.numClasses - 1, P = this.preds.length + 1;
      const v = new Array(nK + 1).fill(0);
      for (let k = 0; k < nK; k++) { let s = this.par[0][k]; for (let j = 1; j < P; j++) s += this.par[j][k] * r[this.preds[j - 1]]; v[k] = s; }
      const mx = Math.max(...v);
      return U.normalize(v.map((x) => Math.exp(x - mx)));
    }
    toString() {
      const h = this.header, cv = h.classAttr.values, nK = cv.length - 1;
      const names = this.preds.map((j) => h.attributes[j].name);
      const colW = Math.max(...cv.slice(0, nK).map((s) => s.length), 10) + 2;
      const nameW = Math.max(...names.map((s) => s.length), 'Intercept'.length, 'Variable'.length) + 2;
      const line = (lbl, vals) => U.padR(lbl, nameW) + vals.map((v) => U.padL(v, colW)).join('');
      const t = ['Logistic Regression with ridge parameter of ' + javaE(this.ridge), 'Coefficients...'];
      t.push(line('', cv.slice(0, nK).map((_, k) => (k === 0 ? 'Class' : ''))));
      t.push(line('Variable', cv.slice(0, nK)));
      t.push('='.repeat(nameW + colW * nK));
      names.forEach((nm, j) => t.push(line(nm, this.par[j + 1].map((v) => U.d2s(v, this.decimals)))));
      t.push(line('Intercept', this.par[0].map((v) => U.d2s(v, this.decimals))));
      t.push('', '', 'Odds Ratios...');
      t.push(line('', cv.slice(0, nK).map((_, k) => (k === 0 ? 'Class' : ''))));
      t.push(line('Variable', cv.slice(0, nK)));
      t.push('='.repeat(nameW + colW * nK));
      names.forEach((nm, j) => t.push(line(nm, this.par[j + 1].map((v) => U.d2s(Math.exp(v), this.decimals)))));
      return t.join('\n') + '\n';
    }
  }
  function javaE(v) { if (v === 0) return '0.0'; const e = Math.floor(Math.log10(Math.abs(v))); if (e >= -3 && e < 7) return W.javaDouble(v); const m = v / Math.pow(10, e); return W.javaDouble(Number(m.toPrecision(15))) + 'E' + e; }
  W.registerClassifier({
    name: 'Logistic', group: 'functions', path: 'weka.classifiers.functions.Logistic',
    desc: L`Πολυωνυμική λογιστική παλινδρόμηση με ridge estimator.`,
    caps: { numericClass: false },
    options: [
      { key: 'R', flag: 'R', label: 'Ridge', type: 'number', default: 1e-8, min: 0 },
      { key: 'M', flag: 'M', label: L`Μέγιστες επαναλήψεις (-1 = μέχρι σύγκλιση)`, type: 'int', default: -1, min: -1 },
    ],
    create: (o) => new Logistic(o),
  });

  // ===================================================================== SMO
  class BinarySMO {
    constructor(C, tol, eps, kernel) { this.C = C; this.tol = tol; this.eps = eps; this.kernel = kernel; }
    train(X, y, w, rand) {
      const n = X.length, C = this.C, K = this.kernel;
      this.X = X; this.y = y;
      const alpha = new Float64Array(n);
      const cache = new Map();
      const kv = (i, j) => { if (i > j) [i, j] = [j, i]; const key = i * n + j; let v = cache.get(key); if (v === undefined) { v = K(X[i], X[j]); if (cache.size < 2e6) cache.set(key, v); } return v; };
      let b = 0;
      const err = new Float64Array(n);
      for (let i = 0; i < n; i++) err[i] = -y[i];
      const Ci = (i) => C * w[i];
      const takeStep = (i1, i2) => {
        if (i1 === i2) return false;
        const a1 = alpha[i1], a2 = alpha[i2], y1 = y[i1], y2 = y[i2], E1 = err[i1], E2 = err[i2], s = y1 * y2;
        const C1 = Ci(i1), C2 = Ci(i2);
        let L, H;
        if (y1 !== y2) { L = Math.max(0, a2 - a1); H = Math.min(C2, C1 + a2 - a1); }
        else { L = Math.max(0, a1 + a2 - C1); H = Math.min(C2, a1 + a2); }
        if (L >= H) return false;
        const k11 = kv(i1, i1), k12 = kv(i1, i2), k22 = kv(i2, i2), eta = 2 * k12 - k11 - k22;
        let a2n;
        if (eta < 0) {
          a2n = a2 - (y2 * (E1 - E2)) / eta;
          if (a2n < L) a2n = L; else if (a2n > H) a2n = H;
        } else {
          const f1 = y1 * (E1 + b) - a1 * k11 - s * a2 * k12, f2 = y2 * (E2 + b) - s * a1 * k12 - a2 * k22;
          const L1 = a1 + s * (a2 - L), H1 = a1 + s * (a2 - H);
          const Lobj = L1 * f1 + L * f2 + 0.5 * L1 * L1 * k11 + 0.5 * L * L * k22 + s * L * L1 * k12;
          const Hobj = H1 * f1 + H * f2 + 0.5 * H1 * H1 * k11 + 0.5 * H * H * k22 + s * H * H1 * k12;
          if (Lobj > Hobj + this.eps) a2n = L; else if (Lobj < Hobj - this.eps) a2n = H; else a2n = a2;
        }
        if (Math.abs(a2n - a2) < this.eps * (a2n + a2 + this.eps)) return false;
        let a1n = a1 + s * (a2 - a2n);
        if (a1n < 0) { a2n += s * a1n; a1n = 0; } else if (a1n > C1) { a2n += s * (a1n - C1); a1n = C1; }
        // νέο b (μορφή f(x) = Σ a y K - b)
        const b1 = E1 + y1 * (a1n - a1) * k11 + y2 * (a2n - a2) * k12 + b;
        const b2 = E2 + y1 * (a1n - a1) * k12 + y2 * (a2n - a2) * k22 + b;
        let bn;
        if (a1n > 0 && a1n < C1) bn = b1; else if (a2n > 0 && a2n < C2) bn = b2; else bn = (b1 + b2) / 2;
        const t1 = y1 * (a1n - a1), t2 = y2 * (a2n - a2);
        for (let i = 0; i < n; i++) err[i] += t1 * kv(i1, i) + t2 * kv(i2, i) + b - bn;
        b = bn;
        alpha[i1] = a1n; alpha[i2] = a2n;
        return true;
      };
      const examine = (i2) => {
        const y2 = y[i2], a2 = alpha[i2], E2 = err[i2], r2 = E2 * y2;
        if ((r2 < -this.tol && a2 < Ci(i2)) || (r2 > this.tol && a2 > 0)) {
          // δεύτερη επιλογή: μέγιστο |E1-E2| μεταξύ non-bound
          let best = -1, mx = -1;
          for (let i = 0; i < n; i++) if (alpha[i] > 0 && alpha[i] < Ci(i)) { const d = Math.abs(err[i] - E2); if (d > mx) { mx = d; best = i; } }
          if (best >= 0 && takeStep(best, i2)) return 1;
          const start = rand.nextInt(n);
          for (let k = 0; k < n; k++) { const i = (start + k) % n; if (alpha[i] > 0 && alpha[i] < Ci(i) && takeStep(i, i2)) return 1; }
          for (let k = 0; k < n; k++) { const i = (start + k) % n; if (takeStep(i, i2)) return 1; }
        }
        return 0;
      };
      let numChanged = 0, examineAll = true, guard = 0;
      while ((numChanged > 0 || examineAll) && guard++ < 10000) {
        numChanged = 0;
        if (examineAll) { for (let i = 0; i < n; i++) numChanged += examine(i); }
        else { for (let i = 0; i < n; i++) if (alpha[i] > 0 && alpha[i] < Ci(i)) numChanged += examine(i); }
        if (examineAll) examineAll = false; else if (numChanged === 0) examineAll = true;
      }
      this.alpha = alpha; this.b = b;
      this.sv = []; for (let i = 0; i < n; i++) if (alpha[i] > 0) this.sv.push(i);
      this.kernelEvals = cache.size;
      if (this.kernel.linear) {
        const p = X[0] ? X[0].length : 0;
        this.weights = U.zeros(p);
        for (const i of this.sv) for (let j = 0; j < p; j++) this.weights[j] += y[i] * alpha[i] * X[i][j];
      }
      return this;
    }
    output(x) {
      if (this.weights) { let s = 0; for (let j = 0; j < x.length; j++) s += this.weights[j] * x[j]; return s - this.b; }
      let s = 0; for (const i of this.sv) s += this.y[i] * this.alpha[i] * this.kernel(this.X[i], x);
      return s - this.b;
    }
  }
  class SMO {
    constructor(o = {}) {
      this.C = o.C ?? 1; this.L = o.L ?? 0.001; this.P = o.P ?? 1e-12; this.N = o.N ?? 0; this.seed = o.W ?? 1;
      this.kernelType = o.kernel || 'poly'; this.E = o.E ?? 1; this.G = o.G ?? 0.01;
    }
    build(data) {
      data = data.withoutMissingClass();
      const fl = [new W.ReplaceMissingValues(), new W.NominalToBinary({ N: true })];
      if (this.N === 0) fl.push(new W.Normalize());
      else if (this.N === 1) fl.push(W.findFilter('Standardize').create({}));
      this.pipe = new Pipeline(fl);
      const d = this.pipe.fit(data);
      this.header = d.emptyCopy();
      const ci = d.classIndex, nc = d.numClasses;
      this.preds = []; for (let j = 0; j < d.numAttributes; j++) if (j !== ci) this.preds.push(j);
      const vec = (r) => this.preds.map((j) => r[j]);
      const E = this.E, G = this.G;
      let kernel;
      if (this.kernelType === 'rbf') { kernel = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) { const t = a[k] - b[k]; s += t * t; } return Math.exp(-G * s); }; }
      else { kernel = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k]; return E === 1 ? s : Math.pow(s, E); }; kernel.linear = E === 1; }
      this.kernel = kernel;
      const rand = new W.JavaRandom(this.seed);
      this.machines = U.matrix(nc, nc, null);
      for (let i = 0; i < nc; i++) for (let j = i + 1; j < nc; j++) {
        const X = [], y = [], w = [];
        for (let k = 0; k < d.rows.length; k++) { const c = d.rows[k][ci]; if (c === i || c === j) { X.push(vec(d.rows[k])); y.push(c === i ? -1 : 1); w.push(d.weights[k]); } }
        if (!X.length) continue;
        this.machines[i][j] = new BinarySMO(this.C, this.L, this.P, kernel).train(X, y, w, rand);
      }
      this.vec = vec;
      return this;
    }
    distributionForRow(row) {
      const x = this.vec(this.pipe.row(row)), nc = this.header.numClasses, res = U.zeros(nc);
      for (let i = 0; i < nc; i++) for (let j = i + 1; j < nc; j++) {
        const m = this.machines[i][j];
        if (!m) continue;
        if (m.output(x) < 0) res[i] += 1; else res[j] += 1;
      }
      return U.normalize(res);
    }
    toString() {
      const h = this.header, cv = h.classAttr.values, nc = cv.length;
      const t = ['SMO', '', 'Kernel used:', this.kernelType === 'rbf' ? `  RBF Kernel: K(x,y) = exp(-${this.G}*(x-y)^2)` : this.E === 1 ? '  Linear Kernel: K(x,y) = <x,y>' : `  Poly Kernel: K(x,y) = <x,y>^${this.E}`, '', 'Classifier for classes: '];
      t.pop();
      const pref = this.N === 0 ? '(normalized) ' : this.N === 1 ? '(standardized) ' : '';
      for (let i = 0; i < nc; i++) for (let j = i + 1; j < nc; j++) {
        const m = this.machines[i][j];
        if (!m) continue;
        t.push('', `Classifier for classes: ${cv[i]}, ${cv[j]}`, '', 'BinarySMO', '');
        if (m.weights) {
          t.push('Machine linear: showing attribute weights, not support vectors.', '');
          const lines = [];
          this.preds.forEach((p, k) => { if (m.weights[k] !== 0) lines.push(dts(m.weights[k], 12, 4) + ' * ' + pref + h.attributes[p].name); });
          t.push(lines.map((l, k) => (k === 0 ? '   ' : ' + ') + l).join('\n'));
          t.push(' - ' + dts(m.b, 12, 4));
        } else {
          t.push(`Support vectors: ${m.sv.length}`, ' - ' + dts(m.b, 12, 4));
        }
        t.push('', 'Number of support vectors: ' + m.sv.length);
      }
      return t.join('\n') + '\n';
    }
  }
  W.registerClassifier({
    name: 'SMO', group: 'functions', path: 'weka.classifiers.functions.SMO',
    desc: L`Support Vector Machine (αλγόριθμος SMO του Platt). Πολυκλασικό με ζεύγη κλάσεων (1-vs-1).`,
    caps: { numericClass: false },
    options: [
      { key: 'C', flag: 'C', label: 'Complexity C', type: 'number', default: 1, min: 0 },
      { key: 'N', flag: 'N', label: L`Προεπεξεργασία`, type: 'select', default: 0, numeric: true, choices: { 0: 'Normalize', 1: 'Standardize', 2: L`Καμία` } },
      { key: 'kernel', label: 'Kernel', type: 'select', default: 'poly', choices: { poly: 'PolyKernel', rbf: 'RBFKernel' } },
      { key: 'E', label: L`Εκθέτης (PolyKernel)`, type: 'number', default: 1, min: 1 },
      { key: 'G', label: 'Gamma (RBFKernel)', type: 'number', default: 0.01, min: 0 },
      { key: 'L', flag: 'L', label: 'Tolerance', type: 'number', default: 0.001, min: 0 },
      { key: 'W', flag: 'W', label: 'Seed', type: 'int', default: 1 },
    ],
    create: (o) => new SMO(o),
  });

  // ===================================================== MultilayerPerceptron
  class MultilayerPerceptron {
    constructor(o = {}) {
      this.H = o.H ?? 'a'; this.lr = o.L ?? 0.3; this.mom = o.M ?? 0.2; this.epochs = o.N ?? 500; this.seed = o.S ?? 0; this.decay = !!o.D;
    }
    build(data) {
      data = data.withoutMissingClass();
      this.pipe = new Pipeline([new W.ReplaceMissingValues(), new W.NominalToBinary({ N: true })]);
      const d = this.pipe.fit(data);
      this.header = d.emptyCopy();
      const ci = d.classIndex, nominal = d.classIsNominal();
      this.nominal = nominal;
      this.preds = []; for (let j = 0; j < d.numAttributes; j++) if (j !== ci) this.preds.push(j);
      // κανονικοποίηση σε [-1,1]
      this.base = []; this.range = [];
      for (const j of this.preds.concat(nominal ? [] : [ci])) {
        let mn = Infinity, mx = -Infinity;
        for (const r of d.rows) { const v = r[j]; if (v === v) { if (v < mn) mn = v; if (v > mx) mx = v; } }
        if (!isFinite(mn)) { mn = 0; mx = 0; }
        this.base[j] = (mx + mn) / 2; this.range[j] = (mx - mn) / 2 || 1;
      }
      const nIn = this.preds.length, nOut = nominal ? d.numClasses : 1;
      const hidden = String(this.H).split(',').map((s) => s.trim()).filter(Boolean).map((s) => {
        if (s === 'a') return Math.max(1, Math.floor((nIn + nOut) / 2));
        if (s === 'i') return nIn; if (s === 'o') return nOut; if (s === 't') return nIn + nOut;
        const v = parseInt(s, 10); return isNaN(v) ? 0 : v;
      }).filter((v) => v > 0);
      this.layers = [nIn, ...hidden, nOut];
      const rand = new W.JavaRandom(this.seed);
      this.W = []; this.dW = [];
      for (let l = 1; l < this.layers.length; l++) {
        const rows = this.layers[l], cols = this.layers[l - 1] + 1;
        this.W.push(Array.from({ length: rows }, () => Array.from({ length: cols }, () => rand.nextDouble() * 0.1 - 0.05)));
        this.dW.push(U.matrix(rows, cols));
      }
      const order = d.copy(); order.randomize(rand);
      const X = order.rows.map((r) => this.input(r));
      const T = order.rows.map((r) => { if (nominal) { const t = U.zeros(nOut); t[r[ci]] = 1; return t; } return [(r[ci] - this.base[ci]) / this.range[ci]]; });
      const ws = order.weights;
      let lr = this.lr;
      for (let ep = 1; ep <= this.epochs; ep++) {
        if (this.decay) lr = this.lr / ep;
        let errSum = 0;
        for (let i = 0; i < X.length; i++) errSum += this.train(X[i], T[i], ws[i], lr);
        this.lastError = errSum / Math.max(1, X.length);
      }
      return this;
    }
    input(r) { return this.preds.map((j) => (r[j] - this.base[j]) / this.range[j]); }
    forward(x) {
      const acts = [x];
      for (let l = 0; l < this.W.length; l++) {
        const prev = acts[l], out = new Array(this.W[l].length), last = l === this.W.length - 1;
        for (let u = 0; u < out.length; u++) {
          const w = this.W[l][u];
          let s = w[prev.length];
          for (let k = 0; k < prev.length; k++) s += w[k] * prev[k];
          out[u] = last && !this.nominal ? s : 1 / (1 + Math.exp(-s));
        }
        acts.push(out);
      }
      return acts;
    }
    train(x, t, wt, lr) {
      const acts = this.forward(x), L = this.W.length;
      let deltas = acts[L].map((o, u) => (this.nominal ? (t[u] - o) * o * (1 - o) : t[u] - o));
      let err = 0; acts[L].forEach((o, u) => { err += 0.5 * (t[u] - o) * (t[u] - o); });
      for (let l = L - 1; l >= 0; l--) {
        const prev = acts[l];
        let nd = null;
        if (l > 0) {
          nd = new Array(prev.length).fill(0);
          for (let u = 0; u < deltas.length; u++) for (let k = 0; k < prev.length; k++) nd[k] += deltas[u] * this.W[l][u][k];
          for (let k = 0; k < prev.length; k++) nd[k] *= prev[k] * (1 - prev[k]);
        }
        for (let u = 0; u < deltas.length; u++) {
          const w = this.W[l][u], dw = this.dW[l][u], g = lr * deltas[u] * wt;
          for (let k = 0; k < prev.length; k++) { const c = g * prev[k] + this.mom * dw[k]; w[k] += c; dw[k] = c; }
          const c = g + this.mom * dw[prev.length]; w[prev.length] += c; dw[prev.length] = c;
        }
        deltas = nd;
      }
      return err * wt;
    }
    distributionForRow(row) {
      const r = this.pipe.row(row), out = this.forward(this.input(r)).pop();
      if (!this.nominal) return [out[0] * this.range[this.header.classIndex] + this.base[this.header.classIndex]];
      const s = U.sum(out);
      return s > 0 ? out.map((v) => v / s) : out.map(() => 1 / out.length);
    }
    toString() {
      const h = this.header, t = [];
      const names = this.preds.map((j) => h.attributes[j].name);
      const L = this.W.length;
      for (let l = L - 1; l >= 0; l--) {
        this.W[l].forEach((w, u) => {
          const isOut = l === L - 1;
          t.push((isOut ? (this.nominal ? 'Sigmoid Node ' : 'Linear Node ') + u : 'Sigmoid Node ' + (u + (l === 0 ? 0 : 0)) + ' (layer ' + (l + 1) + ')'));
          t.push('    Inputs    Weights');
          t.push('    Threshold    ' + U.d2s(-w[w.length - 1], 6));
          for (let k = 0; k < w.length - 1; k++) t.push('    ' + (l === 0 ? 'Attrib ' + names[k] : 'Node ' + k) + '    ' + U.d2s(w[k], 6));
        });
      }
      if (this.nominal) { t.push('Class ' + h.classAttr.values.map((v, k) => k).join(' ')); h.classAttr.values.forEach((v, k) => t.push('    Input', '    Node ' + k + ' → ' + v)); }
      t.push('', L`Τοπολογία: ` + this.layers.join(' - ') + L` · σφάλμα τελευταίας εποχής: ` + U.d2s(this.lastError, 6));
      return t.join('\n') + '\n';
    }
  }
  W.registerClassifier({
    name: 'MultilayerPerceptron', group: 'functions', path: 'weka.classifiers.functions.MultilayerPerceptron',
    desc: L`Νευρωνικό δίκτυο με backpropagation (σιγμοειδείς μονάδες, momentum).`,
    options: [
      { key: 'H', flag: 'H', label: L`Κρυφά επίπεδα (a, i, o, t ή αριθμοί χωρισμένοι με κόμμα)`, type: 'text', default: 'a' },
      { key: 'L', flag: 'L', label: 'Learning rate', type: 'number', default: 0.3, min: 0, step: 0.05 },
      { key: 'M', flag: 'M', label: 'Momentum', type: 'number', default: 0.2, min: 0, step: 0.05 },
      { key: 'N', flag: 'N', label: L`Εποχές`, type: 'int', default: 500, min: 1 },
      { key: 'D', flag: 'D', label: L`Decay του learning rate`, type: 'bool', default: false },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 0 },
    ],
    create: (o) => new MultilayerPerceptron(o),
  });
})();
