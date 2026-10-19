/* Weka Web — ομαδοποίηση: SimpleKMeans, EM, FarthestFirst, HierarchicalClusterer + ClusterEvaluation. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, S = W.Stats;
  W.clusterers = [];
  const reg = (s) => { W.clusterers.push(s); return s; };
  W.findClusterer = (n) => W.clusterers.find((s) => s.name === n);

  /** Ευκλείδεια/Manhattan απόσταση με κανονικοποίηση (όπως weka.core.EuclideanDistance, χωρίς skip κλάσης). */
  class NormDistance {
    constructor(d, manhattan = false) {
      this.atts = d.attributes; this.manhattan = manhattan;
      this.min = []; this.max = [];
      d.attributes.forEach((a, j) => {
        let mn = NaN, mx = NaN;
        if (a.type !== 'nominal') for (const r of d.rows) { const v = r[j]; if (v === v) { if (!(v >= mn)) mn = v; if (!(v <= mx)) mx = v; } }
        this.min.push(mn); this.max.push(mx);
      });
    }
    norm(j, v) { const w = this.max[j] - this.min[j]; return w === 0 || w !== w ? 0 : (v - this.min[j]) / w; }
    diff(j, a, b) {
      if (this.atts[j].type === 'nominal') return a !== a || b !== b || a !== b ? 1 : 0;
      const am = a !== a, bm = b !== b;
      if (am && bm) return 1;
      if (am || bm) { let d = am ? this.norm(j, b) : this.norm(j, a); return d < 0.5 ? 1 - d : d; }
      return this.norm(j, a) - this.norm(j, b);
    }
    distance(x, y) {
      let s = 0;
      for (let j = 0; j < x.length; j++) { const d = this.diff(j, x[j], y[j]); s += this.manhattan ? Math.abs(d) : d * d; }
      return this.manhattan ? s : Math.sqrt(s);
    }
  }
  W.NormDistance = NormDistance;

  function rowKey(r) { return r.map((v) => (v !== v ? '?' : v)).join('\u0001'); }

  // ============================================================ SimpleKMeans
  class SimpleKMeans {
    constructor(o = {}) {
      this.N = o.N ?? 2; this.seed = o.S ?? 10; this.maxIt = o.I ?? 500; this.init = o.init ?? 0; this.manhattan = o.distance === 'manhattan';
      this.dontReplace = !!o.M; this.showStd = !!o.V;
    }
    build(data) {
      let d = data.copy(); d.classIndex = -1;
      if (!this.dontReplace) { this.rmv = new W.ReplaceMissingValues().setInput(d); d = this.rmv.apply(d); }
      this.header = d.emptyCopy();
      this.full = this.centroid(d.rows, d.weights, d);
      this.dist = new NormDistance(d, this.manhattan);
      const rand = new W.JavaRandom(this.seed);
      let cents = [];
      const rows = d.rows.slice();
      if (this.init === 1) cents = this.kmeansPP(d, rand);
      else if (this.init === 3) cents = this.farthestInit(d);
      else {
        const seen = new Set();
        for (let j = rows.length - 1; j >= 0; j--) {
          const idx = rand.nextInt(j + 1);
          const k = rowKey(rows[idx]);
          if (!seen.has(k)) { cents.push(rows[idx].slice()); seen.add(k); }
          const t = rows[j]; rows[j] = rows[idx]; rows[idx] = t;
          if (cents.length === this.N) break;
        }
      }
      this.initial = cents.map((c) => c.slice());
      let k = cents.length;
      const assign = new Array(d.numInstances).fill(0);
      let converged = false, it = 0, groups;
      while (!converged) {
        it++; converged = true;
        for (let i = 0; i < d.numInstances; i++) {
          const c = this.nearest(d.rows[i], cents, k)[0];
          if (c !== assign[i]) converged = false;
          assign[i] = c;
        }
        groups = Array.from({ length: k }, () => ({ rows: [], ws: [], idx: [] }));
        for (let i = 0; i < d.numInstances; i++) { const g = groups[assign[i]]; g.rows.push(d.rows[i]); g.ws.push(d.weights[i]); g.idx.push(i); }
        const newC = [];
        const keep = [];
        groups.forEach((g, gi) => { if (g.rows.length) { newC.push(this.centroid(g.rows, g.ws, d)); keep.push(gi); } });
        if (it === this.maxIt) converged = true;
        if (keep.length < k) {
          // αφαίρεση κενών ομάδων & επαναρίθμηση
          const remap = new Map(keep.map((g, i) => [g, i]));
          for (let i = 0; i < assign.length; i++) assign[i] = remap.has(assign[i]) ? remap.get(assign[i]) : 0;
          groups = keep.map((g) => groups[g]);
          k = keep.length;
        }
        cents = newC;
      }
      this.iterations = it; this.centroids = cents; this.k = k;
      this.sse = U.zeros(k);
      for (let i = 0; i < d.numInstances; i++) {
        const [c, dist] = this.nearest(d.rows[i], cents, k);
        this.sse[c] += this.manhattan ? dist : dist * dist * d.weights[i];
      }
      this.sizes = groups.map((g) => U.sum(g.ws));
      this.groupStd = groups.map((g) => this.stdDevs(g.rows, g.ws, d));
      this.fullStd = this.stdDevs(d.rows, d.weights, d);
      this.fullSize = d.sumOfWeights();
      return this;
    }
    kmeansPP(d, rand) {
      const n = d.numInstances, cents = [d.rows[rand.nextInt(n)].slice()];
      const md = d.rows.map((r) => this.dist.distance(r, cents[0]) ** 2);
      while (cents.length < this.N) {
        const tot = U.sum(md);
        if (tot === 0) break;
        let r = rand.nextDouble() * tot, idx = 0;
        for (; idx < n - 1; idx++) { r -= md[idx]; if (r <= 0) break; }
        cents.push(d.rows[idx].slice());
        for (let i = 0; i < n; i++) md[i] = Math.min(md[i], this.dist.distance(d.rows[i], d.rows[idx]) ** 2);
      }
      return cents;
    }
    farthestInit(d) { const ff = new FarthestFirst({ N: this.N, S: this.seed }); ff.build(d); return ff.centers.map((c) => c.slice()); }
    nearest(r, cents, k) {
      let best = 0, md = 2147483647;
      for (let i = 0; i < k; i++) { const dd = this.dist.distance(r, cents[i]); if (dd < md) { md = dd; best = i; } }
      return [best, md];
    }
    centroid(rows, ws, d) {
      const na = d.numAttributes, vals = U.zeros(na), wm = U.zeros(na), wn = U.zeros(na);
      const nd = d.attributes.map((a) => (a.type === 'nominal' ? U.zeros(a.values.length) : null));
      for (let i = 0; i < rows.length; i++) {
        const r = rows[i], w = ws[i];
        for (let j = 0; j < na; j++) {
          if (r[j] !== r[j]) { wm[j] += w; continue; }
          wn[j] += w;
          if (nd[j]) nd[j][r[j]] += w; else vals[j] += w * r[j];
        }
      }
      for (let j = 0; j < na; j++) {
        if (!nd[j]) vals[j] = wn[j] > 0 ? vals[j] / wn[j] : NaN;
        else { let mx = -Infinity, mi = -1; nd[j].forEach((c, i) => { if (c > mx) { mx = c; mi = i; } }); vals[j] = mx < wm[j] ? NaN : mi; }
      }
      if (this.manhattan) {
        for (let j = 0; j < na; j++) if (!nd[j] && wn[j] > 0) {
          const v = rows.map((r) => r[j]).filter((x) => x === x).sort((a, b) => a - b);
          const m = (v.length - 1) >> 1;
          vals[j] = v.length % 2 === 0 ? (v[m] + v[m + 1]) / 2 : v[m];
        }
      }
      return vals;
    }
    stdDevs(rows, ws, d) {
      return d.attributes.map((a, j) => {
        if (a.type === 'nominal') return NaN;
        let sw = 0, s = 0, ss = 0;
        for (let i = 0; i < rows.length; i++) { const v = rows[i][j]; if (v === v) { sw += ws[i]; s += ws[i] * v; ss += ws[i] * v * v; } }
        return sw > 1 ? Math.sqrt(Math.max(0, (ss - (s * s) / sw) / (sw - 1))) : 0;
      });
    }
    clusterRow(r) { if (this.rmv) r = this.rmv.convertRow(r); return this.nearest(r, this.centroids, this.k)[0]; }
    numberOfClusters() { return this.k; }
    toString() {
      const h = this.header, t = ['', 'kMeans', '======', '', 'Number of iterations: ' + this.iterations];
      t.push((this.manhattan ? 'Sum of within cluster distances: ' : 'Within cluster sum of squared errors: ') + W.javaDouble(U.sum(this.sse)), '');
      t.push('Initial starting points (' + ['random', 'k-means++', 'canopy', 'farthest first'][this.init] + '):', '');
      this.initial.forEach((c, i) => t.push('Cluster ' + i + ': ' + c.map((v, j) => fmtVal(h, j, v)).join(',')));
      t.push('');
      if (!this.dontReplace) t.push('Missing values globally replaced with mean/mode', '');
      t.push('Final cluster centroids:');
      t.push(centroidTable(h, this.full, this.centroids, this.fullSize, this.sizes, this.showStd ? this.fullStd : null, this.showStd ? this.groupStd : null));
      return t.join('\n') + '\n';
    }
  }
  function fmtVal(h, j, v) {
    if (v !== v) return '?';
    const a = h.attributes[j];
    return a.type === 'nominal' ? a.values[v] : W.javaDouble(v);
  }
  /** Πίνακας κέντρων σε μορφή Weka. */
  function centroidTable(h, full, cents, fullSize, sizes, fullStd, std) {
    const k = cents.length;
    const cell = (j, v) => { if (v !== v) return '?'; const a = h.attributes[j]; return a.type === 'nominal' ? a.values[v] : U.d2s(v, 4); };
    let maxW = Math.max('Attribute'.length, ...h.attributes.map((a) => a.name.length)) + 1;
    let colW = Math.max(10, 'Full Data'.length, ...h.attributes.map((a, j) => {
      let m = cell(j, full[j]).length;
      for (const c of cents) m = Math.max(m, cell(j, c[j]).length);
      return m;
    })) + 1;
    colW = Math.max(colW, ('(' + U.d2s(fullSize, 1) + ')').length + 1);
    const pad = (s) => U.padL(s, colW);
    const lines = [];
    lines.push(U.padR('', maxW) + pad('') + U.padL('Cluster#', colW * Math.max(1, k)).slice(-(colW * Math.max(1, k))));
    lines.push(U.padR('Attribute', maxW) + pad('Full Data') + cents.map((_, i) => pad(String(i))).join(''));
    lines.push(U.padR('', maxW) + pad('(' + W.javaDouble(U.roundDouble(fullSize, 2)) + ')') + sizes.map((s) => pad('(' + W.javaDouble(U.roundDouble(s, 2)) + ')')).join(''));
    lines.push('='.repeat(maxW + colW * (k + 1)));
    h.attributes.forEach((a, j) => {
      lines.push(U.padR(a.name, maxW) + pad(cell(j, full[j])) + cents.map((c) => pad(cell(j, c[j]))).join(''));
      if (fullStd && a.type !== 'nominal') lines.push(U.padR('', maxW) + pad('+/-' + U.d2s(fullStd[j], 4)) + std.map((s) => pad('+/-' + U.d2s(s[j], 4))).join(''));
    });
    return lines.join('\n');
  }
  W.centroidTable = centroidTable;
  reg({
    name: 'SimpleKMeans', path: 'weka.clusterers.SimpleKMeans',
    desc: L`Κλασικός k-means με κανονικοποιημένη Ευκλείδεια απόσταση. Τα nominal κέντρα είναι η επικρατούσα τιμή.`,
    options: [
      { key: 'N', flag: 'N', label: L`Πλήθος ομάδων (k)`, type: 'int', default: 2, min: 1 },
      { key: 'init', flag: 'init', label: L`Αρχικοποίηση`, type: 'select', default: 0, numeric: true, choices: { 0: 'Random', 1: 'k-means++', 3: 'Farthest first' } },
      { key: 'distance', label: L`Απόσταση`, type: 'select', default: 'euclidean', choices: { euclidean: L`Ευκλείδεια`, manhattan: L`Manhattan (διάμεσος)` } },
      { key: 'I', flag: 'I', label: L`Μέγιστες επαναλήψεις`, type: 'int', default: 500, min: 1 },
      { key: 'V', flag: 'V', label: L`Εμφάνιση τυπικών αποκλίσεων`, type: 'bool', default: false },
      { key: 'M', flag: 'M', label: L`Χωρίς αντικατάσταση ελλιπών`, type: 'bool', default: false },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 10 },
    ],
    create: (o) => new SimpleKMeans(o),
  });

  // ============================================================ FarthestFirst
  class FarthestFirst {
    constructor(o = {}) { this.N = o.N ?? 2; this.seed = o.S ?? 1; }
    build(data) {
      const d = data.copy(); d.classIndex = -1;
      this.header = d.emptyCopy();
      this.dist = new NormDistance(d);
      const n = d.numInstances, rand = new W.JavaRandom(this.seed);
      const k = Math.min(this.N, n);
      const centers = [];
      const first = rand.nextInt(n);
      centers.push(d.rows[first]);
      const md = d.rows.map((r) => this.dist.distance(r, d.rows[first]));
      const used = new Set([first]);
      while (centers.length < k) {
        let best = -1, bd = -1;
        for (let i = 0; i < n; i++) if (!used.has(i) && md[i] > bd) { bd = md[i]; best = i; }
        if (best < 0) break;
        used.add(best); centers.push(d.rows[best]);
        for (let i = 0; i < n; i++) md[i] = Math.min(md[i], this.dist.distance(d.rows[i], d.rows[best]));
      }
      this.centers = centers;
      return this;
    }
    clusterRow(r) { let best = 0, bd = Infinity; this.centers.forEach((c, i) => { const dd = this.dist.distance(r, c); if (dd < bd) { bd = dd; best = i; } }); return best; }
    numberOfClusters() { return this.centers.length; }
    toString() {
      const h = this.header;
      return 'FarthestFirst\n==============\n\nCluster centroids:\n\n' + this.centers.map((c, i) => 'Cluster ' + i + '\n\t ' + c.map((v, j) => fmtVal(h, j, v)).join(' ')).join('\n') + '\n';
    }
  }
  reg({
    name: 'FarthestFirst', path: 'weka.clusterers.FarthestFirst',
    desc: L`Επιλέγει διαδοχικά ως κέντρο το πιο απομακρυσμένο σημείο (γρήγορη ευρετική του Hochbaum & Shmoys).`,
    options: [
      { key: 'N', flag: 'N', label: L`Πλήθος ομάδων`, type: 'int', default: 2, min: 1 },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 1 },
    ],
    create: (o) => new FarthestFirst(o),
  });

  // ====================================================================== EM
  class EM {
    constructor(o = {}) { this.N = o.N ?? -1; this.maxIt = o.I ?? 100; this.minStd = o.M ?? 1e-6; this.seed = o.S ?? 100; this.maxClusters = o.max ?? -1; this.folds = o.X ?? 10; }
    build(data) { return this.buildSync(data); }
    /** Ασύγχρονη εκδοχή: δίνει χρόνο στο UI ανάμεσα στα folds και αναφέρει πρόοδο. */
    async buildAsync(data, progress) {
      this.progress = progress || (() => {});
      this.async = true;
      return this.buildSync(data);
    }
    buildSync(data) {
      let d = data.copy(); d.classIndex = -1;
      this.header = d.emptyCopy();
      if (this.N > 0) { this.fit(d, this.N); return this.async ? Promise.resolve(this) : this; }
      if (this.async) return this.cvSelectAsync(d);
      else {
        // επιλογή k με cross-validated log-likelihood (όπως το Weka: αυξάνει το k όσο βελτιώνεται)
        let k = 1, bestLL = -Infinity, bestK = 1;
        const maxK = this.maxClusters > 0 ? this.maxClusters : Math.min(d.numInstances, 20);
        const folds = Math.min(this.folds, d.numInstances);
        for (; k <= maxK; k++) {
          const cv = d.copy(); cv.randomize(new W.JavaRandom(this.seed));
          let ll = 0;
          for (let f = 0; f < folds; f++) {
            const tr = cv.trainCV(folds, f), te = cv.testCV(folds, f);
            const m = new EM({ N: k, I: this.maxIt, M: this.minStd, S: this.seed });
            m.header = this.header; m.fit(tr, k);
            for (let i = 0; i < te.numInstances; i++) ll += te.weights[i] * m.logDensity(te.rows[i]);
          }
          ll /= d.sumOfWeights();
          if (ll > bestLL) { bestLL = ll; bestK = k; } else break;
        }
        this.cvK = bestK;
        this.fit(d, bestK);
      }
      return this;
    }
    async cvSelectAsync(d) {
      let bestLL = -Infinity, bestK = 1;
      const maxK = this.maxClusters > 0 ? this.maxClusters : Math.min(d.numInstances, 20);
      const folds = Math.min(this.folds, d.numInstances);
      for (let k = 1; k <= maxK; k++) {
        const cv = d.copy(); cv.randomize(new W.JavaRandom(this.seed));
        let ll = 0;
        for (let f = 0; f < folds; f++) {
          this.progress(Math.min(0.95, (k - 1 + f / folds) / Math.max(4, k + 1)), L`EM: cross-validating k = ${k} (fold ${f + 1}/${folds})`);
          await W.nextFrame();
          const tr = cv.trainCV(folds, f), te = cv.testCV(folds, f);
          const m = new EM({ N: k, I: this.maxIt, M: this.minStd, S: this.seed });
          m.header = this.header; m.fit(tr, k);
          for (let i = 0; i < te.numInstances; i++) ll += te.weights[i] * m.logDensity(te.rows[i]);
        }
        ll /= d.sumOfWeights();
        if (ll > bestLL) { bestLL = ll; bestK = k; } else break;
      }
      this.cvK = bestK;
      this.fit(d, bestK);
      return this;
    }
    fit(d, k) {
      this.k = k;
      const na = d.numAttributes, n = d.numInstances;
      this.nominal = d.attributes.map((a) => a.type === 'nominal');
      // αρχικοποίηση με k-means (όπως το Weka)
      const km = new SimpleKMeans({ N: k, S: this.seed }).build(d);
      this.k = k = km.k;
      const resp = d.rows.map((r) => { const v = U.zeros(k); v[km.clusterRow(r)] = 1; return v; });
      // ελάχιστη τυπική απόκλιση ανά χαρακτηριστικό
      this.minSd = d.attributes.map((a, j) => { if (this.nominal[j]) return 0; const st = d.attributeStats(j); return Math.max(this.minStd, (st.stdDev || 0) * 0.0 + this.minStd); });
      let prevLL = -Infinity;
      this.iterations = 0;
      for (let it = 0; it < this.maxIt; it++) {
        this.mStep(d, resp);
        let ll = 0;
        for (let i = 0; i < n; i++) {
          const lp = this.logJoint(d.rows[i]), ri = resp[i];
          let mx = -Infinity; for (let c = 0; c < k; c++) if (lp[c] > mx) mx = lp[c];
          let s = 0; for (let c = 0; c < k; c++) { const e = Math.exp(lp[c] - mx); ri[c] = e; s += e; }
          for (let c = 0; c < k; c++) ri[c] /= s;
          ll += d.weights[i] * (mx + Math.log(s));
        }
        ll /= d.sumOfWeights();
        this.iterations = it + 1;
        if (it > 0 && ll - prevLL < 1e-6) { prevLL = ll; break; }
        prevLL = ll;
      }
      this.loglik = prevLL;
      this.mStep(d, resp);
      this.respSums = U.zeros(k);
      for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) this.respSums[c] += resp[i][c] * d.weights[i];
      return this;
    }
    mStep(d, resp) {
      const k = this.k, na = d.numAttributes;
      this.prior = U.zeros(k);
      this.model = Array.from({ length: k }, () => d.attributes.map((a) => (a.type === 'nominal' ? { counts: new Array(a.values.length).fill(1), tot: a.values.length } : { sw: 0, s: 0, ss: 0, mean: 0, sd: 0 })));
      for (let i = 0; i < d.numInstances; i++) {
        const r = d.rows[i], w = d.weights[i];
        for (let c = 0; c < k; c++) {
          const p = resp[i][c] * w;
          this.prior[c] += p;
          if (p < 1e-12) continue;
          for (let j = 0; j < na; j++) {
            const v = r[j]; if (v !== v) continue;
            const m = this.model[c][j];
            if (this.nominal[j]) { m.counts[v] += p; m.tot += p; } else { m.sw += p; m.s += p * v; m.ss += p * v * v; }
          }
        }
      }
      const tot = U.sum(this.prior);
      for (let c = 0; c < k; c++) {
        this.prior[c] = (this.prior[c] + 1) / (tot + k);
        for (let j = 0; j < na; j++) {
          const m = this.model[c][j];
          if (this.nominal[j]) continue;
          m.mean = m.sw > 0 ? m.s / m.sw : 0;
          const v = m.sw > 0 ? m.ss / m.sw - m.mean * m.mean : 0;
          m.sd = Math.max(this.minSd[j], Math.sqrt(Math.max(0, v)));
        }
      }
    }
    /** Προϋπολογισμός λογαρίθμων για γρήγορο E-step. */
    cacheLogs() {
      const k = this.k;
      this.lp = this.prior.map((p) => Math.log(p));
      this.lc = []; this.lnorm = []; this.inv2v = [];
      for (let c = 0; c < k; c++) {
        const m = this.model[c];
        this.lc.push(m.map((x, j) => (this.nominal[j] ? x.counts.map((v) => Math.log(v / x.tot)) : null)));
        this.lnorm.push(m.map((x, j) => (this.nominal[j] ? 0 : -Math.log(x.sd) - 0.5 * Math.log(2 * Math.PI))));
        this.inv2v.push(m.map((x, j) => (this.nominal[j] ? 0 : 1 / (2 * x.sd * x.sd))));
      }
    }
    logJoint(r) {
      if (!this.lp || this.lpStamp !== this.model) { this.cacheLogs(); this.lpStamp = this.model; }
      const out = new Array(this.k);
      for (let c = 0; c < this.k; c++) {
        let lp = this.lp[c];
        const lc = this.lc[c], ln = this.lnorm[c], iv = this.inv2v[c], m = this.model[c];
        for (let j = 0; j < r.length; j++) {
          const v = r[j]; if (v !== v) continue;
          if (lc[j]) lp += lc[j][v];
          else { const dlt = v - m[j].mean; lp += ln[j] - dlt * dlt * iv[j]; }
        }
        out[c] = lp;
      }
      return out;
    }
    logDensity(r) { const lp = this.logJoint(r); const mx = Math.max(...lp); let s = 0; for (const v of lp) s += Math.exp(v - mx); return mx + Math.log(s); }
    distributionForRow(r) { const lp = this.logJoint(r); const mx = Math.max(...lp); return U.normalize(lp.map((v) => Math.exp(v - mx))); }
    clusterRow(r) { return U.maxIndex(this.distributionForRow(r)); }
    numberOfClusters() { return this.k; }
    toString() {
      const h = this.header, k = this.k, t = ['EM', '==', ''];
      if (this.cvK) t.push('Number of clusters selected by cross validation: ' + this.cvK);
      t.push('Number of iterations performed: ' + this.iterations, '');
      const colW = 14, nameW = Math.max(12, ...h.attributes.map((a) => a.name.length + 2), ...h.attributes.flatMap((a) => (a.values ? a.values.map((v) => v.length + 4) : []))) ;
      const row = (l, cells) => U.padR(l, nameW) + cells.map((c) => U.padL(c, colW)).join('');
      t.push(row('', Array.from({ length: k }, (_, c) => (c === 0 ? 'Cluster' : ''))));
      t.push(row('Attribute', Array.from({ length: k }, (_, c) => String(c))));
      t.push(row('', this.prior.map((p) => '(' + U.d2s(p, 2) + ')')));
      t.push('='.repeat(nameW + colW * k));
      h.attributes.forEach((a, j) => {
        t.push(a.name);
        if (a.type === 'nominal') {
          a.values.forEach((v, vi) => t.push(row('  ' + v, this.model.map((m) => U.d2s(m[j].counts[vi], 4)))));
          t.push(row('  [total]', this.model.map((m) => U.d2s(m[j].tot, 4))));
        } else {
          t.push(row('  mean', this.model.map((m) => U.d2s(m[j].mean, 4))));
          t.push(row('  std. dev.', this.model.map((m) => U.d2s(m[j].sd, 4))));
        }
        t.push('');
      });
      t.push('Log likelihood: ' + U.d2s(this.loglik, 5));
      return t.join('\n') + '\n';
    }
  }
  reg({
    name: 'EM', path: 'weka.clusterers.EM',
    desc: L`Expectation-Maximization με μίξη κανονικών/διακριτών κατανομών. Με k = −1 επιλέγει τον αριθμό ομάδων με cross-validation.`,
    options: [
      { key: 'N', flag: 'N', label: L`Πλήθος ομάδων (−1 = αυτόματα)`, type: 'int', default: -1, min: -1 },
      { key: 'max', flag: 'max', label: L`Μέγιστο k στην αυτόματη επιλογή (−1 = 20)`, type: 'int', default: -1, min: -1 },
      { key: 'I', flag: 'I', label: L`Μέγιστες επαναλήψεις`, type: 'int', default: 100, min: 1 },
      { key: 'M', flag: 'M', label: L`Ελάχιστη τυπική απόκλιση`, type: 'number', default: 1e-6, min: 0 },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 100 },
    ],
    create: (o) => new EM(o),
  });

  // ==================================================== HierarchicalClusterer
  class HierarchicalClusterer {
    constructor(o = {}) { this.N = o.N ?? 2; this.link = o.L || 'SINGLE'; }
    build(data) {
      const d = data.copy(); d.classIndex = -1;
      this.header = d.emptyCopy();
      const n = d.numInstances;
      if (n > 3000) throw new Error(L`Το HierarchicalClusterer περιορίζεται σε 3.000 εγγραφές στον browser.`);
      this.dist = new NormDistance(d);
      this.rows = d.rows;
      // πίνακας αποστάσεων
      const D = Array.from({ length: n }, () => new Float64Array(n));
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = this.dist.distance(d.rows[i], d.rows[j]); D[i][j] = v; D[j][i] = v; }
      const active = new Set([...Array(n).keys()]);
      const size = new Array(n).fill(1);
      const nodes = d.rows.map((_, i) => ({ id: i, leaf: true, size: 1, height: 0 }));
      const members = d.rows.map((_, i) => [i]);
      const merges = [];
      const L = this.link;
      while (active.size > 1) {
        let bi = -1, bj = -1, bd = Infinity;
        const arr = [...active];
        for (let a = 0; a < arr.length; a++) for (let b = a + 1; b < arr.length; b++) { const v = D[arr[a]][arr[b]]; if (v < bd) { bd = v; bi = arr[a]; bj = arr[b]; } }
        // Lance–Williams
        for (const k of active) {
          if (k === bi || k === bj) continue;
          const di = D[bi][k], dj = D[bj][k], ni = size[bi], nj = size[bj];
          let v;
          if (L === 'SINGLE') v = Math.min(di, dj);
          else if (L === 'COMPLETE') v = Math.max(di, dj);
          else if (L === 'AVERAGE') v = (ni * di + nj * dj) / (ni + nj);
          else if (L === 'WARD') { const nk = size[k]; v = Math.sqrt(Math.max(0, ((ni + nk) * di * di + (nj + nk) * dj * dj - nk * bd * bd) / (ni + nj + nk))); }
          else v = (ni * di + nj * dj) / (ni + nj);
          D[bi][k] = D[k][bi] = v;
        }
        const node = { leaf: false, left: nodes[bi], right: nodes[bj], height: bd, size: size[bi] + size[bj] };
        nodes[bi] = node; size[bi] += size[bj];
        members[bi] = members[bi].concat(members[bj]);
        active.delete(bj);
        merges.push({ height: bd, active: active.size });
      }
      this.root = nodes[[...active][0]];
      // τομή δέντρου σε N ομάδες
      const k = Math.min(this.N, n);
      let clusters = [this.root];
      while (clusters.length < k) {
        let idx = -1, h = -1;
        clusters.forEach((c, i) => { if (!c.leaf && c.height > h) { h = c.height; idx = i; } });
        if (idx < 0) break;
        const c = clusters[idx];
        clusters.splice(idx, 1, c.left, c.right);
      }
      this.assign = new Array(n).fill(0);
      const collect = (node, out) => { if (node.leaf) out.push(node.id); else { collect(node.left, out); collect(node.right, out); } };
      clusters.forEach((c, ci) => { const ids = []; collect(c, ids); ids.forEach((i) => { this.assign[i] = ci; }); c.cluster = ci; });
      this.clusters = clusters;
      this.k = clusters.length;
      return this;
    }
    clusterRow(r) {
      // πλησιέστερη ομάδα κατά τη μέθοδο σύνδεσης (μέση απόσταση για νέα σημεία)
      const sums = U.zeros(this.k), cnt = U.zeros(this.k), mins = new Array(this.k).fill(Infinity), maxs = U.zeros(this.k);
      this.rows.forEach((x, i) => { const c = this.assign[i], dd = this.dist.distance(r, x); sums[c] += dd; cnt[c]++; if (dd < mins[c]) mins[c] = dd; if (dd > maxs[c]) maxs[c] = dd; });
      const score = this.link === 'SINGLE' ? mins : this.link === 'COMPLETE' ? maxs : sums.map((s, c) => s / cnt[c]);
      return U.minIndex(score);
    }
    numberOfClusters() { return this.k; }
    newick(node = this.root) {
      if (node.leaf) return String(node.id + 1);
      return '(' + this.newick(node.left) + ':' + U.d2s(node.height - (node.left.height || 0), 5) + ',' + this.newick(node.right) + ':' + U.d2s(node.height - (node.right.height || 0), 5) + ')';
    }
    toString() {
      return this.clusters.map((c, i) => 'Cluster ' + i + '\n' + this.newick(c)).join('\n\n') + '\n';
    }
  }
  reg({
    name: 'HierarchicalClusterer', path: 'weka.clusterers.HierarchicalClusterer',
    desc: L`Ιεραρχική συσσωρευτική ομαδοποίηση με δενδρόγραμμα (single, complete, average, ward).`, dendrogram: true,
    options: [
      { key: 'N', flag: 'N', label: L`Πλήθος ομάδων`, type: 'int', default: 2, min: 1 },
      { key: 'L', flag: 'L', label: L`Μέθοδος σύνδεσης (linkage)`, type: 'select', default: 'SINGLE', choices: { SINGLE: 'Single', COMPLETE: 'Complete', AVERAGE: 'Average', WARD: 'Ward' } },
    ],
    create: (o) => new HierarchicalClusterer(o),
  });

  // ========================================================= ClusterEvaluation
  /** Βέλτιστη αντιστοίχιση ομάδων σε κλάσεις (ελάχιστα λάθη), όπως το mapClasses του Weka. */
  function mapClasses(counts, k, nc) {
    let best = null, bestErr = Infinity;
    const assign = new Array(k).fill(-1);
    const used = new Array(nc).fill(false);
    const totals = counts.map((r) => U.sum(r));
    const rec = (c, err) => {
      if (err >= bestErr) return;
      if (c === k) { bestErr = err; best = assign.slice(); return; }
      // χωρίς κλάση
      assign[c] = -1; rec(c + 1, err + totals[c]);
      for (let j = 0; j < nc; j++) {
        if (used[j]) continue;
        used[j] = true; assign[c] = j;
        rec(c + 1, err + totals[c] - counts[c][j]);
        used[j] = false;
      }
      assign[c] = -1;
    };
    if (k * nc <= 400) rec(0, 0);
    else { // άπληστη λύση για μεγάλα προβλήματα
      best = counts.map((r) => U.maxIndex(r)); bestErr = 0;
      counts.forEach((r, c) => { bestErr += totals[c] - r[best[c]]; });
    }
    return { map: best, errors: bestErr };
  }

  /**
   * Εκτέλεση clusterer. opts: { data, spec, options, mode: 'train'|'classes'|'split'|'test', ignore: [indices], percentage, seed, testData }
   */
  W.runClusterer = async function (opts) {
    const { spec, options } = opts;
    const data = opts.data;
    const mode = opts.mode || 'train';
    const ignore = new Set(opts.ignore || []);
    let classIdx = -1;
    if (mode === 'classes') {
      classIdx = data.classIndex;
      if (classIdx < 0 || !data.classIsNominal()) throw new Error(L`Η αξιολόγηση classes-to-clusters απαιτεί nominal χαρακτηριστικό κλάσης.`);
      ignore.add(classIdx);
    }
    const keep = []; for (let j = 0; j < data.numAttributes; j++) if (!ignore.has(j)) keep.push(j);
    if (!keep.length) throw new Error(L`Δεν απομένουν χαρακτηριστικά για ομαδοποίηση.`);
    for (const j of keep) if (data.attributes[j].type === 'string') throw new Error(L`Το χαρακτηριστικό "${data.attributes[j].name}" είναι τύπου string· αγνόησέ το ή μετέτρεψέ το.`);
    const rm = new W.Remove({ indices: keep, V: true });
    const base = data.copy(); base.classIndex = -1;
    rm.setInput(base);
    let train = rm.apply(base), test = null, testRowsOrig = null;
    if (mode === 'split') {
      const d = base.copy(); d.randomize(new W.JavaRandom(opts.seed ?? 1));
      const n = Math.round(d.numInstances * (opts.percentage ?? 66) / 100);
      const all = rm.apply(d);
      train = all.withRows(all.rows.slice(0, n), all.weights.slice(0, n));
      test = all.withRows(all.rows.slice(n), all.weights.slice(n));
    } else if (mode === 'test') {
      if (!opts.testData) throw new Error(L`Φόρτωσε πρώτα σύνολο ελέγχου.`);
      const msg = W.equalHeadersMsg(data, opts.testData);
      if (msg) throw new Error(L`Ασύμβατο σύνολο ελέγχου: ` + msg);
      const t = opts.testData.copy(); t.classIndex = -1;
      test = rm.apply(t);
    }
    if (opts.onProgress) opts.onProgress(0.2, L`Εκπαίδευση…`);
    await W.nextFrame();
    const t0 = performance.now();
    const model = spec.create(Object.assign({}, options));
    if (model.buildAsync) await model.buildAsync(train, opts.onProgress); else model.build(train);
    const buildTime = (performance.now() - t0) / 1000;
    const evalSet = test || train;
    const assignments = evalSet.rows.map((r) => model.clusterRow(r));
    const k = model.numberOfClusters();
    const counts = U.zeros(k);
    assignments.forEach((a, i) => { counts[a] += evalSet.weights[i]; });
    const tot = U.sum(counts);
    const optStr = W.optionString(spec, options);
    const lines = ['=== Run information ===', '',
      'Scheme:       ' + spec.path + (optStr ? ' ' + optStr : ''),
      'Relation:     ' + data.relation,
      'Instances:    ' + data.numInstances,
      'Attributes:   ' + data.numAttributes,
      ...data.attributes.map((a, j) => '              ' + a.name + (ignore.has(j) ? L`   [αγνοείται]` : '')),
      'Test mode:    ' + ({ train: 'evaluate on training data', classes: 'Classes to clusters evaluation on training data', split: `split ${opts.percentage ?? 66}% train, remainder test`, test: 'user supplied test set' })[mode], '',
      '=== Clustering model (full training set) ===', '', model.toString(), '',
      'Time taken to build model (full training data) : ' + U.d2s(buildTime, 2) + ' seconds', '',
      '=== Model and evaluation on ' + (test ? 'test set' : 'training set') + ' ===', '', 'Clustered Instances', ''];
    counts.forEach((c, i) => { if (c > 0) lines.push(i + '      ' + U.d2s(c, 0) + ' (' + U.padL(U.d2s(Math.round(100 * c / tot), 0), 3) + '%)'); });
    let classes = null;
    if (mode === 'classes') {
      const ca = data.attributes[classIdx], nc = ca.values.length;
      const m = U.matrix(k, nc);
      let unassigned = 0;
      data.rows.forEach((r, i) => { const c = r[classIdx]; if (c === c) m[assignments[i]][c] += data.weights[i]; else unassigned += data.weights[i]; });
      const map = mapClasses(m, k, nc);
      lines.push('', '', 'Class attribute: ' + ca.name, 'Classes to Clusters:', '');
      const w = Math.max(4, ...m.flat().map((v) => U.d2s(v, 0).length + 1));
      lines.push(Array.from({ length: k }, (_, c) => U.padL(c, w)).join('') + '  <-- assigned to cluster');
      for (let j = 0; j < nc; j++) lines.push(Array.from({ length: k }, (_, c) => U.padL(U.d2s(m[c][j], 0), w)).join('') + ' | ' + ca.values[j]);
      lines.push('');
      map.map.forEach((cls, c) => lines.push('Cluster ' + c + ' <-- ' + (cls < 0 ? 'No class' : ca.values[cls])));
      lines.push('', 'Incorrectly clustered instances :\t' + W.javaDouble(map.errors) + '\t ' + U.d2s(100 * map.errors / U.sum(m.flat()), 4) + ' %');
      classes = { matrix: m, map: map.map, errors: map.errors, labels: ca.values };
    }
    return { model, report: lines.join('\n') + '\n', assignments, counts, k, classes, mode, spec, options, optStr, evalRows: test ? null : data, trainHeader: train, buildTime, keep };
  };
  W.mapClasses = mapClasses;
})();
