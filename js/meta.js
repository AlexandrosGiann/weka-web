/* Weka Web — meta ταξινομητές (AdaBoostM1, Bagging, Vote) και κανόνες PART. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils;

  const baseChoices = () => {
    const o = {};
    for (const s of W.classifiers) if (!s.meta) o[s.name] = s.name;
    return o;
  };
  const makeBase = (name) => {
    const s = W.findClassifier(name);
    if (!s) throw new Error(L`Άγνωστος βασικός ταξινομητής: ` + name);
    return s.create(W.defaultOptions(s));
  };

  // ============================================================ AdaBoostM1
  class AdaBoostM1 {
    constructor(o = {}) { this.I = o.I ?? 10; this.P = o.P ?? 100; this.seed = o.S ?? 1; this.base = o.W || 'DecisionStump'; this.Q = !!o.Q; }
    build(data) {
      data = data.withoutMissingClass();
      this.header = data.emptyCopy();
      const nc = data.numClasses, ci = data.classIndex;
      if (data.numAttributes === 1) { this.zero = new W.ZeroR().build(data); return this; }
      let train = data.copy();
      const sumW = train.sumOfWeights();
      this.models = []; this.betas = [];
      const rand = new W.JavaRandom(this.seed);
      for (let it = 0; it < this.I; it++) {
        // επιλογή εγγραφών με βάση το βάρος (weight threshold P%)
        let sample = train;
        if (this.P < 100) {
          const idx = U.argsort(train.weights.map((w) => -w));
          const target = train.sumOfWeights() * this.P / 100;
          let acc = 0; const keep = [];
          for (const i of idx) { keep.push(i); acc += train.weights[i]; if (acc >= target) break; }
          sample = train.subset(keep.sort((a, b) => a - b));
        }
        if (this.Q) {
          // resampling αντί για reweighting
          const n = sample.numInstances, cum = [];
          let s = 0; for (const w of sample.weights) { s += w; cum.push(s); }
          const rows = [];
          for (let i = 0; i < n; i++) { const r = rand.nextDouble() * s; let k = cum.findIndex((c) => c >= r); if (k < 0) k = n - 1; rows.push(sample.rows[k]); }
          sample = sample.withRows(rows);
        }
        const m = makeBase(this.base).build(sample);
        let err = 0;
        const wrong = [];
        for (let i = 0; i < train.numInstances; i++) {
          const pred = U.maxIndex(m.distributionForRow(train.rows[i]));
          const bad = pred !== train.rows[i][ci];
          wrong.push(bad);
          if (bad) err += train.weights[i];
        }
        err /= train.sumOfWeights();
        if (err >= 0.5 || U.eq(err, 0)) {
          if (this.models.length === 0) { this.models.push(m); this.betas.push(1); }
          break;
        }
        const beta = Math.log((1 - err) / err);
        this.models.push(m); this.betas.push(beta);
        const reweight = (1 - err) / err;
        const nw = train.weights.map((w, i) => (wrong[i] ? w * reweight : w));
        const s = U.sum(nw);
        train = train.withRows(train.rows, nw.map((w) => w * sumW / s));
      }
      return this;
    }
    distributionForRow(r) {
      if (this.zero) return this.zero.distributionForRow(r);
      const nc = this.header.numClasses;
      if (this.models.length === 1) return this.models[0].distributionForRow(r);
      const sums = U.zeros(nc);
      this.models.forEach((m, k) => { sums[U.maxIndex(m.distributionForRow(r))] += this.betas[k]; });
      // logistic transformation όπως στο Weka (Utils.logs2probs)
      const mx = Math.max(...sums);
      return U.normalize(sums.map((v) => Math.exp(v - mx)));
    }
    toString() {
      if (this.zero) return this.zero.toString();
      const t = ['AdaBoostM1: Base classifiers and their weights: ', ''];
      this.models.forEach((m, k) => { t.push(m.toString(), '', 'Weight: ' + U.d2s(this.betas[k], 2), ''); });
      t.push('Number of performed Iterations: ' + this.models.length);
      return t.join('\n') + '\n';
    }
  }
  W.registerClassifier({
    name: 'AdaBoostM1', group: 'meta', meta: true, path: 'weka.classifiers.meta.AdaBoostM1',
    desc: L`Boosting: εκπαιδεύει διαδοχικά μοντέλα δίνοντας βάρος στα λάθη των προηγούμενων.`,
    caps: { numericClass: false },
    options: [
      { key: 'W', flag: 'W', label: L`Βασικός ταξινομητής`, type: 'select', default: 'DecisionStump', choices: baseChoices, cli: null },
      { key: 'I', flag: 'I', label: L`Επαναλήψεις`, type: 'int', default: 10, min: 1 },
      { key: 'P', flag: 'P', label: 'Weight threshold (%)', type: 'int', default: 100, min: 1, max: 100 },
      { key: 'Q', flag: 'Q', label: L`Resampling αντί για reweighting`, type: 'bool', default: false },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 1 },
    ],
    create: (o) => new AdaBoostM1(o),
  });

  // ================================================================ Bagging
  class Bagging {
    constructor(o = {}) { this.I = o.I ?? 10; this.P = o.P ?? 100; this.seed = o.S ?? 1; this.base = o.W || 'REPTree'; }
    build(data) {
      data = data.withoutMissingClass();
      this.header = data.emptyCopy();
      this.nominal = data.classIsNominal();
      const rand = new W.JavaRandom(this.seed), n = data.numInstances, size = Math.max(1, Math.round(n * this.P / 100));
      this.models = [];
      for (let it = 0; it < this.I; it++) {
        const rows = [], ws = [];
        for (let i = 0; i < size; i++) { const k = rand.nextInt(n); rows.push(data.rows[k]); ws.push(data.weights[k]); }
        const spec = W.findClassifier(this.base);
        const opts = W.defaultOptions(spec);
        if ('S' in opts) opts.S = rand.nextInt(1 << 30);
        this.models.push(spec.create(opts).build(data.withRows(rows, ws)));
      }
      return this;
    }
    distributionForRow(r) {
      const out = U.zeros(this.nominal ? this.header.numClasses : 1);
      for (const m of this.models) { const d = m.distributionForRow(r); for (let c = 0; c < out.length; c++) out[c] += d[c]; }
      if (this.nominal) return U.normalize(out);
      out[0] /= this.models.length; return out;
    }
    toString() { return L`Bagging with ${this.I} iterations and base learner\n\n${W.findClassifier(this.base).path}\n\n(Τα ${this.models.length} επιμέρους μοντέλα συνδυάζονται με μέσο όρο κατανομών.)\n`; }
  }
  W.registerClassifier({
    name: 'Bagging', group: 'meta', meta: true, path: 'weka.classifiers.meta.Bagging',
    desc: L`Bootstrap aggregating: πολλά μοντέλα σε τυχαία δείγματα με επανάθεση, με μέσο όρο προβλέψεων.`,
    options: [
      { key: 'W', flag: 'W', label: L`Βασικός ταξινομητής`, type: 'select', default: 'REPTree', choices: baseChoices },
      { key: 'I', flag: 'I', label: L`Επαναλήψεις`, type: 'int', default: 10, min: 1 },
      { key: 'P', flag: 'P', label: L`Μέγεθος bag (%)`, type: 'int', default: 100, min: 1, max: 100 },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 1 },
    ],
    create: (o) => new Bagging(o),
  });

  // =================================================================== Vote
  class Vote {
    constructor(o = {}) { this.list = (o.B || 'J48,NaiveBayes,IBk').split(',').map((s) => s.trim()).filter(Boolean); }
    build(data) {
      data = data.withoutMissingClass();
      this.header = data.emptyCopy();
      this.nominal = data.classIsNominal();
      for (const n of this.list) { const s = W.findClassifier(n); if (!s) throw new Error(L`Άγνωστος ταξινομητής στο Vote: ` + n); W.checkCapabilities(s, data); }
      this.models = this.list.map((n) => makeBase(n).build(data));
      return this;
    }
    distributionForRow(r) {
      const out = U.zeros(this.nominal ? this.header.numClasses : 1);
      for (const m of this.models) { const d = m.distributionForRow(r); for (let c = 0; c < out.length; c++) out[c] += d[c]; }
      if (this.nominal) return U.normalize(out);
      out[0] /= this.models.length; return out;
    }
    toString() { return 'Vote combines the probability distributions of these base learners:\n' + this.list.map((n) => '\t' + W.findClassifier(n).path).join('\n') + '\nusing the \'Average of Probabilities\' combination rule\n'; }
  }
  W.registerClassifier({
    name: 'Vote', group: 'meta', meta: true, path: 'weka.classifiers.meta.Vote',
    desc: L`Συνδυάζει πολλούς ταξινομητές με μέσο όρο των πιθανοτήτων τους.`,
    options: [{ key: 'B', label: L`Ταξινομητές (χωρισμένοι με κόμμα)`, type: 'text', default: 'J48,NaiveBayes,IBk' }],
    create: (o) => new Vote(o),
  });

  // =================================================================== PART
  /**
   * PART: λίστα αποφάσεων. Σε κάθε βήμα χτίζεται pruned C4.5 δέντρο στα εναπομείναντα δεδομένα,
   * το φύλλο με τη μεγαλύτερη κάλυψη γίνεται κανόνας και οι καλυπτόμενες εγγραφές αφαιρούνται.
   */
  class PART {
    constructor(o = {}) { this.CF = o.C ?? 0.25; this.M = o.M ?? 2; this.U = !!o.U; this.B = !!o.B; }
    build(data) {
      data = data.withoutMissingClass();
      this.header = data.emptyCopy();
      const nc = data.numClasses, ci = data.classIndex;
      this.rules = [];
      let rest = data.copy();
      let guard = 0;
      while (rest.numInstances > 0 && guard++ < 1000) {
        const j = new W.J48({ C: this.CF, M: this.M, U: this.U });
        j.build(rest);
        // συλλογή φύλλων: συνθήκες + κάλυψη
        const leaves = [];
        const walk = (node, conds) => {
          if (node.isLeaf) return;
          node.sons.forEach((s, i) => {
            const c = conds.concat([{ model: node.model, i }]);
            if (s.isLeaf) leaves.push({ conds: c, dist: node.model.dist, bag: i });
            else walk(s, c);
          });
        };
        if (j.root.isLeaf) {
          const dist = j.root.model.dist;
          this.rules.push({ conds: [], cls: dist.maxClass(), covered: dist.total, errors: dist.numIncorrect(), dist: dist.perClass.map((v) => v / (dist.total || 1)) });
          break;
        }
        walk(j.root, []);
        let best = null;
        for (const l of leaves) { const cov = l.dist.perBag[l.bag]; if (!best || cov > best.cov) best = { ...l, cov }; }
        const matches = (r) => best.conds.every(({ model, i }) => model.whichSubset(r) === i);
        const keep = [], cover = [];
        rest.rows.forEach((r, k) => (matches(r) ? cover : keep).push(k));
        if (!cover.length) { // εκφυλισμένη περίπτωση: προεπιλεγμένος κανόνας
          const d = W.C45Distribution.of(rest);
          this.rules.push({ conds: [], cls: d.maxClass(), covered: d.total, errors: d.numIncorrect(), dist: d.perClass.map((v) => v / (d.total || 1)) });
          break;
        }
        const counts = U.zeros(nc); let tot = 0;
        for (const k of cover) { counts[rest.rows[k][ci]] += rest.weights[k]; tot += rest.weights[k]; }
        const cls = U.maxIndex(counts);
        this.rules.push({ conds: best.conds.map((c) => ({ model: c.model, i: c.i, text: c.model.leftSide(this.header) + c.model.rightSide(c.i, this.header) })), cls, covered: tot, errors: tot - counts[cls], dist: counts.map((v) => v / tot) });
        rest = rest.subset(keep);
      }
      return this;
    }
    distributionForRow(r) {
      for (const rule of this.rules) {
        if (rule.conds.every(({ model, i }) => model.whichSubset(r) === i || (model.whichSubset(r) === -1 && false))) return rule.dist.slice();
      }
      // ελλιπείς τιμές ή καμία αντιστοίχιση: τελευταίος κανόνας
      return this.rules[this.rules.length - 1].dist.slice();
    }
    toString() {
      const cv = this.header.classAttr.values, t = ['PART decision list', '------------------', ''];
      for (const r of this.rules) {
        const tail = cv[r.cls] + ' (' + W.javaDouble(U.roundDouble(r.covered, 2)) + (U.gr(r.errors, 0) ? '/' + W.javaDouble(U.roundDouble(r.errors, 2)) : '') + ')';
        if (!r.conds.length) t.push(': ' + tail, '');
        else t.push(r.conds.map((c) => c.text).join(' AND\n') + ': ' + tail, '');
      }
      t.push('Number of Rules  : \t' + this.rules.length);
      return t.join('\n') + '\n';
    }
  }
  W.registerClassifier({
    name: 'PART', group: 'rules', path: 'weka.classifiers.rules.PART',
    desc: L`Λίστα κανόνων αποφάσεων από επαναλαμβανόμενα (μερικά) δέντρα C4.5.`,
    caps: { numericClass: false },
    options: [
      { key: 'C', flag: 'C', label: 'Confidence factor', type: 'number', default: 0.25, min: 0.001, max: 0.5, step: 0.05 },
      { key: 'M', flag: 'M', label: L`Ελάχιστες εγγραφές ανά κανόνα`, type: 'int', default: 2, min: 1 },
      { key: 'U', flag: 'U', label: L`Χωρίς pruning`, type: 'bool', default: false },
    ],
    create: (o) => new PART(o),
  });
})();
