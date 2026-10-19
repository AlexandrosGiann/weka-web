/* Weka Web — δέντρα αποφάσεων: J48 (C4.5), RandomTree, RandomForest, REPTree. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, S = W.Stats;

  // ======================================================== J48 / C4.5
  class Distribution {
    constructor(nb, nc) {
      this.pcb = U.matrix(nb, nc); this.perBag = U.zeros(nb); this.perClass = U.zeros(nc); this.total = 0;
    }
    static of(d) {
      const x = new Distribution(1, d.numClasses), ci = d.classIndex;
      for (let i = 0; i < d.rows.length; i++) x.add(0, d.rows[i][ci], d.weights[i]);
      return x;
    }
    static ofModel(d, model) {
      const x = new Distribution(model.numSubsets, d.numClasses), ci = d.classIndex;
      for (let i = 0; i < d.rows.length; i++) {
        const r = d.rows[i], idx = model.whichSubset(r);
        if (idx !== -1) x.add(idx, r[ci], d.weights[i]);
        else x.addWeights(r[ci], d.weights[i], model.weights(r));
      }
      return x;
    }
    clone() {
      const x = new Distribution(this.perBag.length, this.perClass.length);
      x.pcb = this.pcb.map((r) => r.slice()); x.perBag = this.perBag.slice(); x.perClass = this.perClass.slice(); x.total = this.total;
      return x;
    }
    add(b, c, w) { this.pcb[b][c] += w; this.perBag[b] += w; this.perClass[c] += w; this.total += w; }
    addWeights(c, w, ws) {
      for (let i = 0; i < this.perBag.length; i++) { const x = w * ws[i]; this.pcb[i][c] += x; this.perBag[i] += x; this.perClass[c] += x; this.total += x; }
    }
    addRange(b, d, start, end) {
      const ci = d.classIndex;
      for (let i = start; i < end; i++) this.add(b, d.rows[i][ci], d.weights[i]);
    }
    shiftRange(from, to, d, start, end) {
      const ci = d.classIndex;
      for (let i = start; i < end; i++) {
        const c = d.rows[i][ci], w = d.weights[i];
        this.pcb[from][c] -= w; this.pcb[to][c] += w; this.perBag[from] -= w; this.perBag[to] += w;
      }
    }
    addInstWithUnknown(d, att) {
      const nb = this.perBag.length, ci = d.classIndex;
      const probs = new Array(nb);
      for (let j = 0; j < nb; j++) probs[j] = U.eq(this.total, 0) ? 1 / nb : this.perBag[j] / this.total;
      for (let i = 0; i < d.rows.length; i++) {
        const r = d.rows[i];
        if (r[att] === r[att]) continue;
        const c = r[ci], w = d.weights[i];
        this.perClass[c] += w; this.total += w;
        for (let j = 0; j < nb; j++) { const nw = probs[j] * w; this.pcb[j][c] += nw; this.perBag[j] += nw; }
      }
    }
    check(min) { let n = 0; for (const b of this.perBag) if (U.grOrEq(b, min)) n++; return n > 1; }
    maxBag() { let max = 0, idx = -1; for (let i = 0; i < this.perBag.length; i++) if (U.grOrEq(this.perBag[i], max)) { max = this.perBag[i]; idx = i; } return idx; }
    maxClass(bag) {
      let max = 0, idx = 0;
      if (bag === undefined) { for (let i = 0; i < this.perClass.length; i++) if (U.gr(this.perClass[i], max)) { max = this.perClass[i]; idx = i; } return idx; }
      if (U.gr(this.perBag[bag], 0)) { for (let i = 0; i < this.perClass.length; i++) if (U.gr(this.pcb[bag][i], max)) { max = this.pcb[bag][i]; idx = i; } return idx; }
      return this.maxClass();
    }
    numCorrect(bag) { return bag === undefined ? this.perClass[this.maxClass()] : this.pcb[bag][this.maxClass(bag)]; }
    numIncorrect(bag) { return bag === undefined ? this.total - this.numCorrect() : this.perBag[bag] - this.numCorrect(bag); }
    prob(c, bag) {
      if (bag !== undefined && U.gr(this.perBag[bag], 0)) return this.pcb[bag][c] / this.perBag[bag];
      return !U.eq(this.total, 0) ? this.perClass[c] / this.total : 0;
    }
  }
  W.C45Distribution = Distribution;

  // entropy criteria (EntropyBasedSplitCrit)
  const lnF = (n) => (n < 1e-6 ? 0 : n * Math.log(n));
  function oldEnt(b) { let r = 0; for (const v of b.perClass) r += lnF(v); return (lnF(b.total) - r) / Math.LN2; }
  function newEnt(b) { let r = 0; for (let i = 0; i < b.perBag.length; i++) { for (const v of b.pcb[i]) r += lnF(v); r -= lnF(b.perBag[i]); } return -(r / Math.LN2); }
  function infoGainCrit(b, totalNoInst, oe) {
    const unknownRate = (totalNoInst - b.total) / totalNoInst;
    let num = (oe === undefined ? oldEnt(b) : oe) - newEnt(b);
    num = (1 - unknownRate) * num;
    if (U.eq(num, 0)) return 0;
    return num / b.total;
  }
  function gainRatioCrit(b, totalNoInst, numerator) {
    let den = 0;
    if (U.gr(b.total, 0)) {
      for (const v of b.perBag) den -= lnF(v);
      den -= lnF(totalNoInst - b.total);
      den += lnF(totalNoInst);
    }
    den /= Math.LN2;
    if (U.eq(den, 0)) return 0;
    den /= totalNoInst;
    return numerator / den;
  }

  class NoSplit {
    constructor(dist) { this.dist = dist.clone(); this.numSubsets = 1; }
    whichSubset() { return 0; }
    weights() { return null; }
    classProb(c, row, sub) { return sub > -1 ? this.dist.prob(c, sub) : this.dist.prob(c); }
    resetDistribution(d) { this.dist = Distribution.of(d); }
    split(d) { return [d]; }
  }

  class C45Split {
    constructor(att, minNoObj, sumOfWeights, mdl) {
      this.att = att; this.minNoObj = minNoObj; this.sumW = sumOfWeights; this.mdl = mdl;
    }
    build(d) {
      this.numSubsets = 0; this.splitPoint = Number.MAX_VALUE; this.infoGain = 0; this.gainRatio = 0;
      const a = d.attributes[this.att];
      if (a.type === 'nominal') { this.complexity = a.values.length; this.index = this.complexity; this.nominal = true; this.handleEnumerated(d); }
      else { this.complexity = 2; this.index = 0; this.nominal = false; d.sortBy(this.att); this.handleNumeric(d); }
      return this;
    }
    handleEnumerated(d) {
      const att = this.att, ci = d.classIndex;
      this.dist = new Distribution(this.complexity, d.numClasses);
      for (let i = 0; i < d.rows.length; i++) { const v = d.rows[i][att]; if (v === v) this.dist.add(v, d.rows[i][ci], d.weights[i]); }
      if (this.dist.check(this.minNoObj)) {
        this.numSubsets = this.complexity;
        this.infoGain = infoGainCrit(this.dist, this.sumW);
        this.gainRatio = gainRatioCrit(this.dist, this.sumW, this.infoGain);
      }
    }
    handleNumeric(d) {
      const att = this.att, ci = d.classIndex, rows = d.rows;
      this.dist = new Distribution(2, d.numClasses);
      let i = 0;
      for (; i < rows.length; i++) { if (rows[i][att] !== rows[i][att]) break; this.dist.add(1, rows[i][ci], d.weights[i]); }
      const firstMiss = i;
      let minSplit = 0.1 * this.dist.total / d.numClasses;
      if (U.smOrEq(minSplit, this.minNoObj)) minSplit = this.minNoObj; else if (U.gr(minSplit, 25)) minSplit = 25;
      if (U.sm(firstMiss, 2 * minSplit)) return;
      const defaultEnt = oldEnt(this.dist);
      let next = 1, last = 0, splitIndex = -1;
      while (next < firstMiss) {
        if (rows[next - 1][att] + 1e-5 < rows[next][att]) {
          this.dist.shiftRange(1, 0, d, last, next);
          if (U.grOrEq(this.dist.perBag[0], minSplit) && U.grOrEq(this.dist.perBag[1], minSplit)) {
            const g = infoGainCrit(this.dist, this.sumW, defaultEnt);
            if (U.gr(g, this.infoGain)) { this.infoGain = g; splitIndex = next - 1; }
            this.index++;
          }
          last = next;
        }
        next++;
      }
      if (this.index === 0) return;
      if (this.mdl) this.infoGain -= U.log2(this.index) / this.sumW;
      if (U.smOrEq(this.infoGain, 0)) return;
      this.numSubsets = 2;
      this.splitPoint = (rows[splitIndex + 1][att] + rows[splitIndex][att]) / 2;
      if (this.splitPoint === rows[splitIndex + 1][att]) this.splitPoint = rows[splitIndex][att];
      this.dist = new Distribution(2, d.numClasses);
      this.dist.addRange(0, d, 0, splitIndex + 1);
      this.dist.addRange(1, d, splitIndex + 1, firstMiss);
      this.gainRatio = gainRatioCrit(this.dist, this.sumW, this.infoGain);
    }
    checkModel() { return this.numSubsets > 0; }
    setSplitPoint(all) {
      let nsp = -Number.MAX_VALUE;
      if (!this.nominal && this.numSubsets > 1) {
        for (const r of all.rows) { const v = r[this.att]; if (v === v && v > nsp && v <= this.splitPoint) nsp = v; }
        this.splitPoint = nsp;
      }
    }
    whichSubset(r) {
      const v = r[this.att];
      if (v !== v) return -1;
      if (this.nominal) return v;
      return v <= this.splitPoint ? 0 : 1;
    }
    weights(r) {
      if (r[this.att] === r[this.att]) return null;
      const w = new Array(this.numSubsets);
      for (let i = 0; i < this.numSubsets; i++) w[i] = this.dist.perBag[i] / this.dist.total;
      return w;
    }
    classProb(c, r, sub) {
      if (sub <= -1) {
        const w = this.weights(r);
        if (!w) return this.dist.prob(c);
        let p = 0; for (let i = 0; i < w.length; i++) p += w[i] * this.dist.prob(c, i); return p;
      }
      return U.gr(this.dist.perBag[sub], 0) ? this.dist.prob(c, sub) : this.dist.prob(c);
    }
    resetDistribution(d) {
      const keep = [];
      for (let i = 0; i < d.rows.length; i++) if (this.whichSubset(d.rows[i]) > -1) keep.push(i);
      const nd = Distribution.ofModel(d.subset(keep), this);
      nd.addInstWithUnknown(d, this.att);
      this.dist = nd;
    }
    split(d) {
      const n = this.numSubsets, rows = Array.from({ length: n }, () => []), ws = Array.from({ length: n }, () => []);
      for (let i = 0; i < d.rows.length; i++) {
        const r = d.rows[i], s = this.whichSubset(r);
        if (s > -1) { rows[s].push(r); ws[s].push(d.weights[i]); }
        else {
          const w = this.weights(r);
          for (let j = 0; j < n; j++) if (U.gr(w[j], 0)) { rows[j].push(r); ws[j].push(w[j] * d.weights[i]); }
        }
      }
      return rows.map((rr, j) => d.withRows(rr, ws[j]));
    }
    leftSide(h) { return h.attributes[this.att].name; }
    rightSide(i, h) {
      const a = h.attributes[this.att];
      if (a.type === 'nominal') return ' = ' + a.values[i];
      return (i === 0 ? ' <= ' : ' > ') + U.d2s(this.splitPoint, 6);
    }
  }

  class ModelSelection {
    constructor(minNoObj, allData, mdl, noActual) { this.minNoObj = minNoObj; this.all = allData; this.mdl = mdl; this.noActual = noActual; }
    select(d) {
      const check = Distribution.of(d);
      const noSplit = new NoSplit(check);
      if (U.sm(check.total, 2 * this.minNoObj) || U.eq(check.total, check.perClass[check.maxClass()])) return noSplit;
      let multiVal = true;
      if (this.all) {
        for (let j = 0; j < d.numAttributes; j++) {
          if (j === d.classIndex) continue;
          const a = d.attributes[j];
          if (a.type !== 'nominal' || U.sm(a.values.length, 0.3 * this.all.numInstances)) { multiVal = false; break; }
        }
      }
      const models = new Array(d.numAttributes);
      const sumW = d.sumOfWeights();
      let avg = 0, valid = 0;
      for (let j = 0; j < d.numAttributes; j++) {
        if (j === d.classIndex) continue;
        models[j] = new C45Split(j, this.minNoObj, sumW, this.mdl).build(d);
        if (models[j].checkModel()) {
          const a = d.attributes[j];
          if (!this.all || a.type !== 'nominal' || multiVal || U.sm(a.values.length, 0.3 * this.all.numInstances)) { avg += models[j].infoGain; valid++; }
        }
      }
      if (valid === 0) return noSplit;
      avg /= valid;
      let minResult = 0, best = null;
      for (let j = 0; j < d.numAttributes; j++) {
        if (j === d.classIndex || !models[j].checkModel()) continue;
        if (models[j].infoGain >= avg - 1e-3 && U.gr(models[j].gainRatio, minResult)) { best = models[j]; minResult = best.gainRatio; }
      }
      if (U.eq(minResult, 0)) return noSplit;
      best.dist.addInstWithUnknown(d, best.att);
      if (this.all && !this.noActual) best.setSplitPoint(this.all);
      return best;
    }
  }

  class C45Tree {
    constructor(sel, prune, CF, raise, collapse) { this.sel = sel; this.pruneTree = prune; this.CF = CF; this.raise = raise; this.collapseTree = collapse; }
    newTree(d) { const t = new C45Tree(this.sel, this.pruneTree, this.CF, this.raise, this.collapseTree); t.buildTree(d, this.raise); return t; }
    buildTree(d, keep) {
      if (keep) this.train = d;
      this.isLeaf = false; this.isEmpty = false; this.sons = null;
      this.model = this.sel.select(d);
      if (this.model.numSubsets > 1) {
        const parts = this.model.split(d);
        this.sons = parts.map((p) => this.newTree(p));
      } else {
        this.isLeaf = true;
        if (U.eq(d.sumOfWeights(), 0)) this.isEmpty = true;
      }
    }
    collapse() {
      if (this.isLeaf) return;
      const sub = this.trainingErrors(), tree = this.model.dist.numIncorrect();
      if (sub >= tree - 1e-3) { this.sons = null; this.isLeaf = true; this.model = new NoSplit(this.model.dist); }
      else for (const s of this.sons) s.collapse();
    }
    trainingErrors() { if (this.isLeaf) return this.model.dist.numIncorrect(); let e = 0; for (const s of this.sons) e += s.trainingErrors(); return e; }
    errForDist(dist) { if (U.eq(dist.total, 0)) return 0; return dist.numIncorrect() + addErrs(dist.total, dist.numIncorrect(), this.CF); }
    estimatedErrors() { if (this.isLeaf) return this.errForDist(this.model.dist); let e = 0; for (const s of this.sons) e += s.estimatedErrors(); return e; }
    errForBranch(d) {
      if (this.isLeaf) return this.errForDist(Distribution.of(d));
      const saved = this.model.dist;
      this.model.resetDistribution(d);
      const parts = this.model.split(d);
      this.model.dist = saved;
      let e = 0;
      for (let i = 0; i < this.sons.length; i++) e += this.sons[i].errForBranch(parts[i]);
      return e;
    }
    newDistribution(d) {
      this.model.resetDistribution(d);
      this.train = d;
      if (!this.isLeaf) {
        const parts = this.model.split(d);
        for (let i = 0; i < this.sons.length; i++) this.sons[i].newDistribution(parts[i]);
      } else if (!U.eq(d.sumOfWeights(), 0)) this.isEmpty = false;
    }
    prune() {
      if (this.isLeaf) return;
      for (const s of this.sons) s.prune();
      const largest = this.model.dist.maxBag();
      const errLargest = this.raise ? this.sons[largest].errForBranch(this.train) : Number.MAX_VALUE;
      const errLeaf = this.errForDist(this.model.dist);
      const errTree = this.estimatedErrors();
      if (U.smOrEq(errLeaf, errTree + 0.1) && U.smOrEq(errLeaf, errLargest + 0.1)) {
        this.sons = null; this.isLeaf = true; this.model = new NoSplit(this.model.dist); return;
      }
      if (U.smOrEq(errLargest, errTree + 0.1)) {
        const lb = this.sons[largest];
        this.sons = lb.sons; this.model = lb.model; this.isLeaf = lb.isLeaf;
        this.newDistribution(this.train);
        this.prune();
      }
    }
    cleanup() { this.train = null; if (this.sons) for (const s of this.sons) s.cleanup(); }
    getProbs(c, r, w) {
      if (this.isLeaf) return w * this.model.classProb(c, r, -1);
      const idx = this.model.whichSubset(r);
      if (idx === -1) {
        const ws = this.model.weights(r);
        let p = 0;
        for (let i = 0; i < this.sons.length; i++) if (!this.sons[i].isEmpty) p += this.sons[i].getProbs(c, r, ws[i] * w);
        return p;
      }
      if (this.sons[idx].isEmpty) return w * this.model.classProb(c, r, idx);
      return this.sons[idx].getProbs(c, r, w);
    }
    getProbsLaplace(c, r, w) {
      if (this.isLeaf) return w * laplaceProb(this.model.dist, c, -1);
      const idx = this.model.whichSubset(r);
      if (idx === -1) {
        const ws = this.model.weights(r);
        let p = 0;
        for (let i = 0; i < this.sons.length; i++) if (!this.sons[i].isEmpty) p += this.sons[i].getProbsLaplace(c, r, ws[i] * w);
        return p;
      }
      if (this.sons[idx].isEmpty) return w * laplaceProb(this.model.dist, c, idx);
      return this.sons[idx].getProbsLaplace(c, r, w);
    }
    numLeaves() { return this.isLeaf ? 1 : this.sons.reduce((s, x) => s + x.numLeaves(), 0); }
    numNodes() { return this.isLeaf ? 1 : 1 + this.sons.reduce((s, x) => s + x.numNodes(), 0); }
    dumpLabel(dist, i, h) {
      let s = h.classAttr.values[dist.maxClass(i)] + ' (' + W.javaDouble(U.roundDouble(dist.perBag[i], 2));
      if (U.gr(dist.numIncorrect(i), 0)) s += '/' + W.javaDouble(U.roundDouble(dist.numIncorrect(i), 2));
      return s + ')';
    }
    dumpTree(depth, out, h) {
      for (let i = 0; i < this.sons.length; i++) {
        out.push('\n' + '|   '.repeat(depth) + this.model.leftSide(h) + this.model.rightSide(i, h));
        if (this.sons[i].isLeaf) out.push(': ' + this.dumpLabel(this.model.dist, i, h));
        else this.sons[i].dumpTree(depth + 1, out, h);
      }
    }
    toString(h) {
      const out = [];
      if (this.isLeaf) out.push(': ' + this.dumpLabel(this.model.dist, 0, h));
      else this.dumpTree(0, out, h);
      out.push('\n\nNumber of Leaves  : \t' + this.numLeaves() + '\n');
      out.push('\nSize of the tree : \t' + this.numNodes() + '\n');
      return out.join('');
    }
    /** Δομή για γραφική απεικόνιση. */
    graph(h, dist = this.model.dist, bag = 0) {
      if (this.isLeaf) return { leaf: true, label: this.dumpLabel(dist, bag, h), cls: dist.maxClass(bag) };
      return {
        leaf: false, label: this.model.leftSide(h),
        children: this.sons.map((s, i) => ({ edge: this.model.rightSide(i, h).trim(), node: s.isLeaf ? { leaf: true, label: this.dumpLabel(this.model.dist, i, h), cls: this.model.dist.maxClass(i) } : s.graph(h) })),
      };
    }
  }
  function laplaceProb(dist, c, bag) {
    const nc = dist.perClass.length;
    if (bag > -1 && U.gr(dist.perBag[bag], 0)) return (dist.pcb[bag][c] + 1) / (dist.perBag[bag] + nc);
    return (dist.perClass[c] + 1) / (dist.total + nc);
  }
  function addErrs(N, e, CF) {
    if (CF > 0.5) return 0;
    if (e < 1) {
      const base = N * (1 - Math.pow(CF, 1 / N));
      if (e === 0) return base;
      return base + e * (addErrs(N, 1, CF) - base);
    }
    if (e + 0.5 >= N) return Math.max(N - e, 0);
    const z = S.normalInverse(1 - CF);
    const f = (e + 0.5) / N;
    const r = (f + (z * z) / (2 * N) + z * Math.sqrt(f / N - (f * f) / N + (z * z) / (4 * N * N))) / (1 + (z * z) / N);
    return r * N - e;
  }
  W.addErrs = addErrs;

  class J48 {
    constructor(o = {}) {
      this.CF = o.C ?? 0.25; this.M = o.M ?? 2; this.unpruned = !!o.U; this.raise = o.subtreeRaising ?? true;
      this.laplace = !!o.A; this.collapse = o.collapse ?? true; this.mdl = o.mdl ?? true; this.noActual = !!o.doNotMakeSplitPointActualValue;
      this.CF = Math.fround(this.CF); // το Weka αποθηκεύει το CF ως float
    }
    build(data) {
      data = data.withoutMissingClass();
      this.header = data.emptyCopy();
      const sel = new ModelSelection(this.M, data, this.mdl, this.noActual);
      const raise = this.raise && !this.unpruned;
      this.root = new C45Tree(sel, !this.unpruned, this.CF, raise, this.collapse);
      this.root.buildTree(data.copy(), raise);
      if (this.collapse) this.root.collapse();
      if (!this.unpruned) this.root.prune();
      this.root.cleanup();
      return this;
    }
    distributionForRow(r) {
      const nc = this.header.numClasses, out = new Array(nc);
      for (let c = 0; c < nc; c++) out[c] = this.laplace ? this.root.getProbsLaplace(c, r, 1) : this.root.getProbs(c, r, 1);
      return out;
    }
    toString() { return (this.unpruned ? 'J48 unpruned tree' : 'J48 pruned tree') + '\n------------------\n' + this.root.toString(this.header); }
    graph() { return this.root.graph(this.header); }
  }
  W.J48 = J48;
  W.registerClassifier({
    name: 'J48', group: 'trees', path: 'weka.classifiers.trees.J48',
    desc: L`Δέντρο απόφασης C4.5 με pruning βάσει διαστημάτων εμπιστοσύνης. Ο κλασικός αλγόριθμος του Weka.`,
    caps: { numericClass: false }, graph: true,
    options: [
      { key: 'C', flag: 'C', label: 'Confidence factor (pruning)', type: 'number', default: 0.25, min: 0.001, max: 0.5, step: 0.05 },
      { key: 'M', flag: 'M', label: L`Ελάχιστες εγγραφές ανά φύλλο`, type: 'int', default: 2, min: 1 },
      { key: 'U', flag: 'U', label: L`Χωρίς pruning (unpruned)`, type: 'bool', default: false },
      { key: 'subtreeRaising', flagFalse: 'S', label: 'Subtree raising', type: 'bool', default: true },
      { key: 'collapse', flagFalse: 'O', label: 'Collapse tree', type: 'bool', default: true },
      { key: 'A', flag: 'A', label: L`Laplace smoothing στις πιθανότητες`, type: 'bool', default: false },
    ],
    create: (o) => new J48(o),
  });

  // ============================================= RandomTree (Weka-like)
  /**
   * Δέντρο με τυχαία επιλογή K χαρακτηριστικών σε κάθε κόμβο. Ελλιπείς τιμές
   * διαμοιράζονται κλασματικά, όπως στο weka.classifiers.trees.RandomTree.
   */
  class RandomTree {
    constructor(o = {}) {
      this.K = o.K ?? 0; this.minNum = o.M ?? 1; this.minVarProp = o.V ?? 0.001; this.seed = o.S ?? 1; this.maxDepth = o.depth ?? 0;
      this.rand = o.rand || null;
    }
    build(data) {
      data = data.withoutMissingClass();
      this.header = data.emptyCopy();
      this.nominal = data.classIsNominal();
      const rand = this.rand || new W.JavaRandom(this.seed);
      const na = data.numAttributes - 1;
      let K = this.K;
      if (K < 1) K = Math.floor(U.log2(na) + 1);
      if (K > na) K = na;
      this.k = K;
      const ci = data.classIndex;
      const attIdx = []; for (let j = 0; j < data.numAttributes; j++) if (j !== ci) attIdx.push(j);
      const classDist = this.classDist(data.rows, data.weights);
      let totalVar = 0;
      if (!this.nominal) totalVar = variance(data.rows, data.weights, ci);
      this.minVar = totalVar * this.minVarProp;
      this.root = this.grow(data.rows, data.weights, classDist, attIdx, rand, 0);
      return this;
    }
    classDist(rows, ws) {
      const ci = this.header.classIndex;
      if (this.nominal) { const d = U.zeros(this.header.numClasses); for (let i = 0; i < rows.length; i++) d[rows[i][ci]] += ws[i]; return d; }
      let s = 0, w = 0; for (let i = 0; i < rows.length; i++) { s += rows[i][ci] * ws[i]; w += ws[i]; }
      return [w > 0 ? s / w : 0, w];
    }
    grow(rows, ws, dist, attIdx, rand, depth) {
      const ci = this.header.classIndex;
      const totalW = U.sum(ws);
      const node = { dist };
      if (rows.length === 0) { node.leaf = true; return node; }
      let stop;
      if (this.nominal) {
        const mx = Math.max(...dist), tot = U.sum(dist);
        stop = tot < 2 * this.minNum || U.eq(mx, tot);
      } else stop = totalW < 2 * this.minNum || variance(rows, ws, ci) <= this.minVar;
      if (stop || (this.maxDepth > 0 && depth >= this.maxDepth)) { node.leaf = true; return node; }
      // τυχαία μετάθεση & επιλογή K
      const perm = attIdx.slice();
      let best = null, kTried = 0, idx = perm.length;
      while (idx > 0 && (kTried < this.k || (best === null || best.gain <= 0))) {
        const r = rand.nextInt(idx);
        const att = perm[r]; perm[r] = perm[idx - 1]; perm[idx - 1] = att; idx--;
        const s = this.evalSplit(rows, ws, att);
        if (s && (!best || s.gain > best.gain)) best = s;
        kTried++;
        if (kTried >= this.k && best && best.gain > 0) break;
      }
      if (!best || best.gain <= 1e-10) { node.leaf = true; return node; }
      // διαμοιρασμός
      const nb = best.numBranches;
      const brRows = Array.from({ length: nb }, () => []), brW = Array.from({ length: nb }, () => []);
      const props = best.props;
      for (let i = 0; i < rows.length; i++) {
        const v = rows[i][best.att];
        if (v !== v) { for (let b = 0; b < nb; b++) if (props[b] > 0) { brRows[b].push(rows[i]); brW[b].push(ws[i] * props[b]); } }
        else { const b = best.nominal ? v : v < best.split ? 0 : 1; brRows[b].push(rows[i]); brW[b].push(ws[i]); }
      }
      node.att = best.att; node.split = best.split; node.nominal = best.nominal; node.props = props;
      node.children = brRows.map((rr, b) => {
        const cd = rr.length ? this.classDist(rr, brW[b]) : dist;
        return this.grow(rr, brW[b], cd, attIdx, rand, depth + 1);
      });
      return node;
    }
    evalSplit(rows, ws, att) {
      const h = this.header, ci = h.classIndex, nominalAttr = h.attributes[att].type === 'nominal';
      const nc = this.nominal ? h.numClasses : 1;
      const known = [], missW = [];
      for (let i = 0; i < rows.length; i++) { if (rows[i][att] === rows[i][att]) known.push(i); }
      if (!known.length) return null;
      const score = (bags) => (this.nominal ? entropyGainNominal(bags) : varianceGain(bags));
      if (nominalAttr) {
        const nv = h.attributes[att].values.length;
        const bags = Array.from({ length: nv }, () => (this.nominal ? U.zeros(nc) : { s: 0, ss: 0, w: 0 }));
        for (const i of known) addTo(bags[rows[i][att]], rows[i][ci], ws[i], this.nominal);
        const props = bags.map((b) => bagW(b, this.nominal));
        const tot = U.sum(props);
        let ok = 0; for (const p of props) if (p > 0) ok++;
        if (ok < 2) return null;
        return { att, nominal: true, numBranches: nv, gain: score(bags), props: props.map((p) => p / tot) };
      }
      const sorted = known.slice().sort((a, b) => rows[a][att] - rows[b][att]);
      const left = this.nominal ? U.zeros(nc) : { s: 0, ss: 0, w: 0 };
      const right = this.nominal ? U.zeros(nc) : { s: 0, ss: 0, w: 0 };
      for (const i of sorted) addTo(right, rows[i][ci], ws[i], this.nominal);
      let best = -Infinity, split = NaN, propsBest = null;
      for (let k = 0; k < sorted.length - 1; k++) {
        const i = sorted[k];
        addTo(left, rows[i][ci], ws[i], this.nominal); addTo(right, rows[i][ci], -ws[i], this.nominal);
        const a = rows[i][att], b = rows[sorted[k + 1]][att];
        if (a + 1e-5 >= b) continue;
        const g = score([left, right]);
        if (g > best) { best = g; split = (a + b) / 2; const lw = bagW(left, this.nominal), rw = bagW(right, this.nominal); propsBest = [lw / (lw + rw), rw / (lw + rw)]; }
      }
      if (split !== split) return null;
      return { att, nominal: false, numBranches: 2, gain: best, split, props: propsBest };
    }
    nodeDist(node, r, w, out) {
      if (node.leaf || !node.children) {
        if (this.nominal) { const t = U.sum(node.dist); if (t > 0) for (let c = 0; c < out.length; c++) out[c] += w * node.dist[c] / t; }
        else { out[0] += w * node.dist[0]; }
        return;
      }
      const v = r[node.att];
      if (v !== v) { for (let b = 0; b < node.children.length; b++) if (node.props[b] > 0) this.nodeDist(node.children[b], r, w * node.props[b], out); return; }
      const b = node.nominal ? v : v < node.split ? 0 : 1;
      this.nodeDist(node.children[b], r, w, out);
    }
    distributionForRow(r) {
      const out = U.zeros(this.nominal ? this.header.numClasses : 1);
      this.nodeDist(this.root, r, 1, out);
      if (this.nominal) U.normalize(out);
      return out;
    }
    size(n = this.root) { return n.leaf || !n.children ? 1 : 1 + n.children.reduce((s, c) => s + this.size(c), 0); }
    toString() {
      const h = this.header, lines = ['', 'RandomTree', '=========='];
      const lab = (n) => (this.nominal ? h.classAttr.values[U.maxIndex(n.dist)] + ' (' + U.d2s(U.sum(n.dist), 2) + '/' + U.d2s(U.sum(n.dist) - Math.max(...n.dist), 2) + ')' : U.d2s(n.dist[0], 2) + ' (' + U.d2s(n.dist[1] || 0, 2) + ')');
      const rec = (n, depth) => {
        n.children.forEach((c, b) => {
          const a = h.attributes[n.att];
          const cond = n.nominal ? `${a.name} = ${a.values[b]}` : `${a.name} ${b === 0 ? '<' : '>='} ${U.d2s(n.split, 2)}`;
          if (c.leaf || !c.children) lines.push('|   '.repeat(depth) + cond + ' : ' + lab(c));
          else { lines.push('|   '.repeat(depth) + cond); rec(c, depth + 1); }
        });
      };
      if (this.root.leaf) lines.push(': ' + lab(this.root)); else rec(this.root, 0);
      lines.push('', 'Size of the tree : ' + this.size());
      return lines.join('\n');
    }
    graph() {
      const h = this.header;
      const lab = (n) => (this.nominal ? h.classAttr.values[U.maxIndex(n.dist)] + ' (' + U.d2s(U.sum(n.dist), 1) + ')' : U.d2s(n.dist[0], 3));
      const g = (n) => (n.leaf || !n.children ? { leaf: true, label: lab(n), cls: this.nominal ? U.maxIndex(n.dist) : 0 } : {
        leaf: false, label: h.attributes[n.att].name,
        children: n.children.map((c, b) => ({ edge: n.nominal ? '= ' + h.attributes[n.att].values[b] : (b === 0 ? '< ' : '>= ') + U.d2s(n.split, 3), node: g(c) })),
      });
      return g(this.root);
    }
  }
  function addTo(bag, c, w, nominal) { if (nominal) bag[c] += w; else { bag.s += w * c; bag.ss += w * c * c; bag.w += w; } }
  function bagW(bag, nominal) { return nominal ? U.sum(bag) : bag.w; }
  function entropyGainNominal(bags) {
    // prior entropy - conditional entropy (σε bits)
    const nc = bags[0].length, tot = U.zeros(nc);
    for (const b of bags) for (let c = 0; c < nc; c++) tot[c] += b[c];
    return S.entropy(tot) - S.entropyConditionedOnRows(bags);
  }
  function varianceGain(bags) {
    let s = 0, ss = 0, w = 0, within = 0;
    for (const b of bags) { s += b.s; ss += b.ss; w += b.w; if (b.w > 0) within += b.ss - (b.s * b.s) / b.w; }
    const tot = w > 0 ? ss - (s * s) / w : 0;
    return tot - within;
  }
  function variance(rows, ws, ci) {
    let s = 0, ss = 0, w = 0;
    for (let i = 0; i < rows.length; i++) { const v = rows[i][ci]; s += ws[i] * v; ss += ws[i] * v * v; w += ws[i]; }
    return w > 0 ? (ss - (s * s) / w) / w : 0;
  }
  W.RandomTree = RandomTree;
  W.registerClassifier({
    name: 'RandomTree', group: 'trees', path: 'weka.classifiers.trees.RandomTree',
    desc: L`Δέντρο που εξετάζει K τυχαία χαρακτηριστικά σε κάθε κόμβο, χωρίς pruning.`, graph: true,
    options: [
      { key: 'K', flag: 'K', label: L`Τυχαία χαρακτηριστικά ανά κόμβο (0 = log2(M)+1)`, type: 'int', default: 0, min: 0 },
      { key: 'M', flag: 'M', label: L`Ελάχιστο βάρος ανά φύλλο`, type: 'number', default: 1, min: 0 },
      { key: 'depth', flag: 'depth', label: L`Μέγιστο βάθος (0 = απεριόριστο)`, type: 'int', default: 0, min: 0 },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 1 },
    ],
    create: (o) => new RandomTree(o),
  });

  // ============================================================ RandomForest
  class RandomForest {
    constructor(o = {}) {
      this.I = o.I ?? 100; this.K = o.K ?? 0; this.seed = o.S ?? 1; this.depth = o.depth ?? 0; this.P = o.P ?? 100; this.M = o.M ?? 1;
      this.computeOOB = o.oob ?? true;
    }
    build(data) {
      data = data.withoutMissingClass();
      this.header = data.emptyCopy();
      this.nominal = data.classIsNominal();
      const rand = new W.JavaRandom(this.seed);
      const n = data.numInstances, bagSize = Math.max(1, Math.round(n * this.P / 100));
      this.trees = [];
      const inBag = [];
      for (let t = 0; t < this.I; t++) {
        const counts = new Array(n).fill(0);
        const r = new W.JavaRandom(Number(rand.nextLong() & 0x7fffffffn));
        for (let i = 0; i < bagSize; i++) counts[r.nextInt(n)]++;
        const rows = [], ws = [];
        for (let i = 0; i < n; i++) if (counts[i] > 0) { rows.push(data.rows[i]); ws.push(data.weights[i] * counts[i]); }
        const tree = new RandomTree({ K: this.K, M: this.M, depth: this.depth, rand: r });
        tree.build(data.withRows(rows, ws));
        this.trees.push(tree);
        inBag.push(counts);
      }
      this.k = this.trees.length ? this.trees[0].k : 0;
      // out-of-bag σφάλμα
      if (this.computeOOB) {
        let err = 0, tot = 0;
        const ci = data.classIndex;
        for (let i = 0; i < n; i++) {
          const out = U.zeros(this.nominal ? data.numClasses : 1); let cnt = 0;
          for (let t = 0; t < this.trees.length; t++) if (!inBag[t][i]) { const d = this.trees[t].distributionForRow(data.rows[i]); for (let c = 0; c < out.length; c++) out[c] += d[c]; cnt++; }
          if (!cnt) continue;
          const w = data.weights[i];
          if (this.nominal) { if (U.maxIndex(out) !== data.rows[i][ci]) err += w; }
          else { const e = out[0] / cnt - data.rows[i][ci]; err += w * e * e; }
          tot += w;
        }
        this.oob = tot > 0 ? (this.nominal ? err / tot : Math.sqrt(err / tot)) : NaN;
      }
      return this;
    }
    distributionForRow(r) {
      const out = U.zeros(this.nominal ? this.header.numClasses : 1);
      for (const t of this.trees) { const d = t.distributionForRow(r); for (let c = 0; c < out.length; c++) out[c] += d[c]; }
      if (this.nominal) return U.normalize(out);
      out[0] /= this.trees.length;
      return out;
    }
    toString() {
      let s = `RandomForest\n\nBagging with ${this.I} iterations and base learner\n\nweka.classifiers.trees.RandomTree -K ${this.K} -M ${this.M}.0 -V 0.001 -S 1${this.depth ? ' -depth ' + this.depth : ''} -do-not-check-capabilities\n`;
      s += L`\nΤυχαία χαρακτηριστικά ανά κόμβο: ${this.k}`;
      s += L`\nΜέσο μέγεθος δέντρου: ${U.d2s(this.trees.reduce((a, t) => a + t.size(), 0) / Math.max(1, this.trees.length), 1)} κόμβοι`;
      if (this.oob === this.oob) s += this.nominal ? `\n\nOut of bag error: ${U.d2s(this.oob, 4)}\n` : `\n\nOut of bag RMSE: ${U.d2s(this.oob, 4)}\n`;
      return s;
    }
  }
  W.registerClassifier({
    name: 'RandomForest', group: 'trees', path: 'weka.classifiers.trees.RandomForest',
    desc: L`Σύνολο τυχαίων δέντρων με bagging. Συνήθως από τους πιο ακριβείς ταξινομητές.`,
    options: [
      { key: 'I', flag: 'I', label: L`Πλήθος δέντρων`, type: 'int', default: 100, min: 1, max: 1000 },
      { key: 'K', flag: 'K', label: L`Τυχαία χαρακτηριστικά ανά κόμβο (0 = log2(M)+1)`, type: 'int', default: 0, min: 0 },
      { key: 'depth', flag: 'depth', label: L`Μέγιστο βάθος (0 = απεριόριστο)`, type: 'int', default: 0, min: 0 },
      { key: 'P', flag: 'P', label: L`Μέγεθος bag (% του συνόλου)`, type: 'int', default: 100, min: 1, max: 100 },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 1 },
    ],
    create: (o) => new RandomForest(o),
  });

  // ================================================================ REPTree
  /** Γρήγορο δέντρο με information gain / variance και reduced-error pruning σε hold-out fold. */
  class REPTree {
    constructor(o = {}) { this.M = o.M ?? 2; this.V = o.V ?? 0.001; this.N = o.N ?? 3; this.seed = o.S ?? 1; this.noPrune = !!o.P; this.L = o.L ?? -1; }
    build(data) {
      data = data.withoutMissingClass();
      this.header = data.emptyCopy();
      this.nominal = data.classIsNominal();
      const ci = data.classIndex;
      let grow = data, prune = null;
      if (!this.noPrune) {
        const d = data.copy();
        d.randomize(new W.JavaRandom(this.seed));
        if (this.nominal) d.stratify(this.N);
        grow = d.trainCV(this.N, 0);
        prune = d.testCV(this.N, 0);
      }
      const rt = new RandomTree({ K: data.numAttributes - 1, M: this.M, V: this.V, depth: this.L > 0 ? this.L : 0, rand: new W.JavaRandom(this.seed) });
      rt.build(grow);
      this.tree = rt;
      if (prune) {
        // καταμέτρηση hold-out στους κόμβους και reduced-error pruning (bottom-up)
        const assign = (node, r, w, c) => {
          if (!node.pd) node.pd = this.nominal ? U.zeros(data.numClasses) : { s: 0, ss: 0, w: 0, err: 0 };
          if (this.nominal) node.pd[c] += w; else { node.pd.s += w * c; node.pd.ss += w * c * c; node.pd.w += w; node.pd.err += w * (c - node.dist[0]) * (c - node.dist[0]); }
          if (node.leaf || !node.children) return;
          const v = r[node.att];
          if (v !== v) { node.children.forEach((ch, b) => { if (node.props[b] > 0) assign(ch, r, w * node.props[b], c); }); return; }
          assign(node.children[node.nominal ? v : v < node.split ? 0 : 1], r, w, c);
        };
        for (let i = 0; i < prune.numInstances; i++) assign(rt.root, prune.rows[i], prune.weights[i], prune.rows[i][ci]);
        const nodeErr = (n) => {
          if (!n.pd) return 0;
          if (this.nominal) return U.sum(n.pd) - n.pd[U.maxIndex(n.dist)];
          return n.pd.err;
        };
        const pruneRec = (n) => {
          if (n.leaf || !n.children) return nodeErr(n);
          let sub = 0;
          for (const c of n.children) sub += pruneRec(c);
          const leafErr = nodeErr(n);
          if (leafErr <= sub) { n.leaf = true; n.children = null; return leafErr; }
          return sub;
        };
        pruneRec(rt.root);
        // ενσωμάτωση hold-out στις κατανομές των φύλλων (όπως insertHoldOutSet)
        if (this.nominal) {
          const fold = (n) => { if (n.pd) for (let c = 0; c < n.dist.length; c++) n.dist[c] += n.pd[c]; if (n.children) n.children.forEach(fold); };
          fold(rt.root);
        }
      }
      return this;
    }
    distributionForRow(r) { return this.tree.distributionForRow(r); }
    toString() { return '\nREPTree\n============\n' + this.tree.toString().split('\n').slice(3).join('\n'); }
    graph() { return this.tree.graph(); }
  }
  W.registerClassifier({
    name: 'REPTree', group: 'trees', path: 'weka.classifiers.trees.REPTree',
    desc: L`Γρήγορο δέντρο (information gain / variance) με reduced-error pruning σε ξεχωριστό fold.`, graph: true,
    options: [
      { key: 'M', flag: 'M', label: L`Ελάχιστο βάρος ανά φύλλο`, type: 'number', default: 2, min: 0 },
      { key: 'V', flag: 'V', label: L`Ελάχιστη αναλογία διασποράς`, type: 'number', default: 0.001, min: 0, step: 0.001 },
      { key: 'N', flag: 'N', label: L`Folds για pruning`, type: 'int', default: 3, min: 2 },
      { key: 'L', flag: 'L', label: L`Μέγιστο βάθος (-1 = απεριόριστο)`, type: 'int', default: -1, min: -1 },
      { key: 'P', flag: 'P', label: L`Χωρίς pruning`, type: 'bool', default: false },
      { key: 'S', flag: 'S', label: 'Seed', type: 'int', default: 1 },
    ],
    create: (o) => new REPTree(o),
  });

  W.C45 = { ModelSelection, C45Tree, C45Split, NoSplit, Distribution };
})();
