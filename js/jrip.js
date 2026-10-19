/* Weka Web — JRip (RIPPER, Cohen 1995): μεταφορά του weka.classifiers.rules.JRip. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils;
  const log2 = U.log2;
  const MAX_DL_SURPLUS = 64.0, REDUNDANCY_FACTOR = 0.5, MDL_THEORY_WEIGHT = 1.0;

  // ------------------------------------------------------- μικρά «Instances»
  const mk = (rows, ws) => ({ rows, ws });
  const sumW = (d) => { let s = 0; for (const w of d.ws) s += w; return s; };
  function randomize(d, rand) {
    for (let j = d.rows.length - 1; j > 0; j--) {
      const k = rand.nextInt(j + 1);
      let t = d.rows[j]; d.rows[j] = d.rows[k]; d.rows[k] = t;
      t = d.ws[j]; d.ws[j] = d.ws[k]; d.ws[k] = t;
    }
  }
  /** Instances.sort(att) του Weka (quicksort, λείπουσες στο τέλος), επί τόπου. */
  function sortInPlace(d, att) {
    const order = U.wekaSort(d.rows.map((r) => r[att]));
    const rows = order.map((i) => d.rows[i]), ws = order.map((i) => d.ws[i]);
    for (let i = 0; i < rows.length; i++) { d.rows[i] = rows[i]; d.ws[i] = ws[i]; }
  }
  /** Java String.hashCode */
  function javaHash(s) { let h = 0; for (let i = 0; i < s.length; i++) h = (Math.imul(h, 31) + s.charCodeAt(i)) | 0; return h; }
  function instanceString(hdr, row) {
    return row.map((v, j) => {
      if (v !== v) return '?';
      const a = hdr.attributes[j];
      if (a.type === 'nominal' || a.type === 'string') return W.quote(a.values[v]);
      if (a.type === 'date') return W.quote(W.formatDate(v, a.dateFormat));
      return U.d2s(v, 6);
    }).join(',');
  }

  // --------------------------------------------------------- antecedents
  class NumericAntd {
    constructor(att, name) { this.att = att; this.name = name; this.numeric = true; this.value = NaN; this.splitPoint = NaN; this.maxInfoGain = 0; this.accuRate = NaN; this.cover = NaN; this.accu = NaN; }
    copy() { const a = new NumericAntd(this.att, this.name); a.value = this.value; a.splitPoint = this.splitPoint; return a; }
    splitData(data, defAcRt, cl, ci) {
      const att = this.att;
      let total = data.rows.length, split = 1, prev = 0, finalSplit = split;
      this.maxInfoGain = 0; this.value = 0;
      let fstCover = 0, sndCover = 0, fstAccu = 0, sndAccu = 0;
      sortInPlace(data, att);
      for (let x = 0; x < data.rows.length; x++) {
        const r = data.rows[x];
        if (r[att] !== r[att]) { total = x; break; }
        sndCover += data.ws[x];
        if (U.eq(r[ci], cl)) sndAccu += data.ws[x];
      }
      if (total === 0) return null;
      this.splitPoint = data.rows[total - 1][att];
      for (; split <= total; split++) {
        if (split === total || data.rows[split][att] > data.rows[prev][att]) {
          for (let y = prev; y < split; y++) { fstCover += data.ws[y]; if (U.eq(data.rows[y][ci], cl)) fstAccu += data.ws[y]; }
          const fstAccuRate = (fstAccu + 1.0) / (fstCover + 1.0), sndAccuRate = (sndAccu + 1.0) / (sndCover + 1.0);
          const fstInfoGain = fstAccu * (log2(fstAccuRate) - log2(defAcRt));
          const sndInfoGain = sndAccu * (log2(sndAccuRate) - log2(defAcRt));
          let isFirst, infoGain, accRate, accurate, coverage;
          if (fstInfoGain > sndInfoGain) { isFirst = true; infoGain = fstInfoGain; accRate = fstAccuRate; accurate = fstAccu; coverage = fstCover; }
          else { isFirst = false; infoGain = sndInfoGain; accRate = sndAccuRate; accurate = sndAccu; coverage = sndCover; }
          if (infoGain > this.maxInfoGain) {
            this.splitPoint = data.rows[prev][att];
            this.value = isFirst ? 0 : 1;
            this.accuRate = accRate; this.accu = accurate; this.cover = coverage; this.maxInfoGain = infoGain;
            finalSplit = isFirst ? split : prev;
          }
          for (let y = prev; y < split; y++) { sndCover -= data.ws[y]; if (U.eq(data.rows[y][ci], cl)) sndAccu -= data.ws[y]; }
          prev = split;
        }
      }
      return [mk(data.rows.slice(0, finalSplit), data.ws.slice(0, finalSplit)), mk(data.rows.slice(finalSplit, total), data.ws.slice(finalSplit, total))];
    }
    covers(r) {
      const v = r[this.att];
      if (v !== v) return false;
      return this.value === 0 ? !(v > this.splitPoint) : !(v < this.splitPoint);
    }
    toString() { return this.name + (this.value === 0 ? ' <= ' : ' >= ') + U.d2s(this.splitPoint, 6); }
  }
  class NominalAntd {
    constructor(att, name, values) { this.att = att; this.name = name; this.values = values; this.numeric = false; this.value = NaN; this.maxInfoGain = 0; this.accuRate = NaN; this.cover = NaN; this.accu = NaN; }
    copy() { const a = new NominalAntd(this.att, this.name, this.values); a.value = this.value; return a; }
    splitData(data, defAcRt, cl, ci) {
      const bag = this.values.length, att = this.att;
      const split = Array.from({ length: bag }, () => mk([], [])), accurate = U.zeros(bag), coverage = U.zeros(bag);
      for (let x = 0; x < data.rows.length; x++) {
        const r = data.rows[x], v = r[att];
        if (v !== v) continue;
        split[v].rows.push(r); split[v].ws.push(data.ws[x]);
        coverage[v] += data.ws[x];
        if (r[ci] === cl) accurate[v] += data.ws[x];
      }
      for (let x = 0; x < bag; x++) {
        const t = coverage[x] + 1.0, p = accurate[x] + 1.0;
        const infoGain = accurate[x] * (log2(p / t) - log2(defAcRt));
        if (infoGain > this.maxInfoGain) { this.maxInfoGain = infoGain; this.cover = coverage[x]; this.accu = accurate[x]; this.accuRate = p / t; this.value = x; }
      }
      return split;
    }
    covers(r) { const v = r[this.att]; return v === v && v === this.value; }
    toString() { return this.name + ' = ' + this.values[this.value]; }
  }

  // ---------------------------------------------------------------- rule
  class RipperRule {
    constructor(owner) { this.o = owner; this.cons = -1; this.antds = []; }
    copy() { const r = new RipperRule(this.o); r.cons = this.cons; r.antds = this.antds.map((a) => a.copy()); return r; }
    covers(row) { for (const a of this.antds) if (!a.covers(row)) return false; return true; }
    size() { return this.antds.length; }
    defAccu(d) { const ci = this.o.ci; let s = 0; for (let i = 0; i < d.rows.length; i++) if (d.rows[i][ci] === this.cons) s += d.ws[i]; return s; }
    grow(data) {
      const o = this.o, ci = o.ci;
      let growData = data;
      const sw = sumW(growData);
      if (!U.gr(sw, 0)) return;
      let defAcRt = (this.defAccu(growData) + 1.0) / (sw + 1.0);
      const used = new Array(o.hdr.numAttributes).fill(false);
      let numUnused = used.length;
      for (const a of this.antds) if (!a.numeric) { used[a.att] = true; numUnused--; }
      while (U.gr(growData.rows.length, 0) && numUnused > 0 && U.sm(defAcRt, 1.0)) {
        let maxInfoGain = 0, oneAntd = null, coverData = null;
        for (let j = 0; j < o.hdr.numAttributes; j++) {
          if (j === ci) continue;
          const at = o.hdr.attributes[j];
          const antd = W.isNumericAttr(at) ? new NumericAntd(j, at.name) : new NominalAntd(j, at.name, at.values);
          if (used[j]) continue;
          const split = antd.splitData(growData, defAcRt, this.cons, ci);
          if (!split) continue;
          const covered = split[antd.value !== antd.value ? 0 : antd.value];
          if (antd.maxInfoGain > maxInfoGain) { oneAntd = antd; coverData = covered; maxInfoGain = antd.maxInfoGain; }
        }
        if (!oneAntd) break;
        if (U.sm(oneAntd.accu, o.minNo)) break;
        if (!oneAntd.numeric) { used[oneAntd.att] = true; numUnused--; }
        this.antds.push(oneAntd);
        growData = coverData;
        defAcRt = oneAntd.accuRate;
      }
    }
    prune(pruneData, useWhole) {
      const ci = this.o.ci;
      let data = pruneData;
      const total = sumW(data);
      if (!U.gr(total, 0)) return;
      const defAccu = this.defAccu(data);
      const size = this.antds.length;
      if (size === 0) return;
      const worthRt = U.zeros(size), coverage = U.zeros(size), worthValue = U.zeros(size);
      let tn = 0;
      for (let x = 0; x < size; x++) {
        const antd = this.antds[x];
        const nd = data; data = mk([], []);
        for (let y = 0; y < nd.rows.length; y++) {
          const r = nd.rows[y], w = nd.ws[y];
          if (antd.covers(r)) { coverage[x] += w; data.rows.push(r); data.ws.push(w); if (r[ci] === this.cons) worthValue[x] += w; }
          else if (useWhole && r[ci] !== this.cons) tn += w;
        }
        if (useWhole) { worthValue[x] += tn; worthRt[x] = worthValue[x] / total; }
        else worthRt[x] = (worthValue[x] + 1.0) / (coverage[x] + 2.0);
      }
      let maxValue = (defAccu + 1.0) / (total + 2.0), maxIndex = -1;
      for (let i = 0; i < size; i++) if (worthRt[i] > maxValue) { maxValue = worthRt[i]; maxIndex = i; }
      this.antds.length = maxIndex + 1;
    }
    cleanUp(na) {
      const mins = new Array(na).fill(Number.MAX_VALUE), maxs = new Array(na).fill(-Number.MAX_VALUE);
      for (let i = this.antds.length - 1; i >= 0; i--) {
        const a = this.antds[i];
        if (!a.numeric) continue;
        if (a.value === 0) { if (a.splitPoint < mins[a.att]) mins[a.att] = a.splitPoint; else this.antds.splice(i, 1); }
        else if (a.splitPoint > maxs[a.att]) maxs[a.att] = a.splitPoint; else this.antds.splice(i, 1);
      }
    }
    toString(classAttr) {
      let s = this.antds.map((a) => '(' + a.toString() + ')').join(' and ');
      return s + ' => ' + classAttr.name + '=' + classAttr.values[this.cons];
    }
  }

  // ------------------------------------------------------------ RuleStats
  const subsetDL = (t, k, p) => { let rt = U.gr(p, 0) ? -k * log2(p) : 0; rt -= (t - k) * log2(1 - p); return rt; };
  function dataDL(expFPOverErr, cover, uncover, fp, fn) {
    const totalBits = log2(cover + uncover + 1.0);
    let coverBits, uncoverBits, expErr;
    if (U.gr(cover, uncover)) {
      expErr = expFPOverErr * (fp + fn);
      coverBits = subsetDL(cover, fp, expErr / cover);
      uncoverBits = U.gr(uncover, 0) ? subsetDL(uncover, fn, fn / uncover) : 0;
    } else {
      expErr = (1.0 - expFPOverErr) * (fp + fn);
      coverBits = U.gr(cover, 0) ? subsetDL(cover, fp, fp / cover) : 0;
      uncoverBits = subsetDL(uncover, fn, expErr / uncover);
    }
    return totalBits + coverBits + uncoverBits;
  }
  class RuleStats {
    constructor(o, data = null, rules = null) { this.o = o; this.data = data; this.ruleset = rules; this.simple = null; this.filtered = null; this.dists = null; this.total = -1; }
    setNumAllConds(t) { this.total = t; }
    compute(index, insts, stats, dist) {
      const rule = this.ruleset[index], ci = this.o.ci;
      const cov = mk([], []), unc = mk([], []);
      for (let i = 0; i < insts.rows.length; i++) {
        const r = insts.rows[i], w = insts.ws[i];
        if (rule.covers(r)) {
          cov.rows.push(r); cov.ws.push(w); stats[0] += w;
          if (r[ci] === rule.cons) stats[2] += w; else stats[4] += w;
          if (dist) dist[r[ci]] += w;
        } else {
          unc.rows.push(r); unc.ws.push(w); stats[1] += w;
          if (r[ci] !== rule.cons) stats[3] += w; else stats[5] += w;
        }
      }
      return [cov, unc];
    }
    countData() {
      if (this.filtered || !this.ruleset || !this.data) return;
      this.filtered = []; this.simple = []; this.dists = [];
      let d = mk(this.data.rows.slice(), this.data.ws.slice());
      for (let i = 0; i < this.ruleset.length; i++) {
        const st = U.zeros(6), cc = U.zeros(this.o.nc);
        const f = this.compute(i, d, st, cc);
        this.filtered.push(f); this.simple.push(st); this.dists.push(cc);
        d = f[1];
      }
    }
    countDataFrom(index, uncovered, prev) {
      if (this.filtered || !this.ruleset) return;
      this.filtered = []; this.simple = [];
      let data = [null, uncovered];
      for (let i = 0; i < index; i++) { this.simple.push(prev[i]); this.filtered.push(i + 1 === index ? data : []); }
      for (let j = index; j < this.ruleset.length; j++) {
        const st = U.zeros(6);
        const f = this.compute(j, data[1], st, null);
        this.filtered.push(f); this.simple.push(st);
        data = f;
      }
    }
    addAndUpdate(rule) {
      if (!this.ruleset) this.ruleset = [];
      this.ruleset.push(rule);
      const data = this.filtered === null ? this.data : this.filtered[this.filtered.length - 1][1];
      const st = U.zeros(6), cc = U.zeros(this.o.nc);
      const f = this.compute(this.ruleset.length - 1, data, st, cc);
      (this.filtered = this.filtered || []).push(f);
      (this.simple = this.simple || []).push(st);
      (this.dists = this.dists || []).push(cc);
    }
    theoryDL(index) {
      const k = this.ruleset[index].size();
      if (k === 0) return 0;
      let tdl = log2(k);
      if (k > 1) tdl += 2.0 * log2(tdl);
      tdl += subsetDL(this.total, k, k / this.total);
      return MDL_THEORY_WEIGHT * REDUNDANCY_FACTOR * tdl;
    }
    potential(index, exp, rs, r, checkErr) {
      const pcov = rs[0] - r[0], puncov = rs[1] + r[0], pfp = rs[4] - r[4], pfn = rs[5] + r[2];
      const with_ = dataDL(exp, rs[0], rs[1], rs[4], rs[5]);
      const theory = this.theoryDL(index);
      const without = dataDL(exp, pcov, puncov, pfp, pfn);
      const pot = with_ + theory - without;
      const err = r[4] / r[0];
      const overErr = checkErr ? U.grOrEq(err, 0.5) : false;
      if (U.grOrEq(pot, 0) || overErr) { rs[0] = pcov; rs[1] = puncov; rs[4] = pfp; rs[5] = pfn; return pot; }
      return NaN;
    }
    minDataDLIfDeleted(index, exp, checkErr) {
      const rs = U.zeros(6), more = this.ruleset.length - 1 - index, plus = [];
      for (let j = 0; j < index; j++) { rs[0] += this.simple[j][0]; rs[2] += this.simple[j][2]; rs[4] += this.simple[j][4]; }
      let data = index === 0 ? this.data : this.filtered[index - 1][1];
      for (let j = index + 1; j < this.ruleset.length; j++) {
        const st = U.zeros(6);
        const sp = this.compute(j, data, st, null);
        plus.push(st); rs[0] += st[0]; rs[2] += st[2]; rs[4] += st[4];
        data = sp[1];
      }
      if (more > 0) { const l = plus[plus.length - 1]; rs[1] = l[1]; rs[3] = l[3]; rs[5] = l[5]; }
      else if (index > 0) { const l = this.simple[index - 1]; rs[1] = l[1]; rs[3] = l[3]; rs[5] = l[5]; }
      else { const s0 = this.simple[0]; rs[1] = s0[0] + s0[1]; rs[3] = s0[3] + s0[4]; rs[5] = s0[2] + s0[5]; }
      let pot = 0;
      for (let k = index + 1; k < this.ruleset.length; k++) { const v = this.potential(k, exp, rs, plus[k - index - 1], checkErr); if (v === v) pot += v; }
      return dataDL(exp, rs[0], rs[1], rs[4], rs[5]) - pot;
    }
    minDataDLIfExists(index, exp, checkErr) {
      const rs = U.zeros(6), n = this.simple.length;
      for (let j = 0; j < n; j++) {
        rs[0] += this.simple[j][0]; rs[2] += this.simple[j][2]; rs[4] += this.simple[j][4];
        if (j === n - 1) { rs[1] = this.simple[j][1]; rs[3] = this.simple[j][3]; rs[5] = this.simple[j][5]; }
      }
      let pot = 0;
      for (let k = index + 1; k < n; k++) { const v = this.potential(k, exp, rs, this.simple[k], checkErr); if (v === v) pot += v; }
      return dataDL(exp, rs[0], rs[1], rs[4], rs[5]) - pot;
    }
    relativeDL(index, exp, checkErr) { return this.minDataDLIfExists(index, exp, checkErr) + this.theoryDL(index) - this.minDataDLIfDeleted(index, exp, checkErr); }
    reduceDL(exp, checkErr) {
      let needUpdate = false;
      const rs = U.zeros(6), n = this.simple.length;
      for (let j = 0; j < n; j++) {
        rs[0] += this.simple[j][0]; rs[2] += this.simple[j][2]; rs[4] += this.simple[j][4];
        if (j === n - 1) { rs[1] = this.simple[j][1]; rs[3] = this.simple[j][3]; rs[5] = this.simple[j][5]; }
      }
      for (let k = this.simple.length - 1; k >= 0; k--) {
        const v = this.potential(k, exp, rs, this.simple[k], checkErr);
        if (v === v) {
          if (k === this.simple.length - 1) this.removeLast();
          else { this.ruleset.splice(k, 1); needUpdate = true; }
        }
      }
      if (needUpdate) { this.filtered = null; this.simple = null; this.countData(); }
    }
    removeLast() {
      const last = this.ruleset.length - 1;
      this.ruleset.splice(last, 1); this.filtered.splice(last, 1); this.simple.splice(last, 1);
      if (this.dists) this.dists.splice(last, 1);
    }
  }
  function stratify(o, data, folds, rand) {
    const bags = Array.from({ length: o.nc }, () => mk([], []));
    for (let j = 0; j < data.rows.length; j++) { const b = bags[data.rows[j][o.ci]]; b.rows.push(data.rows[j]); b.ws.push(data.ws[j]); }
    for (const b of bags) randomize(b, rand);
    const res = mk([], []);
    for (let k = 0; k < folds; k++) {
      let offset = k, bag = 0;
      outer: for (;;) {
        while (offset >= bags[bag].rows.length) { offset -= bags[bag].rows.length; if (++bag >= bags.length) break outer; }
        res.rows.push(bags[bag].rows[offset]); res.ws.push(bags[bag].ws[offset]);
        offset += folds;
      }
    }
    return res;
  }
  function partition(data, folds) {
    const splits = Math.trunc((data.rows.length * (folds - 1)) / folds);
    return [mk(data.rows.slice(0, splits), data.ws.slice(0, splits)), mk(data.rows.slice(splits), data.ws.slice(splits))];
  }
  function rmCoveredBySuccessives(data, rules, index) {
    const rt = mk([], []);
    for (let i = 0; i < data.rows.length; i++) {
      let cov = false;
      for (let j = index + 1; j < rules.length; j++) if (rules[j].covers(data.rows[i])) { cov = true; break; }
      if (!cov) { rt.rows.push(data.rows[i]); rt.ws.push(data.ws[i]); }
    }
    return rt;
  }

  // --------------------------------------------------------------- JRip
  class JRip {
    constructor(o = {}) { this.folds = o.F ?? 3; this.minNo = o.N ?? 2; this.optim = o.O ?? 2; this.seed = o.S ?? 1; this.checkErr = !o.E; this.usePruning = !o.P; }
    build(data) {
      data = data.withoutMissingClass();
      const n = data.numInstances, ci = data.classIndex, nc = data.numClasses;
      // Instances.getRandomNumberGenerator(seed)
      const r0 = new W.JavaRandom(this.seed);
      const pick = r0.nextInt(n);
      const rand = new W.JavaRandom(javaHash(instanceString(data, data.rows[pick])) + this.seed);
      this.rand = rand;
      // numAllConditions
      let total = 0;
      data.attributes.forEach((a, j) => {
        if (j === ci) return;
        if (a.type === 'nominal') total += a.values.length;
        else { const s = new Set(); for (const r of data.rows) if (r[j] === r[j]) s.add(r[j]); total += 2.0 * s.size; }
      });
      this.total = total;
      // ClassOrder (FREQ_ASCEND)
      const coRand = new W.JavaRandom(rand.nextInt());
      const counts = U.zeros(nc);
      for (let i = 0; i < n; i++) counts[data.rows[i][ci]] += data.weights[i];
      const ri = [...Array(nc).keys()];
      for (let j = nc - 1; j > 0; j--) { const k = coRand.nextInt(j + 1); const t = ri[j]; ri[j] = ri[k]; ri[k] = t; }
      const sorted = U.wekaSort(ri.map((i) => counts[i]));
      const conv = sorted.map((s) => ri[s]); // νέα κλάση i = παλιά conv[i]
      const ordered = conv.map((c) => counts[c]);
      const oldToNew = new Array(nc); conv.forEach((old, nw) => { oldToNew[old] = nw; });
      this.conv = conv;
      const ca = data.classAttr;
      this.classAttr = { name: ca.name, values: conv.map((c) => ca.values[c]) };
      this.hdr = data; this.ci = ci; this.nc = nc;
      let d = mk(data.rows.map((r) => { const c = r.slice(); c[ci] = oldToNew[r[ci]]; return c; }), data.weights.slice());
      this.ruleset = []; this.statsList = []; this.distributions = [];
      for (let y = 0; y < nc - 1; y++) {
        if (U.eq(ordered[y], 0)) continue;
        let all = 0; for (let i = y; i < nc; i++) all += ordered[i];
        const expFPRate = ordered[y] / all;
        let classYW = 0, totW = 0;
        for (let j = 0; j < d.rows.length; j++) { totW += d.ws[j]; if (d.rows[j][ci] === y) classYW += d.ws[j]; }
        if (!(classYW > 0)) continue;
        const defDL = dataDL(expFPRate, 0, totW, 0, classYW);
        d = this.rulesetForOneClass(expFPRate, d, y, defDL);
      }
      for (const r of this.ruleset) r.cleanUp(data.numAttributes);
      const defRule = new RipperRule(this); defRule.cons = nc - 1;
      this.ruleset.push(defRule);
      const ds = new RuleStats(this, d); ds.setNumAllConds(this.total); ds.addAndUpdate(defRule);
      this.statsList.push(ds);
      for (const rs of this.statsList) {
        for (let k = 0; k < rs.ruleset.length; k++) {
          const cd = rs.dists[k].slice(); U.normalize(cd);
          const orig = new Array(nc); for (let i = 0; i < nc; i++) orig[i] = cd[oldToNew[i]];
          this.distributions.push(orig);
        }
      }
      return this;
    }
    rulesetForOneClass(exp, data, cls, defDL) {
      let newData = data, growData, pruneData, stop = false, hasPositive = true;
      let ruleset = [], dl = defDL, minDL = defDL, rstats = null, rst;
      const rand = this.rand;
      while (!stop && hasPositive) {
        let one;
        if (this.usePruning) {
          newData = stratify(this, newData, this.folds, rand);
          [growData, pruneData] = partition(newData, this.folds);
          one = new RipperRule(this); one.cons = cls;
          one.grow(growData);
          one.prune(pruneData, false);
        } else { one = new RipperRule(this); one.cons = cls; one.grow(newData); }
        if (rstats === null) { rstats = new RuleStats(this); rstats.setNumAllConds(this.total); rstats.data = newData; }
        rstats.addAndUpdate(one);
        const last = rstats.ruleset.length - 1;
        dl += rstats.relativeDL(last, exp, this.checkErr);
        if (dl < minDL) minDL = dl;
        rst = rstats.simple[last];
        stop = this.checkStop(rst, minDL, dl);
        if (!stop) { ruleset.push(one); newData = rstats.filtered[last][1]; hasPositive = U.gr(rst[5], 0); }
        else rstats.removeLast();
      }
      if (this.usePruning) {
        for (let z = 0; z < this.optim; z++) {
          newData = data;
          const fin = new RuleStats(this); fin.data = newData; fin.setNumAllConds(this.total);
          let position = 0; stop = false; hasPositive = true; dl = minDL = defDL;
          while (!stop && hasPositive) {
            const isResidual = position >= ruleset.length;
            newData = stratify(this, newData, this.folds, rand);
            [growData, pruneData] = partition(newData, this.folds);
            let finalRule;
            if (isResidual) {
              const nr = new RipperRule(this); nr.cons = cls; nr.grow(growData); nr.prune(pruneData, false);
              finalRule = nr;
            } else {
              const oldRule = ruleset[position];
              let covers = false;
              for (const r of newData.rows) if (oldRule.covers(r)) { covers = true; break; }
              if (!covers) { fin.addAndUpdate(oldRule); position++; continue; }
              const replace = new RipperRule(this); replace.cons = cls; replace.grow(growData);
              pruneData = rmCoveredBySuccessives(pruneData, ruleset, position);
              replace.prune(pruneData, true);
              const revision = oldRule.copy();
              const ng = mk([], []);
              for (let b = 0; b < growData.rows.length; b++) if (revision.covers(growData.rows[b])) { ng.rows.push(growData.rows[b]); ng.ws.push(growData.ws[b]); }
              revision.grow(ng);
              revision.prune(pruneData, true);
              const prev = []; for (let c = 0; c < position; c++) prev.push(fin.simple[c]);
              const temp = ruleset.map((r) => r.copy());
              temp[position] = replace;
              const repStat = new RuleStats(this, data, temp); repStat.setNumAllConds(this.total); repStat.countDataFrom(position, newData, prev);
              const repDL = repStat.relativeDL(position, exp, this.checkErr);
              temp[position] = revision;
              const revStat = new RuleStats(this, data, temp); revStat.setNumAllConds(this.total); revStat.countDataFrom(position, newData, prev);
              const revDL = revStat.relativeDL(position, exp, this.checkErr);
              const oStat = new RuleStats(this, data, ruleset); oStat.setNumAllConds(this.total); oStat.countDataFrom(position, newData, prev);
              const oldDL = oStat.relativeDL(position, exp, this.checkErr);
              if (oldDL <= revDL && oldDL <= repDL) finalRule = oldRule;
              else if (revDL <= repDL) finalRule = revision;
              else finalRule = replace;
            }
            fin.addAndUpdate(finalRule);
            rst = fin.simple[position];
            if (isResidual) {
              dl += fin.relativeDL(position, exp, this.checkErr);
              if (dl < minDL) minDL = dl;
              stop = this.checkStop(rst, minDL, dl);
              if (!stop) ruleset.push(finalRule);
              else { fin.removeLast(); position--; }
            } else ruleset[position] = finalRule;
            if (fin.ruleset.length > 0) newData = fin.filtered[position][1];
            hasPositive = U.gr(rst[5], 0);
            position++;
          }
          if (ruleset.length > position + 1) for (let k = position + 1; k < ruleset.length; k++) fin.addAndUpdate(ruleset[k]);
          fin.reduceDL(exp, this.checkErr);
          ruleset = fin.ruleset;
          rstats = fin;
        }
      }
      this.ruleset.push(...ruleset);
      this.statsList.push(rstats);
      if (ruleset.length > 0) return rstats.filtered[ruleset.length - 1][1];
      return data;
    }
    checkStop(rst, minDL, dl) {
      if (dl > minDL + MAX_DL_SURPLUS) return true;
      if (!U.gr(rst[2], 0)) return true;
      if (rst[4] / rst[0] >= 0.5) return this.checkErr;
      return false;
    }
    distributionForRow(row) {
      for (let i = 0; i < this.ruleset.length; i++) if (this.ruleset[i].covers(row)) return this.distributions[i].slice();
      return U.zeros(this.nc);
    }
    toString() {
      let s = 'JRIP rules:\n===========\n\n';
      for (const rs of this.statsList) {
        if (!rs) continue;
        rs.ruleset.forEach((r, k) => { const st = rs.simple[k]; s += r.toString(this.classAttr) + ' (' + W.javaDouble(st[0]) + '/' + W.javaDouble(st[4]) + ')\n'; });
      }
      return s + '\nNumber of Rules : ' + this.ruleset.length + '\n';
    }
  }
  W.registerClassifier({
    name: 'JRip', group: 'rules', path: 'weka.classifiers.rules.JRip',
    desc: L`RIPPER propositional rule learner (Cohen): grows and prunes if-then rules, then optimises the rule set with MDL.`,
    caps: { numericClass: false },
    options: [
      { key: 'F', flag: 'F', label: L`Folds (one used for pruning)`, type: 'int', default: 3, min: 2 },
      { key: 'N', flag: 'N', label: L`Minimum total weight of instances in a rule`, type: 'number', default: 2, min: 0 },
      { key: 'O', flag: 'O', label: L`Optimisation runs`, type: 'int', default: 2, min: 0 },
      { key: 'E', flag: 'E', label: L`Do not check error rate ≥ 0.5 in stopping criterion`, type: 'bool', default: false },
      { key: 'P', flag: 'P', label: L`No pruning`, type: 'bool', default: false },
      { key: 'S', flag: 'S', label: L`Seed`, type: 'int', default: 1 },
    ],
    create: (o) => new JRip(o),
  });
})();
