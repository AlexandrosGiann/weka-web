/* Weka Web — κανόνες συσχέτισης: Apriori (πιστή μεταφορά) και FPGrowth. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils;
  W.associators = [];
  const reg = (s) => { W.associators.push(s); return s; };
  W.findAssociator = (n) => W.associators.find((s) => s.name === n);

  // =============================================================== Apriori
  const key = (items) => items.join(',');
  class Apriori {
    constructor(o = {}) {
      this.N = o.N ?? 10; this.T = o.T ?? 0; this.C = o.C ?? 0.9; this.D = o.D ?? 0.05; this.U = o.U ?? 1.0; this.M = o.M ?? 0.1;
      this.Z = !!o.Z; this.removeMissingCols = !!o.R;
    }
    build(data) {
      const d = data.copy(); d.classIndex = -1;
      for (const a of d.attributes) if (a.type !== 'nominal') throw new Error(L`Το Apriori δέχεται μόνο nominal χαρακτηριστικά ("${a.name}" είναι ${a.type}). Χρησιμοποίησε πρώτα το φίλτρο Discretize ή NumericToNominal.`);
      this.data = d;
      const n = d.numInstances, na = d.numAttributes;
      this.n = n;
      const lower = this.M * n < 1 ? 1 / n : this.M;
      let minSup = 1.0 - this.D;
      if (minSup < lower) minSup = lower;
      this.cycles = 0;
      let rules, necSupport;
      do {
        this.minSupport = minSup;
        const res = this.findLarge(d, minSup);
        this.Ls = res.Ls; this.hashtables = res.hashtables;
        rules = [];
        for (let j = 1; j < this.Ls.length; j++) for (const set of this.Ls[j]) {
          if (this.T === 0) this.generateRules(set, j + 1, rules); else this.bruteForce(set, j + 1, rules);
        }
        if (this.U < 1.0) { const maxS = Math.trunc(this.U * n + 0.5); rules = rules.filter((r) => r.support <= maxS); }
        // ταξινόμηση: όπως το Weka (stable sort σε support και μετά στη μετρική)
        const m = rules.length - 1;
        const sup = rules.map((r) => -r.support);
        let idx = stableSort(sup);
        const sorted = [];
        for (let i = 0; i <= m; i++) sorted.push(rules[idx[m - i]]);
        const metric = sorted.map((r) => r.metrics[this.T]);
        idx = stableSort(metric);
        const out = [];
        for (let i = sorted.length - 1; i >= sorted.length - this.N && i >= 0; i--) out.push(sorted[idx[i]]);
        rules = out;
        if (minSup === lower || minSup - this.D > lower) minSup -= this.D; else minSup = lower;
        necSupport = U.rint(minSup * n);
        this.cycles++;
      } while (rules.length < this.N && U.grOrEq(minSup, lower) && necSupport >= 1);
      this.rules = rules;
      this.printSup = minSup + this.D;
      return this;
    }
    findLarge(d, minSup) {
      const n = d.numInstances, na = d.numAttributes;
      const nec = Math.trunc(minSup * n + 0.5);
      const Ls = [], hashtables = [];
      let k = [];
      for (let i = 0; i < na; i++) {
        for (let j = this.Z ? 1 : 0; j < d.attributes[i].values.length; j++) {
          const items = new Array(na).fill(-1); items[i] = j; k.push({ items, count: 0 });
        }
      }
      this.count(k, d);
      k = k.filter((s) => s.count >= nec && s.count <= n);
      let size = 0;
      while (k.length) {
        Ls.push(k);
        const prev = k;
        let next = mergeAll(prev, size, na);
        const ht = new Map(prev.map((s) => [key(s.items), s.count]));
        hashtables.push(ht);
        next = next.filter((s) => {
          for (let j = 0; j < na; j++) if (s.items[j] !== -1) { const h = s.items[j]; s.items[j] = -1; const ok = ht.has(key(s.items)); s.items[j] = h; if (!ok) return false; }
          return true;
        });
        this.count(next, d);
        k = next.filter((s) => s.count >= nec && s.count <= n);
        size++;
      }
      return { Ls, hashtables };
    }
    count(sets, d) {
      const z = this.Z;
      const compact = sets.map((s) => { const a = []; s.items.forEach((v, i) => { if (v > -1) a.push(i, v); }); return a; });
      for (const r of d.rows) {
        for (let k = 0; k < sets.length; k++) {
          const c = compact[k];
          let ok = true;
          for (let p = 0; p < c.length; p += 2) { const v = r[c[p]]; if (v !== c[p + 1] || (z && v === 0)) { ok = false; break; } }
          if (ok) sets[k].count++;
        }
      }
    }
    bruteForce(set, numItems, out) {
      const items = set.items, max = Math.pow(2, numItems);
      for (let j = 1; j < max; j++) {
        let np = 0, h = j; while (h > 0) { if (h % 2 === 1) np++; h = Math.floor(h / 2); }
        if (np >= numItems) continue;
        const prem = new Array(items.length).fill(-1), cons = new Array(items.length).fill(-1);
        h = j;
        for (let i = 0; i < items.length; i++) {
          if (items[i] === -1) continue;
          if (h % 2 === 1) prem[i] = items[i]; else cons[i] = items[i];
          h = Math.floor(h / 2);
        }
        const r = this.makeRule({ items: prem, count: this.hashtables[np - 1].get(key(prem)) }, cons, set.count, this.hashtables[numItems - np - 1].get(key(cons)));
        if (!(r.metrics[this.T] < this.C)) out.push(r);
      }
    }
    makeRule(premise, consequence, total, conseqCount) {
      const n = this.n;
      const conf = total / premise.count;
      const lift = conf / (conseqCount / n);
      const lev = total / n - (premise.count / n) * (conseqCount / n);
      const conv = (premise.count * (n - conseqCount) / n) / ((premise.count - total) + 1);
      return { premise: premise.items, premiseCount: premise.count, consequence: consequence, support: total, conseqCount, metrics: [conf, lift, lev, conv] };
    }
    generateRules(set, numItems, out) {
      const items = set.items, ht = this.hashtables[numItems - 2], ht0 = this.hashtables[0];
      let rules = [];
      for (let i = 0; i < items.length; i++) {
        if (items[i] === -1) continue;
        const prem = items.slice(); prem[i] = -1;
        const cons = new Array(items.length).fill(-1); cons[i] = items[i];
        rules.push(this.makeRule({ items: prem, count: ht.get(key(prem)) }, cons, set.count, ht0.get(key(cons))));
      }
      rules = rules.filter((r) => r.metrics[0] >= this.C);
      out.push(...rules);
      this.moreComplex(set, rules, numItems, 1, out);
    }
    moreComplex(set, rules, numItems, numInCons, out) {
      if (!(numItems > numInCons + 1)) return;
      const ht = this.hashtables[numItems - numInCons - 2], htc = this.hashtables[numInCons];
      const conses = mergeAll(rules.map((r) => ({ items: r.consequence })), numInCons - 1, set.items.length);
      let res = conses.map((c) => {
        const prem = set.items.map((v, i) => (c.items[i] !== -1 ? -1 : v));
        return this.makeRule({ items: prem, count: ht.get(key(prem)) }, c.items, set.count, htc.get(key(c.items)));
      });
      res = res.filter((r) => r.metrics[0] >= this.C);
      out.push(...res);
      this.moreComplex(set, res, numItems, numInCons + 1, out);
    }
    itemsText(items, count) {
      const d = this.data, parts = [];
      items.forEach((v, i) => { if (v !== -1) parts.push(d.attributes[i].name + '=' + d.attributes[i].values[v]); });
      return parts.join(' ') + ' ' + count;
    }
    toString() {
      const t = ['', 'Apriori', '=======', ''];
      const ms = this.printSup;
      t.push('Minimum support: ' + U.d2s(ms, 2) + ' (' + Math.trunc(ms * this.n + 0.5) + ' instances)');
      t.push('Minimum metric <' + ['confidence', 'lift', 'leverage', 'conviction'][this.T] + '>: ' + W.javaDouble(this.C));
      t.push('Number of cycles performed: ' + this.cycles, '', 'Generated sets of large itemsets:', '');
      this.Ls.forEach((L, i) => t.push('Size of set of large itemsets L(' + (i + 1) + '): ' + L.length, ''));
      t.push('Best rules found:', '');
      const w = Math.trunc(Math.log(this.N) / Math.log(10) + 1);
      this.rules.forEach((r, i) => {
        const m = r.metrics;
        const T = this.T;
        const f = (k, nm) => (T === k ? (k === 0 ? '<' : ' <') : '') + (k === 0 ? '' : ' ') + nm + ':(' + U.d2s(m[k], 2) + ')' + (T === k ? '>' : '');
        t.push(U.padL(String(i + 1), w) + '. ' + this.itemsText(r.premise, r.premiseCount) + ' ==> ' + this.itemsText(r.consequence, r.support) + '    ' + f(0, 'conf') + f(1, 'lift') + (T === 2 ? ' < lev:(' + U.d2s(m[2], 2) + ') [' + Math.trunc(m[2] * this.n) + ']>' : ' lev:(' + U.d2s(m[2], 2) + ') [' + Math.trunc(m[2] * this.n) + ']') + f(3, 'conv'));
      });
      return t.join('\n') + '\n';
    }
    ruleList() {
      return this.rules.map((r) => ({ premise: this.itemsText(r.premise, '').trim(), consequence: this.itemsText(r.consequence, '').trim(), premiseCount: r.premiseCount, support: r.support, conf: r.metrics[0], lift: r.metrics[1], lev: r.metrics[2], conv: r.metrics[3] }));
    }
  }
  function mergeAll(sets, size, na) {
    const out = [];
    for (let i = 0; i < sets.length; i++) {
      const a = sets[i].items;
      outer: for (let j = i + 1; j < sets.length; j++) {
        const b = sets[j].items, res = new Array(na);
        let found = 0, k = 0;
        while (found < size) {
          if (a[k] === b[k]) { if (a[k] !== -1) found++; res[k] = a[k]; } else break outer;
          k++;
        }
        for (; k < na; k++) {
          if (a[k] !== -1 && b[k] !== -1) break;
          res[k] = a[k] !== -1 ? a[k] : b[k];
        }
        if (k === na) out.push({ items: res, count: 0 });
      }
    }
    return out;
  }
  /** Utils.stableSort του Weka: αύξουσα σειρά, ισοβαθμίες κατά αρχικό δείκτη. */
  function stableSort(arr) {
    return arr.map((v, i) => [v !== v ? Number.MAX_VALUE : v, i]).sort((x, y) => (x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : x[1] - y[1])).map((x) => x[1]);
  }
  reg({
    name: 'Apriori', path: 'weka.associations.Apriori',
    desc: L`Κλασικός αλγόριθμος Apriori: μειώνει σταδιακά το minimum support μέχρι να βρει τους N καλύτερους κανόνες.`,
    options: [
      { key: 'N', flag: 'N', label: L`Πλήθος κανόνων`, type: 'int', default: 10, min: 1 },
      { key: 'T', flag: 'T', label: L`Μετρική ταξινόμησης`, type: 'select', default: 0, numeric: true, choices: { 0: 'Confidence', 1: 'Lift', 2: 'Leverage', 3: 'Conviction' } },
      { key: 'C', flag: 'C', label: L`Ελάχιστη τιμή μετρικής`, type: 'number', default: 0.9, min: 0, step: 0.05 },
      { key: 'D', flag: 'D', label: L`Βήμα μείωσης support (delta)`, type: 'number', default: 0.05, min: 0.001, step: 0.01 },
      { key: 'U', flag: 'U', label: L`Άνω όριο support`, type: 'number', default: 1.0, min: 0, max: 1, step: 0.05 },
      { key: 'M', flag: 'M', label: L`Κάτω όριο support`, type: 'number', default: 0.1, min: 0, max: 1, step: 0.05 },
      { key: 'Z', flag: 'Z', label: L`Η πρώτη τιμή θεωρείται ελλιπής`, type: 'bool', default: false },
    ],
    create: (o) => new Apriori(o),
  });

  // ============================================================== FPGrowth
  class FPGrowth {
    constructor(o = {}) { this.P = o.P ?? 2; this.N = o.N ?? 10; this.T = o.T ?? 0; this.C = o.C ?? 0.9; this.D = o.D ?? 0.05; this.U = o.U ?? 1.0; this.M = o.M ?? 0.1; this.I = o.I ?? -1; }
    build(data) {
      const d = data.copy(); d.classIndex = -1;
      this.data = d;
      // items: θετική τιμή κάθε nominal χαρακτηριστικού
      this.pos = d.attributes.map((a) => {
        if (a.type !== 'nominal') throw new Error(L`Το FPGrowth δέχεται μόνο nominal/δυαδικά χαρακτηριστικά ("${a.name}").`);
        if (a.values.length > 2) throw new Error(L`Το FPGrowth απαιτεί δυαδικά χαρακτηριστικά ("${a.name}" έχει ${a.values.length} τιμές). Δοκίμασε NominalToBinary.`);
        return a.values.length === 1 ? 0 : Math.min(this.P - 1, a.values.length - 1);
      });
      const n = d.numInstances;
      const tx = d.rows.map((r) => { const t = []; r.forEach((v, j) => { if (v === this.pos[j]) t.push(j); }); return t; });
      this.freq = new Array(d.numAttributes).fill(0);
      for (const t of tx) for (const j of t) this.freq[j]++;
      const lower = this.M * n < 1 ? 1 / n : this.M;
      let minSup = this.U - this.D;
      if (minSup < lower) minSup = lower;
      this.cycles = 0;
      let rules = [];
      for (;;) {
        this.cycles++;
        const nec = Math.ceil(minSup * n - 1e-9);
        const sets = mineFP(tx, Math.max(1, nec), this.I);
        this.numSets = sets.length;
        rules = this.rulesFrom(sets, n);
        if (rules.length >= this.N || minSup <= lower + 1e-12) break;
        minSup = Math.max(lower, minSup - this.D);
      }
      this.minSupport = minSup;
      this.total = rules.length;
      rules.sort((a, b) => b.metrics[this.T] - a.metrics[this.T] || b.support - a.support);
      this.rules = rules.slice(0, this.N);
      return this;
    }
    rulesFrom(sets, n) {
      const sup = new Map(sets.map((s) => [s.items.join(','), s.count]));
      const rules = [];
      for (const s of sets) {
        if (s.items.length < 2) continue;
        const k = s.items.length;
        for (let mask = 1; mask < (1 << k) - 1; mask++) {
          const prem = [], cons = [];
          s.items.forEach((it, i) => ((mask >> i) & 1 ? cons : prem).push(it));
          const pc = sup.get(prem.join(',')), cc = sup.get(cons.join(','));
          if (!pc || !cc) continue;
          const conf = s.count / pc, lift = conf / (cc / n), lev = s.count / n - (pc / n) * (cc / n), conv = (pc * (n - cc) / n) / (pc - s.count + 1);
          const metrics = [conf, lift, lev, conv];
          if (metrics[this.T] < this.C) continue;
          rules.push({ prem, cons, premCount: pc, support: s.count, metrics });
        }
      }
      return rules;
    }
    itemList(items) { const d = this.data; items = items.slice().sort((a, b) => this.freq[b] - this.freq[a] || a - b); return '[' + items.map((j) => d.attributes[j].name + '=' + d.attributes[j].values[this.pos[j]]).join(', ') + ']'; }
    toString() {
      const t = [`FPGrowth found ${this.total} rules (displaying top ${Math.min(this.N, this.rules.length)})`, ''];
      const T = this.T;
      this.rules.forEach((r, i) => {
        const f = (k, nm) => (T === k ? '<' : '') + nm + ':(' + U.d2s(r.metrics[k], 2) + ')' + (T === k ? '>' : '');
        t.push(U.padL(String(i + 1), 2) + '. ' + this.itemList(r.prem) + ': ' + r.premCount + ' ==> ' + this.itemList(r.cons) + ': ' + r.support + '   ' + f(0, 'conf') + ' ' + f(1, 'lift') + ' ' + f(2, 'lev') + ' ' + f(3, 'conv') + ' ');
      });
      return t.join('\n') + '\n';
    }
    ruleList() {
      return this.rules.map((r) => ({ premise: this.itemList(r.prem).slice(1, -1), consequence: this.itemList(r.cons).slice(1, -1), premiseCount: r.premCount, support: r.support, conf: r.metrics[0], lift: r.metrics[1], lev: r.metrics[2], conv: r.metrics[3] }));
    }
  }
  /** FP-tree mining: επιστρέφει όλα τα συχνά itemsets {items (ταξινομημένα), count}. */
  function mineFP(tx, minCount, maxItems) {
    const freq = new Map();
    for (const t of tx) for (const it of t) freq.set(it, (freq.get(it) || 0) + 1);
    const out = [];
    const grow = (transactions, suffix) => {
      // transactions: [{items, count}]
      const f = new Map();
      for (const t of transactions) for (const it of t.items) f.set(it, (f.get(it) || 0) + t.count);
      const items = [...f.entries()].filter(([, c]) => c >= minCount).sort((a, b) => b[1] - a[1] || a[0] - b[0]);
      for (let idx = items.length - 1; idx >= 0; idx--) {
        const [it, c] = items[idx];
        const set = suffix.concat([it]);
        out.push({ items: set.slice().sort((a, b) => a - b), count: c });
        if (maxItems > 0 && set.length >= maxItems) continue;
        // conditional pattern base: συναλλαγές που περιέχουν το it, με items υψηλότερης συχνότητας
        const rank = new Map(items.map(([x], i) => [x, i]));
        const cond = [];
        for (const t of transactions) {
          if (!t.items.includes(it)) continue;
          const rest = t.items.filter((x) => rank.has(x) && rank.get(x) < idx);
          if (rest.length) cond.push({ items: rest, count: t.count });
        }
        if (cond.length) grow(cond, set);
      }
    };
    grow(tx.map((t) => ({ items: t, count: 1 })), []);
    return out;
  }
  reg({
    name: 'FPGrowth', path: 'weka.associations.FPGrowth',
    desc: L`Εξόρυξη συχνών itemsets με FP-tree, χωρίς παραγωγή υποψηφίων. Για δυαδικά δεδομένα (market basket).`,
    options: [
      { key: 'P', flag: 'P', label: L`Δείκτης θετικής τιμής (1-based)`, type: 'int', default: 2, min: 1 },
      { key: 'N', flag: 'N', label: L`Πλήθος κανόνων`, type: 'int', default: 10, min: 1 },
      { key: 'T', flag: 'T', label: L`Μετρική`, type: 'select', default: 0, numeric: true, choices: { 0: 'Confidence', 1: 'Lift', 2: 'Leverage', 3: 'Conviction' } },
      { key: 'C', flag: 'C', label: L`Ελάχιστη τιμή μετρικής`, type: 'number', default: 0.9, min: 0, step: 0.05 },
      { key: 'D', flag: 'D', label: L`Βήμα μείωσης support`, type: 'number', default: 0.05, min: 0.001, step: 0.01 },
      { key: 'U', flag: 'U', label: L`Άνω όριο support`, type: 'number', default: 1.0, min: 0, max: 1, step: 0.05 },
      { key: 'M', flag: 'M', label: L`Κάτω όριο support`, type: 'number', default: 0.1, min: 0, max: 1, step: 0.05 },
      { key: 'I', flag: 'I', label: L`Μέγιστο μέγεθος itemset (−1 = χωρίς όριο)`, type: 'int', default: -1, min: -1 },
    ],
    create: (o) => new FPGrowth(o),
  });

  W.runAssociator = async function (opts) {
    const { spec, options, data } = opts;
    await W.nextFrame();
    const t0 = performance.now();
    const m = spec.create(Object.assign({}, options));
    m.build(data);
    const optStr = W.optionString(spec, options);
    const head = ['=== Run information ===', '', 'Scheme:       ' + spec.path + (optStr ? ' ' + optStr : ''), 'Relation:     ' + data.relation, 'Instances:    ' + data.numInstances, 'Attributes:   ' + data.numAttributes, ...data.attributes.map((a) => '              ' + a.name), '=== Associator model (full training set) ===', ''];
    return { model: m, report: head.join('\n') + '\n' + m.toString(), rules: m.ruleList(), time: (performance.now() - t0) / 1000, spec, options, optStr };
  };
})();
