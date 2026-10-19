/* Weka Web — πυρήνας: namespace, βοηθητικά, java.util.Random, μοντέλο δεδομένων, ARFF/CSV. */
(function () {
  'use strict';
  const W = (window.WEKA = window.WEKA || {});
  W.datasets = {};
  W.registerDataset = (id, text, format) => { W.datasets[id] = text; if (format) (W.datasetFormats = W.datasetFormats || {})[id] = format; };
  W.parseSample = (id) => ((W.datasetFormats || {})[id] === 'csv' ? W.parseCSV(W.datasets[id], id + '.csv') : W.parseARFF(W.datasets[id]));

  // ---------------------------------------------------------------- Utils
  const U = (W.Utils = {});
  U.SMALL = 1e-6;
  U.eq = (a, b) => a - b < U.SMALL && b - a < U.SMALL;
  U.gr = (a, b) => a - b > U.SMALL;
  U.grOrEq = (a, b) => b - a < U.SMALL || a >= b;
  U.sm = (a, b) => b - a > U.SMALL;
  U.smOrEq = (a, b) => a - b < U.SMALL || a <= b;
  U.log2 = (x) => Math.log(x) / Math.LN2;
  U.isMissing = (v) => v !== v; // NaN
  U.sum = (a) => { let s = 0; for (const v of a) s += v; return s; };
  U.maxIndex = (a) => { let m = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[m]) m = i; return m; };
  U.minIndex = (a) => { let m = 0; for (let i = 1; i < a.length; i++) if (a[i] < a[m]) m = i; return m; };
  U.normalize = (a) => { const s = U.sum(a); if (s > 0 && isFinite(s)) for (let i = 0; i < a.length; i++) a[i] /= s; return a; };
  U.zeros = (n) => new Array(n).fill(0);
  U.matrix = (r, c, v = 0) => Array.from({ length: r }, () => new Array(c).fill(v));
  U.lnFunc = (n) => (n < 1e-6 ? 0 : n * Math.log(n));

  /** Ισοδύναμο του Utils.doubleToString(value, afterDecimal) του Weka. */
  U.d2s = function (value, afterDecimal = 6) {
    if (value !== value) return '?';
    if (!isFinite(value)) return value > 0 ? 'Infinity' : '-Infinity';
    let s = value.toFixed(afterDecimal);
    if (s.indexOf('.') >= 0) s = s.replace(/0+$/, '').replace(/\.$/, '');
    if (s === '-0') s = '0';
    return s;
  };
  /** Utils.doubleToString(value, width, afterDecimal): σταθερά δεκαδικά, δεξιά στοίχιση. */
  U.d2sw = function (value, width, afterDecimal) {
    let s;
    if (value !== value) s = '?';
    else {
      s = value.toFixed(afterDecimal);
      if (/^-0\.?0*$/.test(s)) s = s.slice(1);
    }
    return s.length < width ? ' '.repeat(width - s.length) + s : s;
  };
  U.padL = (s, w) => { s = String(s); return s.length < w ? ' '.repeat(w - s.length) + s : s; };
  U.padR = (s, w) => { s = String(s); return s.length < w ? s + ' '.repeat(w - s.length) : s; };
  U.roundDouble = (v, d) => { const p = Math.pow(10, d); return Math.round(v * p) / p; };
  U.fmt = (v, d = 4) => (v === null || v === undefined || v !== v ? '—' : U.d2s(v, d));

  /** Επιλογή k-οστού / ταξινόμηση με σταθερότητα (stable). */
  U.argsort = function (vals) {
    const idx = vals.map((_, i) => i);
    idx.sort((a, b) => {
      const va = vals[a], vb = vals[b];
      const ma = va !== va, mb = vb !== vb;
      if (ma || mb) return ma === mb ? a - b : ma ? 1 : -1;
      return va < vb ? -1 : va > vb ? 1 : a - b;
    });
    return idx;
  };

  /** Αναπαραγωγή του weka.core.Utils.sort(double[]) (quicksort, μη σταθερό) για ίδια σειρά ισοβαθμιών. */
  U.wekaSort = function (arr) {
    const a = arr.map((v) => (v !== v ? Number.MAX_VALUE : v));
    const index = a.map((_, i) => i);
    const cswap = (l, r) => { if (a[index[l]] > a[index[r]]) { const h = index[l]; index[l] = index[r]; index[r] = h; } };
    const sw = (l, r) => { const h = index[l]; index[l] = index[r]; index[r] = h; };
    const partition = (l, r, pivot) => {
      r--;
      for (;;) {
        while (a[index[++l]] < pivot);
        while (a[index[--r]] > pivot);
        if (l >= r) return l;
        sw(l, r);
      }
    };
    const qs = (left, right) => {
      const diff = right - left;
      if (diff <= 0) return;
      if (diff === 1) { cswap(left, right); return; }
      if (diff === 2) { cswap(left, left + 1); cswap(left, right); cswap(left + 1, right); return; }
      const c = Math.floor((left + right) / 2);
      cswap(left, c); cswap(left, right); cswap(c, right);
      sw(c, right - 1);
      const center = partition(left, right, a[index[right - 1]]);
      sw(center, right - 1);
      qs(left, center - 1); qs(center + 1, right);
    };
    if (a.length > 1) qs(0, a.length - 1);
    return index;
  };

  /** Ανάλυση εύρους τύπου Weka ("first-3,5,last") σε δείκτες 0-based. */
  U.parseRange = function (text, n) {
    const out = new Set();
    const tok = (t) => {
      t = t.trim().toLowerCase();
      if (t === 'first') return 0;
      if (t === 'last') return n - 1;
      const v = parseInt(t, 10);
      if (!(v >= 1 && v <= n)) throw new Error(L`Μη έγκυρο εύρος: ` + t);
      return v - 1;
    };
    for (const part of String(text).split(',')) {
      if (!part.trim()) continue;
      const [a, b] = part.split('-');
      if (b === undefined) out.add(tok(a));
      else { const s = tok(a), e = tok(b); for (let i = Math.min(s, e); i <= Math.max(s, e); i++) out.add(i); }
    }
    return [...out].sort((a, b) => a - b);
  };

  // ------------------------------------------------- java.util.Random clone
  /** Ακριβής αναπαραγωγή του java.util.Random (48-bit LCG) ώστε τα seeds να ταυτίζονται με το Weka. */
  class JavaRandom {
    constructor(seed = 1) { this.setSeed(seed); }
    setSeed(seed) {
      // (seed ^ 0x5DEECE66D) & ((1<<48)-1), με seed ως 64-bit long
      let s = BigInt.asUintN(64, BigInt(Math.trunc(seed)));
      s = (s ^ 0x5DEECE66Dn) & 0xFFFFFFFFFFFFn;
      this.hi = Number(s >> 24n);
      this.lo = Number(s & 0xFFFFFFn);
      this.haveNextNextGaussian = false;
    }
    next(bits) {
      const MLO = 0xECE66D, MHI = 0x5DE;
      const loProd = this.lo * MLO + 0xB;
      const carry = Math.floor(loProd / 0x1000000);
      const newLo = loProd - carry * 0x1000000;
      const cross = this.hi * MLO + this.lo * MHI + carry;
      const newHi = cross % 0x1000000;
      this.hi = newHi; this.lo = newLo;
      // seed >>> (48 - bits)
      const full = newHi * 0x1000000 + newLo;
      const r = Math.floor(full / Math.pow(2, 48 - bits));
      return r | 0;
    }
    nextInt(bound) {
      if (bound === undefined) return this.next(32);
      if (bound <= 0) throw new Error('bound must be positive');
      if ((bound & -bound) === bound) {
        return Number((BigInt(bound) * BigInt(this.next(31))) >> 31n);
      }
      let bits, val;
      do {
        bits = this.next(31);
        val = bits % bound;
      } while (bits - val + (bound - 1) > 0x7fffffff);
      return val;
    }
    nextDouble() {
      return (this.next(26) * 134217728 + this.next(27)) * Math.pow(2, -53);
    }
    nextBoolean() { return this.next(1) !== 0; }
    nextGaussian() {
      if (this.haveNextNextGaussian) { this.haveNextNextGaussian = false; return this.nextNextGaussian; }
      let v1, v2, s;
      do {
        v1 = 2 * this.nextDouble() - 1;
        v2 = 2 * this.nextDouble() - 1;
        s = v1 * v1 + v2 * v2;
      } while (s >= 1 || s === 0);
      const mul = Math.sqrt(-2 * Math.log(s) / s);
      this.nextNextGaussian = v2 * mul;
      this.haveNextNextGaussian = true;
      return v1 * mul;
    }
    nextLong() {
      // ((long)next(32) << 32) + next(32) — χρειάζεται για seeds που δίνονται σε υπο-μοντέλα
      const a = BigInt(this.next(32)), b = BigInt(this.next(32));
      return BigInt.asIntN(64, (a << 32n) + b);
    }
  }
  W.JavaRandom = JavaRandom;

  // ---------------------------------------------------------- Dataset model
  /**
   * Attribute: { name, type: 'numeric'|'nominal'|'string'|'date', values?: string[], dateFormat? }
   * Οι nominal/string τιμές αποθηκεύονται ως δείκτης στο values. Λείπουσα τιμή = NaN.
   */
  class Dataset {
    constructor(relation, attributes, rows = [], weights = null, classIndex = -1) {
      this.relation = relation;
      this.attributes = attributes;
      this.rows = rows;
      this.weights = weights || rows.map(() => 1);
      this.classIndex = classIndex;
    }
    get numInstances() { return this.rows.length; }
    get numAttributes() { return this.attributes.length; }
    get classAttr() { return this.classIndex >= 0 ? this.attributes[this.classIndex] : null; }
    get numClasses() { const c = this.classAttr; return !c ? 0 : c.type === 'nominal' ? c.values.length : 1; }
    classIsNominal() { const c = this.classAttr; return !!c && c.type === 'nominal'; }
    isNominal(i) { return this.attributes[i].type === 'nominal'; }
    isNumeric(i) { const t = this.attributes[i].type; return t === 'numeric' || t === 'date'; }
    sumOfWeights() { let s = 0; for (const w of this.weights) s += w; return s; }
    /** Αντίγραφο της δομής (header) χωρίς εγγραφές. */
    emptyCopy() {
      return new Dataset(this.relation, this.attributes.map(cloneAttr), [], [], this.classIndex);
    }
    /** Ρηχό αντίγραφο: νέοι πίνακες γραμμών/βαρών, ίδιοι πίνακες τιμών. */
    copy() {
      return new Dataset(this.relation, this.attributes, this.rows.slice(), this.weights.slice(), this.classIndex);
    }
    deepCopy() {
      return new Dataset(this.relation, this.attributes.map(cloneAttr), this.rows.map((r) => r.slice()), this.weights.slice(), this.classIndex);
    }
    subset(indices) {
      return new Dataset(this.relation, this.attributes, indices.map((i) => this.rows[i]), indices.map((i) => this.weights[i]), this.classIndex);
    }
    withRows(rows, weights) {
      return new Dataset(this.relation, this.attributes, rows, weights || rows.map(() => 1), this.classIndex);
    }
    /** Αφαίρεση εγγραφών με λείπουσα κλάση. */
    withoutMissingClass() {
      const ci = this.classIndex;
      if (ci < 0) return this.copy();
      const idx = [];
      for (let i = 0; i < this.rows.length; i++) if (this.rows[i][ci] === this.rows[i][ci]) idx.push(i);
      return idx.length === this.rows.length ? this.copy() : this.subset(idx);
    }
    value(i, j) { return this.rows[i][j]; }
    stringValue(row, j) {
      const v = row[j];
      if (v !== v) return '?';
      const a = this.attributes[j];
      if (a.type === 'nominal' || a.type === 'string') return a.values[v];
      if (a.type === 'date') return formatDate(v, a.dateFormat);
      return U.d2s(v, 6);
    }
    /** java Instances.randomize(Random) */
    randomize(rand) {
      for (let j = this.rows.length - 1; j > 0; j--) {
        const k = rand.nextInt(j + 1);
        swap(this.rows, j, k); swap(this.weights, j, k);
      }
    }
    swap(i, j) { swap(this.rows, i, j); swap(this.weights, i, j); }
    /** java Instances.stratify(numFolds) */
    stratify(numFolds) {
      if (!this.classIsNominal()) return;
      const ci = this.classIndex, n = this.rows.length;
      let index = 1;
      while (index < n) {
        const c1 = this.rows[index - 1][ci];
        for (let j = index; j < n; j++) {
          const c2 = this.rows[j][ci];
          if (c1 === c2 || (c1 !== c1 && c2 !== c2)) { this.swap(index, j); index++; }
        }
        index++;
      }
      const nr = [], nw = [];
      let start = 0;
      while (nr.length < n) {
        for (let j = start; j < n; j += numFolds) { nr.push(this.rows[j]); nw.push(this.weights[j]); }
        start++;
      }
      this.rows = nr; this.weights = nw;
    }
    foldBounds(numFolds, fold) {
      const n = this.rows.length;
      let numInstForFold = Math.floor(n / numFolds), offset;
      if (fold < n % numFolds) { numInstForFold++; offset = fold; } else offset = n % numFolds;
      const first = fold * Math.floor(n / numFolds) + offset;
      return [first, numInstForFold];
    }
    testCV(numFolds, fold) {
      const [first, cnt] = this.foldBounds(numFolds, fold);
      return this.withRows(this.rows.slice(first, first + cnt), this.weights.slice(first, first + cnt));
    }
    trainCV(numFolds, fold, rand) {
      const [first, cnt] = this.foldBounds(numFolds, fold);
      const rows = this.rows.slice(0, first).concat(this.rows.slice(first + cnt));
      const ws = this.weights.slice(0, first).concat(this.weights.slice(first + cnt));
      const d = this.withRows(rows, ws);
      if (rand) d.randomize(rand);
      return d;
    }
    /** java Instances.sort(attIndex): σταθερή ταξινόμηση, λείπουσες στο τέλος. */
    sortBy(att) {
      const vals = this.rows.map((r) => r[att]);
      const order = U.argsort(vals);
      this.rows = order.map((i) => this.rows[i]);
      this.weights = order.map((i) => this.weights[i]);
    }
    /** Στατιστικά χαρακτηριστικού για την καρτέλα προεπεξεργασίας. */
    attributeStats(j) {
      const a = this.attributes[j];
      let missing = 0;
      const counts = new Map();
      for (const r of this.rows) {
        const v = r[j];
        if (v !== v) { missing++; continue; }
        counts.set(v, (counts.get(v) || 0) + 1);
      }
      let unique = 0;
      for (const c of counts.values()) if (c === 1) unique++;
      const st = { missing, distinct: counts.size, unique, total: this.rows.length };
      if (a.type === 'nominal' || a.type === 'string') {
        st.counts = a.values.map((_, k) => counts.get(k) || 0);
      } else {
        let sw = 0, s = 0, s2 = 0, mn = Infinity, mx = -Infinity;
        for (let i = 0; i < this.rows.length; i++) {
          const v = this.rows[i][j];
          if (v !== v) continue;
          const w = this.weights[i];
          sw += w; s += w * v; s2 += w * v * v;
          if (v < mn) mn = v; if (v > mx) mx = v;
        }
        const mean = sw > 0 ? s / sw : NaN;
        let sd = NaN;
        if (sw > 1) { sd = (s2 - (s * s) / sw) / (sw - 1); sd = sd <= 0 ? 0 : Math.sqrt(sd); }
        st.min = sw > 0 ? mn : NaN; st.max = sw > 0 ? mx : NaN; st.mean = mean; st.stdDev = sd; st.sumWeights = sw;
      }
      return st;
    }
  }
  function swap(a, i, j) { const t = a[i]; a[i] = a[j]; a[j] = t; }
  function cloneAttr(a) { const c = Object.assign({}, a); if (a.values) c.values = a.values.slice(); return c; }
  W.Dataset = Dataset;
  W.cloneAttr = cloneAttr;

  // ---------------------------------------------------------------- dates
  function pad(n, w = 2) { return String(n).padStart(w, '0'); }
  function formatDate(ms, fmt) {
    const d = new Date(ms);
    fmt = fmt || "yyyy-MM-dd'T'HH:mm:ss";
    return fmt.replace(/yyyy|MM|dd|HH|mm|ss|'T'/g, (t) => ({
      yyyy: d.getFullYear(), MM: pad(d.getMonth() + 1), dd: pad(d.getDate()), HH: pad(d.getHours()), mm: pad(d.getMinutes()), ss: pad(d.getSeconds()), "'T'": 'T',
    })[t]);
  }
  function parseDate(s) { const t = Date.parse(s.replace(' ', 'T')); return isNaN(t) ? NaN : t; }

  // ----------------------------------------------------------- ARFF parser
  /** Tokenizer για ARFF: σέβεται εισαγωγικά ' και " και διαχωριστικά , { } */
  function tokenizeLine(line) {
    const out = [];
    let i = 0;
    const n = line.length;
    while (i < n) {
      const c = line[i];
      if (c === ' ' || c === '\t' || c === ',') { i++; continue; }
      if (c === '%') break;
      if (c === '{' || c === '}') { out.push({ t: c }); i++; continue; }
      if (c === "'" || c === '"') {
        let j = i + 1, s = '';
        while (j < n && line[j] !== c) {
          if (line[j] === '\\' && j + 1 < n) { const e = line[j + 1]; s += e === 'n' ? '\n' : e === 't' ? '\t' : e; j += 2; }
          else s += line[j++];
        }
        out.push({ v: s, q: true });
        i = j + 1;
        continue;
      }
      let j = i;
      while (j < n && !' \t,{}'.includes(line[j])) j++;
      out.push({ v: line.slice(i, j) });
      i = j;
    }
    return out;
  }

  W.parseARFF = function (text) {
    const lines = text.split(/\r\n|\r|\n/);
    let relation = 'data';
    const attrs = [];
    const rows = [], weights = [];
    let inData = false, li = 0;
    for (; li < lines.length; li++) {
      const raw = lines[li].trim();
      if (!raw || raw[0] === '%') continue;
      const low = raw.toLowerCase();
      if (!inData) {
        if (low.startsWith('@relation')) {
          const t = tokenizeLine(raw.slice(9));
          relation = t.length ? t[0].v : 'data';
        } else if (low.startsWith('@attribute')) {
          const rest = raw.slice(10).trim();
          let name, typePart;
          if (rest[0] === "'" || rest[0] === '"') {
            const q = rest[0]; let j = 1;
            while (j < rest.length && rest[j] !== q) { if (rest[j] === '\\') j++; j++; }
            name = rest.slice(1, j).replace(/\\(.)/g, '$1');
            typePart = rest.slice(j + 1).trim();
          } else {
            const m = rest.match(/^(\S+?)(\s+|(?=\{))(.*)$/);
            if (!m) throw new Error(L`Μη έγκυρη δήλωση χαρακτηριστικού (γραμμή ` + (li + 1) + ')');
            name = m[1]; typePart = m[3].trim();
          }
          const tl = typePart.toLowerCase();
          if (typePart[0] === '{') {
            const inner = typePart.slice(1, typePart.lastIndexOf('}'));
            const vals = tokenizeLine(inner).filter((t) => t.v !== undefined).map((t) => t.v);
            attrs.push({ name, type: 'nominal', values: vals });
          } else if (/^(numeric|real|integer)\b/.test(tl)) {
            attrs.push({ name, type: 'numeric', integer: tl.startsWith('integer') });
          } else if (tl.startsWith('string')) {
            attrs.push({ name, type: 'string', values: [] });
          } else if (tl.startsWith('date')) {
            const t = tokenizeLine(typePart.slice(4));
            attrs.push({ name, type: 'date', dateFormat: t.length ? t[0].v : "yyyy-MM-dd'T'HH:mm:ss" });
          } else if (tl.startsWith('relational')) {
            throw new Error(L`Τα relational χαρακτηριστικά δεν υποστηρίζονται (` + name + ').');
          } else {
            throw new Error(L`Άγνωστος τύπος χαρακτηριστικού: ` + typePart);
          }
        } else if (low.startsWith('@data')) {
          inData = true;
        } else {
          throw new Error(L`Μη αναγνωρίσιμη γραμμή κεφαλίδας ` + (li + 1) + ': ' + raw.slice(0, 60));
        }
        continue;
      }
      // data section
      const toks = tokenizeLine(raw);
      if (!toks.length) continue;
      const row = new Array(attrs.length);
      let w = 1;
      if (toks[0].t === '{') {
        // sparse: default 0 for numeric, first value for nominal
        for (let j = 0; j < attrs.length; j++) row[j] = 0;
        let k = 1;
        for (; k < toks.length && toks[k].t !== '}'; k += 2) {
          const idx = parseInt(toks[k].v, 10);
          row[idx] = convertValue(attrs[idx], toks[k + 1], li);
        }
        k++;
        if (k < toks.length && toks[k].t === '{') w = parseFloat(toks[k + 1].v);
      } else {
        let k = 0;
        for (let j = 0; j < attrs.length; j++, k++) {
          if (k >= toks.length || toks[k].t) throw new Error(L`Λιγότερες τιμές από χαρακτηριστικά στη γραμμή ` + (li + 1));
          row[j] = convertValue(attrs[j], toks[k], li);
        }
        if (k < toks.length && toks[k].t === '{') w = parseFloat(toks[k + 1].v);
      }
      rows.push(row);
      weights.push(w);
    }
    if (!attrs.length) throw new Error(L`Δεν βρέθηκαν χαρακτηριστικά (@attribute).`);
    return new Dataset(relation, attrs, rows, weights, attrs.length - 1);
  };

  function convertValue(a, tok, li) {
    const s = tok.v;
    if (!tok.q && s === '?') return NaN;
    switch (a.type) {
      case 'numeric': {
        const v = Number(s);
        if (s === '' || isNaN(v)) throw new Error(L`Μη αριθμητική τιμή "${s}" για ${a.name} (γραμμή ${li + 1})`);
        return v;
      }
      case 'nominal': {
        const k = a.values.indexOf(s);
        if (k < 0) throw new Error(L`Τιμή "${s}" εκτός πεδίου ορισμού του ${a.name} (γραμμή ${li + 1})`);
        return k;
      }
      case 'string': {
        let k = a._index ? a._index.get(s) : undefined;
        if (!a._index) Object.defineProperty(a, '_index', { value: new Map(), enumerable: false, writable: true });
        if (k === undefined) { k = a.values.length; a.values.push(s); a._index.set(s, k); }
        return k;
      }
      case 'date': return parseDate(s);
    }
    return NaN;
  }

  // ------------------------------------------------------------ CSV parser
  W.parseCSV = function (text, name = 'csv') {
    const firstLine = text.split(/\r?\n/, 1)[0] || '';
    const counts = { ',': 0, ';': 0, '\t': 0 };
    for (const ch of firstLine) if (ch in counts) counts[ch]++;
    const sep = Object.keys(counts).reduce((a, b) => (counts[b] > counts[a] ? b : a), ',');
    const records = [];
    let field = '', rec = [], q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) {
        if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; }
        else field += c;
      } else if (c === '"') q = true;
      else if (c === sep) { rec.push(field); field = ''; }
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        rec.push(field); field = '';
        if (rec.length > 1 || rec[0].trim() !== '') records.push(rec);
        rec = [];
      } else field += c;
    }
    if (field !== '' || rec.length) { rec.push(field); if (rec.length > 1 || rec[0].trim() !== '') records.push(rec); }
    if (records.length < 2) throw new Error(L`Το CSV χρειάζεται γραμμή κεφαλίδας και τουλάχιστον μία εγγραφή.`);
    const header = records[0].map((h, i) => h.trim() || 'attr' + (i + 1));
    const body = records.slice(1);
    const isMiss = (s) => s === undefined || s.trim() === '' || s.trim() === '?';
    const attrs = header.map((h, j) => {
      let numeric = true;
      for (const r of body) { const s = r[j]; if (isMiss(s)) continue; if (isNaN(Number(s.trim()))) { numeric = false; break; } }
      if (numeric) return { name: h, type: 'numeric' };
      const vals = [];
      const seen = new Set();
      for (const r of body) { const s = r[j]; if (isMiss(s)) continue; const t = s.trim(); if (!seen.has(t)) { seen.add(t); vals.push(t); } }
      return { name: h, type: 'nominal', values: vals };
    });
    const rows = body.map((r) => attrs.map((a, j) => {
      const s = r[j];
      if (isMiss(s)) return NaN;
      return a.type === 'numeric' ? Number(s.trim()) : a.values.indexOf(s.trim());
    }));
    return new Dataset(name.replace(/\.[^.]+$/, ''), attrs, rows, null, attrs.length - 1);
  };

  // ------------------------------------------------------------ ARFF writer
  function quote(s) {
    s = String(s);
    if (s === '') return "''";
    if (/[\s,{}'"%?\\]/.test(s)) return "'" + s.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n').replace(/\t/g, '\\t') + "'";
    return s;
  }
  W.quote = quote;
  W.attrDecl = function (a) {
    let t;
    if (a.type === 'nominal') t = '{' + a.values.map(quote).join(',') + '}';
    else if (a.type === 'string') t = 'string';
    else if (a.type === 'date') t = 'date ' + quote(a.dateFormat);
    else t = 'numeric';
    return '@attribute ' + quote(a.name) + ' ' + t;
  };
  W.toARFF = function (d, maxRows = Infinity) {
    const out = ['@relation ' + quote(d.relation), ''];
    for (const a of d.attributes) out.push(W.attrDecl(a));
    out.push('', '@data');
    const n = Math.min(d.rows.length, maxRows);
    for (let i = 0; i < n; i++) {
      const r = d.rows[i];
      let line = r.map((v, j) => {
        if (v !== v) return '?';
        const a = d.attributes[j];
        if (a.type === 'nominal' || a.type === 'string') return quote(a.values[v]);
        if (a.type === 'date') return quote(formatDate(v, a.dateFormat));
        return String(v);
      }).join(',');
      if (d.weights[i] !== 1) line += ',{' + d.weights[i] + '}';
      out.push(line);
    }
    return out.join('\n') + '\n';
  };
  W.toCSV = function (d) {
    const esc = (s) => (/[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s);
    const out = [d.attributes.map((a) => esc(a.name)).join(',')];
    for (const r of d.rows) out.push(r.map((v, j) => (v !== v ? '?' : esc(d.stringValue(r, j)))).join(','));
    return out.join('\n') + '\n';
  };

  /** Ελέγχει αν δύο σύνολα έχουν συμβατή δομή (για supplied test set). */
  W.equalHeadersMsg = function (a, b) {
    if (a.numAttributes !== b.numAttributes) return L`διαφορετικός αριθμός χαρακτηριστικών`;
    for (let j = 0; j < a.numAttributes; j++) {
      const x = a.attributes[j], y = b.attributes[j];
      if (x.name !== y.name) return L`διαφορετικό όνομα στη θέση ${j + 1} (${x.name} / ${y.name})`;
      if (x.type !== y.type) return L`διαφορετικός τύπος για ${x.name}`;
      if (x.type === 'nominal' && x.values.join('\u0001') !== y.values.join('\u0001')) return L`διαφορετικές τιμές για ${x.name}`;
    }
    return null;
  };

  // ------------------------------------------------------ Statistics helpers
  const S = (W.Stats = {});
  /** Αντίστροφη κανονική κατανομή (Wichura AS241, ακρίβεια ~1e-16). */
  S.normalInverse = function (p) {
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    const q = p - 0.5;
    let r, val;
    if (Math.abs(q) <= 0.425) {
      r = 0.180625 - q * q;
      val = q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r + 133.14166789178437745) * r + 3.387132872796366608) /
        (((((((r * 5226.495278852545925 + 28729.085735721942674) * r + 39307.89580009271061) * r + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r + 42.313330701600911252) * r + 1);
      return val;
    }
    r = q < 0 ? p : 1 - p;
    r = Math.sqrt(-Math.log(r));
    if (r <= 5) {
      r -= 1.6;
      val = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + 0.24178072517745061177) * r + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r + 4.6303378461565452959) * r + 1.42343711074968357734) /
        (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r + 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r + 2.05319162663775882187) * r + 1);
    } else {
      r -= 5;
      val = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r + 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r + 5.4637849111641143699) * r + 6.6579046435011037772) /
        (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r + 0.59983220655588793769) * r + 1);
    }
    return q < 0 ? -val : val;
  };
  S.normalCDF = function (x) {
    // Abramowitz-Stegun 7.1.26 μέσω erf, αρκετό για εμφάνιση
    const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x / 2);
    return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
  };
  S.entropy = function (counts) {
    let tot = 0, r = 0;
    for (const c of counts) { r -= U.lnFunc(c); tot += c; }
    if (tot === 0) return 0;
    return (r + U.lnFunc(tot)) / (tot * Math.LN2);
  };
  /** Εντροπία υπό συνθήκη ως προς γραμμές (ContingencyTables.entropyConditionedOnRows) */
  S.entropyConditionedOnRows = function (m) {
    let ret = 0, tot = 0;
    for (const row of m) { let s = 0; for (const v of row) { ret += U.lnFunc(v); s += v; } ret -= U.lnFunc(s); tot += s; }
    if (tot === 0) return 0;
    return -ret / (tot * Math.LN2);
  };
  S.entropyOverColumns = function (m) {
    const nc = m[0].length; let ret = 0, tot = 0;
    for (let j = 0; j < nc; j++) { let s = 0; for (const row of m) s += row[j]; ret -= U.lnFunc(s); tot += s; }
    if (tot === 0) return 0;
    return (ret + U.lnFunc(tot)) / (tot * Math.LN2);
  };
  S.entropyOverRows = function (m) {
    let ret = 0, tot = 0;
    for (const row of m) { const s = U.sum(row); ret -= U.lnFunc(s); tot += s; }
    if (tot === 0) return 0;
    return (ret + U.lnFunc(tot)) / (tot * Math.LN2);
  };

  // --------------------------------------------------------- Misc helpers
  /** Εκτίμηση "μεγέθους" αποτελέσματος για async UI */
  W.nextFrame = () => new Promise((r) => setTimeout(r, 16));
  W.formatDate = formatDate;
})();
