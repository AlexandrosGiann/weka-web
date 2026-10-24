/* Weka Web — automated planning in the browser.
 * Host modules for MiniPy so that Colab notebooks with OR-Tools CP-SAT, Pyro/torch, numpy, math, random and
 * matplotlib run unchanged:
 *   - a small CP-SAT compatible solver (integer variables, linear constraints, intervals, NoOverlap,
 *     Max/MinEquality, AllDifferent, enforcement literals; depth-first branch & bound with bounds propagation);
 *   - Pyro sampling (Normal, Uniform, Exponential, LogNormal, Gamma, Beta, Bernoulli, Poisson, Categorical)
 *     with float32 torch-like tensors;
 *   - matplotlib figures captured as data and drawn by the Python page.
 * Every solve, sample site and figure is reported through interp.emit() so the page can draw Gantt charts and histograms. */
(function () {
  'use strict';
  const W = window.WEKA;
  const P = window.MiniPy;
  const { F, num, isNum, native, PyTuple, PyDict, PyModule, PyError, repr, str, NOTIMPL } = P;
  const fl = (v) => F(v);
  const kwOr = (a, kw, i, name, d) => (a.length > i ? a[i] : name in kw ? kw[name] : d);
  const asList = (interp, x) => (Array.isArray(x) ? x : interp.iterate(x));

  // =============================================================== CP-SAT
  let VAR_ID = 0;
  class LinExpr {
    constructor(terms, c) { this.terms = terms || new Map(); this.c = c || 0; }
    static of(x) {
      if (x instanceof LinExpr) return x;
      if (isNum(x)) { const v = num(x); if (!Number.isInteger(v)) throw new PyError('TypeError', 'CP-SAT expressions need integer coefficients, got ' + v); return new LinExpr(new Map(), v); }
      if (x && x.__tensor__) return LinExpr.of(Math.round(x.__float__()));
      throw new PyError('TypeError', `cannot use '${P.typeName(x)}' in a linear expression`);
    }
    plus(o, k = 1) {
      const b = LinExpr.of(o), t = new Map(this.terms);
      for (const [v, c] of b.terms) { const n = (t.get(v) || 0) + k * c; if (n) t.set(v, n); else t.delete(v); }
      return new LinExpr(t, this.c + k * b.c);
    }
    times(k) {
      if (k instanceof LinExpr) { if (k.terms.size === 0) return this.times(k.c); if (this.terms.size === 0) return k.times(this.c); throw new PyError('TypeError', 'CP-SAT: products of variables are not linear (use AddMultiplicationEquality in OR-Tools)'); }
      const m = num(k);
      if (!Number.isInteger(m)) throw new PyError('TypeError', 'CP-SAT coefficients must be integers');
      const t = new Map(); for (const [v, c] of this.terms) if (c * m) t.set(v, c * m);
      return new LinExpr(t, this.c * m);
    }
    __add__(o) { return this.plus(o); } __radd__(o) { return this.plus(o); }
    __sub__(o) { return this.plus(o, -1); } __rsub__(o) { return LinExpr.of(o).plus(this, -1); }
    __mul__(o) { return this.times(o); } __rmul__(o) { return this.times(o); }
    __neg__() { return this.times(-1); }
    bound(lo, hi, ne) { return new Bounded(this, lo, hi, ne); }
    // the bounds apply to the whole difference (constant included): d <= 0, d >= 0, d == 0 …
    __le__(o) { return this.plus(o, -1).bound(-Infinity, 0); }
    __lt__(o) { return this.plus(o, -1).bound(-Infinity, -1); }
    __ge__(o) { return this.plus(o, -1).bound(0, Infinity); }
    __gt__(o) { return this.plus(o, -1).bound(1, Infinity); }
    __eq__(o) { if (!(o instanceof LinExpr) && !isNum(o)) return false; return this.plus(o, -1).bound(0, 0); }
    __ne__(o) { if (!(o instanceof LinExpr) && !isNum(o)) return true; return this.plus(o, -1).bound(0, 0, true); }
    __hash__() { return 'lin' + [...this.terms].join(';') + '|' + this.c; }
    __str__() {
      const parts = [];
      for (const [v, c] of this.terms) { const n = VARS.get(v); const name = n ? n.name : 'x' + v; parts.push((c === 1 ? '' : c === -1 ? '-' : c + ' * ') + name); }
      if (this.c || !parts.length) parts.push(String(this.c));
      return parts.join(' + ').replace(/\+ -/g, '- ');
    }
  }
  const VARS = new Map();
  class IntVar extends LinExpr {
    constructor(model, lb, ub, name, isBool) {
      const id = VAR_ID++;
      super(new Map([[id, 1]]), 0);
      Object.assign(this, { id, lb: num(lb), ub: num(ub), name: name || 'x' + id, model, isBool: !!isBool, __typename__: 'IntVar' });
      VARS.set(id, this);
    }
    __hash__() { return 'v' + this.id; }
    __str__() { return this.name; }
    __repr__() { return `${this.name}(${this.lb}..${this.ub})`; }
    Name() { return this.name; } Index() { return this.id; }
    Not() { return new NotLit(this); }
  }
  ['Name', 'Index', 'Not'].forEach((m) => { IntVar.prototype[m.toLowerCase()] = IntVar.prototype[m]; });
  class NotLit extends LinExpr { constructor(v) { super(new Map([[v.id, -1]]), 1); this.v = v; this.negated = true; } }
  class Bounded {
    constructor(expr, lo, hi, ne) { this.expr = expr; this.lo = lo; this.hi = hi; this.ne = !!ne; this.__typename__ = 'BoundedLinearExpression'; }
    __bool__() { throw new PyError('NotImplementedError', 'Evaluating a CP-SAT constraint as a boolean is not supported'); }
    __str__() { return `${this.expr.__str__()} ${this.ne ? '!=' : 'in'} [${this.lo}, ${this.hi}]`; }
  }
  class Constraint {
    constructor(model, c) { this.model = model; this.c = c; this.__typename__ = 'Constraint'; }
    OnlyEnforceIf(...args) { const lits = args.flat(); const flat = []; for (const x of lits) { if (Array.isArray(x)) flat.push(...x); else if (x instanceof PyTuple) flat.push(...x.items); else flat.push(x); } this.c.enf = flat.map((l) => (l instanceof NotLit ? { id: l.v.id, val: 0 } : { id: l.id, val: 1 })); return this; }
    WithName(n) { this.c.name = n; return this; }
    Index() { return this.model.cons.indexOf(this.c); }
  }
  Constraint.prototype.only_enforce_if = Constraint.prototype.OnlyEnforceIf;
  Constraint.prototype.with_name = Constraint.prototype.WithName;
  class IntervalVar {
    constructor(start, size, end, name, presence) { Object.assign(this, { start: LinExpr.of(start), size: LinExpr.of(size), end: LinExpr.of(end), name, presence, __typename__: 'IntervalVar' }); }
    StartExpr() { return this.start; } SizeExpr() { return this.size; } EndExpr() { return this.end; } Name() { return this.name; }
    __str__() { return this.name; }
  }
  ['StartExpr', 'SizeExpr', 'EndExpr', 'Name'].forEach((m) => { IntervalVar.prototype[m.replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase()] = IntervalVar.prototype[m]; });

  class CpModel {
    constructor() { this.vars = []; this.cons = []; this.intervals = []; this.objective = null; this.__typename__ = 'CpModel'; this.hint = new Map(); }
    NewIntVar(lb, ub, name) { const v = new IntVar(this, lb, ub, str(name ?? '')); this.vars.push(v); return v; }
    NewBoolVar(name) { const v = new IntVar(this, 0, 1, str(name ?? ''), true); this.vars.push(v); return v; }
    NewConstant(v) { return this.NewIntVar(v, v, String(num(v))); }
    NewIntVarFromDomain(dom, name) { return this.NewIntVar(dom.lo, dom.hi, name); }
    NewIntervalVar(start, size, end, name) {
      const iv = new IntervalVar(start, size, end, str(name ?? 'interval'), null);
      this.intervals.push(iv);
      this.addLin(iv.start.plus(iv.size).plus(iv.end, -1), 0, 0, 'interval ' + iv.name);
      return iv;
    }
    NewFixedSizeIntervalVar(start, size, name) { return this.NewIntervalVar(start, size, LinExpr.of(start).plus(size), name); }
    NewOptionalIntervalVar(start, size, end, presence, name) {
      const iv = new IntervalVar(start, size, end, str(name ?? 'interval'), presence instanceof IntVar ? { id: presence.id, val: 1 } : null);
      this.intervals.push(iv);
      const c = this.addLin(iv.start.plus(iv.size).plus(iv.end, -1), 0, 0, 'interval ' + iv.name);
      if (iv.presence) c.enf = [iv.presence];
      return iv;
    }
    addLin(expr, lo, hi, name) {
      const c = { kind: 'lin', terms: [...expr.terms], lo: lo - expr.c, hi: hi - expr.c, name };
      this.cons.push(c); return c;
    }
    Add(b) {
      if (b === true) return new Constraint(this, { kind: 'true' });
      if (b === false) { const c = { kind: 'false' }; this.cons.push(c); return new Constraint(this, c); }
      if (!(b instanceof Bounded)) throw new PyError('TypeError', 'model.Add() expects a linear constraint such as x + y <= 10');
      if (b.ne) { const c = { kind: 'ne', terms: [...b.expr.terms], val: b.lo - b.expr.c }; this.cons.push(c); return new Constraint(this, c); }
      return new Constraint(this, this.addLin(b.expr, b.lo, b.hi));
    }
    AddLinearConstraint(expr, lo, hi) { return new Constraint(this, this.addLin(LinExpr.of(expr), num(lo), num(hi))); }
    AddNoOverlap(intervals) { const c = { kind: 'nooverlap', intervals: listOf(intervals) }; this.cons.push(c); return new Constraint(this, c); }
    AddCumulative(intervals, demands, capacity) { const c = { kind: 'cumulative', intervals: listOf(intervals), demands: listOf(demands).map(LinExpr.of), capacity: LinExpr.of(capacity) }; this.cons.push(c); return new Constraint(this, c); }
    AddMaxEquality(target, exprs) { return this.addMinMax('max', target, exprs); }
    AddMinEquality(target, exprs) { return this.addMinMax('min', target, exprs); }
    addMinMax(kind, target, exprs) {
      const t = LinExpr.of(target), es = listOf(exprs).map(LinExpr.of);
      for (const e of es) { if (kind === 'max') this.addLin(e.plus(t, -1), -Infinity, 0); else this.addLin(e.plus(t, -1), 0, Infinity); }
      const c = { kind, t, es }; this.cons.push(c); return new Constraint(this, c);
    }
    AddAllDifferent(...args) { const es = (args.length === 1 ? listOf(args[0]) : args).map(LinExpr.of); const c = { kind: 'alldiff', es }; this.cons.push(c); return new Constraint(this, c); }
    AddBoolOr(lits) { const ls = listOf(lits); let e = new LinExpr(); for (const l of ls) e = e.plus(l); return new Constraint(this, this.addLin(e, 1, Infinity)); }
    AddBoolAnd(lits) { const ls = listOf(lits); let e = new LinExpr(); for (const l of ls) e = e.plus(l); return new Constraint(this, this.addLin(e, ls.length, ls.length)); }
    AddExactlyOne(lits) { const ls = listOf(lits); let e = new LinExpr(); for (const l of ls) e = e.plus(l); return new Constraint(this, this.addLin(e, 1, 1)); }
    AddAtMostOne(lits) { const ls = listOf(lits); let e = new LinExpr(); for (const l of ls) e = e.plus(l); return new Constraint(this, this.addLin(e, -Infinity, 1)); }
    AddImplication(a, b) { return new Constraint(this, this.addLin(LinExpr.of(b).plus(a, -1), 0, Infinity)); }
    Minimize(e) { this.objective = { e: LinExpr.of(e), sense: 1 }; return null; }
    Maximize(e) { this.objective = { e: LinExpr.of(e), sense: -1 }; return null; }
    AddHint(v, val) { this.hint.set(v.id, num(val)); return null; }
    Validate() { return ''; }
    ModelStats() { return `#Variables: ${this.vars.length}\n#Constraints: ${this.cons.length}\n#Intervals: ${this.intervals.length}`; }
  }
  // snake_case aliases (OR-Tools ≥ 9.8 style)
  for (const m of Object.getOwnPropertyNames(CpModel.prototype)) {
    if (/^[A-Z]/.test(m)) { const snake = m.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase(); CpModel.prototype[snake] = CpModel.prototype[m]; }
  }
  function listOf(x) { if (Array.isArray(x)) return x; if (x instanceof PyTuple) return x.items; if (x instanceof PyDict) return x.values(); if (x && x.__iter__) return x.__iter__(); return [x]; }

  // ---------------------------------------------------------- solver
  const STATUS = { UNKNOWN: 0, MODEL_INVALID: 1, FEASIBLE: 2, INFEASIBLE: 3, OPTIMAL: 4 };
  const STATUS_NAME = ['UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', 'OPTIMAL'];

  function solve(model, params) {
    const t0 = performance.now();
    const ids = new Map(model.vars.map((v, i) => [v.id, i]));
    const n = model.vars.length;
    const idx = (id) => { const i = ids.get(id); if (i === undefined) throw new PyError('ValueError', 'CP-SAT: a variable from another model was used'); return i; };
    const mapTerms = (terms) => terms.map(([id, c]) => [idx(id), c]);
    const lins = [], others = [];
    const enfOf = (c) => (c.enf || []).map((l) => ({ i: idx(l.id), val: l.val }));
    for (const c of model.cons) {
      if (c.kind === 'lin') lins.push({ t: mapTerms(c.terms), lo: c.lo, hi: c.hi, enf: enfOf(c) });
      else if (c.kind === 'false') return { status: STATUS.INFEASIBLE };
      else if (c.kind !== 'true') others.push(c);
    }
    const LE = (e) => ({ t: mapTerms([...e.terms]), c: e.c });
    const nos = [], cums = [], minmax = [], nes = [], alldiffs = [];
    for (const c of others) {
      if (c.kind === 'nooverlap') nos.push(c.intervals.map((iv) => ({ s: LE(iv.start), d: LE(iv.size), e: LE(iv.end), pres: iv.presence ? { i: idx(iv.presence.id), val: 1 } : null, name: iv.name })));
      else if (c.kind === 'cumulative') cums.push({ ivs: c.intervals.map((iv) => ({ s: LE(iv.start), d: LE(iv.size), e: LE(iv.end), pres: iv.presence ? { i: idx(iv.presence.id), val: 1 } : null })), dem: c.demands.map(LE), cap: LE(c.capacity) });
      else if (c.kind === 'max' || c.kind === 'min') minmax.push({ kind: c.kind, t: LE(c.t), es: c.es.map(LE) });
      else if (c.kind === 'ne') nes.push({ t: mapTerms(c.terms), val: c.val, enf: enfOf(c) });
      else if (c.kind === 'alldiff') alldiffs.push(c.es.map(LE));
    }
    const obj = model.objective ? { t: mapTerms([...model.objective.e.terms]), c: model.objective.e.c, sense: model.objective.sense } : null;
    // pairs for no-overlap: order decisions live in the node state
    const pairs = [];
    nos.forEach((ivs) => { for (let a = 0; a < ivs.length; a++) for (let b = a + 1; b < ivs.length; b++) pairs.push([ivs[a], ivs[b]]); });

    const minE = (e, lo, hi) => { let s = e.c; for (const [i, c] of e.t) s += c > 0 ? c * lo[i] : c * hi[i]; return s; };
    const maxE = (e, lo, hi) => { let s = e.c; for (const [i, c] of e.t) s += c > 0 ? c * hi[i] : c * lo[i]; return s; };
    const active = (enf, lo, hi) => { let all = true; for (const l of enf) { if (lo[l.i] === hi[l.i]) { if (lo[l.i] !== l.val) return -1; } else all = false; } return all ? 1 : 0; };
    /** Bounds propagation of sum t + c in [L, U]; returns false on failure. */
    function propLin(t, c, L, U, lo, hi, changed) {
      let mn = c, mx = c;
      for (const [i, k] of t) { if (k > 0) { mn += k * lo[i]; mx += k * hi[i]; } else { mn += k * hi[i]; mx += k * lo[i]; } }
      if (mn > U || mx < L) return false;
      if (mn >= L && mx <= U) return true;
      for (const [i, k] of t) {
        const tmin = k > 0 ? k * lo[i] : k * hi[i], tmax = k > 0 ? k * hi[i] : k * lo[i];
        const restMin = mn - tmin, restMax = mx - tmax;
        // k*x in [L - restMax, U - restMin]
        const a = L - restMax, b = U - restMin;
        let nlo, nhi;
        if (k > 0) { nlo = Math.ceil(a / k); nhi = Math.floor(b / k); } else { nlo = Math.ceil(b / k); nhi = Math.floor(a / k); }
        if (nlo > lo[i]) { lo[i] = nlo; changed.v = true; }
        if (nhi < hi[i]) { hi[i] = nhi; changed.v = true; }
        if (lo[i] > hi[i]) return false;
      }
      return true;
    }
    let bestObj = null;
    const nodeLimit = 400000, timeLimit = (params.max_time_in_seconds ? num(params.max_time_in_seconds) : 20) * 1000;
    let nodes = 0, best = null, aborted = false;

    function propagate(lo, hi, order) {
      for (let iter = 0; iter < 10000; iter++) {
        const ch = { v: false };
        for (const c of lins) {
          if (c.enf.length) { const a = active(c.enf, lo, hi); if (a < 0) continue; if (a === 0) { // inactive unless literal forced; prune literal if violated
              let mn = 0, mx = 0; for (const [i, k] of c.t) { mn += k > 0 ? k * lo[i] : k * hi[i]; mx += k > 0 ? k * hi[i] : k * lo[i]; }
              if ((mn > c.hi || mx < c.lo) && c.enf.length === 1) { const l = c.enf[0]; const v = 1 - l.val; if (lo[l.i] > v || hi[l.i] < v) return false; if (lo[l.i] !== v || hi[l.i] !== v) { lo[l.i] = hi[l.i] = v; ch.v = true; } }
              continue; } }
          if (!propLin(c.t, 0, c.lo, c.hi, lo, hi, ch)) return false;
        }
        for (const m of minmax) {
          if (m.kind === 'max') { let M = -Infinity; for (const e of m.es) M = Math.max(M, maxE(e, lo, hi)); if (!propLin(m.t.t, m.t.c, -Infinity, M, lo, hi, ch)) return false; }
          else { let M = Infinity; for (const e of m.es) M = Math.min(M, minE(e, lo, hi)); if (!propLin(m.t.t, m.t.c, M, Infinity, lo, hi, ch)) return false; }
        }
        for (const c of nes) {
          if (c.enf.length && active(c.enf, lo, hi) !== 1) continue;
          if (c.t.every(([i]) => lo[i] === hi[i])) { let s = 0; for (const [i, k] of c.t) s += k * lo[i]; if (s === c.val) return false; }
          else if (c.t.length === 1) { const [i, k] = c.t[0]; const v = c.val / k; if (Number.isInteger(v)) { if (lo[i] === v) { lo[i]++; ch.v = true; } if (hi[i] === v) { hi[i]--; ch.v = true; } if (lo[i] > hi[i]) return false; } }
        }
        for (const es of alldiffs) {
          const fixed = es.filter((e) => e.t.every(([i]) => lo[i] === hi[i])).map((e) => minE(e, lo, hi));
          if (new Set(fixed).size !== fixed.length) return false;
        }
        if (obj && bestObj !== null) { if (obj.sense > 0) { if (!propLin(obj.t, obj.c, -Infinity, bestObj - 1, lo, hi, ch)) return false; } else if (!propLin(obj.t, obj.c, bestObj + 1, Infinity, lo, hi, ch)) return false; }
        // no-overlap: decided orders become precedences; impossible orders are pruned
        for (let p = 0; p < pairs.length; p++) {
          const [A, B] = pairs[p];
          const pa = A.pres ? (lo[A.pres.i] === hi[A.pres.i] ? lo[A.pres.i] : -1) : 1, pb = B.pres ? (lo[B.pres.i] === hi[B.pres.i] ? lo[B.pres.i] : -1) : 1;
          if (pa === 0 || pb === 0) continue;
          if (pa < 0 || pb < 0) continue; // optional and undecided: checked at the leaf
          let o = order[p];
          if (o === 0) {
            const abOk = minE(A.e, lo, hi) <= maxE(B.s, lo, hi), baOk = minE(B.e, lo, hi) <= maxE(A.s, lo, hi);
            if (!abOk && !baOk) return false;
            if (!abOk) o = order[p] = 2; else if (!baOk) o = order[p] = 1;
          }
          if (o === 1) { if (!propLin(A.e.t.concat(B.s.t.map(([i, k]) => [i, -k])), A.e.c - B.s.c, -Infinity, 0, lo, hi, ch)) return false; }
          else if (o === 2) { if (!propLin(B.e.t.concat(A.s.t.map(([i, k]) => [i, -k])), B.e.c - A.s.c, -Infinity, 0, lo, hi, ch)) return false; }
        }
        if (!ch.v) return true;
      }
      return true;
    }
    function leafOk(v) {
      const val = (e) => { let s = e.c; for (const [i, k] of e.t) s += k * v[i]; return s; };
      for (const m of minmax) { const ts = val(m.t), vs = m.es.map(val); if (ts !== (m.kind === 'max' ? Math.max(...vs) : Math.min(...vs))) return false; }
      for (const ivs of nos) {
        const on = ivs.filter((iv) => !iv.pres || v[iv.pres.i] === 1);
        for (let a = 0; a < on.length; a++) for (let b = a + 1; b < on.length; b++) { const A = on[a], B = on[b]; if (val(A.d) > 0 && val(B.d) > 0 && val(A.s) < val(B.e) && val(B.s) < val(A.e)) return false; }
      }
      for (const c of cums) {
        const on = c.ivs.map((iv, k) => ({ s: val(iv.s), e: val(iv.e), d: val(c.dem[k]), p: !iv.pres || v[iv.pres.i] === 1 })).filter((x) => x.p);
        const cap = val(c.cap);
        for (const x of on) { let load = 0; for (const y of on) if (y.s <= x.s && x.s < y.e) load += y.d; if (load > cap) return false; }
      }
      for (const es of alldiffs) { const vals = es.map(val); if (new Set(vals).size !== vals.length) return false; }
      for (const c of nes) { if (c.enf.length && !c.enf.every((l) => v[l.i] === l.val)) continue; let s = 0; for (const [i, k] of c.t) s += k * v[i]; if (s === c.val) return false; }
      for (const c of lins) { if (!c.enf.length) continue; if (!c.enf.every((l) => v[l.i] === l.val)) continue; let s = 0; for (const [i, k] of c.t) s += k * v[i]; if (s < c.lo || s > c.hi) return false; }
      return true;
    }
    const lo0 = new Float64Array(n), hi0 = new Float64Array(n);
    model.vars.forEach((v, i) => { lo0[i] = v.lb; hi0[i] = v.ub; });
    function dfs(lo, hi, order) {
      if (aborted) return;
      if (++nodes > nodeLimit || (nodes % 512 === 0 && performance.now() - t0 > timeLimit)) { aborted = true; return; }
      if (!propagate(lo, hi, order)) return;
      // branch on an undecided overlap first
      for (let p = 0; p < pairs.length; p++) {
        if (order[p] !== 0) continue;
        const [A, B] = pairs[p];
        if ((A.pres && lo[A.pres.i] !== hi[A.pres.i]) || (B.pres && lo[B.pres.i] !== hi[B.pres.i])) continue;
        if ((A.pres && lo[A.pres.i] === 0) || (B.pres && lo[B.pres.i] === 0)) continue;
        if (maxE(A.e, lo, hi) <= minE(B.s, lo, hi) || maxE(B.e, lo, hi) <= minE(A.s, lo, hi)) continue;
        for (const o of [1, 2]) { const o2 = order.slice(); o2[p] = o; dfs(lo.slice(), hi.slice(), o2); if (aborted) return; if (bestObj !== null && !obj) return; }
        return;
      }
      let k = -1;
      for (let i = 0; i < n; i++) if (lo[i] !== hi[i]) { k = i; break; }
      if (k < 0) {
        if (!leafOk(lo)) return;
        best = Array.from(lo);
        if (obj) { let s = obj.c; for (const [i, c] of obj.t) s += c * lo[i]; bestObj = s; }
        else bestObj = 0;
        return;
      }
      const v = lo[k];
      const hint = model.hint.get(model.vars[k].id);
      const first = hint !== undefined && hint >= lo[k] && hint <= hi[k] ? hint : v;
      { const l2 = lo.slice(), h2 = hi.slice(); l2[k] = first; h2[k] = first; dfs(l2, h2, order.slice()); }
      if (aborted || (!obj && best)) return;
      if (first === v) { const l2 = lo.slice(); l2[k] = v + 1; dfs(l2, hi.slice(), order.slice()); }
      else { const h2 = hi.slice(); h2[k] = first - 1; dfs(lo.slice(), h2, order.slice()); if (aborted || (!obj && best)) return; const l3 = lo.slice(); l3[k] = first + 1; dfs(l3, hi.slice(), order.slice()); }
    }
    dfs(lo0, hi0, new Int8Array(pairs.length));
    const wall = (performance.now() - t0) / 1000;
    if (!best) return { status: aborted ? STATUS.UNKNOWN : STATUS.INFEASIBLE, wall, nodes };
    return { status: aborted ? STATUS.FEASIBLE : STATUS.OPTIMAL, values: best, ids, objective: bestObj, wall, nodes };
  }

  class CpSolver {
    constructor(interp) { this.interp = interp; this.parameters = { max_time_in_seconds: 20, num_workers: 8, num_search_workers: 8, log_search_progress: false, __typename__: 'SatParameters' }; this.res = null; this.__typename__ = 'CpSolver'; }
    Solve(model, cb) {
      if (!(model instanceof CpModel)) throw new PyError('TypeError', 'solver.Solve() expects a cp_model.CpModel');
      if (cb) throw new PyError('NotImplementedError', 'solution callbacks are not supported in the browser solver — read the values after Solve() instead');
      this.model = model;
      this.res = solve(model, this.parameters);
      this.interp.emit && this.interp.emit({ type: 'cpsat', model, solver: this, status: STATUS_NAME[this.res.status] });
      return this.res.status;
    }
    val(e) {
      if (!this.res || !this.res.values) throw new PyError('RuntimeError', 'solver.Value(): no solution is available (check the status first)');
      const x = LinExpr.of(e); let s = x.c;
      for (const [id, c] of x.terms) { const i = this.res.ids.get(id); if (i === undefined) throw new PyError('ValueError', 'variable not in the solved model'); s += c * this.res.values[i]; }
      return s;
    }
    Value(e) { return this.val(e); }
    BooleanValue(e) { return !!this.val(e); }
    ObjectiveValue() { return fl(this.res && this.res.objective !== undefined ? this.res.objective : 0); }
    BestObjectiveBound() { return this.ObjectiveValue(); }
    StatusName(s) { return STATUS_NAME[s === undefined || s === null ? this.res.status : num(s)]; }
    WallTime() { return fl(this.res ? this.res.wall : 0); }
    NumBranches() { return this.res ? this.res.nodes : 0; }
    NumConflicts() { return 0; }
    ResponseStats() { return `CpSolverResponse summary:\nstatus: ${this.StatusName()}\nobjective: ${this.res && this.res.objective}\nwalltime: ${this.res && this.res.wall.toFixed(4)}\nbranches: ${this.res && this.res.nodes}\n(engine: Weka Web browser CP solver)`; }
  }
  for (const m of ['Solve', 'Value', 'BooleanValue', 'ObjectiveValue', 'BestObjectiveBound', 'StatusName', 'WallTime', 'NumBranches', 'NumConflicts', 'ResponseStats']) CpSolver.prototype[m.replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase()] = CpSolver.prototype[m];

  function cpModelModule(interp) {
    const m = new PyModule('ortools.sat.python.cp_model', {
      CpModel: native(() => new CpModel(), 'CpModel'),
      CpSolver: native(() => new CpSolver(interp), 'CpSolver'),
      LinearExpr: { Sum: native((a) => listOf(a[0]).reduce((s, x) => s.plus(x), new LinExpr()), 'Sum'), WeightedSum: native((a) => listOf(a[0]).reduce((s, x, i) => s.plus(LinExpr.of(x).times(listOf(a[1])[i])), new LinExpr()), 'WeightedSum'), __typename__: 'LinearExpr' },
      Domain: native((a) => ({ lo: num(a[0]), hi: num(a[1]), __typename__: 'Domain' }), 'Domain'),
      CpSolverSolutionCallback: native(() => { throw new PyError('NotImplementedError', 'solution callbacks are not supported in the browser solver'); }, 'CpSolverSolutionCallback'),
    });
    Object.assign(m, STATUS);
    m.LinearExpr.sum = m.LinearExpr.Sum; m.LinearExpr.weighted_sum = m.LinearExpr.WeightedSum;
    return m;
  }

  // =============================================================== random numbers
  class RNG {
    constructor(seed) { this.seed(seed); }
    seed(s) { this.s = (s === undefined || s === null ? (Date.now() ^ (Math.random() * 1e9)) : Math.floor(num(s))) >>> 0; this.spare = null; }
    next() { let t = (this.s += 0x6d2b79f5) >>> 0; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
    normal() { if (this.spare !== null) { const s = this.spare; this.spare = null; return s; } let u, v, r; do { u = 2 * this.next() - 1; v = 2 * this.next() - 1; r = u * u + v * v; } while (r >= 1 || r === 0); const f = Math.sqrt(-2 * Math.log(r) / r); this.spare = v * f; return u * f; }
    gamma(k) { if (k < 1) return this.gamma(k + 1) * Math.pow(this.next(), 1 / k); const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d); for (;;) { let x, v; do { x = this.normal(); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = this.next(); if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; } }
    poisson(l) { if (l > 30) return Math.max(0, Math.round(l + Math.sqrt(l) * this.normal())); const L = Math.exp(-l); let k = 0, p = 1; do { k++; p *= this.next(); } while (p > L); return k - 1; }
  }

  // =============================================================== torch-like tensors (float32)
  const f32 = Math.fround;
  class Tensor {
    constructor(data, shape, dtype) { this.data = data; this.shape = shape; this.dtype = dtype || 'float32'; this.__tensor__ = true; this.__typename__ = 'Tensor'; }
    static scalar(v, dtype) { return new Tensor([dtype === 'bool' ? v : dtype === 'int64' ? Math.trunc(v) : f32(v)], [], dtype); }
    static from(x, interp) {
      if (x instanceof Tensor) return x;
      if (isNum(x)) return Tensor.scalar(num(x), typeof x === 'number' && Number.isInteger(x) && !(x instanceof P.PyFloat) ? 'float32' : 'float32');
      if (Array.isArray(x) || x instanceof PyTuple) { const items = Array.isArray(x) ? x : x.items; const flat = items.map((v) => (v instanceof Tensor ? v.data[0] : num(v))); return new Tensor(flat.map((v) => f32(v)), [flat.length]); }
      throw new PyError('TypeError', 'cannot convert to tensor');
    }
    get isScalar() { return this.shape.length === 0; }
    map(fn, dtype) { return new Tensor(this.data.map((v) => (dtype === 'bool' ? fn(v) : f32(fn(v)))), this.shape.slice(), dtype || this.dtype); }
    zip(o, fn, dtype) {
      const b = o instanceof Tensor ? o : isNum(o) ? Tensor.scalar(num(o)) : null;
      if (!b) return NOTIMPL;
      const n = Math.max(this.data.length, b.data.length);
      const out = [];
      for (let i = 0; i < n; i++) { const r = fn(this.data[this.data.length === 1 ? 0 : i], b.data[b.data.length === 1 ? 0 : i]); out.push(dtype === 'bool' ? r : f32(r)); }
      return new Tensor(out, this.data.length >= b.data.length ? this.shape.slice() : b.shape.slice(), dtype);
    }
    __add__(o) { return this.zip(o, (a, b) => a + b); } __radd__(o) { return this.__add__(o); }
    __sub__(o) { return this.zip(o, (a, b) => a - b); } __rsub__(o) { return this.zip(o, (a, b) => b - a); }
    __mul__(o) { return this.zip(o, (a, b) => a * b); } __rmul__(o) { return this.__mul__(o); }
    __truediv__(o) { return this.zip(o, (a, b) => a / b); } __rtruediv__(o) { return this.zip(o, (a, b) => b / a); }
    __pow__(o) { return this.zip(o, (a, b) => Math.pow(a, b)); }
    __neg__() { return this.map((v) => -v); }
    __lt__(o) { return this.zip(o, (a, b) => a < b, 'bool'); } __le__(o) { return this.zip(o, (a, b) => a <= b, 'bool'); }
    __gt__(o) { return this.zip(o, (a, b) => a > b, 'bool'); } __ge__(o) { return this.zip(o, (a, b) => a >= b, 'bool'); }
    __eq__(o) { return this.zip(o, (a, b) => a === b, 'bool'); } __ne__(o) { return this.zip(o, (a, b) => a !== b, 'bool'); }
    __bool__() { if (this.data.length !== 1) throw new PyError('RuntimeError', 'Boolean value of Tensor with more than one value is ambiguous'); return !!this.data[0]; }
    __float__() { if (this.data.length !== 1) throw new PyError('ValueError', 'only one element tensors can be converted to Python scalars'); return +this.data[0]; }
    __int__() { return Math.trunc(this.__float__()); }
    __len__() { return this.shape.length ? this.shape[0] : 0; }
    __iter__() { return this.data.map((v) => Tensor.scalar(v, this.dtype)); }
    __getitem__(k) { if (k && k.__slice__) { const lo = k.lo === null ? 0 : num(k.lo), hi = k.hi === null ? this.data.length : num(k.hi); return new Tensor(this.data.slice(lo, hi), [Math.max(0, Math.min(hi, this.data.length) - lo)], this.dtype); } let i = num(k); if (i < 0) i += this.data.length; return Tensor.scalar(this.data[i], this.dtype); }
    __format__(spec) { return P.formatValue(fl(this.__float__()), spec); }
    __abs__() { return this.map(Math.abs); }
    item() { const v = this.__float__(); return this.dtype === 'bool' ? !!v : this.dtype === 'int64' ? v : fl(v); }
    tolist() { return this.isScalar ? this.item() : this.data.map((v) => (this.dtype === 'bool' ? !!v : fl(v))); }
    numpy() { return this.tolist(); }
    float() { return this.map((v) => +v, 'float32'); }
    detach() { return this; } cpu() { return this; } clone() { return this.map((v) => v); }
    size() { return new PyTuple(this.shape.slice()); } dim() { return this.shape.length; } numel() { return this.data.length; }
    mean() { return Tensor.scalar(this.data.reduce((s, v) => s + +v, 0) / this.data.length); }
    sum() { return Tensor.scalar(this.data.reduce((s, v) => s + +v, 0)); }
    std(unbiased = true) { const m = this.data.reduce((s, v) => s + v, 0) / this.data.length; const ss = this.data.reduce((s, v) => s + (v - m) * (v - m), 0); return Tensor.scalar(Math.sqrt(ss / (this.data.length - (unbiased === false ? 0 : 1)))); }
    var() { const s = this.std(); return Tensor.scalar(s.data[0] ** 2); }
    min() { return Tensor.scalar(Math.min(...this.data)); } max() { return Tensor.scalar(Math.max(...this.data)); }
    argmin() { return Tensor.scalar(this.data.indexOf(Math.min(...this.data)), 'int64'); } argmax() { return Tensor.scalar(this.data.indexOf(Math.max(...this.data)), 'int64'); }
    exp() { return this.map(Math.exp); } log() { return this.map(Math.log); } sqrt() { return this.map(Math.sqrt); } abs() { return this.map(Math.abs); }
    __repr__() {
      const fmt = (v) => {
        if (this.dtype === 'bool') return v ? 'True' : 'False';
        if (this.dtype === 'int64') return String(v);
        const allInt = this.data.every((x) => Number.isInteger(x));
        return allInt ? v.toFixed(0) + '.' : v.toFixed(4);
      };
      if (this.isScalar) return `tensor(${fmt(this.data[0])})`;
      const items = this.data.length > 12 ? this.data.slice(0, 3).map(fmt).concat(['...']).concat(this.data.slice(-3).map(fmt)) : this.data.map(fmt);
      return `tensor([${items.join(', ')}])`;
    }
  }
  function torchModule(interp, rng) {
    const T = (x) => Tensor.from(x, interp);
    const m = new PyModule('torch', {
      tensor: native((a, kw) => { const t = T(a[0]); if (kw.dtype && kw.dtype.__dtype__ === 'int64') return t.map(Math.trunc, 'int64'); return t; }, 'tensor'),
      as_tensor: native((a) => T(a[0]), 'as_tensor'),
      stack: native((a) => { const items = listOf(a[0]); return new Tensor(items.map((x) => (x instanceof Tensor ? x.data[0] : f32(num(x)))), [items.length]); }, 'stack'),
      cat: native((a) => { const items = listOf(a[0]); const d = []; items.forEach((x) => d.push(...T(x).data)); return new Tensor(d, [d.length]); }, 'cat'),
      mean: native((a) => T(a[0]).mean(), 'mean'), sum: native((a) => T(a[0]).sum(), 'sum'), std: native((a) => T(a[0]).std(), 'std'),
      exp: native((a) => T(a[0]).exp(), 'exp'), log: native((a) => T(a[0]).log(), 'log'), sqrt: native((a) => T(a[0]).sqrt(), 'sqrt'), abs: native((a) => T(a[0]).abs(), 'abs'),
      clamp: native((a, kw) => { const lo = kwOr(a, kw, 1, 'min', null), hi = kwOr(a, kw, 2, 'max', null); return T(a[0]).map((v) => Math.min(hi === null ? Infinity : num(hi), Math.max(lo === null ? -Infinity : num(lo), v))); }, 'clamp'),
      maximum: native((a) => T(a[0]).zip(T(a[1]), Math.max), 'maximum'), minimum: native((a) => T(a[0]).zip(T(a[1]), Math.min), 'minimum'),
      zeros: native((a) => new Tensor(new Array(num(a[0])).fill(0), [num(a[0])]), 'zeros'), ones: native((a) => new Tensor(new Array(num(a[0])).fill(1), [num(a[0])]), 'ones'),
      arange: native((a) => { const [s, e, st] = a.length === 1 ? [0, num(a[0]), 1] : [num(a[0]), num(a[1]), a.length > 2 ? num(a[2]) : 1]; const d = []; for (let x = s; x < e; x += st) d.push(f32(x)); return new Tensor(d, [d.length]); }, 'arange'),
      randn: native((a) => { const k = a.length ? num(a[0]) : 1; return new Tensor(Array.from({ length: k }, () => f32(rng.normal())), [k]); }, 'randn'),
      rand: native((a) => { const k = a.length ? num(a[0]) : 1; return new Tensor(Array.from({ length: k }, () => f32(rng.next())), [k]); }, 'rand'),
      manual_seed: native((a) => { rng.seed(a[0]); return null; }, 'manual_seed'),
      Size: native((a) => new PyTuple(listOf(a[0] || [])), 'Size'),
      no_grad: native(() => ({ __enter__: () => null, __exit__: () => null }), 'no_grad'),
      is_tensor: native((a) => a[0] instanceof Tensor, 'is_tensor'),
      float32: { __dtype__: 'float32', __str__: () => 'torch.float32' }, float64: { __dtype__: 'float32', __str__: () => 'torch.float64' }, int64: { __dtype__: 'int64', __str__: () => 'torch.int64' }, long: { __dtype__: 'int64' },
      __version__: '2.x (Weka Web browser shim)',
    });
    return m;
  }

  // =============================================================== Pyro
  function pyroModules(interp, rng, trace) {
    const T = (x) => Tensor.from(x, interp);
    const sval = (x) => (x instanceof Tensor ? x.data[0] : num(x));
    const shapeN = (a, kw) => { const s = a.length ? a[0] : kw.sample_shape; if (s === undefined || s === null) return 0; const l = listOf(s); return l.length ? num(l[0]) : 0; };
    class Dist {
      constructor(name, params, sampler, logp, mean, sd) { Object.assign(this, { name, params, sampler, logp, __typename__: name }); this.mean = mean; this.stddev = sd; }
      sample(a = [], kw = {}) { const n = shapeN(a, kw); if (!n) return Tensor.scalar(this.sampler(), this.dtype); return new Tensor(Array.from({ length: n }, () => f32(this.sampler())), [n], this.dtype); }
      rsample(a, kw) { return this.sample(a, kw); }
      log_prob(x) { return T(x).map((v) => this.logp(v)); }
      __repr__() { return `${this.name}(${Object.entries(this.params).map(([k, v]) => `${k}: ${P.repr(fl(v))}`).join(', ')})`; }
    }
    // methods called from Python receive (args, kwargs)
    Dist.prototype.sample = native(function () { return null; });
    const wrapDist = (d) => {
      d.sample = native((a, kw) => { const n = shapeN(a, kw); if (!n) return Tensor.scalar(d.sampler(), d.dtype); return new Tensor(Array.from({ length: n }, () => (d.dtype ? d.sampler() : f32(d.sampler()))), [n], d.dtype); }, 'sample');
      d.rsample = d.sample;
      d.log_prob = native((a) => T(a[0]).map((v) => d.logp(v)), 'log_prob');
      return d;
    };
    const LN2PI = Math.log(2 * Math.PI);
    const D = {
      Normal: (loc, scale) => new Dist('Normal', { loc, scale }, () => loc + scale * rng.normal(), (x) => -0.5 * ((x - loc) / scale) ** 2 - Math.log(scale) - 0.5 * LN2PI, T(loc), T(scale)),
      Uniform: (low, high) => new Dist('Uniform', { low, high }, () => low + (high - low) * rng.next(), (x) => (x >= low && x < high ? -Math.log(high - low) : -Infinity), T((low + high) / 2), T((high - low) / Math.sqrt(12))),
      Exponential: (rate) => new Dist('Exponential', { rate }, () => -Math.log(1 - rng.next()) / rate, (x) => (x < 0 ? -Infinity : Math.log(rate) - rate * x), T(1 / rate), T(1 / rate)),
      LogNormal: (loc, scale) => new Dist('LogNormal', { loc, scale }, () => Math.exp(loc + scale * rng.normal()), (x) => (x <= 0 ? -Infinity : -0.5 * ((Math.log(x) - loc) / scale) ** 2 - Math.log(scale * x) - 0.5 * LN2PI), T(Math.exp(loc + scale * scale / 2)), T(Math.sqrt((Math.exp(scale * scale) - 1) * Math.exp(2 * loc + scale * scale)))),
      Gamma: (c, r) => new Dist('Gamma', { concentration: c, rate: r }, () => rng.gamma(c) / r, (x) => (x <= 0 ? -Infinity : c * Math.log(r) + (c - 1) * Math.log(x) - r * x - W.lnGamma(c)), T(c / r), T(Math.sqrt(c) / r)),
      Beta: (a, b) => new Dist('Beta', { concentration1: a, concentration0: b }, () => { const x = rng.gamma(a), y = rng.gamma(b); return x / (x + y); }, (x) => (x <= 0 || x >= 1 ? -Infinity : (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - (W.lnGamma(a) + W.lnGamma(b) - W.lnGamma(a + b))), T(a / (a + b)), T(Math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))),
      Bernoulli: (p) => { const d = new Dist('Bernoulli', { probs: p }, () => (rng.next() < p ? 1 : 0), (x) => (x ? Math.log(p) : Math.log(1 - p)), T(p), T(Math.sqrt(p * (1 - p)))); return d; },
      Poisson: (l) => new Dist('Poisson', { rate: l }, () => rng.poisson(l), (x) => x * Math.log(l) - l - W.lnGamma(x + 1), T(l), T(Math.sqrt(l))),
      Categorical: (probs) => { const p = listOf(probs instanceof Tensor ? probs.tolist() : probs).map(sval); const s = p.reduce((a, b) => a + b, 0); const d = new Dist('Categorical', { probs: p }, () => { let r = rng.next() * s; for (let i = 0; i < p.length; i++) { r -= p[i]; if (r < 0) return i; } return p.length - 1; }, (x) => Math.log(p[x] / s)); d.dtype = 'int64'; return d; },
    };
    const distMod = new PyModule('pyro.distributions', {});
    const argNames = { Normal: ['loc', 'scale'], Uniform: ['low', 'high'], Exponential: ['rate'], LogNormal: ['loc', 'scale'], Gamma: ['concentration', 'rate'], Beta: ['concentration1', 'concentration0'], Bernoulli: ['probs'], Poisson: ['rate'], Categorical: ['probs'] };
    for (const [name, mk] of Object.entries(D)) {
      distMod[name] = native((a, kw) => {
        const names = argNames[name];
        const vals = names.map((k, i) => (a.length > i ? a[i] : kw[k]));
        if (name === 'Bernoulli' && vals[0] === undefined && kw.logits !== undefined) vals[0] = 1 / (1 + Math.exp(-sval(kw.logits)));
        if (vals.some((v) => v === undefined)) throw new PyError('TypeError', `${name}() missing arguments: ${names.join(', ')}`);
        const d = mk(...vals.map((v) => (name === 'Categorical' ? v : sval(v))));
        if (name !== 'Categorical' && name !== 'Bernoulli' && name !== 'Uniform' && vals.slice(1).some((v) => sval(v) <= 0) && name !== 'Exponential' && name !== 'Poisson') throw new PyError('ValueError', `${name}: scale/rate parameters must be positive`);
        return wrapDist(d);
      }, name);
    }
    const pyro = new PyModule('pyro', {
      sample: native((a, kw) => {
        const name = str(a[0]), d = a[1];
        const obs = kw.obs;
        const v = obs !== undefined && obs !== null ? T(obs) : interp.call(d.sample, [], {});
        if (obs === undefined || obs === null) {
          const arr = trace.get(name) || [];
          if (arr.length < 200000) { if (v.isScalar) arr.push(+v.data[0]); else arr.push(...v.data.map(Number)); }
          trace.set(name, arr);
        }
        return v;
      }, 'sample'),
      clear_param_store: native(() => null, 'clear_param_store'),
      set_rng_seed: native((a) => { rng.seed(a[0]); return null; }, 'set_rng_seed'),
      param: native((a) => T(a[1]), 'param'),
      deterministic: native((a) => T(a[1]), 'deterministic'),
      plate: native(() => ({ __enter__: () => null, __exit__: () => null }), 'plate'),
      distributions: distMod,
      __version__: '1.9 (Weka Web browser shim)',
    });
    return { pyro, distMod };
  }

  // =============================================================== numpy (small subset, values as Python lists/floats)
  function numpyModule(interp, rng) {
    const vals = (x) => (Array.isArray(x) ? x : x instanceof PyTuple ? x.items : x instanceof Tensor ? x.data : [x]).map((v) => (v instanceof Tensor ? v.__float__() : num(v)));
    const mean = (a) => { const v = vals(a); return v.reduce((s, x) => s + x, 0) / v.length; };
    const np = new PyModule('numpy', {
      array: native((a) => (Array.isArray(a[0]) ? a[0].slice() : a[0] instanceof PyTuple ? a[0].items.slice() : a[0]), 'array'),
      mean: native((a) => fl(mean(a[0])), 'mean'),
      sum: native((a) => { const v = vals(a[0]); const s = v.reduce((x, y) => x + y, 0); return v.every(Number.isInteger) ? s : fl(s); }, 'sum'),
      std: native((a, kw) => { const v = vals(a[0]), m = mean(v), dd = kw.ddof ? num(kw.ddof) : 0; return fl(Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - dd))); }, 'std'),
      var: native((a) => { const v = vals(a[0]), m = mean(v); return fl(v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length); }, 'var'),
      median: native((a) => { const v = vals(a[0]).slice().sort((x, y) => x - y), k = v.length; return fl(k % 2 ? v[(k - 1) / 2] : (v[k / 2 - 1] + v[k / 2]) / 2); }, 'median'),
      percentile: native((a) => { const v = vals(a[0]).slice().sort((x, y) => x - y), q = num(a[1]) / 100 * (v.length - 1), i = Math.floor(q); return fl(v[i] + (v[Math.min(i + 1, v.length - 1)] - v[i]) * (q - i)); }, 'percentile'),
      max: native((a) => Math.max(...vals(a[0])), 'max'), min: native((a) => Math.min(...vals(a[0])), 'min'),
      argmax: native((a) => { const v = vals(a[0]); return v.indexOf(Math.max(...v)); }, 'argmax'), argmin: native((a) => { const v = vals(a[0]); return v.indexOf(Math.min(...v)); }, 'argmin'),
      sqrt: native((a) => (Array.isArray(a[0]) ? vals(a[0]).map((x) => fl(Math.sqrt(x))) : fl(Math.sqrt(num(a[0])))), 'sqrt'),
      exp: native((a) => (Array.isArray(a[0]) ? vals(a[0]).map((x) => fl(Math.exp(x))) : fl(Math.exp(num(a[0])))), 'exp'),
      log: native((a) => (Array.isArray(a[0]) ? vals(a[0]).map((x) => fl(Math.log(x))) : fl(Math.log(num(a[0])))), 'log'),
      abs: native((a) => (Array.isArray(a[0]) ? vals(a[0]).map((x) => Math.abs(x)) : Math.abs(num(a[0]))), 'abs'),
      round: native((a) => { const m = Math.pow(10, a.length > 1 ? num(a[1]) : 0); return fl(Math.round(num(a[0]) * m) / m); }, 'round'),
      arange: native((a) => { const [s, e, st] = a.length === 1 ? [0, num(a[0]), 1] : [num(a[0]), num(a[1]), a.length > 2 ? num(a[2]) : 1]; const r = []; for (let x = s; x < e - 1e-12; x += st) r.push(Number.isInteger(st) && Number.isInteger(s) ? x : fl(x)); return r; }, 'arange'),
      linspace: native((a) => { const s = num(a[0]), e = num(a[1]), k = a.length > 2 ? num(a[2]) : 50; return Array.from({ length: k }, (_, i) => fl(s + (e - s) * i / (k - 1))); }, 'linspace'),
      zeros: native((a) => new Array(num(a[0])).fill(fl(0)), 'zeros'), ones: native((a) => new Array(num(a[0])).fill(fl(1)), 'ones'),
      cumsum: native((a) => { let s = 0; return vals(a[0]).map((x) => (s += x)); }, 'cumsum'),
      pi: fl(Math.PI), e: fl(Math.E), inf: fl(Infinity), nan: fl(NaN),
    });
    np.random = new PyModule('numpy.random', {
      seed: native((a) => { rng.seed(a[0]); return null; }, 'seed'),
      normal: native((a, kw) => { const loc = num(kwOr(a, kw, 0, 'loc', 0)), sc = num(kwOr(a, kw, 1, 'scale', 1)), size = kwOr(a, kw, 2, 'size', null); if (size === null) return fl(loc + sc * rng.normal()); return Array.from({ length: num(size) }, () => fl(loc + sc * rng.normal())); }, 'normal'),
      uniform: native((a, kw) => { const lo = num(kwOr(a, kw, 0, 'low', 0)), hi = num(kwOr(a, kw, 1, 'high', 1)), size = kwOr(a, kw, 2, 'size', null); const f = () => fl(lo + (hi - lo) * rng.next()); return size === null ? f() : Array.from({ length: num(size) }, f); }, 'uniform'),
      rand: native((a) => (a.length ? Array.from({ length: num(a[0]) }, () => fl(rng.next())) : fl(rng.next())), 'rand'),
      randn: native((a) => (a.length ? Array.from({ length: num(a[0]) }, () => fl(rng.normal())) : fl(rng.normal())), 'randn'),
      randint: native((a, kw) => { const lo = num(a[0]), hi = a.length > 1 ? num(a[1]) : null; const size = kwOr(a, kw, 2, 'size', null); const f = () => (hi === null ? Math.floor(rng.next() * lo) : lo + Math.floor(rng.next() * (hi - lo))); return size === null ? f() : Array.from({ length: num(size) }, f); }, 'randint'),
      choice: native((a) => { const v = Array.isArray(a[0]) ? a[0] : Array.from({ length: num(a[0]) }, (_, i) => i); return v[Math.floor(rng.next() * v.length)]; }, 'choice'),
      exponential: native((a, kw) => { const sc = num(kwOr(a, kw, 0, 'scale', 1)), size = kwOr(a, kw, 1, 'size', null); const f = () => fl(-Math.log(1 - rng.next()) * sc); return size === null ? f() : Array.from({ length: num(size) }, f); }, 'exponential'),
    });
    return np;
  }
  function mathModule() {
    const f1 = (fn) => native((a) => fl(fn(...a.map((x) => (x && x.__float__ ? x.__float__() : num(x))))));
    return new PyModule('math', {
      sqrt: f1(Math.sqrt), exp: f1(Math.exp), log: native((a) => fl(a.length > 1 ? Math.log(num(a[0])) / Math.log(num(a[1])) : Math.log(num(a[0])))), log10: f1(Math.log10), log2: f1(Math.log2),
      sin: f1(Math.sin), cos: f1(Math.cos), tan: f1(Math.tan), atan: f1(Math.atan), atan2: f1(Math.atan2), fabs: f1(Math.abs), pow: f1(Math.pow), hypot: f1(Math.hypot),
      ceil: native((a) => Math.ceil(num(a[0]))), floor: native((a) => Math.floor(num(a[0]))), trunc: native((a) => Math.trunc(num(a[0]))),
      factorial: native((a) => { let r = 1; for (let i = 2; i <= num(a[0]); i++) r *= i; return r; }), gcd: native((a) => { let x = Math.abs(num(a[0])), y = Math.abs(num(a[1])); while (y) [x, y] = [y, x % y]; return x; }),
      isclose: native((a, kw) => Math.abs(num(a[0]) - num(a[1])) <= Math.max((kw.rel_tol ? num(kw.rel_tol) : 1e-9) * Math.max(Math.abs(num(a[0])), Math.abs(num(a[1]))), kw.abs_tol ? num(kw.abs_tol) : 0)),
      isnan: native((a) => Number.isNaN(num(a[0]))), isinf: native((a) => !Number.isFinite(num(a[0])) && !Number.isNaN(num(a[0]))),
      pi: fl(Math.PI), e: fl(Math.E), inf: fl(Infinity), nan: fl(NaN), tau: fl(2 * Math.PI),
    });
  }
  function randomModule(interp, rng) {
    return new PyModule('random', {
      seed: native((a) => { rng.seed(a.length ? a[0] : undefined); return null; }), random: native(() => fl(rng.next())),
      uniform: native((a) => fl(num(a[0]) + (num(a[1]) - num(a[0])) * rng.next())), gauss: native((a) => fl(num(a[0]) + num(a[1]) * rng.normal())), normalvariate: native((a) => fl(num(a[0]) + num(a[1]) * rng.normal())),
      randint: native((a) => num(a[0]) + Math.floor(rng.next() * (num(a[1]) - num(a[0]) + 1))), randrange: native((a) => (a.length === 1 ? Math.floor(rng.next() * num(a[0])) : num(a[0]) + Math.floor(rng.next() * (num(a[1]) - num(a[0]))))),
      choice: native((a) => { const v = interp.iterate(a[0]); return v[Math.floor(rng.next() * v.length)]; }),
      shuffle: native((a) => { const v = a[0]; for (let i = v.length - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); [v[i], v[j]] = [v[j], v[i]]; } return null; }),
      sample: native((a) => { const v = interp.iterate(a[0]).slice(); for (let i = v.length - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); [v[i], v[j]] = [v[j], v[i]]; } return v.slice(0, num(a[1])); }),
      expovariate: native((a) => fl(-Math.log(1 - rng.next()) / num(a[0]))),
    });
  }
  function timeModule() { const t0 = performance.now(); return new PyModule('time', { time: native(() => fl(Date.now() / 1000)), perf_counter: native(() => fl((performance.now() - t0) / 1000)), sleep: native(() => null) }); }

  // =============================================================== matplotlib (captured figures)
  function pyplotModule(interp) {
    const vals = (x) => (x === undefined || x === null ? [] : listOf(x instanceof Tensor ? x.tolist() : x).map((v) => (v instanceof Tensor ? v.__float__() : isNum(v) ? num(v) : v)));
    let fig = null;
    const cur = () => (fig = fig || { series: [], title: '', xlabel: '', ylabel: '', yticks: null, xticks: null, invertY: false, legend: false });
    const emit = () => { if (fig && fig.series.length) interp.emit && interp.emit({ type: 'figure', fig }); fig = null; return null; };
    const color = (kw) => (kw.color !== undefined ? (listOf(kw.color).length > 1 ? listOf(kw.color) : str(listOf(kw.color)[0])) : null);
    const axes = {
      barh: native((a, kw) => { const y = vals(a[0]), w = vals(kwOr(a, kw, 1, 'width', [])); const left = kw.left !== undefined ? vals(kw.left) : null; cur().series.push({ kind: 'barh', y, w: w.length === 1 && y.length > 1 ? y.map(() => w[0]) : w, left: left && left.length === 1 && y.length > 1 ? y.map(() => left[0]) : left, color: color(kw), label: kw.label ? str(kw.label) : null }); return null; }, 'barh'),
      bar: native((a, kw) => { const x = vals(a[0]), hgt = vals(kwOr(a, kw, 1, 'height', [])); cur().series.push({ kind: 'bar', x, h: hgt.length === 1 && x.length > 1 ? x.map(() => hgt[0]) : hgt, color: color(kw), label: kw.label ? str(kw.label) : null }); return null; }, 'bar'),
      plot: native((a, kw) => { let x = vals(a[0]), y = a.length > 1 && typeof a[1] !== 'string' ? vals(a[1]) : null; if (!y) { y = x; x = y.map((_, i) => i); } cur().series.push({ kind: 'line', x, y, color: color(kw), label: kw.label ? str(kw.label) : null, marker: typeof a[2] === 'string' ? a[2] : kw.marker || null }); return null; }, 'plot'),
      scatter: native((a, kw) => { cur().series.push({ kind: 'scatter', x: vals(a[0]), y: vals(a[1]), color: color(kw), label: kw.label ? str(kw.label) : null }); return null; }, 'scatter'),
      hist: native((a, kw) => { cur().series.push({ kind: 'hist', data: vals(a[0]), bins: kw.bins !== undefined ? num(kw.bins) : a.length > 1 ? num(a[1]) : 10, color: color(kw), label: kw.label ? str(kw.label) : null, alpha: kw.alpha !== undefined ? num(kw.alpha) : 1 }); return null; }, 'hist'),
      axvline: native((a, kw) => { cur().series.push({ kind: 'vline', x: num(a.length ? a[0] : kw.x), color: color(kw), label: kw.label ? str(kw.label) : null }); return null; }, 'axvline'),
      set_title: native((a) => { cur().title = str(a[0]); return null; }, 'set_title'), set_xlabel: native((a) => { cur().xlabel = str(a[0]); return null; }, 'set_xlabel'), set_ylabel: native((a) => { cur().ylabel = str(a[0]); return null; }, 'set_ylabel'),
      set_yticks: native((a) => { cur().yticksPos = vals(a[0]); if (a[1]) cur().yticks = vals(a[1]).map(str); return null; }, 'set_yticks'), set_yticklabels: native((a) => { cur().yticks = vals(a[0]).map(str); return null; }, 'set_yticklabels'),
      set_xticks: native(() => null, 'set_xticks'), set_xlim: native(() => null, 'set_xlim'), set_ylim: native(() => null, 'set_ylim'), grid: native(() => null, 'grid'),
      legend: native(() => { cur().legend = true; return null; }, 'legend'), invert_yaxis: native(() => { cur().invertY = true; return null; }, 'invert_yaxis'),
    };
    const plt = new PyModule('matplotlib.pyplot', Object.assign({}, axes, {
      figure: native(() => { emit(); cur(); return { __typename__: 'Figure' }; }, 'figure'),
      subplots: native(() => { emit(); cur(); const ax = Object.assign({ __typename__: 'Axes' }, axes); return new PyTuple([{ __typename__: 'Figure', tight_layout: native(() => null), savefig: native(() => null) }, ax]); }, 'subplots'),
      title: axes.set_title, xlabel: axes.set_xlabel, ylabel: axes.set_ylabel, yticks: native((a) => { if (a.length > 1) { cur().yticksPos = vals(a[0]); cur().yticks = vals(a[1]).map(str); } return null; }, 'yticks'), xticks: native(() => null, 'xticks'),
      xlim: native(() => null), ylim: native(() => null), tight_layout: native(() => null), savefig: native(() => null), close: native(() => null),
      gca: native(() => Object.assign({ __typename__: 'Axes' }, axes)), show: native(() => emit(), 'show'),
    }));
    plt.__flush__ = emit;
    return plt;
  }

  // =============================================================== interpreter factory
  /** Creates an interpreter with all host modules. onEvent receives {type:'cpsat'|'samples'|'figure', …}. */
  function create({ out, onEvent } = {}) {
    const rng = new RNG();
    const trace = new Map();
    const interp = new P.Interpreter({ out });
    interp.emit = (e) => onEvent && onEvent(e);
    interp.trace = trace;
    const lazy = (fn) => fn;
    let pyroMods = null;
    const getPyro = () => (pyroMods = pyroMods || pyroModules(interp, rng, trace));
    let torchMod = null, pltMod = null;
    const getTorch = () => (torchMod = torchMod || torchModule(interp, rng));
    const getPlt = () => (pltMod = pltMod || pyplotModule(interp));
    const ortools = new PyModule('ortools', {});
    const sat = new PyModule('ortools.sat', {});
    const satpy = new PyModule('ortools.sat.python', {});
    const cp = cpModelModule(interp);
    satpy.cp_model = cp; sat.python = satpy; ortools.sat = sat;
    interp.modules = {
      'ortools': ortools, 'ortools.sat': sat, 'ortools.sat.python': satpy, 'ortools.sat.python.cp_model': cp,
      get 'pyro'() { return getPyro().pyro; }, get 'pyro.distributions'() { return getPyro().distMod; },
      get 'torch'() { return getTorch(); },
      'numpy': numpyModule(interp, rng), get 'numpy.random'() { return this.numpy.random; },
      'math': mathModule(), 'random': randomModule(interp, rng), 'time': timeModule(),
      get 'matplotlib'() { const m = new PyModule('matplotlib', { pyplot: getPlt(), use: native(() => null) }); return m; }, get 'matplotlib.pyplot'() { return getPlt(); },
      'statistics': new PyModule('statistics', { mean: native((a) => fl(interp.iterate(a[0]).reduce((s, x) => s + num(x), 0) / interp.iterate(a[0]).length)), stdev: native((a) => { const v = interp.iterate(a[0]).map(num); const m = v.reduce((s, x) => s + x, 0) / v.length; return fl(Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1))); }) }),
      'warnings': new PyModule('warnings', { filterwarnings: native(() => null), simplefilter: native(() => null) }),
      'itertools': new PyModule('itertools', {
        permutations: native((a) => { const v = interp.iterate(a[0]), k = a.length > 1 ? num(a[1]) : v.length, out = []; const rec = (pre, rest) => { if (pre.length === k) { out.push(new PyTuple(pre)); return; } rest.forEach((x, i) => rec(pre.concat([x]), rest.slice(0, i).concat(rest.slice(i + 1)))); }; rec([], v); return out; }),
        combinations: native((a) => { const v = interp.iterate(a[0]), k = num(a[1]), out = []; const rec = (s, pre) => { if (pre.length === k) { out.push(new PyTuple(pre)); return; } for (let i = s; i < v.length; i++) rec(i + 1, pre.concat([v[i]])); }; rec(0, []); return out; }),
        product: native((a, kw) => { let lists = a.map((x) => interp.iterate(x)); if (kw.repeat) { const r = num(kw.repeat); const base = lists; lists = []; for (let i = 0; i < r; i++) lists.push(...base); } let out = [[]]; for (const l of lists) { const n2 = []; for (const p of out) for (const x of l) n2.push(p.concat([x])); out = n2; } return out.map((p) => new PyTuple(p)); }),
      }),
    };
    // flush figures and sample sites at the end of each cell
    interp.endCell = () => {
      if (pltMod) pltMod.__flush__();
      if (trace.size) { interp.emit({ type: 'samples', sites: new Map(trace) }); trace.clear(); }
    };
    return interp;
  }

  /** Runs a whole cell (shell lines skipped) and returns {text, events, error}. */
  function runCell(interp, code) {
    let text = '';
    const events = [];
    const prevOut = interp.out, prevEmit = interp.emit;
    interp.out = (s) => { text += s; if (text.length > 200000) throw new PyError('RuntimeError', 'output limit reached'); };
    const userEmit = interp.emit;
    interp.emit = (e) => events.push(e);
    let error = null;
    try {
      P.runCell(interp, code, { skipped: (lines) => '⚙ ' + lines.join(' · ') + '  →  skipped: the browser already provides these packages' });
    } catch (e) {
      error = e instanceof PyError ? { type: e.pyType, message: e.message, line: e.line } : { type: 'InternalError', message: e.message || String(e) };
    }
    try { interp.endCell(); } catch (e) { /* ignore */ }
    interp.out = prevOut; interp.emit = prevEmit || userEmit;
    return { text, events, error };
  }

  // helpers for the page: turn a solved model into Gantt rows and a task table
  function ganttRows(ev) {
    const { model, solver } = ev;
    if (!solver.res || !solver.res.values) return [];
    const value = (e) => solver.val(e);
    const rows = [];
    const clean = (n) => n.replace(/^(interval|iv|task|int)[_\s-]*/i, '');
    if (model.intervals.length) {
      for (const iv of model.intervals) {
        if (iv.presence) { const i = solver.res.ids.get(iv.presence.id); if (solver.res.values[i] !== 1) continue; }
        rows.push({ name: clean(iv.name), start: value(iv.start), end: value(iv.end) });
      }
      return rows;
    }
    // no intervals: pair start_X / end_X variables, or start_X with a constraint end = start + d
    const byName = new Map(model.vars.map((v) => [v.name.toLowerCase(), v]));
    for (const v of model.vars) {
      const m = v.name.match(/^(?:start|s)[_\s-]?(.+)$/i);
      if (!m) continue;
      const e = byName.get(('end_' + m[1]).toLowerCase()) || byName.get(('e_' + m[1]).toLowerCase());
      if (e) rows.push({ name: m[1], start: value(v), end: value(e) });
    }
    return rows;
  }
  /** Reconstructs tasks (name, duration, predecessors) from a solved model — used by “Edit in planner”. */
  function tasksFromModel(model) {
    const tasks = [];
    const startOf = new Map();
    const fixedSize = (iv) => (iv.size.terms.size === 0 ? iv.size.c : null);
    model.intervals.forEach((iv) => {
      const d = fixedSize(iv);
      if (d === null || iv.start.terms.size !== 1) return;
      const [id] = [...iv.start.terms.keys()];
      const name = iv.name.replace(/^(interval|iv|task)[_\s-]*/i, '');
      tasks.push({ name, d, pred: [] }); startOf.set(id, tasks[tasks.length - 1]);
    });
    if (!tasks.length) return null;
    for (const c of model.cons) {
      if (c.kind !== 'lin' || c.terms.length !== 2 || c.lo !== -Infinity) continue;
      const [[a, ca], [b, cb]] = c.terms;
      // s_a - s_b <= -d  →  a before b
      const pairs = ca === 1 && cb === -1 ? [a, b] : ca === -1 && cb === 1 ? [b, a] : null;
      if (!pairs) continue;
      const A = startOf.get(pairs[0]), B = startOf.get(pairs[1]);
      if (A && B && c.hi <= -A.d && !B.pred.includes(A.name)) B.pred.push(A.name);
    }
    const noOverlap = model.cons.some((c) => c.kind === 'nooverlap');
    return { tasks, oneWorker: noOverlap };
  }

  W.Py = { create, runCell, ganttRows, tasksFromModel, CpModel, CpSolver, LinExpr, STATUS_NAME };
})();
