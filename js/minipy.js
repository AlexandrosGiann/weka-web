/* Weka Web — MiniPy: a small Python interpreter for notebook cells (planning and probabilistic examples).
 * Supports the everyday subset: numbers, strings, f-strings, lists, tuples, dicts, sets, comprehensions,
 * if/elif/else, for/while, def/return/lambda, try/except, import/from-import, operator overloading through
 * __add__/__le__/… methods on host objects. Host modules (OR-Tools CP-SAT, Pyro, torch, matplotlib…) are
 * registered from planning.js. No network, no eval: the code is tokenised, parsed and walked. */
(function (global) {
  'use strict';

  // ================================================================ values
  class PyFloat { constructor(v) { this.v = v; } }
  const F = (v) => new PyFloat(v);
  const isF = (x) => x instanceof PyFloat;
  const isNum = (x) => typeof x === 'number' || x instanceof PyFloat || typeof x === 'boolean';
  const num = (x) => (x instanceof PyFloat ? x.v : x === true ? 1 : x === false ? 0 : x);
  class PyTuple { constructor(items) { this.items = items; } }
  class PyDict {
    constructor() { this.m = new Map(); }
    static key(k) { return typeof k === 'string' ? 's' + k : isNum(k) ? 'n' + num(k) : k instanceof PyTuple ? 't' + k.items.map(PyDict.key).join('\u0001') : k === null ? 'N' : k && k.__hash__ ? 'o' + k.__hash__() : 'x' + String(k); }
    get(k) { const e = this.m.get(PyDict.key(k)); return e ? e[1] : undefined; }
    set(k, v) { this.m.set(PyDict.key(k), [k, v]); }
    has(k) { return this.m.has(PyDict.key(k)); }
    del(k) { this.m.delete(PyDict.key(k)); }
    keys() { return [...this.m.values()].map((e) => e[0]); }
    values() { return [...this.m.values()].map((e) => e[1]); }
    entries() { return [...this.m.values()]; }
    get size() { return this.m.size; }
  }
  class PySet { constructor(items = []) { this.d = new PyDict(); items.forEach((x) => this.d.set(x, true)); } items() { return this.d.keys(); } }
  class PyFunction { constructor(name, params, body, closure, interp, isLambda) { Object.assign(this, { name, params, body, closure, interp, isLambda }); } }
  class PyModule { constructor(name, attrs) { this.__name__ = name; Object.assign(this, attrs); } }
  class PyError extends Error { constructor(type, msg, line) { super(msg); this.pyType = type; this.line = line; } }
  /** Host callable: fn(args, kwargs). */
  const native = (fn, name) => { fn.__native__ = true; fn.__pyname__ = name || fn.name; return fn; };

  // ================================================================ tokenizer
  const OPS = ['**=', '//=', '>>=', '<<=', '...', '**', '//', '==', '!=', '<=', '>=', '->', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', ':=', '<<', '>>',
    '+', '-', '*', '/', '%', '<', '>', '=', '(', ')', '[', ']', '{', '}', ',', ':', '.', ';', '@', '&', '|', '^', '~'];
  const KEYWORDS = new Set(['False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']);
  function tokenize(src) {
    const toks = [], indents = [0];
    let i = 0, line = 1, depth = 0, atLineStart = true;
    const push = (type, value, extra) => toks.push(Object.assign({ type, value, line }, extra));
    while (i < src.length) {
      if (atLineStart && depth === 0) {
        let col = 0, j = i;
        while (j < src.length && (src[j] === ' ' || src[j] === '\t')) { col += src[j] === '\t' ? 8 - (col % 8) : 1; j++; }
        if (j >= src.length) { i = j; break; }
        if (src[j] === '\n' || src[j] === '#' || src[j] === '\r') { // blank or comment-only line
          while (j < src.length && src[j] !== '\n') j++;
          i = j + 1; line++; continue;
        }
        if (col > indents[indents.length - 1]) { indents.push(col); push('INDENT'); }
        while (col < indents[indents.length - 1]) { indents.pop(); push('DEDENT'); }
        if (col !== indents[indents.length - 1]) throw new PyError('IndentationError', 'unindent does not match any outer indentation level', line);
        i = j; atLineStart = false;
      }
      const c = src[i];
      if (c === '\n') { if (depth === 0) { push('NEWLINE'); atLineStart = true; } i++; line++; continue; }
      if (c === '\r' || c === ' ' || c === '\t' || c === '\f') { i++; continue; }
      if (c === '\\' && src[i + 1] === '\n') { i += 2; line++; continue; }
      if (c === '#') { while (i < src.length && src[i] !== '\n') i++; continue; }
      // strings (with optional prefixes r, f, b, u, rf, fr)
      const pm = src.slice(i, i + 3).match(/^([rRfFbBuU]{1,2})?(['"])/);
      if (pm && (pm[1] ? /^[rfbu]{1,2}$/i.test(pm[1]) : true) && (pm[1] || c === '"' || c === "'")) {
        const prefix = (pm[1] || '').toLowerCase();
        let j = i + prefix.length;
        const q = src[j];
        const triple = src.slice(j, j + 3) === q + q + q;
        j += triple ? 3 : 1;
        let s = '';
        const startLine = line;
        for (;;) {
          if (j >= src.length) throw new PyError('SyntaxError', 'unterminated string literal', startLine);
          const ch = src[j];
          if (triple ? src.slice(j, j + 3) === q + q + q : ch === q) { j += triple ? 3 : 1; break; }
          if (!triple && ch === '\n') throw new PyError('SyntaxError', 'unterminated string literal', startLine);
          if (ch === '\n') line++;
          if (ch === '\\' && !prefix.includes('r')) {
            const n = src[j + 1];
            const map = { n: '\n', t: '\t', r: '\r', '\\': '\\', "'": "'", '"': '"', '0': '\0', a: '\x07', b: '\b', f: '\f', v: '\v' };
            if (n === '\n') { j += 2; line++; continue; }
            if (n in map) { s += map[n]; j += 2; continue; }
            if (n === 'x') { s += String.fromCharCode(parseInt(src.substr(j + 2, 2), 16)); j += 4; continue; }
            if (n === 'u') { s += String.fromCharCode(parseInt(src.substr(j + 2, 4), 16)); j += 6; continue; }
            if ((n === '{' || n === '}') && prefix.includes('f')) { s += '\\' + n; j += 2; continue; }
            s += '\\'; j++; continue;
          }
          s += ch; j++;
        }
        const last = toks[toks.length - 1];
        if (last && last.type === 'STRING' && !last.fstring && !prefix.includes('f')) last.value += s; // implicit concatenation
        else push('STRING', s, { fstring: prefix.includes('f'), line: startLine });
        i = j; continue;
      }
      const nm = src.slice(i).match(/^(0[xX][0-9a-fA-F_]+|0[bB][01_]+|(\d[\d_]*\.?[\d_]*|\.\d[\d_]*)([eE][+-]?\d+)?j?)/);
      if (nm && /\d/.test(c + (src[i + 1] || ''))) {
        if (!(c === '.' && !/\d/.test(src[i + 1]))) {
          const t = nm[0].replace(/_/g, '');
          let v;
          if (/^0[xX]/.test(t)) v = parseInt(t, 16); else if (/^0[bB]/.test(t)) v = parseInt(t.slice(2), 2);
          else if (/[.eE]/.test(t)) v = F(parseFloat(t)); else v = parseInt(t, 10);
          push('NUMBER', v); i += nm[0].length; continue;
        }
      }
      const id = src.slice(i).match(/^[A-Za-z_À-￿][\wÀ-￿]*/);
      if (id) { push(KEYWORDS.has(id[0]) ? 'KW' : 'NAME', id[0]); i += id[0].length; continue; }
      const op = OPS.find((o) => src.startsWith(o, i));
      if (op) {
        if ('([{'.includes(op)) depth++; else if (')]}'.includes(op)) depth = Math.max(0, depth - 1);
        push('OP', op); i += op.length; continue;
      }
      throw new PyError('SyntaxError', `invalid character '${c}'`, line);
    }
    if (toks.length && toks[toks.length - 1].type !== 'NEWLINE') push('NEWLINE');
    while (indents.length > 1) { indents.pop(); push('DEDENT'); }
    push('EOF');
    return toks;
  }

  // ================================================================ parser
  function parse(src) {
    const toks = tokenize(src);
    let p = 0;
    const peek = (k = 0) => toks[p + k];
    const at = (type, value) => { const t = toks[p]; return t.type === type && (value === undefined || t.value === value); };
    const isOp = (v) => at('OP', v), isKw = (v) => at('KW', v);
    const next = () => toks[p++];
    const err = (msg) => { throw new PyError('SyntaxError', msg || `invalid syntax near '${toks[p].value ?? toks[p].type}'`, toks[p].line); };
    const expect = (type, value) => { if (!at(type, value)) err(`expected '${value || type}'`); return next(); };
    const N = (type, props, line) => Object.assign({ type, line: line ?? toks[p].line }, props);

    function program() {
      const body = [];
      while (!at('EOF')) { if (at('NEWLINE')) { next(); continue; } body.push(...statement()); }
      return body;
    }
    function block() {
      if (!at('NEWLINE')) { const s = simpleStatements(); return s; }
      expect('NEWLINE');
      while (at('NEWLINE')) next();
      expect('INDENT');
      const body = [];
      while (!at('DEDENT') && !at('EOF')) { if (at('NEWLINE')) { next(); continue; } body.push(...statement()); }
      if (at('DEDENT')) next();
      return body;
    }
    function statement() {
      const t = peek(), line = t.line;
      if (t.type === 'OP' && t.value === '@') { // decorators are ignored
        while (!at('NEWLINE')) next(); next(); return statement();
      }
      if (t.type === 'KW') {
        switch (t.value) {
          case 'if': return [ifStmt()];
          case 'while': { next(); const test = expr(); expect('OP', ':'); const body = block(); let orelse = []; if (isKw('else')) { next(); expect('OP', ':'); orelse = block(); } return [N('While', { test, body, orelse }, line)]; }
          case 'for': {
            next(); const target = targetList(); expect('KW', 'in'); const iter = exprList(); expect('OP', ':'); const body = block();
            let orelse = []; if (isKw('else')) { next(); expect('OP', ':'); orelse = block(); }
            return [N('For', { target, iter, body, orelse }, line)];
          }
          case 'def': {
            next(); const name = expect('NAME').value; expect('OP', '('); const params = paramList(')'); expect('OP', ')');
            if (isOp('->')) { next(); expr(); }
            expect('OP', ':'); const body = block();
            return [N('FunctionDef', { name, params, body }, line)];
          }
          case 'class': {
            next(); const name = expect('NAME').value; let bases = [];
            if (isOp('(')) { next(); if (!isOp(')')) bases = argList(')').args; expect('OP', ')'); }
            expect('OP', ':'); const body = block();
            return [N('ClassDef', { name, bases, body }, line)];
          }
          case 'try': {
            next(); expect('OP', ':'); const body = block(); const handlers = []; let orelse = [], fin = [];
            while (isKw('except')) {
              next(); let etype = null, name = null;
              if (!isOp(':')) { etype = expr(); if (isKw('as')) { next(); name = expect('NAME').value; } }
              expect('OP', ':'); handlers.push({ etype, name, body: block() });
            }
            if (isKw('else')) { next(); expect('OP', ':'); orelse = block(); }
            if (isKw('finally')) { next(); expect('OP', ':'); fin = block(); }
            return [N('Try', { body, handlers, orelse, fin }, line)];
          }
          case 'with': {
            next(); const items = [];
            do { const ctx = expr(); let name = null; if (isKw('as')) { next(); name = targetList(); } items.push({ ctx, name }); } while (isOp(',') && next());
            expect('OP', ':'); return [N('With', { items, body: block() }, line)];
          }
        }
      }
      return simpleStatements();
    }
    function ifStmt() {
      const line = peek().line; next();
      const test = expr(); expect('OP', ':'); const body = block(); let orelse = [];
      if (isKw('elif')) orelse = [ifStmt()];
      else if (isKw('else')) { next(); expect('OP', ':'); orelse = block(); }
      return N('If', { test, body, orelse }, line);
    }
    function simpleStatements() {
      const out = [small()];
      while (isOp(';')) { next(); if (at('NEWLINE')) break; out.push(small()); }
      if (!at('EOF')) expect('NEWLINE');
      return out;
    }
    function small() {
      const t = peek(), line = t.line;
      if (t.type === 'KW') {
        switch (t.value) {
          case 'pass': next(); return N('Pass', {}, line);
          case 'break': next(); return N('Break', {}, line);
          case 'continue': next(); return N('Continue', {}, line);
          case 'return': next(); return N('Return', { value: at('NEWLINE') || isOp(';') ? null : exprList() }, line);
          case 'raise': next(); return N('Raise', { exc: at('NEWLINE') ? null : expr() }, line);
          case 'global': case 'nonlocal': { next(); const names = [expect('NAME').value]; while (isOp(',')) { next(); names.push(expect('NAME').value); } return N('Global', { names }, line); }
          case 'del': { next(); const targets = [expr()]; while (isOp(',')) { next(); targets.push(expr()); } return N('Del', { targets }, line); }
          case 'assert': { next(); const test = expr(); let msg = null; if (isOp(',')) { next(); msg = expr(); } return N('Assert', { test, msg }, line); }
          case 'import': {
            next(); const names = [];
            do { let mod = expect('NAME').value; while (isOp('.')) { next(); mod += '.' + expect('NAME').value; } let as = null; if (isKw('as')) { next(); as = expect('NAME').value; } names.push({ mod, as }); } while (isOp(',') && next());
            return N('Import', { names }, line);
          }
          case 'from': {
            next(); let mod = ''; while (isOp('.')) { next(); mod += '.'; } mod += expect('NAME').value; while (isOp('.')) { next(); mod += '.' + expect('NAME').value; }
            expect('KW', 'import'); const names = [];
            const paren = isOp('('); if (paren) next();
            if (isOp('*')) { next(); names.push({ name: '*' }); }
            else do { if (paren && isOp(')')) break; const name = expect('NAME').value; let as = null; if (isKw('as')) { next(); as = expect('NAME').value; } names.push({ name, as }); } while (isOp(',') && next());
            if (paren) expect('OP', ')');
            return N('ImportFrom', { mod, names }, line);
          }
        }
      }
      const e = exprList(true);
      if (isOp('=')) {
        const targets = [e];
        let value;
        for (;;) { next(); value = exprList(true); if (isOp('=')) targets.push(value); else break; }
        return N('Assign', { targets, value }, line);
      }
      const aug = ['+=', '-=', '*=', '/=', '//=', '%=', '**=', '&=', '|=', '^='].find((o) => isOp(o));
      if (aug) { next(); return N('AugAssign', { target: e, op: aug.slice(0, -1), value: exprList() }, line); }
      if (isOp(':')) { next(); expr(); let value = null; if (isOp('=')) { next(); value = exprList(); } return value ? N('Assign', { targets: [e], value }, line) : N('Pass', {}, line); }
      return N('Expr', { value: e }, line);
    }
    function paramList(close) {
      const params = [];
      while (!isOp(close)) {
        if (isOp('*')) { next(); if (isOp(',')) { next(); continue; } params.push({ name: expect('NAME').value, star: 1 }); }
        else if (isOp('**')) { next(); params.push({ name: expect('NAME').value, star: 2 }); }
        else if (isOp('/')) { next(); }
        else {
          const name = expect('NAME').value; let def = null;
          if (isOp(':') && close === ')') { next(); expr(); }
          if (isOp('=')) { next(); def = expr(); }
          params.push({ name, def });
        }
        if (!isOp(close)) expect('OP', ',');
      }
      return params;
    }
    function targetList() {
      const line = peek().line, items = [targetAtom()];
      let tuple = false;
      while (isOp(',')) { next(); tuple = true; if (isKw('in') || isOp('=')) break; items.push(targetAtom()); }
      return tuple ? N('Tuple', { elts: items }, line) : items[0];
    }
    function targetAtom() { if (isOp('*')) { const line = peek().line; next(); return N('Starred', { value: postfix() }, line); } return postfix(); }
    function exprList(allowStar) {
      const line = peek().line;
      const first = allowStar && isOp('*') ? (next(), N('Starred', { value: expr() }, line)) : expr();
      if (!isOp(',')) return first;
      const elts = [first];
      while (isOp(',')) { next(); if (at('NEWLINE') || isOp('=') || isOp(')') || isOp(';') || at('EOF') || isOp(':')) break; elts.push(allowStar && isOp('*') ? (next(), N('Starred', { value: expr() }, line)) : expr()); }
      return N('Tuple', { elts }, line);
    }
    function expr() {
      if (isKw('lambda')) {
        const line = peek().line; next(); const params = paramList(':'); expect('OP', ':');
        return N('Lambda', { params, body: expr() }, line);
      }
      const line = peek().line;
      const e = orExpr();
      if (isKw('if') ) {
        const save = p; next();
        const test = orExpr();
        if (isKw('else')) { next(); return N('IfExp', { test, body: e, orelse: expr() }, line); }
        p = save; // a comprehension "if"
      }
      if (isOp(':=') && e.type === 'Name') { next(); return N('NamedExpr', { target: e, value: expr() }, line); }
      return e;
    }
    function orExpr() { let l = andExpr(); while (isKw('or')) { const line = peek().line; next(); l = N('BoolOp', { op: 'or', left: l, right: andExpr() }, line); } return l; }
    function andExpr() { let l = notExpr(); while (isKw('and')) { const line = peek().line; next(); l = N('BoolOp', { op: 'and', left: l, right: notExpr() }, line); } return l; }
    function notExpr() { if (isKw('not')) { const line = peek().line; next(); return N('Unary', { op: 'not', operand: notExpr() }, line); } return comparison(); }
    function comparison() {
      const line = peek().line, left = bitor(), ops = [], comps = [];
      for (;;) {
        let op = null;
        if (['<', '>', '==', '!=', '<=', '>='].some((o) => isOp(o))) op = next().value;
        else if (isKw('in')) { next(); op = 'in'; }
        else if (isKw('not') && peek(1).type === 'KW' && peek(1).value === 'in') { next(); next(); op = 'not in'; }
        else if (isKw('is')) { next(); if (isKw('not')) { next(); op = 'is not'; } else op = 'is'; }
        if (!op) break;
        ops.push(op); comps.push(bitor());
      }
      return ops.length ? N('Compare', { left, ops, comps }, line) : left;
    }
    function bin(nextFn, ops) { return function f() { let l = nextFn(); while (ops.some((o) => isOp(o))) { const line = peek().line; const op = next().value; l = N('BinOp', { op, left: l, right: nextFn() }, line); } return l; }; }
    const term = () => {
      let l = factor();
      while (['*', '/', '//', '%', '@'].some((o) => isOp(o))) { const line = peek().line; const op = next().value; l = N('BinOp', { op, left: l, right: factor() }, line); }
      return l;
    };
    const arith = () => { let l = term(); while (isOp('+') || isOp('-')) { const line = peek().line; const op = next().value; l = N('BinOp', { op, left: l, right: term() }, line); } return l; };
    const shift = bin(arith, ['<<', '>>']);
    const band = bin(shift, ['&']);
    const bxor = bin(band, ['^']);
    const bitor = bin(bxor, ['|']);
    function factor() {
      if (isOp('-') || isOp('+') || isOp('~')) { const line = peek().line; const op = next().value; return N('Unary', { op, operand: factor() }, line); }
      return power();
    }
    function power() {
      const line = peek().line, base = postfix();
      if (isOp('**')) { next(); return N('BinOp', { op: '**', left: base, right: factor() }, line); }
      return base;
    }
    function postfix() {
      let e = atom();
      for (;;) {
        const line = peek().line;
        if (isOp('(')) { next(); const a = argList(')'); expect('OP', ')'); e = N('Call', { func: e, args: a.args, kwargs: a.kwargs, starargs: a.starargs, kwstar: a.kwstar }, line); }
        else if (isOp('[')) { next(); const idx = subscript(); expect('OP', ']'); e = N('Subscript', { value: e, index: idx }, line); }
        else if (isOp('.')) { next(); e = N('Attribute', { value: e, attr: expect('NAME').value }, line); }
        else return e;
      }
    }
    function subscript() {
      const line = peek().line;
      const one = () => {
        let lo = null, hi = null, step = null;
        if (!isOp(':')) { lo = expr(); if (!isOp(':')) return lo; }
        next(); if (!isOp(']') && !isOp(':') && !isOp(',')) hi = expr();
        if (isOp(':')) { next(); if (!isOp(']') && !isOp(',')) step = expr(); }
        return N('Slice', { lo, hi, step }, line);
      };
      const first = one();
      if (!isOp(',')) return first;
      const elts = [first]; while (isOp(',')) { next(); if (isOp(']')) break; elts.push(one()); }
      return N('Tuple', { elts }, line);
    }
    function argList(close) {
      const args = [], kwargs = [], starargs = [], kwstar = [];
      while (!isOp(close)) {
        if (isOp('*')) { next(); starargs.push(expr()); }
        else if (isOp('**')) { next(); kwstar.push(expr()); }
        else if (at('NAME') && peek(1).type === 'OP' && peek(1).value === '=') { const k = next().value; next(); kwargs.push({ k, v: expr() }); }
        else {
          const e = expr();
          if (isKw('for')) { args.push(comprehension('GenExp', e)); }
          else args.push(e);
        }
        if (!isOp(close)) expect('OP', ',');
      }
      return { args, kwargs, starargs, kwstar };
    }
    function comprehension(kind, elt, value) {
      const line = peek().line, gens = [];
      while (isKw('for')) {
        next(); const target = targetList(); expect('KW', 'in'); const iter = orExpr(); const ifs = [];
        while (isKw('if')) { next(); ifs.push(orExprNoCond()); }
        gens.push({ target, iter, ifs });
      }
      return N(kind, { elt, value, gens }, line);
    }
    function orExprNoCond() { return orExpr(); }
    function atom() {
      const t = peek(), line = t.line;
      if (t.type === 'NAME') { next(); return N('Name', { id: t.value }, line); }
      if (t.type === 'NUMBER') { next(); return N('Const', { value: t.value }, line); }
      if (t.type === 'STRING') {
        next();
        if (t.fstring) return fstring(t.value, line);
        let v = t.value;
        while (at('STRING')) { const s = next(); if (s.fstring) return N('JoinedStr', { parts: [N('Const', { value: v }, line), fstring(s.value, line)] }, line); v += s.value; }
        return N('Const', { value: v }, line);
      }
      if (t.type === 'KW') {
        if (t.value === 'True') { next(); return N('Const', { value: true }, line); }
        if (t.value === 'False') { next(); return N('Const', { value: false }, line); }
        if (t.value === 'None') { next(); return N('Const', { value: null }, line); }
      }
      if (t.type === 'OP') {
        if (t.value === '...') { next(); return N('Const', { value: null }, line); }
        if (t.value === '(') {
          next();
          if (isOp(')')) { next(); return N('Tuple', { elts: [] }, line); }
          const first = isOp('*') ? (next(), N('Starred', { value: expr() }, line)) : expr();
          if (isKw('for')) { const c = comprehension('GenExp', first); expect('OP', ')'); return c; }
          if (isOp(')')) { next(); return first; }
          const elts = [first];
          while (isOp(',')) { next(); if (isOp(')')) break; elts.push(isOp('*') ? (next(), N('Starred', { value: expr() }, line)) : expr()); }
          expect('OP', ')');
          return N('Tuple', { elts }, line);
        }
        if (t.value === '[') {
          next();
          if (isOp(']')) { next(); return N('List', { elts: [] }, line); }
          const first = isOp('*') ? (next(), N('Starred', { value: expr() }, line)) : expr();
          if (isKw('for')) { const c = comprehension('ListComp', first); expect('OP', ']'); return c; }
          const elts = [first];
          while (isOp(',')) { next(); if (isOp(']')) break; elts.push(isOp('*') ? (next(), N('Starred', { value: expr() }, line)) : expr()); }
          expect('OP', ']');
          return N('List', { elts }, line);
        }
        if (t.value === '{') {
          next();
          if (isOp('}')) { next(); return N('Dict', { keys: [], values: [] }, line); }
          if (isOp('**')) { next(); const keys = [null], values = [expr()]; while (isOp(',')) { next(); if (isOp('}')) break; if (isOp('**')) { next(); keys.push(null); values.push(expr()); } else { keys.push(expr()); expect('OP', ':'); values.push(expr()); } } expect('OP', '}'); return N('Dict', { keys, values }, line); }
          const k = expr();
          if (isOp(':')) {
            next(); const v = expr();
            if (isKw('for')) { const c = comprehension('DictComp', k, v); expect('OP', '}'); return c; }
            const keys = [k], values = [v];
            while (isOp(',')) { next(); if (isOp('}')) break; if (isOp('**')) { next(); keys.push(null); values.push(expr()); continue; } keys.push(expr()); expect('OP', ':'); values.push(expr()); }
            expect('OP', '}');
            return N('Dict', { keys, values }, line);
          }
          if (isKw('for')) { const c = comprehension('SetComp', k); expect('OP', '}'); return c; }
          const elts = [k]; while (isOp(',')) { next(); if (isOp('}')) break; elts.push(expr()); }
          expect('OP', '}');
          return N('SetLit', { elts }, line);
        }
      }
      err();
    }
    function fstring(s, line) {
      const parts = [];
      let i = 0, lit = '';
      while (i < s.length) {
        const c = s[i];
        if (c === '\\' && (s[i + 1] === '{' || s[i + 1] === '}')) { lit += s[i + 1]; i += 2; continue; }
        if (c === '{' && s[i + 1] === '{') { lit += '{'; i += 2; continue; }
        if (c === '}' && s[i + 1] === '}') { lit += '}'; i += 2; continue; }
        if (c === '{') {
          if (lit) { parts.push(N('Const', { value: lit }, line)); lit = ''; }
          let depth = 1, j = i + 1, q = null;
          while (j < s.length && depth) { const ch = s[j]; if (q) { if (ch === q) q = null; } else if (ch === '"' || ch === "'") q = ch; else if ('([{'.includes(ch)) depth++; else if (')]}'.includes(ch)) depth--; j++; }
          let inner = s.slice(i + 1, j - 1);
          let spec = null, conv = null, selfDoc = false;
          // split off the format spec at the top-level ':'
          let d = 0, qq = null;
          for (let k = 0; k < inner.length; k++) {
            const ch = inner[k];
            if (qq) { if (ch === qq) qq = null; continue; }
            if (ch === '"' || ch === "'") qq = ch;
            else if ('([{'.includes(ch)) d++; else if (')]}'.includes(ch)) d--;
            else if (ch === ':' && d === 0 && inner[k + 1] !== '=') { spec = inner.slice(k + 1); inner = inner.slice(0, k); break; }
            else if (ch === '!' && d === 0 && inner[k + 1] !== '=') { conv = inner[k + 1]; inner = inner.slice(0, k); }
          }
          if (/=\s*$/.test(inner) && !/[=!<>]=\s*$/.test(inner)) { selfDoc = true; inner = inner.replace(/=\s*$/, ''); }
          const e = parseExpression(inner, line);
          const specNode = spec !== null ? (spec.includes('{') ? fstring(spec, line) : N('Const', { value: spec }, line)) : null;
          if (selfDoc) parts.push(N('Const', { value: inner + '=' }, line));
          parts.push(N('FormattedValue', { value: e, spec: specNode, conv: conv || (selfDoc && !spec ? 'r' : null) }, line));
          i = j; continue;
        }
        lit += c; i++;
      }
      if (lit) parts.push(N('Const', { value: lit }, line));
      return N('JoinedStr', { parts }, line);
    }
    return program();
  }
  function parseExpression(src, line) {
    const body = parse(src.trim());
    if (body.length !== 1 || body[0].type !== 'Expr') throw new PyError('SyntaxError', 'f-string: invalid expression', line);
    return body[0].value;
  }

  // ================================================================ formatting
  function floatRepr(v) {
    if (Number.isNaN(v)) return 'nan';
    if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
    if (Number.isInteger(v) && Math.abs(v) < 1e16) return v.toFixed(1);
    let s = String(v);
    if (/e/.test(s)) { s = s.replace(/e([+-])(\d)$/, 'e$10$2'); if (!/\./.test(s.split('e')[0]) && false) s = s; }
    return s;
  }
  function repr(x) {
    if (x === null || x === undefined) return 'None';
    if (x === true) return 'True';
    if (x === false) return 'False';
    if (typeof x === 'number') return String(x);
    if (x instanceof PyFloat) return floatRepr(x.v);
    if (typeof x === 'string') { const q = x.includes("'") && !x.includes('"') ? '"' : "'"; return q + x.replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/\t/g, '\\t').replace(new RegExp(q, 'g'), '\\' + q) + q; }
    if (Array.isArray(x)) return '[' + x.map(repr).join(', ') + ']';
    if (x instanceof PyTuple) return '(' + x.items.map(repr).join(', ') + (x.items.length === 1 ? ',' : '') + ')';
    if (x instanceof PyDict) return '{' + x.entries().map(([k, v]) => repr(k) + ': ' + repr(v)).join(', ') + '}';
    if (x instanceof PySet) return x.d.size ? '{' + x.items().map(repr).join(', ') + '}' : 'set()';
    if (x instanceof PyFunction) return `<function ${x.name}>`;
    if (x instanceof PyModule) return `<module '${x.__name__}'>`;
    if (x.__repr__) return x.__repr__();
    if (x.__str__) return x.__str__();
    if (typeof x === 'function') return `<built-in function ${x.__pyname__ || x.name}>`;
    if (x.__pyclass__) return `<${x.__pyclass__.name} object>`;
    return '<object>';
  }
  function str(x) {
    if (typeof x === 'string') return x;
    if (x && typeof x === 'object' && !(x instanceof PyFloat) && !Array.isArray(x) && !(x instanceof PyTuple) && !(x instanceof PyDict) && x.__str__) return x.__str__();
    return repr(x);
  }
  /** Python format-spec mini-language. */
  function formatValue(v, spec) {
    if (!spec) return str(v);
    if (v && v.__format__) return v.__format__(spec);
    const m = spec.match(/^(?:(.)?([<>=^]))?([+\- ])?(#)?(0)?(\d+)?([,_])?(?:\.(\d+))?([bcdeEfFgGnosxX%])?$/);
    if (!m) throw new PyError('ValueError', `Invalid format specifier '${spec}'`);
    let [, fill, align, sign, , zero, width, group, prec, type] = m;
    let s;
    const n = isNum(v) ? num(v) : null;
    if (type === undefined && n === null) s = str(v);
    else if (n === null) throw new PyError('ValueError', `Unknown format code '${type}' for object of type 'str'`);
    else {
      const p = prec !== undefined ? +prec : 6;
      switch (type) {
        case 'f': case 'F': s = n.toFixed(p); break;
        case 'e': case 'E': s = n.toExponential(p).replace(/e([+-])(\d)$/, 'e$10$2'); if (type === 'E') s = s.toUpperCase(); break;
        case '%': s = (n * 100).toFixed(p) + '%'; break;
        case 'd': s = String(Math.trunc(n)); break;
        case 'x': s = Math.trunc(n).toString(16); break;
        case 'b': s = Math.trunc(n).toString(2); break;
        case 'g': case 'G': s = String(Number(n.toPrecision(prec !== undefined ? Math.max(1, +prec) : 6))); break;
        default: s = prec !== undefined ? String(Number(n.toPrecision(Math.max(1, +prec)))) : str(v);
      }
      if (group) { const [ip, fp] = s.split('.'); s = ip.replace(/\B(?=(\d{3})+(?!\d))/g, group) + (fp !== undefined ? '.' + fp : ''); }
      if (sign === '+' && n >= 0) s = '+' + s; else if (sign === ' ' && n >= 0) s = ' ' + s;
    }
    if (width) {
      const w = +width;
      if (zero && !align) { fill = '0'; align = '='; }
      fill = fill || ' ';
      if (!align) align = n !== null ? '>' : '<';
      const padN = Math.max(0, w - s.length);
      if (align === '<') s = s + fill.repeat(padN);
      else if (align === '>') s = fill.repeat(padN) + s;
      else if (align === '^') s = fill.repeat(Math.floor(padN / 2)) + s + fill.repeat(Math.ceil(padN / 2));
      else if (align === '=') { const sg = /^[+-]/.test(s) ? s[0] : ''; s = sg + fill.repeat(padN) + s.slice(sg.length); }
    }
    return s;
  }

  // ================================================================ interpreter
  const BREAK = { t: 'break' }, CONTINUE = { t: 'continue' };
  class ReturnSignal { constructor(v) { this.v = v; } }
  class Scope {
    constructor(parent, globals) { this.vars = new Map(); this.parent = parent; this.globals = globals || this; this.globalNames = null; }
    lookup(name) {
      for (let s = this; s; s = s.parent) if (s.vars.has(name)) return s.vars.get(name);
      return undefined;
    }
    assign(name, v) { if (this.globalNames && this.globalNames.has(name)) this.globals.vars.set(name, v); else this.vars.set(name, v); }
  }

  class Interpreter {
    constructor(opts = {}) {
      this.out = opts.out || ((s) => console.log(s));
      this.modules = opts.modules || {};
      this.maxSteps = opts.maxSteps || 2e7;
      this.steps = 0;
      this.globals = new Scope(null);
      this.globals.vars.set('__name__', '__main__');
      this.builtins = makeBuiltins(this);
    }
    // ---------------------------------------------------------- running
    run(src) {
      const body = parse(src);
      this.steps = 0;
      return this.execBlock(body, this.globals);
    }
    execBlock(body, scope) {
      for (const s of body) {
        const r = this.exec(s, scope);
        if (r) return r;
      }
      return null;
    }
    tick(node) { if (++this.steps > this.maxSteps) throw new PyError('RuntimeError', 'execution limit reached (possible infinite loop)', node.line); }
    exec(n, sc) {
      this.tick(n);
      this.line = n.line;
      switch (n.type) {
        case 'Expr': { const v = this.eval(n.value, sc); if (this.echo && v !== null && v !== undefined && n === this.lastExpr) this.out(repr(v) + '\n', true); return null; }
        case 'Assign': { const v = this.eval(n.value, sc); for (const t of n.targets) this.assignTo(t, v, sc); return null; }
        case 'AugAssign': {
          const cur = this.eval(n.target, sc), v = this.eval(n.value, sc);
          const iop = '__i' + OPNAME[n.op] + '__';
          let r;
          if (Array.isArray(cur) && n.op === '+') { cur.push(...this.iterate(v)); r = cur; }
          else if (cur && cur[iop]) r = cur[iop](v);
          else r = this.binop(n.op, cur, v);
          this.assignTo(n.target, r, sc); return null;
        }
        case 'If': return this.truthy(this.eval(n.test, sc)) ? this.execBlock(n.body, sc) : this.execBlock(n.orelse, sc);
        case 'While': {
          while (this.truthy(this.eval(n.test, sc))) {
            this.tick(n);
            const r = this.execBlock(n.body, sc);
            if (r === BREAK) return null;
            if (r instanceof ReturnSignal) return r;
          }
          return this.execBlock(n.orelse, sc);
        }
        case 'For': {
          const it = this.iterate(this.eval(n.iter, sc));
          for (const v of it) {
            this.tick(n);
            this.assignTo(n.target, v, sc);
            const r = this.execBlock(n.body, sc);
            if (r === BREAK) return null;
            if (r instanceof ReturnSignal) return r;
          }
          return this.execBlock(n.orelse, sc);
        }
        case 'FunctionDef': sc.assign(n.name, new PyFunction(n.name, this.evalParams(n.params, sc), n.body, sc, this)); return null;
        case 'ClassDef': sc.assign(n.name, this.makeClass(n, sc)); return null;
        case 'Return': return new ReturnSignal(n.value ? this.eval(n.value, sc) : null);
        case 'Pass': case 'Global': if (n.type === 'Global') { sc.globalNames = sc.globalNames || new Set(); n.names.forEach((x) => sc.globalNames.add(x)); } return null;
        case 'Break': return BREAK;
        case 'Continue': return CONTINUE;
        case 'Del': for (const t of n.targets) { if (t.type === 'Subscript') { const o = this.eval(t.value, sc), k = this.eval(t.index, sc); if (o instanceof PyDict) o.del(k); else if (Array.isArray(o)) o.splice(this.index(o, k), 1); } else if (t.type === 'Name') sc.vars.delete(t.id); } return null;
        case 'Assert': if (!this.truthy(this.eval(n.test, sc))) throw new PyError('AssertionError', n.msg ? str(this.eval(n.msg, sc)) : '', n.line); return null;
        case 'Raise': {
          if (!n.exc) throw new PyError('RuntimeError', 'No active exception to reraise', n.line);
          const e = this.eval(n.exc, sc);
          if (e && e.__exctype__) throw new PyError(e.__exctype__, e.msg || '', n.line);
          throw new PyError('Exception', str(e), n.line);
        }
        case 'Import':
          for (const { mod, as } of n.names) {
            const m = this.importModule(mod, n.line);
            if (as) sc.assign(as, m);
            else { const top = mod.split('.')[0]; sc.assign(top, this.importModule(top, n.line)); }
          }
          return null;
        case 'ImportFrom': {
          const m = this.importModule(n.mod, n.line);
          for (const { name, as } of n.names) {
            if (name === '*') { for (const k of Object.keys(m)) if (!k.startsWith('_')) sc.assign(k, m[k]); continue; }
            let v = m[name];
            if (v === undefined) { try { v = this.importModule(n.mod + '.' + name, n.line); } catch (e) { throw new PyError('ImportError', `cannot import name '${name}' from '${n.mod}'`, n.line); } }
            sc.assign(as || name, v);
          }
          return null;
        }
        case 'Try': {
          try {
            const r = this.execBlock(n.body, sc);
            if (r) return r;
          } catch (e) {
            if (!(e instanceof PyError)) throw e;
            for (const hnd of n.handlers) {
              const want = hnd.etype ? this.eval(hnd.etype, sc) : null;
              const names = !want ? null : (want instanceof PyTuple ? want.items : [want]).map((w) => w.__exctype__);
              if (!names || names.includes(e.pyType) || names.includes('Exception') || names.includes('BaseException')) {
                if (hnd.name) sc.assign(hnd.name, { __exctype__: e.pyType, msg: e.message, __str__: () => e.message });
                return this.execBlock(hnd.body, sc);
              }
            }
            throw e;
          } finally { if (n.fin.length) this.execBlock(n.fin, sc); }
          return this.execBlock(n.orelse, sc);
        }
        case 'With': {
          for (const it of n.items) { const c = this.eval(it.ctx, sc); const v = c && c.__enter__ ? c.__enter__() : c; if (it.name) this.assignTo(it.name, v, sc); }
          const r = this.execBlock(n.body, sc);
          for (const it of n.items) { const c = this.eval(it.ctx, sc); if (c && c.__exit__) c.__exit__(); }
          return r;
        }
      }
      throw new PyError('SyntaxError', 'unsupported statement ' + n.type, n.line);
    }
    importModule(name, line) {
      const m = this.modules[name];
      if (m) return typeof m === 'function' && !m.__native__ ? (this.modules[name] = m(this)) : m;
      throw new PyError('ModuleNotFoundError', `No module named '${name}' (Weka Web runs a subset of Python in the browser; available: ${Object.keys(this.modules).filter((k) => !k.includes('.')).sort().join(', ')})`, line);
    }
    makeClass(n, sc) {
      const cls = { name: n.name, attrs: new Map(), __class__: true };
      const csc = new Scope(sc, sc.globals);
      this.execBlock(n.body, csc);
      for (const [k, v] of csc.vars) cls.attrs.set(k, v);
      const self = this;
      const ctor = native((args, kw) => {
        const obj = { __pyclass__: cls, __dict__: new Map() };
        const init = cls.attrs.get('__init__');
        if (init) self.callFunction(init, [obj, ...args], kw);
        const strF = cls.attrs.get('__str__') || cls.attrs.get('__repr__');
        if (strF) obj.__str__ = () => self.callFunction(strF, [obj], {});
        return obj;
      }, n.name);
      ctor.__cls__ = cls;
      return ctor;
    }
    evalParams(params, sc) { return params.map((pp) => Object.assign({}, pp, { defv: pp.def ? this.eval(pp.def, sc) : undefined })); }
    assignTo(t, v, sc) {
      switch (t.type) {
        case 'Name': sc.assign(t.id, v); return;
        case 'Tuple': case 'List': {
          const items = this.iterate(v);
          const star = t.elts.findIndex((e) => e.type === 'Starred');
          if (star < 0) {
            if (items.length !== t.elts.length) throw new PyError('ValueError', items.length > t.elts.length ? `too many values to unpack (expected ${t.elts.length})` : `not enough values to unpack (expected ${t.elts.length}, got ${items.length})`, t.line);
            t.elts.forEach((e, i) => this.assignTo(e, items[i], sc));
          } else {
            const after = t.elts.length - star - 1;
            t.elts.slice(0, star).forEach((e, i) => this.assignTo(e, items[i], sc));
            this.assignTo(t.elts[star].value, items.slice(star, items.length - after), sc);
            t.elts.slice(star + 1).forEach((e, i) => this.assignTo(e, items[items.length - after + i], sc));
          }
          return;
        }
        case 'Subscript': {
          const o = this.eval(t.value, sc), k = this.eval(t.index, sc);
          if (o instanceof PyDict) o.set(k, v);
          else if (Array.isArray(o)) o[this.index(o, k)] = v;
          else if (o && o.__setitem__) o.__setitem__(k, v);
          else throw new PyError('TypeError', `'${typeName(o)}' object does not support item assignment`, t.line);
          return;
        }
        case 'Attribute': {
          const o = this.eval(t.value, sc);
          if (o && o.__dict__ instanceof Map) o.__dict__.set(t.attr, v);
          else if (o && typeof o === 'object') o[t.attr] = v;
          else throw new PyError('AttributeError', `cannot set attribute '${t.attr}'`, t.line);
          return;
        }
      }
      throw new PyError('SyntaxError', 'cannot assign to expression', t.line);
    }
    index(arr, k) { let i = num(k); if (typeof i !== 'number' || !Number.isInteger(i)) throw new PyError('TypeError', 'list indices must be integers', this.line); if (i < 0) i += arr.length; if (i < 0 || i >= arr.length) throw new PyError('IndexError', 'list index out of range', this.line); return i; }
    truthy(v) {
      if (v === null || v === undefined || v === false || v === 0 || v === '') return false;
      if (v instanceof PyFloat) return v.v !== 0;
      if (Array.isArray(v)) return v.length > 0;
      if (v instanceof PyTuple) return v.items.length > 0;
      if (v instanceof PyDict) return v.size > 0;
      if (v instanceof PySet) return v.d.size > 0;
      if (v && v.__bool__) return v.__bool__();
      return true;
    }
    iterate(v) {
      if (Array.isArray(v)) return v.slice();
      if (v instanceof PyTuple) return v.items.slice();
      if (typeof v === 'string') return [...v];
      if (v instanceof PyDict) return v.keys();
      if (v instanceof PySet) return v.items();
      if (v && v.__iter__) return v.__iter__();
      throw new PyError('TypeError', `'${typeName(v)}' object is not iterable`, this.line);
    }
    // ---------------------------------------------------------- expressions
    eval(n, sc) {
      switch (n.type) {
        case 'Const': return n.value;
        case 'Name': {
          const v = sc.lookup(n.id);
          if (v !== undefined) return v;
          if (n.id in this.builtins) return this.builtins[n.id];
          throw new PyError('NameError', `name '${n.id}' is not defined`, n.line);
        }
        case 'List': return this.evalElts(n.elts, sc);
        case 'Tuple': return new PyTuple(this.evalElts(n.elts, sc));
        case 'SetLit': return new PySet(this.evalElts(n.elts, sc));
        case 'Dict': {
          const d = new PyDict();
          n.keys.forEach((k, i) => { const v = this.eval(n.values[i], sc); if (k === null) { for (const [kk, vv] of v.entries()) d.set(kk, vv); } else d.set(this.eval(k, sc), v); });
          return d;
        }
        case 'BinOp': return this.binop(n.op, this.eval(n.left, sc), this.eval(n.right, sc));
        case 'Unary': {
          const v = this.eval(n.operand, sc);
          if (n.op === 'not') return !this.truthy(v);
          if (n.op === '-') { if (v && v.__neg__) return v.__neg__(); return isF(v) ? F(-v.v) : -num(v); }
          if (n.op === '+') return v;
          if (n.op === '~') return ~num(v);
          break;
        }
        case 'BoolOp': { const l = this.eval(n.left, sc); if (n.op === 'and') return this.truthy(l) ? this.eval(n.right, sc) : l; return this.truthy(l) ? l : this.eval(n.right, sc); }
        case 'Compare': {
          let l = this.eval(n.left, sc);
          let result = true;
          for (let i = 0; i < n.ops.length; i++) {
            const r = this.eval(n.comps[i], sc);
            const c = this.compare(n.ops[i], l, r);
            if (n.ops.length === 1) return c;
            if (!this.truthy(c)) return false;
            result = c; l = r;
          }
          return result;
        }
        case 'IfExp': return this.truthy(this.eval(n.test, sc)) ? this.eval(n.body, sc) : this.eval(n.orelse, sc);
        case 'NamedExpr': { const v = this.eval(n.value, sc); sc.assign(n.target.id, v); return v; }
        case 'Lambda': return new PyFunction('<lambda>', this.evalParams(n.params, sc), [{ type: 'Return', value: n.body, line: n.line }], sc, this, true);
        case 'Attribute': return this.getattr(this.eval(n.value, sc), n.attr, n.line);
        case 'Subscript': return this.subscript(this.eval(n.value, sc), n.index, sc);
        case 'Call': {
          const f = this.eval(n.func, sc);
          const args = [];
          for (const a of n.args) { if (a.type === 'Starred') args.push(...this.iterate(this.eval(a.value, sc))); else args.push(this.eval(a, sc)); }
          for (const s of n.starargs) args.push(...this.iterate(this.eval(s, sc)));
          const kw = {};
          for (const { k, v } of n.kwargs) kw[k] = this.eval(v, sc);
          for (const s of n.kwstar) { const d = this.eval(s, sc); for (const [k, v] of d.entries()) kw[k] = v; }
          this.line = n.line;
          return this.call(f, args, kw, n.line);
        }
        case 'JoinedStr': return n.parts.map((pt) => (pt.type === 'Const' ? pt.value : this.eval(pt, sc))).join('');
        case 'FormattedValue': {
          let v = this.eval(n.value, sc);
          if (n.conv === 'r') v = repr(v); else if (n.conv === 's') v = str(v);
          const spec = n.spec ? this.eval(n.spec, sc) : '';
          return formatValue(v, spec);
        }
        case 'ListComp': case 'GenExp': { const out = []; this.comp(n.gens, 0, new Scope(sc, sc.globals), (s) => out.push(this.eval(n.elt, s))); return out; }
        case 'SetComp': { const out = []; this.comp(n.gens, 0, new Scope(sc, sc.globals), (s) => out.push(this.eval(n.elt, s))); return new PySet(out); }
        case 'DictComp': { const d = new PyDict(); this.comp(n.gens, 0, new Scope(sc, sc.globals), (s) => d.set(this.eval(n.elt, s), this.eval(n.value, s))); return d; }
        case 'Slice': return { __slice__: true, lo: n.lo ? this.eval(n.lo, sc) : null, hi: n.hi ? this.eval(n.hi, sc) : null, step: n.step ? this.eval(n.step, sc) : null };
        case 'Starred': return this.eval(n.value, sc);
      }
      throw new PyError('SyntaxError', 'unsupported expression ' + n.type, n.line);
    }
    evalElts(elts, sc) { const out = []; for (const e of elts) { if (e.type === 'Starred') out.push(...this.iterate(this.eval(e.value, sc))); else out.push(this.eval(e, sc)); } return out; }
    comp(gens, i, sc, emit) {
      if (i === gens.length) { emit(sc); return; }
      const g = gens[i];
      for (const v of this.iterate(this.eval(g.iter, i === 0 ? sc.parent : sc))) {
        this.tick(g.iter);
        this.assignTo(g.target, v, sc);
        if (g.ifs.every((c) => this.truthy(this.eval(c, sc)))) this.comp(gens, i + 1, sc, emit);
      }
    }
    subscript(o, idxNode, sc) {
      const k = this.eval(idxNode, sc);
      if (k && k.__slice__) {
        if (o && o.__getitem__) return o.__getitem__(k);
        const seq = typeof o === 'string' ? [...o] : o instanceof PyTuple ? o.items : o;
        const L = seq.length, step = k.step === null ? 1 : num(k.step);
        let lo = k.lo === null ? (step > 0 ? 0 : L - 1) : num(k.lo), hi = k.hi === null ? (step > 0 ? L : -L - 1) : num(k.hi);
        if (lo < 0) lo += L; if (hi < 0) hi += L;
        lo = Math.max(step > 0 ? 0 : -1, Math.min(lo, step > 0 ? L : L - 1)); hi = Math.max(-1, Math.min(hi, L));
        const res = [];
        if (step > 0) for (let i = lo; i < hi; i += step) res.push(seq[i]); else for (let i = lo; i > hi; i += step) res.push(seq[i]);
        return typeof o === 'string' ? res.join('') : o instanceof PyTuple ? new PyTuple(res) : res;
      }
      if (o instanceof PyDict) { const v = o.get(k); if (v === undefined) throw new PyError('KeyError', repr(k), this.line); return v; }
      if (Array.isArray(o)) return o[this.index(o, k)];
      if (o instanceof PyTuple) return o.items[this.index(o.items, k)];
      if (typeof o === 'string') return o[this.index(o, k)];
      if (o && o.__getitem__) return o.__getitem__(k);
      throw new PyError('TypeError', `'${typeName(o)}' object is not subscriptable`, this.line);
    }
    getattr(o, name, line) {
      if (o === null || o === undefined) throw new PyError('AttributeError', `'NoneType' object has no attribute '${name}'`, line);
      if (o.__pyclass__) {
        if (o.__dict__.has(name)) return o.__dict__.get(name);
        const v = o.__pyclass__.attrs.get(name);
        if (v instanceof PyFunction) { const self = this; return native((a, kw) => self.callFunction(v, [o, ...a], kw), name); }
        if (v !== undefined) return v;
      }
      const m = methodOf(this, o, name);
      if (m) return m;
      if (typeof o === 'object' || typeof o === 'function') {
        const v = o[name];
        if (v !== undefined && !(typeof name === 'string' && name.startsWith('__') && typeof v === 'function' && !v.__native__)) {
          if (typeof v === 'function' && !v.__native__) return native((a, kw) => v.call(o, ...a, Object.keys(kw).length ? kw : undefined), name);
          return v;
        }
      }
      throw new PyError('AttributeError', `'${typeName(o)}' object has no attribute '${name}'`, line);
    }
    call(f, args, kw, line) {
      if (f instanceof PyFunction) return this.callFunction(f, args, kw);
      if (typeof f === 'function') {
        if (f.__native__) return f(args, kw || {});
        return f(...args);
      }
      if (f && f.__call__) return f.__call__(args, kw || {});
      throw new PyError('TypeError', `'${typeName(f)}' object is not callable`, line);
    }
    callFunction(f, args, kw) {
      const sc = new Scope(f.closure, f.closure.globals);
      const params = f.params;
      let ai = 0;
      for (const pp of params) {
        if (pp.star === 1) { sc.vars.set(pp.name, new PyTuple(args.slice(ai))); ai = args.length; continue; }
        if (pp.star === 2) { const d = new PyDict(); for (const [k, v] of Object.entries(kw)) if (!params.some((q) => q.name === k)) d.set(k, v); sc.vars.set(pp.name, d); continue; }
        if (ai < args.length) sc.vars.set(pp.name, args[ai++]);
        else if (pp.name in kw) sc.vars.set(pp.name, kw[pp.name]);
        else if (pp.defv !== undefined) sc.vars.set(pp.name, pp.defv);
        else throw new PyError('TypeError', `${f.name}() missing required argument: '${pp.name}'`, this.line);
      }
      if (ai < args.length && !params.some((q) => q.star === 1)) throw new PyError('TypeError', `${f.name}() takes ${params.length} positional arguments but ${args.length} were given`, this.line);
      if (this.depth > 300) throw new PyError('RecursionError', 'maximum recursion depth exceeded', this.line);
      this.depth = (this.depth || 0) + 1;
      try {
        const r = this.execBlock(f.body, sc);
        return r instanceof ReturnSignal ? r.v : null;
      } finally { this.depth--; }
    }
    binop(op, a, b) {
      const name = OPNAME[op];
      if (a !== null && typeof a === 'object' && !(a instanceof PyFloat) && a['__' + name + '__']) { const r = a['__' + name + '__'](b); if (r !== NOTIMPL) return r; }
      if (b !== null && typeof b === 'object' && !(b instanceof PyFloat) && b['__r' + name + '__']) { const r = b['__r' + name + '__'](a); if (r !== NOTIMPL) return r; }
      if (op === '+') {
        if (typeof a === 'string' && typeof b === 'string') return a + b;
        if (Array.isArray(a) && Array.isArray(b)) return a.concat(b);
        if (a instanceof PyTuple && b instanceof PyTuple) return new PyTuple(a.items.concat(b.items));
      }
      if (op === '*') {
        if (typeof a === 'string' && typeof b === 'number') return a.repeat(Math.max(0, b));
        if (typeof b === 'string' && typeof a === 'number') return b.repeat(Math.max(0, a));
        if (Array.isArray(a) && typeof b === 'number') { let r = []; for (let i = 0; i < b; i++) r = r.concat(a); return r; }
        if (Array.isArray(b) && typeof a === 'number') { let r = []; for (let i = 0; i < a; i++) r = r.concat(b); return r; }
      }
      if (op === '%' && typeof a === 'string') return percentFormat(a, b);
      if (!isNum(a) || !isNum(b)) throw new PyError('TypeError', `unsupported operand type(s) for ${op}: '${typeName(a)}' and '${typeName(b)}'`, this.line);
      const fl = isF(a) || isF(b), x = num(a), y = num(b);
      const wrap = (v) => (fl ? F(v) : v);
      switch (op) {
        case '+': return wrap(x + y);
        case '-': return wrap(x - y);
        case '*': return wrap(x * y);
        case '/': if (y === 0) throw new PyError('ZeroDivisionError', 'division by zero', this.line); return F(x / y);
        case '//': if (y === 0) throw new PyError('ZeroDivisionError', 'integer division or modulo by zero', this.line); return wrap(Math.floor(x / y));
        case '%': if (y === 0) throw new PyError('ZeroDivisionError', 'integer division or modulo by zero', this.line); return wrap(((x % y) + y) % y);
        case '**': { const r = Math.pow(x, y); return fl || y < 0 ? F(r) : r; }
        case '&': return x & y; case '|': return x | y; case '^': return x ^ y; case '<<': return x << y; case '>>': return x >> y;
      }
      throw new PyError('TypeError', 'unsupported operator ' + op, this.line);
    }
    compare(op, a, b) {
      switch (op) {
        case 'in': case 'not in': {
          let r;
          if (typeof b === 'string') r = b.includes(a);
          else if (b instanceof PyDict) r = b.has(a);
          else if (b instanceof PySet) r = b.d.has(a);
          else r = this.iterate(b).some((x) => this.eq(x, a));
          return op === 'in' ? r : !r;
        }
        case 'is': return a === b || (a === null && b === undefined);
        case 'is not': return !(a === b);
      }
      const name = { '<': 'lt', '>': 'gt', '<=': 'le', '>=': 'ge', '==': 'eq', '!=': 'ne' }[op];
      const refl = { lt: 'gt', gt: 'lt', le: 'ge', ge: 'le', eq: 'eq', ne: 'ne' }[name];
      if (a !== null && typeof a === 'object' && !(a instanceof PyFloat) && a['__' + name + '__']) { const r = a['__' + name + '__'](b); if (r !== NOTIMPL) return r; }
      if (b !== null && typeof b === 'object' && !(b instanceof PyFloat) && b['__' + refl + '__']) { const r = b['__' + refl + '__'](a); if (r !== NOTIMPL) return r; }
      if (name === 'eq') return this.eq(a, b);
      if (name === 'ne') return !this.eq(a, b);
      const c = this.cmp(a, b);
      return name === 'lt' ? c < 0 : name === 'gt' ? c > 0 : name === 'le' ? c <= 0 : c >= 0;
    }
    eq(a, b) {
      if (isNum(a) && isNum(b)) return num(a) === num(b);
      if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => this.eq(x, b[i]));
      if (a instanceof PyTuple && b instanceof PyTuple) return a.items.length === b.items.length && a.items.every((x, i) => this.eq(x, b.items[i]));
      return a === b;
    }
    cmp(a, b) {
      if (isNum(a) && isNum(b)) return num(a) - num(b);
      if (typeof a === 'string' && typeof b === 'string') return a < b ? -1 : a > b ? 1 : 0;
      const seqA = Array.isArray(a) ? a : a instanceof PyTuple ? a.items : null, seqB = Array.isArray(b) ? b : b instanceof PyTuple ? b.items : null;
      if (seqA && seqB) { for (let i = 0; i < Math.min(seqA.length, seqB.length); i++) { const c = this.cmp(seqA[i], seqB[i]); if (c) return c; } return seqA.length - seqB.length; }
      const fa = a && a.__float__ ? a.__float__() : isNum(a) ? num(a) : null, fb = b && b.__float__ ? b.__float__() : isNum(b) ? num(b) : null;
      if (fa !== null && fb !== null) return fa - fb;
      throw new PyError('TypeError', `'<' not supported between instances of '${typeName(a)}' and '${typeName(b)}'`, this.line);
    }
  }
  const NOTIMPL = { notImplemented: true };
  const OPNAME = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'truediv', '//': 'floordiv', '%': 'mod', '**': 'pow', '@': 'matmul', '&': 'and', '|': 'or', '^': 'xor', '<<': 'lshift', '>>': 'rshift' };
  function typeName(x) {
    if (x === null || x === undefined) return 'NoneType';
    if (typeof x === 'boolean') return 'bool';
    if (typeof x === 'number') return 'int';
    if (x instanceof PyFloat) return 'float';
    if (typeof x === 'string') return 'str';
    if (Array.isArray(x)) return 'list';
    if (x instanceof PyTuple) return 'tuple';
    if (x instanceof PyDict) return 'dict';
    if (x instanceof PySet) return 'set';
    if (x instanceof PyFunction) return 'function';
    if (x instanceof PyModule) return 'module';
    if (x.__pyclass__) return x.__pyclass__.name;
    if (x.__typename__) return x.__typename__;
    return typeof x === 'function' ? 'builtin_function_or_method' : 'object';
  }
  function percentFormat(fmt, args) {
    const vals = args instanceof PyTuple ? args.items.slice() : [args];
    return fmt.replace(/%([-+ 0#]*\d*(?:\.\d+)?)([sdifrx%])/g, (m, flags, t) => {
      if (t === '%') return '%';
      const v = vals.shift();
      const spec = flags + (t === 'i' ? 'd' : t === 's' || t === 'r' ? '' : t);
      if (t === 's') return formatValue(str(v), flags.replace(/^-/, '<'));
      if (t === 'r') return repr(v);
      return formatValue(v, spec.replace(/^-/, '<'));
    });
  }
  function strFormat(interp, s, args, kw) {
    let auto = 0;
    return s.replace(/\{\{|\}\}|\{([^{}]*)\}/g, (m, inner) => {
      if (m === '{{') return '{'; if (m === '}}') return '}';
      let [field, spec] = inner.split(/:(.*)/s);
      let conv = null; const cm = field.match(/!([rs])$/); if (cm) { conv = cm[1]; field = field.slice(0, -2); }
      let v;
      const base = field.match(/^[^.[]*/)[0];
      if (base === '') v = args[auto++]; else if (/^\d+$/.test(base)) v = args[+base]; else v = kw[base];
      const rest = field.slice(base.length);
      for (const part of rest.match(/\.\w+|\[[^\]]+\]/g) || []) v = part[0] === '.' ? interp.getattr(v, part.slice(1)) : interp.subscript(v, { type: 'Const', value: /^\d+$/.test(part.slice(1, -1)) ? +part.slice(1, -1) : part.slice(1, -1) }, null);
      if (conv === 'r') v = repr(v);
      return formatValue(v, spec || '');
    });
  }
  /** Methods of built-in types. */
  function methodOf(interp, o, name) {
    const N_ = (fn) => native(fn, name);
    if (typeof o === 'string') {
      const s = o;
      const map = {
        upper: () => s.toUpperCase(), lower: () => s.toLowerCase(), strip: (a) => (a[0] ? s.replace(new RegExp(`^[${esc(a[0])}]+|[${esc(a[0])}]+$`, 'g'), '') : s.trim()),
        lstrip: () => s.replace(/^\s+/, ''), rstrip: () => s.replace(/\s+$/, ''), title: () => s.replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase()), capitalize: () => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase(),
        split: (a) => (a[0] === undefined || a[0] === null ? s.trim().split(/\s+/).filter(Boolean) : s.split(a[0])), join: (a) => interp.iterate(a[0]).map((x) => { if (typeof x !== 'string') throw new PyError('TypeError', 'sequence item: expected str instance', interp.line); return x; }).join(s),
        replace: (a) => s.split(a[0]).join(a[1]), startswith: (a) => (a[0] instanceof PyTuple ? a[0].items.some((x) => s.startsWith(x)) : s.startsWith(a[0])), endswith: (a) => (a[0] instanceof PyTuple ? a[0].items.some((x) => s.endsWith(x)) : s.endsWith(a[0])),
        find: (a) => s.indexOf(a[0]), index: (a) => { const i = s.indexOf(a[0]); if (i < 0) throw new PyError('ValueError', 'substring not found', interp.line); return i; }, count: (a) => s.split(a[0]).length - 1,
        format: (a, kw) => strFormat(interp, s, a, kw), ljust: (a) => s.padEnd(num(a[0]), a[1] || ' '), rjust: (a) => s.padStart(num(a[0]), a[1] || ' '), center: (a) => formatValue(s, (a[1] || ' ') + '^' + num(a[0])), zfill: (a) => s.padStart(num(a[0]), '0'),
        isdigit: () => /^\d+$/.test(s), isalpha: () => /^[^\W\d_]+$/u.test(s), isspace: () => /^\s+$/.test(s), isupper: () => s === s.toUpperCase() && s !== s.toLowerCase(), islower: () => s === s.toLowerCase() && s !== s.toUpperCase(),
        splitlines: () => s.split(/\r?\n/), encode: () => s,
      };
      return map[name] ? N_(map[name]) : null;
    }
    if (Array.isArray(o)) {
      const a = o;
      const map = {
        append: (x) => { a.push(x[0]); return null; }, extend: (x) => { a.push(...interp.iterate(x[0])); return null; }, insert: (x) => { a.splice(num(x[0]), 0, x[1]); return null; },
        pop: (x) => { if (!a.length) throw new PyError('IndexError', 'pop from empty list', interp.line); return x.length ? a.splice(interp.index(a, x[0]), 1)[0] : a.pop(); },
        remove: (x) => { const i = a.findIndex((v) => interp.eq(v, x[0])); if (i < 0) throw new PyError('ValueError', 'list.remove(x): x not in list', interp.line); a.splice(i, 1); return null; },
        index: (x) => { const i = a.findIndex((v) => interp.eq(v, x[0])); if (i < 0) throw new PyError('ValueError', `${repr(x[0])} is not in list`, interp.line); return i; },
        count: (x) => a.filter((v) => interp.eq(v, x[0])).length, clear: () => { a.length = 0; return null; }, copy: () => a.slice(),
        reverse: () => { a.reverse(); return null; }, sort: (x, kw) => { const s = sortList(interp, a, kw); a.splice(0, a.length, ...s); return null; },
      };
      return map[name] ? N_(map[name]) : null;
    }
    if (o instanceof PyTuple) { const map = { index: (x) => o.items.findIndex((v) => interp.eq(v, x[0])), count: (x) => o.items.filter((v) => interp.eq(v, x[0])).length }; return map[name] ? N_(map[name]) : null; }
    if (o instanceof PyDict) {
      const d = o;
      const map = {
        keys: () => d.keys(), values: () => d.values(), items: () => d.entries().map(([k, v]) => new PyTuple([k, v])),
        get: (x) => { const v = d.get(x[0]); return v === undefined ? (x.length > 1 ? x[1] : null) : v; },
        update: (x, kw) => { if (x[0]) for (const [k, v] of x[0].entries()) d.set(k, v); for (const [k, v] of Object.entries(kw)) d.set(k, v); return null; },
        setdefault: (x) => { if (!d.has(x[0])) d.set(x[0], x.length > 1 ? x[1] : null); return d.get(x[0]); },
        pop: (x) => { const v = d.get(x[0]); if (v === undefined) { if (x.length > 1) return x[1]; throw new PyError('KeyError', repr(x[0]), interp.line); } d.del(x[0]); return v; },
        copy: () => { const n = new PyDict(); for (const [k, v] of d.entries()) n.set(k, v); return n; }, clear: () => { d.m.clear(); return null; },
      };
      return map[name] ? N_(map[name]) : null;
    }
    if (o instanceof PySet) { const map = { add: (x) => { o.d.set(x[0], true); return null; }, remove: (x) => { o.d.del(x[0]); return null; }, discard: (x) => { o.d.del(x[0]); return null; }, union: (x) => new PySet(o.items().concat(interp.iterate(x[0]))) }; return map[name] ? N_(map[name]) : null; }
    if (o instanceof PyFloat || typeof o === 'number') { const map = { is_integer: () => Number.isInteger(num(o)), real: () => o }; return map[name] ? N_(map[name]) : null; }
    return null;
  }
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\-]/g, '\\$&');
  function sortList(interp, arr, kw = {}) {
    const key = kw.key ? (x) => interp.call(kw.key, [x], {}) : (x) => x;
    const decorated = arr.map((x, i) => [key(x), i, x]);
    decorated.sort((p, q) => interp.cmp(p[0], q[0]) || p[1] - q[1]);
    const out = decorated.map((d) => d[2]);
    return kw.reverse && interp.truthy(kw.reverse) ? out.reverse() : out;
  }

  function makeBuiltins(interp) {
    const B = {};
    const def = (name, fn) => { B[name] = native(fn, name); };
    def('print', (args, kw) => {
      const sep = kw.sep !== undefined && kw.sep !== null ? kw.sep : ' ', end = kw.end !== undefined && kw.end !== null ? kw.end : '\n';
      interp.out(args.map(str).join(sep) + end, true);
      return null;
    });
    def('range', (a) => { let [s, e, st] = a.length === 1 ? [0, num(a[0]), 1] : [num(a[0]), num(a[1]), a.length > 2 ? num(a[2]) : 1]; const r = []; if (st > 0) for (let i = s; i < e; i += st) r.push(i); else for (let i = s; i > e; i += st) r.push(i); if (r.length > 5e6) throw new PyError('MemoryError', 'range too large', interp.line); return r; });
    def('len', (a) => { const x = a[0]; if (typeof x === 'string' || Array.isArray(x)) return x.length; if (x instanceof PyTuple) return x.items.length; if (x instanceof PyDict) return x.size; if (x instanceof PySet) return x.d.size; if (x && x.__len__) return x.__len__(); throw new PyError('TypeError', `object of type '${typeName(x)}' has no len()`, interp.line); });
    def('sum', (a) => { let acc = a.length > 1 ? a[1] : 0; for (const x of interp.iterate(a[0])) acc = interp.binop('+', acc, x); return acc; });
    const minmax = (sign) => (a, kw) => {
      const items = a.length === 1 ? interp.iterate(a[0]) : a.slice();
      if (!items.length) { if ('default' in kw) return kw.default; throw new PyError('ValueError', `${sign > 0 ? 'max' : 'min'}() arg is an empty sequence`, interp.line); }
      const key = kw.key ? (x) => interp.call(kw.key, [x], {}) : (x) => x;
      let best = items[0], bk = key(best);
      for (const x of items.slice(1)) { const k = key(x); if (sign * interp.cmp(k, bk) > 0) { best = x; bk = k; } }
      return best;
    };
    def('max', minmax(1)); def('min', minmax(-1));
    def('abs', (a) => (a[0] && a[0].__abs__ ? a[0].__abs__() : isF(a[0]) ? F(Math.abs(a[0].v)) : Math.abs(num(a[0]))));
    def('round', (a) => { const x = a[0] && a[0].__float__ ? a[0].__float__() : num(a[0]); if (a.length < 2 || a[1] === null) { const r = Math.round(x); return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r; } const m = Math.pow(10, num(a[1])); return F(Math.round(x * m) / m); });
    def('int', (a) => { const x = a[0]; if (x === undefined) return 0; if (typeof x === 'string') { const v = parseInt(x.trim(), a[1] ? num(a[1]) : 10); if (Number.isNaN(v)) throw new PyError('ValueError', `invalid literal for int(): ${repr(x)}`, interp.line); return v; } if (x && x.__int__) return x.__int__(); if (x && x.__float__) return Math.trunc(x.__float__()); return Math.trunc(num(x)); });
    def('float', (a) => { const x = a[0]; if (x === undefined) return F(0); if (typeof x === 'string') { const t = x.trim().toLowerCase(); if (t === 'inf' || t === 'infinity') return F(Infinity); if (t === '-inf') return F(-Infinity); if (t === 'nan') return F(NaN); const v = Number(t); if (Number.isNaN(v)) throw new PyError('ValueError', `could not convert string to float: ${repr(x)}`, interp.line); return F(v); } if (x && x.__float__) return F(x.__float__()); return F(num(x)); });
    def('str', (a) => (a.length ? str(a[0]) : ''));
    def('repr', (a) => repr(a[0]));
    def('bool', (a) => interp.truthy(a[0]));
    def('list', (a) => (a.length ? interp.iterate(a[0]) : []));
    def('tuple', (a) => new PyTuple(a.length ? interp.iterate(a[0]) : []));
    def('set', (a) => new PySet(a.length ? interp.iterate(a[0]) : []));
    def('dict', (a, kw) => { const d = new PyDict(); if (a[0]) { if (a[0] instanceof PyDict) for (const [k, v] of a[0].entries()) d.set(k, v); else for (const p of interp.iterate(a[0])) { const it = interp.iterate(p); d.set(it[0], it[1]); } } for (const [k, v] of Object.entries(kw)) d.set(k, v); return d; });
    def('sorted', (a, kw) => sortList(interp, interp.iterate(a[0]), kw));
    def('reversed', (a) => interp.iterate(a[0]).reverse());
    def('enumerate', (a, kw) => { const s = a.length > 1 ? num(a[1]) : kw.start ? num(kw.start) : 0; return interp.iterate(a[0]).map((x, i) => new PyTuple([i + s, x])); });
    def('zip', (a) => { const its = a.map((x) => interp.iterate(x)); const n = its.length ? Math.min(...its.map((x) => x.length)) : 0; const out = []; for (let i = 0; i < n; i++) out.push(new PyTuple(its.map((x) => x[i]))); return out; });
    def('map', (a) => { const its = a.slice(1).map((x) => interp.iterate(x)); const n = Math.min(...its.map((x) => x.length)); const out = []; for (let i = 0; i < n; i++) out.push(interp.call(a[0], its.map((x) => x[i]), {})); return out; });
    def('filter', (a) => interp.iterate(a[1]).filter((x) => (a[0] === null ? interp.truthy(x) : interp.truthy(interp.call(a[0], [x], {})))));
    def('any', (a) => interp.iterate(a[0]).some((x) => interp.truthy(x)));
    def('all', (a) => interp.iterate(a[0]).every((x) => interp.truthy(x)));
    def('isinstance', (a) => { const types = a[1] instanceof PyTuple ? a[1].items : [a[1]]; return types.some((t) => t && (t.__pyname__ === typeName(a[0]) || (t.__cls__ && a[0] && a[0].__pyclass__ === t.__cls__) || (t.__pyname__ === 'float' && typeof a[0] === 'number' && false))); });
    def('type', (a) => ({ __typename__: 'type', __str__: () => `<class '${typeName(a[0])}'>`, __name__: typeName(a[0]) }));
    def('hasattr', (a) => { try { interp.getattr(a[0], a[1]); return true; } catch (e) { return false; } });
    def('getattr', (a) => { try { return interp.getattr(a[0], a[1]); } catch (e) { if (a.length > 2) return a[2]; throw e; } });
    def('setattr', (a) => { interp.assignTo({ type: 'Attribute', value: { type: 'Const', value: a[0] }, attr: a[1] }, a[2], interp.globals); return null; });
    def('divmod', (a) => new PyTuple([interp.binop('//', a[0], a[1]), interp.binop('%', a[0], a[1])]));
    def('pow', (a) => interp.binop('**', a[0], a[1]));
    def('chr', (a) => String.fromCharCode(num(a[0]))); def('ord', (a) => a[0].charCodeAt(0));
    def('format', (a) => formatValue(a[0], a[1] || ''));
    def('input', () => { throw new PyError('RuntimeError', 'input() is not available in the browser notebook', interp.line); });
    def('open', () => { throw new PyError('RuntimeError', 'files are not available in the browser notebook — load data on the Preprocess page instead', interp.line); });
    def('display', (a) => { a.forEach((x) => interp.out(str(x) + '\n', true)); return null; });
    for (const e of ['Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError', 'ZeroDivisionError', 'RuntimeError', 'NameError', 'AttributeError', 'AssertionError', 'ImportError', 'ModuleNotFoundError', 'StopIteration', 'NotImplementedError', 'BaseException']) {
      const ctor = native((a) => ({ __exctype__: e, msg: a.length ? str(a[0]) : '', __str__() { return this.msg; } }), e);
      ctor.__exctype__ = e;
      B[e] = ctor;
    }
    B.int.__pyname__ = 'int'; B.float.__pyname__ = 'float'; B.str.__pyname__ = 'str'; B.list.__pyname__ = 'list'; B.dict.__pyname__ = 'dict'; B.tuple.__pyname__ = 'tuple'; B.bool.__pyname__ = 'bool'; B.set.__pyname__ = 'set';
    B.NotImplemented = NOTIMPL;
    return B;
  }

  /** Runs one notebook cell. Shell lines (!pip …, %magic, bare "pip install …") are skipped with a note. */
  function runCell(interp, code, notes = {}) {
    const lines = code.split(/\r?\n/);
    const skipped = [];
    const kept = lines.map((l) => {
      const t = l.trim();
      if (/^[!%]/.test(t) || /^pip\s+install\b/.test(t) || /^(apt|apt-get|conda)\s/.test(t)) { skipped.push(t); return ''; }
      return l;
    }).join('\n');
    if (skipped.length) interp.out((notes.skipped ? notes.skipped(skipped) : '[skipped shell command: ' + skipped.join('; ') + ']') + '\n', true, 'note');
    if (!kept.trim()) return;
    const body = parse(kept);
    // like Jupyter, echo the value of a bare last expression
    const last = body[body.length - 1];
    interp.echo = true;
    interp.lastExpr = last && last.type === 'Expr' && !(last.value.type === 'Call' && last.value.func.type === 'Name' && last.value.func.id === 'print') ? last : null;
    interp.steps = 0;
    try { interp.execBlock(body, interp.globals); } finally { interp.echo = false; interp.lastExpr = null; }
  }

  global.MiniPy = { Interpreter, parse, tokenize, runCell, PyFloat, PyTuple, PyDict, PySet, PyModule, PyError, F, num, isNum, isF, native, repr, str, formatValue, typeName, NOTIMPL };
})(typeof window !== 'undefined' ? window : globalThis);
