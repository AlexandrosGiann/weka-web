/* Weka Web — Needle 2 tool-call runner.
 * Wraps the pure-JS Needle 2 engine (cact-full.js, tokenizer-full.js, engine-full.js) and decodes ONE tool call
 * with a finite grammar: the model may only produce one of the candidate JSON calls supplied by the app,
 * so it can never invent an action. Works in a Web Worker and on the main thread (yield() keeps the page alive). */
(function (global) {
  'use strict';
  global.NEEDLE_CQ2_NEON_MASK = 0;
  var BOS = 2, THINK = 6, END_THINK = 7, CALL = 10, END_CALL = 11, FIRST_TEXT = 14;

  function Runner(buffer) {
    this.model = new global.NeedleCact.CactModel(buffer);
    this.tok = new global.NeedleTokenizer.Tokenizer(this.model);
    this.engine = new global.NeedleFull.Engine(this.model);
    this.bodies = {};
  }
  Runner.prototype.enc = function (text) {
    var x = this.tok.encode(text);
    if (x && x.ids) x = x.ids;
    return Array.prototype.slice.call(x);
  };
  Runner.prototype.body = function (json) {
    if (this.bodies[json]) return this.bodies[json];
    var a = this.enc('<tool_call>' + json + '</tool_call>');
    if (a[0] === BOS) a.shift();
    if (a[0] !== CALL || a[a.length - 1] !== END_CALL) throw new Error('tool_call markers are not single tokens');
    return (this.bodies[json] = a.slice(1, -1));
  };
  Runner.prototype.piece = function (id) { return this.tok.pieceText(id); };
  function clean(parts) { return parts.join('').replace(/<0x0A>/g, ' ').replace(/▁/g, ' ').replace(/\s+/g, ' ').trim(); }
  function argmax(logits, from, extra) {
    var best = extra, bv = extra >= 0 ? Number(logits[extra]) : -Infinity;
    for (var i = from; i < logits.length; i++) { var v = logits[i]; if (v > bv) { bv = v; best = i; } }
    return best;
  }

  /** req: {system, tools, user, candidates:[json strings]}; onProgress(fraction, stage); yieldFn() → Promise. */
  Runner.prototype.run = async function (req, onProgress, yieldFn) {
    var self = this, eng = this.engine, t0 = Date.now();
    var prompt = '<|im_start|>system\n' + req.system + '<|im_end|>\n<|im_start|>user\n<tools>' + JSON.stringify(req.tools) + '</tools>\n' + req.user + '<|im_end|>\n<|im_start|>assistant\n';
    var ids = this.enc(prompt);
    if (!ids.length || ids[0] !== BOS) ids.unshift(BOS);
    var cands = req.candidates.map(function (j) { return { json: j, ids: self.body(j) }; });
    eng.reset(); eng.confidenceReset();
    var r = null, i;
    for (i = 0; i < ids.length; i++) {
      r = eng.forwardToken(ids[i], i === ids.length - 1);
      if (i % 8 === 7) { onProgress && onProgress(0.85 * (i + 1) / ids.length, 'prefill'); if (yieldFn) await yieldFn(); }
    }
    var L = r.logits, plain = argmax(L, FIRST_TEXT, -1);
    var root = [[CALL, Number(L[CALL])], [THINK, Number(L[THINK])], [plain, Number(L[plain])]].sort(function (a, b) { return b[1] - a[1]; })[0][0];
    var think = [], post = [];
    if (root === THINK) {
      r = eng.forwardToken(THINK, true);
      for (var s = 0; s < 32; s++) {
        var b = argmax(r.logits, FIRST_TEXT, END_THINK);
        r = eng.forwardToken(b, true);
        if (b === END_THINK) break;
        think.push(this.piece(b));
        if (yieldFn && s % 4 === 3) await yieldFn();
      }
      onProgress && onProgress(0.92, 'think');
      root = -1;
      for (var p = 0; p < 8; p++) {
        var c = argmax(r.logits, FIRST_TEXT, CALL);
        if (c === CALL) { root = CALL; break; }
        post.push(this.piece(c));
        r = eng.forwardToken(c, true);
      }
    }
    if (root !== CALL) return { call: null, plain: root === plain ? this.piece(plain) : clean(post), think: clean(think), ms: Date.now() - t0, tokens: ids.length };
    r = eng.forwardToken(CALL, true);
    var active = cands, pos = 0;
    for (;;) {
      var allowed = [];
      active.forEach(function (cd) { var n = pos < cd.ids.length ? cd.ids[pos] : END_CALL; if (allowed.indexOf(n) < 0) allowed.push(n); });
      var best = allowed[0];
      allowed.forEach(function (a) { if (Number(r.logits[a]) > Number(r.logits[best])) best = a; });
      if (best === END_CALL) {
        var done = active.filter(function (cd) { return pos >= cd.ids.length; });
        return { call: JSON.parse(done[0].json), json: done[0].json, think: clean(think), confidence: eng.confidenceValue(), ms: Date.now() - t0, tokens: ids.length };
      }
      active = active.filter(function (cd) { return pos < cd.ids.length && cd.ids[pos] === best; });
      r = eng.forwardToken(best, true);
      pos++;
      if (yieldFn && pos % 4 === 0) await yieldFn();
    }
  };
  global.NeedleRunner = Runner;
}(this));
