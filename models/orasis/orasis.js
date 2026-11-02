// Orasis-S image encoder in plain ES5 + TypedArrays. No WASM, no dependencies.
// Works in old Android browsers, Web Workers and Node.js. Same format as runtime/rust.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Orasis = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  var MEAN = [0.48145466, 0.4578275, 0.40821073];
  var STD = [0.26862954, 0.26130258, 0.27577711];
  var RESIZE_SHORT = 256;

  function f16(v) {
    var s = (v & 0x8000) ? -1 : 1, e = (v >> 10) & 31, m = v & 1023;
    if (e === 0) return s * m * 5.960464477539063e-8; // 2^-24
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * Math.pow(2, e - 15);
  }

  function decodeF16(buffer) {
    var u8 = new Uint8Array(buffer), n = u8.length >> 1, out = new Float32Array(n), i;
    // lookup table: decoding megabytes via Math.pow is slow on old phones
    var lut = new Float32Array(65536);
    for (i = 0; i < 65536; i++) lut[i] = f16(i);
    for (i = 0; i < n; i++) out[i] = lut[u8[i * 2] | (u8[i * 2 + 1] << 8)];
    return out;
  }

  function Model(buffer) {
    var u8 = new Uint8Array(buffer);
    if (u8.length < 12 || u8[0] !== 79 || u8[1] !== 82 || u8[2] !== 83 || u8[3] !== 49) throw new Error('bad magic');
    function u16(p) { return u8[p] | (u8[p + 1] << 8); }
    var nops = u16(4), nreg = u16(6), size = u16(8), dim = u16(10), end = 12 + nops * 20;
    if (u8.length < end || !nops || nreg < 2 || !size) throw new Error('truncated op table');
    var shapes = new Array(nreg), lastUse = new Array(nreg), ops = [], pos = end, i, wTotal = 0;
    shapes[0] = { c: 3, h: size, w: size };
    for (i = 0; i < nreg; i++) lastUse[i] = -1;
    for (i = 0; i < nops; i++) {
      var q = 12 + i * 20;
      var op = { kind: u16(q), a: u16(q + 2), b: u16(q + 4), o: u16(q + 6), ci: u16(q + 8), co: u16(q + 10),
        k: u16(q + 12), s: u16(q + 14), g: u16(q + 16), act: u16(q + 18), w: 0 };
      if (op.a >= nreg || op.b >= nreg || op.o >= nreg || !shapes[op.a] || shapes[op.o] || op.act > 3) throw new Error('op ' + i + ': bad registers');
      var sa = shapes[op.a], so;
      if (op.kind === 0) {
        if (!op.g || op.ci !== sa.c || !op.co || op.ci % op.g || op.co % op.g || !(op.k % 2) || !op.s) throw new Error('op ' + i + ': bad conv');
        var n = op.co * (op.ci / op.g) * op.k * op.k + op.co;
        if (pos + n * 2 > u8.length) throw new Error('truncated weights');
        op.w = wTotal; wTotal += n; pos += n * 2;
        so = { c: op.co, h: Math.floor((sa.h - 1) / op.s) + 1, w: Math.floor((sa.w - 1) / op.s) + 1 };
      } else if (op.kind === 1 || op.kind === 2) {
        var sb = shapes[op.b];
        if (!sb) throw new Error('op ' + i + ': bad registers');
        if (op.kind === 1 ? (sb.c !== sa.c || sb.h !== sa.h || sb.w !== sa.w) : (sb.c !== sa.c || sb.h !== 1 || sb.w !== 1)) throw new Error('op ' + i + ': shape mismatch');
        lastUse[op.b] = i;
        so = sa;
      } else if (op.kind === 3) {
        so = { c: sa.c, h: 1, w: 1 };
      } else throw new Error('op ' + i + ': unknown kind');
      lastUse[op.a] = i;
      shapes[op.o] = so;
      ops.push(op);
    }
    if (pos !== u8.length) throw new Error('trailing bytes');
    var last = shapes[nreg - 1];
    if (!last || last.c !== dim || last.h !== 1 || last.w !== 1) throw new Error('bad output shape');
    lastUse[nreg - 1] = Infinity;
    this.weights = decodeF16(u8.subarray(end).slice ? u8.slice(end).buffer : buffer.slice(end));
    this.ops = ops; this.shapes = shapes; this.lastUse = lastUse;
    this.size = size; this.dim = dim; this.nreg = nreg;
    this.pool = {};
  }

  function conv(op, W, src, si, dst, so) {
    var k = op.k, s = op.s, pad = k >> 1, g = op.g, ci = op.ci / g, co = op.co / g;
    var ih = si.h, iw = si.w, oh = so.h, ow = so.w, ip = ih * iw, op_ = oh * ow;
    var w = op.w, gi, oc, ic, ky, kx, x, y, v, i, oBase, iBase, x0, x1, sy, d, de, r;
    for (gi = 0; gi < g; gi++) {
      for (oc = 0; oc < co; oc++) {
        oBase = (gi * co + oc) * op_;
        for (ic = 0; ic < ci; ic++) {
          iBase = (gi * ci + ic) * ip;
          if (k === 1 && s === 1) {
            v = W[w++];
            for (i = 0; i < op_; i++) dst[oBase + i] += v * src[iBase + i];
            continue;
          }
          for (ky = 0; ky < k; ky++) {
            for (kx = 0; kx < k; kx++) {
              v = W[w++];
              x0 = kx < pad ? Math.floor((pad - kx + s - 1) / s) : 0;
              x1 = Math.min(ow, Math.floor((iw + pad - kx - 1) / s) + 1);
              if (x0 >= x1) continue;
              for (y = 0; y < oh; y++) {
                sy = y * s + ky - pad;
                if (sy < 0 || sy >= ih) continue;
                r = iBase + sy * iw + x0 * s + kx - pad;
                d = oBase + y * ow + x0;
                de = oBase + y * ow + x1;
                if (s === 1) { for (; d < de; d++) dst[d] += v * src[r++]; }
                else { for (; d < de; d++, r += s) dst[d] += v * src[r]; }
              }
            }
          }
        }
      }
    }
    var bias = op.w + op.co * ci * k * k;
    for (oc = 0; oc < op.co; oc++) {
      v = W[bias + oc];
      oBase = oc * op_;
      for (i = 0; i < op_; i++) dst[oBase + i] += v;
    }
  }

  Model.prototype.take = function (n) {
    var list = this.pool[n];
    if (list && list.length) {
      var b = list.pop();
      for (var i = 0; i < n; i++) b[i] = 0;
      return b;
    }
    return new Float32Array(n);
  };

  Model.prototype.give = function (b) {
    var n = b.length;
    (this.pool[n] || (this.pool[n] = [])).push(b);
  };

  // rgb: size*size*3 interleaved bytes. onProgress(done, total) is optional.
  Model.prototype.encode = function (rgb, onProgress) {
    var size = this.size, n = size * size, c, i, j;
    if (rgb.length !== n * 3) throw new Error('expected ' + n * 3 + ' bytes, got ' + rgb.length);
    var input = new Float32Array(n * 3);
    for (c = 0; c < 3; c++) {
      var m = MEAN[c], sd = STD[c], base = c * n;
      for (i = 0, j = c; i < n; i++, j += 3) input[base + i] = (rgb[j] / 255 - m) / sd;
    }
    var regs = new Array(this.nreg), ops = this.ops, shapes = this.shapes;
    regs[0] = input;
    for (i = 0; i < ops.length; i++) {
      var op = ops[i], sa = shapes[op.a], so = shapes[op.o], nd = so.c * so.h * so.w;
      var dst = this.take(nd), src = regs[op.a], plane = sa.h * sa.w, x, y, q, t;
      if (op.kind === 0) conv(op, this.weights, src, sa, dst, so);
      else if (op.kind === 1) {
        var bb = regs[op.b];
        for (x = 0; x < nd; x++) dst[x] = src[x] + bb[x];
      } else if (op.kind === 2) {
        var gate = regs[op.b];
        for (y = 0; y < sa.c; y++) {
          var gv = gate[y], from = y * plane;
          for (x = 0; x < plane; x++) dst[from + x] = src[from + x] * gv;
        }
      } else {
        for (y = 0; y < sa.c; y++) {
          t = 0; q = y * plane;
          for (x = 0; x < plane; x++) t += src[q + x];
          dst[y] = t / plane;
        }
      }
      if (op.act === 1) { for (x = 0; x < nd; x++) if (dst[x] < 0) dst[x] = 0; }
      else if (op.act === 2) {
        for (x = 0; x < nd; x++) { t = dst[x] / 6 + 0.5; dst[x] = t < 0 ? 0 : t > 1 ? 1 : t; }
      } else if (op.act === 3) {
        for (x = 0; x < nd; x++) { t = dst[x] / 6 + 0.5; dst[x] *= t < 0 ? 0 : t > 1 ? 1 : t; }
      }
      regs[op.o] = dst;
      if (this.lastUse[op.a] === i && regs[op.a]) { this.give(regs[op.a]); regs[op.a] = null; }
      if (op.b !== op.a && this.lastUse[op.b] === i && regs[op.b]) { this.give(regs[op.b]); regs[op.b] = null; }
      if (onProgress) onProgress(i + 1, ops.length);
    }
    var out = regs[this.nreg - 1], norm = 0;
    for (i = 0; i < out.length; i++) norm += out[i] * out[i];
    norm = Math.sqrt(norm) || 1e-12;
    var res = new Float32Array(out.length);
    for (i = 0; i < out.length; i++) res[i] = out[i] / norm;
    this.give(out);
    return res;
  };

  function filter(from, to) {
    var scale = from / to, sup = Math.max(scale, 1), res = new Array(to), i, j;
    for (i = 0; i < to; i++) {
      var c = (i + 0.5) * scale, l = Math.max(0, Math.floor(c - sup)), r = Math.min(from, Math.ceil(c + sup));
      var w = new Float32Array(r - l), t = 0;
      for (j = l; j < r; j++) { w[j - l] = Math.max(0, 1 - Math.abs((j + 0.5 - c) / sup)); t += w[j - l]; }
      for (j = 0; j < w.length; j++) w[j] /= t;
      res[i] = { l: l, w: w };
    }
    return res;
  }

  // rgb: w*h*3 interleaved bytes. Bilinear antialiased resize to short side 256, center crop.
  function preprocess(rgb, w, h, size) {
    if (!w || !h || w * h * 3 !== rgb.length) throw new Error('bad image dimensions');
    var short = Math.max(RESIZE_SHORT, size), r = short / Math.min(w, h);
    var nw = Math.max(short, Math.round(w * r)), nh = Math.max(short, Math.round(h * r));
    var fx = filter(w, nw), fy = filter(h, nh), L = (nw - size) >> 1, T = (nh - size) >> 1;
    var tmp = new Float32Array(size * h * 3), out = new Uint8Array(size * size * 3), x, y, j, c, a, v;
    for (y = 0; y < h; y++) {
      var rowBase = y * w * 3;
      for (x = 0; x < size; x++) {
        a = fx[x + L];
        for (c = 0; c < 3; c++) {
          v = 0;
          for (j = 0; j < a.w.length; j++) v += a.w[j] * rgb[rowBase + (a.l + j) * 3 + c];
          tmp[(y * size + x) * 3 + c] = v;
        }
      }
    }
    var line = size * 3;
    for (y = 0; y < size; y++) {
      a = fy[y + T];
      for (x = 0; x < line; x++) {
        v = 0;
        for (j = 0; j < a.w.length; j++) v += a.w[j] * tmp[(a.l + j) * line + x];
        v += 0.5;
        out[y * line + x] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
    return out;
  }

  // Best k {index, score} matches of unit vector z in an fp32 bank.
  function topK(z, bank, k) {
    var d = z.length, n = d ? Math.floor(bank.length / d) : 0, res = [], i, j, s;
    for (i = 0; i < n; i++) {
      s = 0;
      for (j = 0; j < d; j++) s += z[j] * bank[i * d + j];
      res.push({ index: i, score: s });
    }
    res.sort(function (a, b) { return b.score - a.score; });
    return res.slice(0, k);
  }

  return { Model: Model, preprocess: preprocess, topK: topK, decodeF16: decodeF16 };
}));
