/*
 * Whistle — pure JavaScript runtime for Cactus Compute's Whistle speech-to-text model.
 *
 * No WASM, no native addon, no dependencies. Reads the official `whistle.cact`
 * archive (tag 0x05E12A84) and runs the whole pipeline: log-mel front end,
 * convolutional stem, 8-block audio encoder, 8-block decoder with gated cross
 * attention and engram lookups, beam search and the SentencePiece detokeniser.
 *
 * The numerics follow the reference engine: CQ weights are consumed as int8
 * centroids, activations are Hadamard-rotated per group of 128 and quantised
 * to int8 before every matmul.
 *
 * Works in browsers (window / worker: `self.Whistle`) and in Node (`require`).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Whistle = factory();
})(typeof self !== 'undefined' ? self : this, function () {
'use strict';

var TAG = 0x05e12a84;
var FP16 = 1, FP32 = 2, CQ = 3, RAW = 4;

// ---------------------------------------------------------------- utilities

var F16_TABLE = null;
function f16Table() {
  if (F16_TABLE) return F16_TABLE;
  var t = new Float32Array(65536);
  for (var h = 0; h < 65536; h++) {
    var s = h >>> 15, e = (h >>> 10) & 31, f = h & 1023, v;
    if (e === 0) v = f * Math.pow(2, -24);
    else if (e === 31) v = f ? NaN : Infinity;
    else v = (1 + f / 1024) * Math.pow(2, e - 15);
    t[h] = s ? -v : v;
  }
  return (F16_TABLE = t);
}

// In-place unnormalised fast Walsh-Hadamard transform of a[off .. off+n).
function fwht(a, off, n) {
  for (var len = 1; len < n; len <<= 1) {
    for (var i = 0; i < n; i += len << 1) {
      for (var j = i; j < i + len; j++) {
        var x = a[off + j], y = a[off + j + len];
        a[off + j] = x + y;
        a[off + j + len] = x - y;
      }
    }
  }
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// ------------------------------------------------------------- .cact reader

function Cact(arrayBuffer) {
  if (!(arrayBuffer instanceof ArrayBuffer)) {
    // Node Buffer / typed array view
    var v = arrayBuffer;
    arrayBuffer = v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength);
  }
  this.ab = arrayBuffer;
  this.u8 = new Uint8Array(arrayBuffer);
  this.dv = new DataView(arrayBuffer);
  var dv = this.dv;
  if (this.u8.length < 196) throw new Error('short .cact file');
  if (dv.getUint32(0, true) !== TAG) {
    throw new Error('not a Needle 3 / Whistle .cact archive (tag 0x' +
      dv.getUint32(0, true).toString(16) + ')');
  }
  var u = function (i) { return dv.getUint32(i * 4, true); };
  this.numTensors = u(1);
  this.codebookLen = u(2);
  this.header = {
    kvWindow: u(3), kvBits: u(4), vocab: u(5), outVocab: u(6), width: u(7),
    heads: u(8), kvHeads: u(9), layers: u(10), qkDim: u(11), vDim: u(12),
    maxSeq: u(13), hadaN: u(14), lanes: u(15), slidingWindow: u(16),
    taps: u(19), engramSlots: u(20), engramSubDim: u(21), engramTables: u(22),
    engramTaps: u(23), engramDilation: u(24), engramSeedHeads: u(25),
    engramOrders: [], engramSites: [], ropeTheta: dv.getFloat32(192, true)
  };
  var i;
  for (i = 0; i < u(26); i++) this.header.engramOrders.push(u(27 + i));
  for (i = 0; i < u(31); i++) this.header.engramSites.push(u(32 + i));

  var cb = new Float32Array(this.codebookLen);
  for (i = 0; i < cb.length; i++) cb[i] = dv.getFloat32(196 + i * 4, true);
  this.codebooks = { 2: cb.subarray(0, 4), 3: cb.subarray(4, 12), 4: cb.subarray(12, 28) };

  var dir = 196 + this.codebookLen * 4;
  this.tensors = [];
  for (i = 0; i < this.numTensors; i++) {
    var o = dir + i * 44, ndim = this.u8[o + 1], shape = [];
    for (var k = 0; k < ndim && k < 4; k++) shape.push(dv.getUint32(o + 4 + k * 4, true));
    var hi = dv.getUint32(o + 24, true), hi2 = dv.getUint32(o + 32, true);
    if (hi || hi2) throw new Error('tensor offset beyond 4 GiB');
    var t = {
      index: i, dtype: this.u8[o], shape: shape,
      offset: dv.getUint32(o + 20, true), nbytes: dv.getUint32(o + 28, true),
      group: dv.getUint32(o + 36, true), bits: dv.getUint32(o + 40, true)
    };
    if (t.offset + t.nbytes > this.u8.length) throw new Error('tensor #' + i + ' out of bounds');
    this.tensors.push(t);
  }
}

Cact.prototype.fp16 = function (i) {
  var t = this.tensors[i];
  if (t.dtype !== FP16) throw new Error('tensor #' + i + ' is not FP16');
  var n = t.nbytes >> 1, out = new Float32Array(n), tab = f16Table(), dv = this.dv;
  for (var j = 0; j < n; j++) out[j] = tab[dv.getUint16(t.offset + 2 * j, true)];
  return out;
};

Cact.prototype.fp32 = function (i) {
  var t = this.tensors[i];
  if (t.dtype !== FP32) throw new Error('tensor #' + i + ' is not FP32');
  var n = t.nbytes >> 2, out = new Float32Array(n), dv = this.dv;
  for (var j = 0; j < n; j++) out[j] = dv.getFloat32(t.offset + 4 * j, true);
  return out;
};

Cact.prototype.raw = function (i) {
  var t = this.tensors[i];
  return this.u8.subarray(t.offset, t.offset + t.nbytes);
};

// ------------------------------------------------------------ audio front end

function hannSymmetric(n) {
  var w = new Float32Array(n);
  for (var i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
  return w;
}

function FrontEnd(cfg, melFilterbank) {
  this.rate = cfg.sampleRate; this.nFft = cfg.nFft; this.win = cfg.windowSamples;
  this.hop = cfg.hopSamples; this.mels = cfg.melBins;
  this.window = hannSymmetric(this.win);
  var bins = (this.nFft >> 1) + 1, m = this.mels, c, b;
  this.bins = bins;
  // Filterbank is stored [bin][mel]; keep it mel-major with each filter's bin range.
  this.fb = new Float32Array(m * bins);
  this.lo = new Int32Array(m); this.hi = new Int32Array(m);
  for (c = 0; c < m; c++) {
    var lo = bins, hi = 0;
    for (b = 0; b < bins; b++) {
      var v = melFilterbank[b * m + c];
      this.fb[c * bins + b] = v;
      if (v !== 0) { if (b < lo) lo = b; hi = b + 1; }
    }
    this.lo[c] = lo; this.hi[c] = hi;
  }
  // Mel channels whose whole filter lies inside 250..3500 Hz: the loudness band.
  this.bandLo = m; this.bandHi = 0;
  for (c = 0; c < m; c++) {
    if (this.lo[c] * this.rate / this.nFft < 250) continue;
    if (this.hi[c] * this.rate / this.nFft > 3500) continue;
    if (c < this.bandLo) this.bandLo = c;
    this.bandHi = c + 1;
  }
  // FFT tables
  var n = this.nFft;
  this.cos = new Float64Array(n >> 1); this.sin = new Float64Array(n >> 1);
  for (b = 0; b < n >> 1; b++) {
    this.cos[b] = Math.cos(2 * Math.PI * b / n);
    this.sin[b] = Math.sin(2 * Math.PI * b / n);
  }
  this.rev = new Int32Array(n);
  var bitsN = Math.round(Math.log(n) / Math.LN2);
  for (b = 0; b < n; b++) {
    var r = 0;
    for (var k = 0; k < bitsN; k++) if (b & (1 << k)) r |= 1 << (bitsN - 1 - k);
    this.rev[b] = r;
  }
}

// Loudness normalisation gain: the 99th percentile RMS of 20 ms blocks maps to 0.1.
FrontEnd.prototype.gain = function (pcm, n) {
  var block = this.hop * 2, k = Math.floor(n / block);
  if (k <= 0) return 1;
  var rms = new Float32Array(k);
  for (var e = 0; e < k; e++) {
    var s = 0;
    for (var i = e * block; i < (e + 1) * block; i++) {
      var x = pcm[i];
      if (x > -65536 && x < 65536) s += x * x;
    }
    rms[e] = Math.sqrt(s / block);
  }
  rms.sort();
  var q = rms[Math.floor(Math.fround(Math.fround(k - 1) * Math.fround(0.99)))];
  return q > 1e-6 ? Math.fround(0.1) / q : 1;
};

// pcm: Float32Array of 16 kHz mono samples. Returns { feat: [T*mels], T, db: [T] }.
FrontEnd.prototype.run = function (pcm) {
  var n = pcm.length;
  if (n < this.win) {
    var padded = new Float32Array(this.win);
    padded.set(pcm);
    pcm = padded; n = this.win;
  }
  var T = Math.floor(n / this.hop), m = this.mels, N = this.nFft, half = N >> 1;
  var gain = this.gain(pcm, n), win = this.window, W = this.win;
  var feat = new Float32Array(T * m), db = new Float32Array(T);
  var re = new Float64Array(N), im = new Float64Array(N), pow = new Float64Array(half + 1);
  var rev = this.rev, cs = this.cos, sn = this.sin, fb = this.fb, bins = this.bins;
  var t, a, c, b;
  for (t = 0; t < T; t++) {
    var start = this.hop * t - (W >> 1);
    for (a = 0; a < N; a++) { re[rev[a]] = 0; im[a] = 0; }
    for (a = 0; a < W; a++) {
      var idx = a + start;
      if (idx < 0 || idx >= n) continue;
      var x = pcm[idx];
      if (!(x > -65536 && x < 65536)) continue;
      re[rev[a]] = gain * x * win[a];
    }
    for (var len = 2; len <= N; len <<= 1) {
      var h2 = len >> 1, step = N / len;
      for (var i = 0; i < N; i += len) {
        for (var j = 0, k = 0; j < h2; j++, k += step) {
          var wr = cs[k], wi = -sn[k];
          var p = i + j, q = p + h2;
          var xr = re[q] * wr - im[q] * wi, xi = re[q] * wi + im[q] * wr;
          re[q] = re[p] - xr; im[q] = im[p] - xi;
          re[p] += xr; im[p] += xi;
        }
      }
    }
    for (b = 0; b <= half; b++) pow[b] = re[b] * re[b] + im[b] * im[b];
    var band = 0;
    for (c = 0; c < m; c++) {
      var e = 0, row = c * bins;
      for (b = this.lo[c]; b < this.hi[c]; b++) e += fb[row + b] * pow[b];
      feat[t * m + c] = Math.log(e + 5.960464477539063e-08);
      if (c >= this.bandLo && c < this.bandHi) band += e;
    }
    db[t] = this.bandHi > 0 ? 10 * Math.log10(band + 1e-30) : -300;
  }
  // Per-channel mean / variance normalisation over the clip.
  var denom = 1 / (Math.max(T, 2) - 1);
  for (c = 0; c < m; c++) {
    var mean = 0, v = 0, d;
    for (t = 0; t < T; t++) mean += feat[t * m + c];
    mean /= T;
    for (t = 0; t < T; t++) { d = feat[t * m + c] - mean; v += d * d; }
    var inv = 1 / (Math.sqrt(v * denom) + 1e-5);
    for (t = 0; t < T; t++) feat[t * m + c] = (feat[t * m + c] - mean) * inv;
  }
  return { feat: feat, T: T, db: db };
};

// --------------------------------------------------- quantised linear algebra

var INV127 = Math.fround(0.007874015718698502);

// A CQ matrix [out, in], kept the way the engine consumes it: one int8 centroid
// per weight (in the Hadamard-rotated domain) plus one float scale per row group.
function QMat(cact, index) {
  var t = cact.tensors[index];
  if (t.dtype !== CQ || t.shape.length !== 2) throw new Error('tensor #' + index + ' is not a CQ matrix');
  var out = t.shape[0], inp = t.shape[1], G = t.group, bits = t.bits;
  var nG = Math.ceil(inp / G), inPad = nG * G, rowBytes = (inPad * bits) >> 3;
  if (out * rowBytes + out * nG * 2 !== t.nbytes) throw new Error('tensor #' + index + ' has an unexpected CQ layout');
  var cb = cact.codebooks[bits];
  if (!cb) throw new Error('unsupported CQ width ' + bits);
  var mx = 0, i;
  for (i = 0; i < cb.length; i++) mx = Math.max(mx, Math.abs(cb[i]));
  var cbScale = Math.fround(mx / 127), table = new Int8Array(cb.length);
  for (i = 0; i < cb.length; i++) table[i] = Math.round(cb[i] / cbScale);

  this.index = index; this.out = out; this.inp = inp; this.G = G; this.nG = nG;
  this.inPad = inPad; this.bits = bits; this.rowBytes = rowBytes;
  this.table = table; this.cbScale = cbScale;
  this.u8 = cact.u8; this.base = t.offset;
  var dv = cact.dv, tab = f16Table(), nb = t.offset + out * rowBytes;
  this.rawNorm = new Float32Array(out * nG);          // fp16 group norms
  this.nf = new Float32Array(out * nG);               // norm * codebook scale
  for (i = 0; i < out * nG; i++) {
    this.rawNorm[i] = tab[dv.getUint16(nb + 2 * i, true)];
    this.nf[i] = this.rawNorm[i] * cbScale;
  }
  this.w = null;
}

QMat.prototype.decodeRowInto = function (r, dst, dOff) {
  var u8 = this.u8, p = this.base + r * this.rowBytes, table = this.table;
  var n = this.inPad, bits = this.bits, k, b;
  if (bits === 2) {
    for (k = 0; k < n; k += 4) {
      b = u8[p + (k >> 2)];
      dst[dOff + k] = table[b & 3]; dst[dOff + k + 1] = table[(b >> 2) & 3];
      dst[dOff + k + 2] = table[(b >> 4) & 3]; dst[dOff + k + 3] = table[b >> 6];
    }
  } else if (bits === 4) {
    for (k = 0; k < n; k += 2) {
      b = u8[p + (k >> 1)];
      dst[dOff + k] = table[b & 15]; dst[dOff + k + 1] = table[b >> 4];
    }
  } else {
    var mask = (1 << bits) - 1;
    for (k = 0; k < n; k++) {
      var bit = k * bits, byte = p + (bit >> 3), sh = bit & 7;
      dst[dOff + k] = table[((u8[byte] | (u8[byte + 1] << 8)) >> sh) & mask];
    }
  }
};

QMat.prototype.weights = function () {
  if (!this.w) {
    var w = new Int8Array(this.out * this.inPad);
    for (var r = 0; r < this.out; r++) this.decodeRowInto(r, w, r * this.inPad);
    this.w = w;
  }
  return this.w;
};

// Rows below this count are multiplied directly; wider matrices go through the
// per-activation lookup tables, which read the packed indices as they sit in the file.
var LUT_MIN_ROWS = { 2: 96, 4: 256 };

// out[oOff + i] = row (rowStart + i) of W times the activation held by `act`.
QMat.prototype.apply = function (act, out, oOff, rowStart, rowCount) {
  rowStart = rowStart || 0;
  if (rowCount === undefined) rowCount = this.out - rowStart;
  var min = LUT_MIN_ROWS[this.bits];
  if (min !== undefined && (rowCount >= min || act.hasLut(this.bits))) this.applyLut(act, out, oOff, rowStart, rowCount);
  else this.applyDirect(act, out, oOff, rowStart, rowCount);
};

QMat.prototype.applyDirect = function (act, out, oOff, rowStart, rowCount) {
  var w = this.weights(), nf = this.nf, G = this.G, nG = this.nG, q = act.q, s = act.s;
  var wi = rowStart * this.inPad, ni = rowStart * nG;
  for (var r = 0; r < rowCount; r++) {
    var acc = 0;
    for (var g = 0; g < nG; g++) {
      var d = 0, qi = g * G, end = qi + G;
      for (; qi < end; qi += 4, wi += 4) {
        d += w[wi] * q[qi] + w[wi + 1] * q[qi + 1] + w[wi + 2] * q[qi + 2] + w[wi + 3] * q[qi + 3];
      }
      acc += d * nf[ni++] * s[g];
    }
    out[oOff + r] = acc;
  }
};

QMat.prototype.applyLut = function (act, out, oOff, rowStart, rowCount) {
  var T = act.lut(this.bits, this.table), u8 = this.u8, nf = this.nf, nG = this.nG, s = act.s;
  var bpg = (this.G * this.bits) >> 3, p = this.base + rowStart * this.rowBytes, ni = rowStart * nG;
  for (var r = 0; r < rowCount; r++) {
    var acc = 0, t = 0;
    for (var g = 0; g < nG; g++) {
      var d = 0, end = p + bpg;
      for (; p < end; p += 4, t += 1024) {
        d += T[t + u8[p]] + T[t + 256 + u8[p + 1]] + T[t + 512 + u8[p + 2]] + T[t + 768 + u8[p + 3]];
      }
      acc += d * nf[ni++] * s[g];
    }
    out[oOff + r] = acc;
  }
};

// One fully dequantised row (for table lookups such as the engrams).
QMat.prototype.rowFloat = function (r, dst, dOff) {
  var G = this.G, nG = this.nG, tmp = this._tmpI || (this._tmpI = new Int8Array(this.inPad));
  var f = this._tmpF || (this._tmpF = new Float32Array(G)), inv = 1 / Math.sqrt(G);
  this.decodeRowInto(r, tmp, 0);
  for (var g = 0; g < nG; g++) {
    var n = this.nf[r * nG + g];
    for (var k = 0; k < G; k++) f[k] = tmp[g * G + k] * n;
    fwht(f, 0, G);
    for (k = 0; k < G && g * G + k < this.inp; k++) dst[dOff + g * G + k] = f[k] * inv;
  }
};

// The activation side of a matmul: each group of G inputs is Hadamard-rotated and
// quantised to int8 with its own scale. One instance holds the current activation
// (q, s) plus, built on demand, the lookup tables that turn a packed weight byte
// into its partial dot product with a single read.
function Quantizer(G, maxDim) {
  this.G = G; this.tmp = new Float32Array(G); this.inv = Math.fround(1 / Math.sqrt(G));
  var maxG = Math.ceil((maxDim || 4096) / G);
  this.q = new Int8Array(maxG * G); this.s = new Float32Array(maxG); this.nG = 0;
  this.luts = {}; this.lutOk = {};
  this.lo = new Int32Array(16); this.hi = new Int32Array(16);
}

Quantizer.prototype.set = function (x, xOff, dim) {
  var G = this.G, tmp = this.tmp, inv = this.inv, nG = Math.ceil(dim / G), q = this.q, s = this.s;
  for (var g = 0; g < nG; g++) {
    var k, base = g * G, n = Math.min(G, dim - base);
    for (k = 0; k < n; k++) tmp[k] = x[xOff + base + k];
    for (; k < G; k++) tmp[k] = 0;
    fwht(tmp, 0, G);
    var mx = 0;
    for (k = 0; k < G; k++) {
      var v = (tmp[k] = tmp[k] * inv);
      if (v < 0) v = -v;
      if (v > mx) mx = v;
    }
    var sc = mx > 0 ? Math.fround(mx * INV127) : 1, r = Math.fround(1 / sc);
    s[g] = sc;
    for (k = 0; k < G; k++) {
      var y = Math.fround(tmp[k] * r);
      y = y < 0 ? -Math.floor(0.5 - y) : Math.floor(y + 0.5);
      q[base + k] = y > 127 ? 127 : y < -127 ? -127 : y;
    }
  }
  this.nG = nG;
  this.lutOk[2] = this.lutOk[4] = false;
  return nG;
};

Quantizer.prototype.hasLut = function (bits) { return this.lutOk[bits] === true; };

// T[pos * 256 + byte] = sum over the weights packed in `byte` of centroid * activation.
Quantizer.prototype.lut = function (bits, table) {
  var n = this.nG * this.G, per = 8 / bits, positions = n / per;
  var T = this.luts[bits];
  if (!T || T.length < positions * 256) T = this.luts[bits] = new Int32Array(positions * 256);
  if (this.lutOk[bits]) return T;
  var q = this.q, j, b, o = 0;
  if (bits === 2) {
    var lo = this.lo, hi = this.hi;
    for (j = 0; j < n; j += 4) {
      var q0 = q[j], q1 = q[j + 1], q2 = q[j + 2], q3 = q[j + 3];
      for (b = 0; b < 16; b++) {
        lo[b] = table[b & 3] * q0 + table[b >> 2] * q1;
        hi[b] = table[b & 3] * q2 + table[b >> 2] * q3;
      }
      for (b = 0; b < 256; b++) T[o++] = lo[b & 15] + hi[b >> 4];
    }
  } else if (bits === 4) {
    for (j = 0; j < n; j += 2) {
      var a0 = q[j], a1 = q[j + 1];
      for (b = 0; b < 256; b++) T[o++] = table[b & 15] * a0 + table[b >> 4] * a1;
    }
  } else {
    throw new Error('no lookup path for CQ' + bits);
  }
  this.lutOk[bits] = true;
  return T;
};

// Symmetric int8 fake quantisation of x[off .. off+n) with one scale (KV cache numerics).
function fakeQuant8(x, off, n) {
  var mx = 0, i, v;
  for (i = 0; i < n; i++) { v = x[off + i]; if (v < 0) v = -v; if (v > mx) mx = v; }
  if (!(mx > 0)) return;
  var sc = Math.fround(mx * INV127), r = Math.fround(1 / sc);
  for (i = 0; i < n; i++) {
    v = Math.fround(x[off + i] * r);
    v = v < 0 ? -Math.floor(0.5 - v) : Math.floor(v + 0.5);
    x[off + i] = v * sc;
  }
}

// y = (1 + scale) * x / sqrt(mean(x^2) + 1e-6)
function rmsNorm(x, xOff, n, scale, out, oOff) {
  var ss = 0, i;
  for (i = 0; i < n; i++) ss += x[xOff + i] * x[xOff + i];
  var inv = 1 / Math.sqrt(ss / n + 1e-6);
  if (scale) for (i = 0; i < n; i++) out[oOff + i] = (1 + scale[i]) * x[xOff + i] * inv;
  else for (i = 0; i < n; i++) out[oOff + i] = x[xOff + i] * inv;
}

function silu(x) { return x / (1 + Math.exp(-x)); }

// --------------------------------------------------------- Hadamard MLP

function HadaMLP(cact, base, p1, p2) {
  // base: index of d1; order d1 d2 b2 d3 d4 w1a w1b w2a w2b w3a w3b cond_v cond_u
  var f = function (k) { return cact.fp16(base + k); };
  this.d1 = f(0); this.d2 = f(1); this.b2 = f(2); this.d3 = f(3); this.d4 = f(4);
  this.wa = [f(5), f(7), f(9)]; this.wb = [f(6), f(8), f(10)];
  this.condV = f(11); this.condU = f(12);
  this.n = this.d1.length;
  this.ba = Math.round(Math.sqrt(this.wa[0].length));
  this.bb = Math.round(Math.sqrt(this.wb[0].length));
  this.rank = this.condU.length / this.n;
  this.p1 = p1; this.p2 = p2;
  this.z = new Float32Array(this.n); this.t = new Float32Array(this.n); this.y = new Float32Array(this.n);
  this.sm = new Float64Array(this.rank);
}

// out[k,l] = sum_ij z[i,j] a[i,k] b[j,l]
HadaMLP.prototype.kron = function (z, a, b, out) {
  var A = this.ba, B = this.bb, t = this.t, i, j, k, l;
  for (i = 0; i < A; i++) {
    var zi = i * B;
    for (l = 0; l < B; l++) t[zi + l] = 0;
    for (j = 0; j < B; j++) {
      var v = z[zi + j], bj = j * B;
      if (v === 0) continue;
      for (l = 0; l < B; l++) t[zi + l] += v * b[bj + l];
    }
  }
  for (k = 0; k < A * B; k++) out[k] = 0;
  for (i = 0; i < A; i++) {
    for (k = 0; k < A; k++) {
      var w = a[i * A + k], ti = i * B, ok = k * B;
      for (l = 0; l < B; l++) out[ok + l] += w * t[ti + l];
    }
  }
};

HadaMLP.prototype.run = function (x, xOff, out, oOff) {
  var n = this.n, R = this.rank, z = this.z, y = this.y, sm = this.sm, i, r;
  var mx = -Infinity, sum = 0;
  for (r = 0; r < R; r++) {
    var a = 0;
    for (i = 0; i < n; i++) a += x[xOff + i] * this.condV[i * R + r];
    sm[r] = a; if (a > mx) mx = a;
  }
  for (r = 0; r < R; r++) { sm[r] = Math.exp(sm[r] - mx); sum += sm[r]; }
  for (r = 0; r < R; r++) sm[r] /= sum;
  for (i = 0; i < n; i++) z[i] = this.d1[i] * x[xOff + i];
  this.kron(z, this.wa[0], this.wb[0], y);
  var p1 = this.p1, p2 = this.p2;
  for (i = 0; i < n; i++) {
    var c = 1;
    for (r = 0; r < R; r++) c += sm[r] * this.condU[r * n + i];
    z[i] = silu(this.d2[i] * c * y[p1[i]] + this.b2[i]);
  }
  this.kron(z, this.wa[1], this.wb[1], y);
  for (i = 0; i < n; i++) z[i] = this.d3[i] * y[p2[i]];
  this.kron(z, this.wa[2], this.wb[2], y);
  for (i = 0; i < n; i++) out[oOff + i] = this.d4[i] * y[i];
};

// ------------------------------------------------- multi-lane residual (mHC)

function Mhc(cact, base, layers, lanes, width, quant) {
  // base: a_pre a_post a_res b_pre b_post b_res phi_pre phi_post phi_res
  this.aPre = cact.fp16(base); this.aPost = cact.fp16(base + 1); this.aRes = cact.fp16(base + 2);
  this.bPre = cact.fp16(base + 3); this.bPost = cact.fp16(base + 4); this.bRes = cact.fp16(base + 5);
  this.phiPre = new QMat(cact, base + 6); this.phiPost = new QMat(cact, base + 7);
  this.phiRes = new QMat(cact, base + 8);
  this.layers = layers; this.lanes = lanes; this.width = width; this.quant = quant;
  var nC = lanes * width;
  this.nx = new Float32Array(nC);
  this.tmpPre = new Float32Array(this.phiPre.out); this.tmpPost = new Float32Array(this.phiPost.out);
  this.tmpRes = new Float32Array(this.phiRes.out);
}

// Reads the lane stream [lanes*width], writes the block input u and the mixing
// state { hpost[lanes], hres[lanes*lanes] } for finish().
Mhc.prototype.begin = function (layer, stream, sOff, u, uOff, st) {
  var n = this.lanes, C = this.width, nC = n * C, i, j, c;
  rmsNorm(stream, sOff, nC, null, this.nx, 0);
  this.quant.set(this.nx, 0, nC);
  this.phiPre.apply(this.quant, this.tmpPre, layer * n, layer * n, n);
  this.phiPost.apply(this.quant, this.tmpPost, layer * n, layer * n, n);
  this.phiRes.apply(this.quant, this.tmpRes, layer * n * n, layer * n * n, n * n);
  var lane = layer % n;
  for (c = 0; c < C; c++) u[uOff + c] = 0;
  for (i = 0; i < n; i++) {
    var hpre = sigmoid(this.aPre[layer] * this.tmpPre[layer * n + i] + this.bPre[layer * n + i] +
      (i === lane ? 4 : -4));
    var so = sOff + i * C;
    for (c = 0; c < C; c++) u[uOff + c] += hpre * stream[so + c];
    st.hpost[i] = 2 * sigmoid(this.aPost[layer] * this.tmpPost[layer * n + i] +
      this.bPost[layer * n + i] + (i === lane ? 0 : -4));
  }
  // Sinkhorn in the log domain, 20 rounds of row then column normalisation.
  var K = st.logK, nn = n * n, it, m, sum;
  for (i = 0; i < nn; i++) {
    K[i] = this.aRes[layer] * this.tmpRes[layer * nn + i] + this.bRes[layer * nn + i];
  }
  for (it = 0; it < 20; it++) {
    for (i = 0; i < n; i++) {
      m = -Infinity; for (j = 0; j < n; j++) if (K[i * n + j] > m) m = K[i * n + j];
      sum = 0; for (j = 0; j < n; j++) sum += Math.exp(K[i * n + j] - m);
      sum = m + Math.log(sum); for (j = 0; j < n; j++) K[i * n + j] -= sum;
    }
    for (j = 0; j < n; j++) {
      m = -Infinity; for (i = 0; i < n; i++) if (K[i * n + j] > m) m = K[i * n + j];
      sum = 0; for (i = 0; i < n; i++) sum += Math.exp(K[i * n + j] - m);
      sum = m + Math.log(sum); for (i = 0; i < n; i++) K[i * n + j] -= sum;
    }
  }
  for (i = 0; i < nn; i++) st.hres[i] = Math.exp(K[i]);
};

// stream <- hres @ stream + hpost * (x - u)
Mhc.prototype.finish = function (stream, sOff, x, xOff, u, uOff, st) {
  var n = this.lanes, C = this.width, col = st.col, i, j, c;
  for (c = 0; c < C; c++) {
    var y = x[xOff + c] - u[uOff + c];
    for (j = 0; j < n; j++) col[j] = stream[sOff + j * C + c];
    for (i = 0; i < n; i++) {
      var a = st.hpost[i] * y;
      for (j = 0; j < n; j++) a += st.hres[i * n + j] * col[j];
      stream[sOff + i * C + c] = a;
    }
  }
};

Mhc.prototype.state = function () {
  var n = this.lanes;
  return { hpost: new Float64Array(n), hres: new Float64Array(n * n), logK: new Float64Array(n * n),
    col: new Float64Array(n) };
};

// --------------------------------------------------------------- rotary

function Rope(dim, theta) {
  this.half = dim >> 1; this.freq = new Float64Array(this.half);
  for (var i = 0; i < this.half; i++) this.freq[i] = 1 / Math.pow(theta, (2 * i) / dim);
}
Rope.prototype.apply = function (x, off, pos) {
  var h = this.half;
  for (var i = 0; i < h; i++) {
    var ang = pos * this.freq[i], c = Math.cos(ang), s = Math.sin(ang);
    var a = x[off + i], b = x[off + h + i];
    x[off + i] = a * c - b * s;
    x[off + h + i] = b * c + a * s;
  }
};

// ------------------------------------------------------------------ stem

function Stem(cact, base, cfg, quant) {
  // base: w, then (dw, pw) per further stage, then out
  this.C = cfg.stemWidth; this.stages = cfg.stemStages; this.mels = cfg.melBins;
  this.w = cact.fp16(base);
  this.dw = []; this.pw = [];
  for (var s = 1; s < this.stages; s++) {
    this.dw.push(cact.fp16(base + 2 * s - 1));
    this.pw.push(new QMat(cact, base + 2 * s));
  }
  this.out = new QMat(cact, base + 2 * this.stages - 1);
  this.quant = quant;
}

// 3x3, stride 2, pad 1 over (time, freq). inC === 1 for the first stage, else depthwise.
function conv3s2(inp, T, F, inC, w, C) {
  var To = (T + 1) >> 1, Fo = (F + 1) >> 1, out = new Float32Array(To * Fo * C);
  for (var s = 0; s < To; s++) {
    for (var p = 0; p < Fo; p++) {
      var o = (s * Fo + p) * C;
      for (var m = 0; m < 3; m++) {
        var t = 2 * s + m - 1;
        if (t < 0 || t >= T) continue;
        for (var kf = 0; kf < 3; kf++) {
          var f = 2 * p + kf - 1;
          if (f < 0 || f >= F) continue;
          var wo = (m * 3 + kf) * C, c;
          if (inC === 1) {
            var v = inp[t * F + f];
            for (c = 0; c < C; c++) out[o + c] += w[wo + c] * v;
          } else {
            var io = (t * F + f) * C;
            for (c = 0; c < C; c++) out[o + c] += w[wo + c] * inp[io + c];
          }
        }
      }
    }
  }
  return { data: out, T: To, F: Fo };
}

// feat: [T, mels] -> { data: [T', width], T: T' }
Stem.prototype.run = function (feat, T) {
  var C = this.C, i, n;
  var x = conv3s2(feat, T, this.mels, 1, this.w, C);
  for (i = 0, n = x.data.length; i < n; i++) x.data[i] = silu(x.data[i]);
  for (var s = 0; s < this.dw.length; s++) {
    var y = conv3s2(x.data, x.T, x.F, C, this.dw[s], C), pw = this.pw[s];
    var rows = y.T * y.F, z = new Float32Array(rows * pw.out);
    for (i = 0; i < rows; i++) {
      this.quant.set(y.data, i * C, C);
      pw.apply(this.quant, z, i * pw.out);
    }
    for (i = 0, n = z.length; i < n; i++) z[i] = silu(z[i]);
    x = { data: z, T: y.T, F: y.F };
  }
  var F = x.F, flat = new Float32Array(C * F), W = this.out.out, out = new Float32Array(x.T * W);
  for (var t = 0; t < x.T; t++) {
    for (var f = 0; f < F; f++) {
      var io = (t * F + f) * C;
      for (var c = 0; c < C; c++) flat[c * F + f] = x.data[io + c];
    }
    this.quant.set(flat, 0, C * F);
    this.out.apply(this.quant, out, t * W);
  }
  return { data: out, T: x.T };
};

// ---------------------------------------------------------------- encoder

function softmaxInPlace(a, n) {
  var m = -Infinity, i, s = 0;
  for (i = 0; i < n; i++) if (a[i] > m) m = a[i];
  for (i = 0; i < n; i++) { a[i] = Math.exp(a[i] - m); s += a[i]; }
  for (i = 0; i < n; i++) a[i] /= s;
}

function EncoderLayer(cact, b, perms) {
  this.preHada0 = cact.fp16(b); this.normIn = cact.fp16(b + 1); this.postAttn = cact.fp16(b + 2);
  this.convNorm = cact.fp16(b + 3); this.convOutNorm = cact.fp16(b + 4); this.preHada = cact.fp16(b + 5);
  this.q = new QMat(cact, b + 6); this.k = new QMat(cact, b + 7); this.v = new QMat(cact, b + 8);
  this.gate = new QMat(cact, b + 9); this.outProj = new QMat(cact, b + 10);
  this.qNorm = cact.fp16(b + 11); this.kNorm = cact.fp16(b + 12);
  this.attnGate = sigmoid(cact.fp16(b + 13)[0]);
  this.pw1 = new QMat(cact, b + 14); this.pw2 = new QMat(cact, b + 15); this.dw = cact.fp16(b + 16);
  this.hada0 = new HadaMLP(cact, b + 17, perms[0], perms[1]);
  this.hada = new HadaMLP(cact, b + 30, perms[0], perms[1]);
}

function Encoder(cact, base, cfg, hdr, perms, quant) {
  this.L = cfg.encLayers; this.W = hdr.width; this.lanes = cfg.encLanes;
  this.heads = hdr.heads; this.kvHeads = hdr.kvHeads; this.qk = hdr.qkDim; this.vd = hdr.vDim;
  this.kernel = cfg.convKernel;
  this.layers = [];
  for (var l = 0; l < this.L; l++) this.layers.push(new EncoderLayer(cact, base + 43 * l, perms));
  this.mhc = new Mhc(cact, base + 43 * this.L, this.L, this.lanes, this.W, quant);
  this.finalNorm = cact.fp16(base + 43 * this.L + 9);
  this.rope = new Rope(this.qk, hdr.ropeTheta);
  this.quant = quant;
}
Encoder.TENSORS_PER_LAYER = 43;

// x: [T, W] stem output. Returns [T, W].
Encoder.prototype.run = function (x, T, onProgress) {
  var W = this.W, n = this.lanes, nC = n * W, H = this.heads, KH = this.kvHeads;
  var qk = this.qk, vd = this.vd, rep = H / KH, K = this.kernel, half = K >> 1;
  var stream = new Float32Array(T * nC), u = new Float32Array(T * W), h = new Float32Array(T * W);
  var tmp = new Float32Array(W), t, c, i, l, hd;
  for (t = 0; t < T; t++) for (i = 0; i < n; i++) stream.set(x.subarray(t * W, (t + 1) * W), t * nC + i * W);
  var states = [];
  for (t = 0; t < T; t++) states.push(this.mhc.state());
  var qd = H * qk, kd = KH * qk, vdim = KH * vd, od = H * vd;
  var Q = new Float32Array(T * qd), Kk = new Float32Array(T * kd), V = new Float32Array(T * vdim);
  var Gt = new Float32Array(od), att = new Float32Array(T * od), y = new Float32Array(T * W);
  var p1 = new Float32Array(T * 2 * W), glu = new Float32Array(T * W), cv = new Float32Array(T * W);
  var logit = new Float64Array(T), scale = 1 / Math.sqrt(qk);
  var quant = this.quant;

  for (l = 0; l < this.L; l++) {
    var ly = this.layers[l];
    for (t = 0; t < T; t++) this.mhc.begin(l, stream, t * nC, u, t * W, states[t]);
    h.set(u);
    // half-step Hadamard MLP
    for (t = 0; t < T; t++) {
      rmsNorm(h, t * W, W, ly.preHada0, tmp, 0);
      ly.hada0.run(tmp, 0, y, 0);
      for (c = 0; c < W; c++) h[t * W + c] += 0.5 * y[c];
    }
    // self attention (not causal)
    for (t = 0; t < T; t++) {
      rmsNorm(h, t * W, W, ly.normIn, tmp, 0);
      quant.set(tmp, 0, W);
      ly.q.apply(quant, Q, t * qd); ly.k.apply(quant, Kk, t * kd); ly.v.apply(quant, V, t * vdim);
      ly.gate.apply(quant, att, t * od);                 // gate logits parked in att
      for (hd = 0; hd < H; hd++) {
        rmsNorm(Q, t * qd + hd * qk, qk, ly.qNorm, Q, t * qd + hd * qk);
        this.rope.apply(Q, t * qd + hd * qk, t);
        fakeQuant8(Q, t * qd + hd * qk, qk);
      }
      for (hd = 0; hd < KH; hd++) {
        rmsNorm(Kk, t * kd + hd * qk, qk, ly.kNorm, Kk, t * kd + hd * qk);
        this.rope.apply(Kk, t * kd + hd * qk, t);
        fakeQuant8(Kk, t * kd + hd * qk, qk);
        fakeQuant8(V, t * vdim + hd * vd, vd);
      }
    }
    for (t = 0; t < T; t++) {
      for (c = 0; c < od; c++) Gt[c] = att[t * od + c];
      for (hd = 0; hd < H; hd++) {
        var kh = (hd / rep) | 0, qo = t * qd + hd * qk, s2, d;
        for (s2 = 0; s2 < T; s2++) {
          var ko = s2 * kd + kh * qk, dot = 0;
          for (d = 0; d < qk; d++) dot += Q[qo + d] * Kk[ko + d];
          logit[s2] = dot * scale;
        }
        softmaxInPlace(logit, T);
        var oo = t * od + hd * vd;
        for (d = 0; d < vd; d++) att[oo + d] = 0;
        for (s2 = 0; s2 < T; s2++) {
          var wgt = logit[s2], vo = s2 * vdim + kh * vd;
          for (d = 0; d < vd; d++) att[oo + d] += wgt * V[vo + d];
        }
      }
      for (c = 0; c < od; c++) att[t * od + c] *= sigmoid(Gt[c]);
    }
    for (t = 0; t < T; t++) {
      quant.set(att, t * od, od);
      ly.outProj.apply(quant, y, 0);
      rmsNorm(y, 0, W, ly.postAttn, y, 0);
      for (c = 0; c < W; c++) h[t * W + c] += ly.attnGate * y[c];
    }
    // convolution module: norm, pointwise x2, GLU, depthwise over time, norm, SiLU, pointwise
    for (t = 0; t < T; t++) {
      rmsNorm(h, t * W, W, ly.convNorm, tmp, 0);
      quant.set(tmp, 0, W);
      ly.pw1.apply(quant, p1, t * 2 * W);
      for (c = 0; c < W; c++) glu[t * W + c] = p1[t * 2 * W + c] * sigmoid(p1[t * 2 * W + W + c]);
    }
    for (t = 0; t < T; t++) {
      var co = t * W;
      for (c = 0; c < W; c++) cv[co + c] = 0;
      for (var kx = 0; kx < K; kx++) {
        var src = t - half + kx;
        if (src < 0 || src >= T) continue;
        var wo = kx * W, so = src * W;
        for (c = 0; c < W; c++) cv[co + c] += ly.dw[wo + c] * glu[so + c];
      }
      rmsNorm(cv, co, W, ly.convOutNorm, cv, co);
      for (c = 0; c < W; c++) cv[co + c] = silu(cv[co + c]);
    }
    for (t = 0; t < T; t++) {
      quant.set(cv, t * W, W);
      ly.pw2.apply(quant, y, 0);
      for (c = 0; c < W; c++) h[t * W + c] += y[c];
      // second half-step Hadamard MLP
      rmsNorm(h, t * W, W, ly.preHada, tmp, 0);
      ly.hada.run(tmp, 0, y, 0);
      for (c = 0; c < W; c++) h[t * W + c] += 0.5 * y[c];
      this.mhc.finish(stream, t * nC, h, t * W, u, t * W, states[t]);
    }
    if (onProgress) onProgress((l + 1) / this.L);
  }
  var out = new Float32Array(T * W);
  for (t = 0; t < T; t++) {
    for (c = 0; c < W; c++) {
      var a = 0;
      for (i = 0; i < n; i++) a += stream[t * nC + i * W + c];
      tmp[c] = a / n;
    }
    rmsNorm(tmp, 0, W, this.finalNorm, out, t * W);
  }
  return out;
};

// ---------------------------------------------------------------- decoder

function Engram(cact, base, hdr, quant) {
  this.tables = new QMat(cact, base); this.keyProj = new QMat(cact, base + 1);
  this.valueProj = new QMat(cact, base + 2); this.taps = cact.fp16(base + 3);
  this.slots = hdr.engramSlots; this.sub = hdr.engramSubDim; this.nTables = hdr.engramTables;
  this.orders = hdr.engramOrders; this.heads = this.nTables / this.orders.length;
  this.stride = hdr.engramSeedHeads || this.heads;
  this.nTaps = hdr.engramTaps; this.dilation = hdr.engramDilation; this.W = hdr.width;
  this.quant = quant;
  this.e = new Float32Array(this.nTables * this.sub);
}

// Key and (int8 cached) value projection of the n-gram rows at position pos.
Engram.prototype.project = function (tokens, pos) {
  var e = this.e, k, j, W = this.W;
  for (k = 0; k < this.nTables; k++) {
    var oi = (k / this.heads) | 0, h = k - oi * this.heads, order = this.orders[oi];
    var off = k * this.sub;
    if (pos < order - 1) { for (j = 0; j < this.sub; j++) e[off + j] = 0; continue; }
    var acc = Math.imul(0x9E3779B9 | 0, oi * this.stride + h + 1);
    for (j = 0; j < order; j++) acc = Math.imul(acc ^ (pos - j >= 0 ? tokens[pos - j] : 0), 16777619);
    acc = (acc ^ (acc >>> 15)) >>> 0;
    this.tables.rowFloat(k * this.slots + (acc % this.slots), e, off);
  }
  this.quant.set(e, 0, e.length);
  var key = new Float32Array(W), val = new Float32Array(W);
  this.keyProj.apply(this.quant, key, 0);
  this.valueProj.apply(this.quant, val, 0);
  for (j = 0; j + 32 <= W; j += 32) fakeQuant8(val, j, 32);
  return { key: key, val: val };
};

// vals: per-position cached values up to pos. Returns the tapped value at pos.
Engram.prototype.tapped = function (vals, pos) {
  var W = this.W, out = new Float32Array(W);
  for (var p = 0; p < this.nTaps; p++) {
    var src = pos - p * this.dilation;
    if (src < 0) continue;
    var v = vals[src], to = p * W;
    for (var c = 0; c < W; c++) out[c] += v[c] * this.taps[to + c];
  }
  return out;
};

function DecoderLayer(cact, b, c, perms, hasTaps) {
  this.normIn = cact.fp16(b);
  this.q = new QMat(cact, b + 1); this.k = new QMat(cact, b + 2); this.v = new QMat(cact, b + 3);
  var o = 4;
  if (hasTaps) { this.qT = cact.fp16(b + 4); this.kT = cact.fp16(b + 5); this.vT = cact.fp16(b + 6); o = 7; }
  this.qNorm = cact.fp16(b + o); this.kNorm = cact.fp16(b + o + 1);
  this.gate = new QMat(cact, b + o + 2); this.outProj = new QMat(cact, b + o + 3);
  this.postAttn = cact.fp16(b + o + 4); this.attnGate = sigmoid(cact.fp16(b + o + 5)[0]);
  this.preHada = cact.fp16(b + o + 6);
  this.hada = new HadaMLP(cact, b + o + 7, perms[0], perms[1]);
  this.crossNorm = cact.fp16(c);
  this.cq = new QMat(cact, c + 1); this.ck = new QMat(cact, c + 2); this.cv = new QMat(cact, c + 3);
  this.cGate = new QMat(cact, c + 4); this.cOut = new QMat(cact, c + 5);
  this.cqNorm = cact.fp16(c + 6); this.ckNorm = cact.fp16(c + 7);
  this.postCross = cact.fp16(c + 8); this.crossGate = sigmoid(cact.fp16(c + 9)[0]);
}

function Decoder(cact, idx, hdr, perms, quant) {
  var L = this.L = hdr.layers, W = this.W = hdr.width, l;
  this.H = hdr.heads; this.KH = hdr.kvHeads; this.qk = hdr.qkDim; this.vd = hdr.vDim;
  this.lanes = hdr.lanes; this.taps = hdr.taps; this.vocab = hdr.outVocab || hdr.vocab;
  this.embedding = new QMat(cact, idx.embedding);
  this.layers = [];
  for (l = 0; l < L; l++) {
    this.layers.push(new DecoderLayer(cact, idx.decLayer0 + idx.decPerLayer * l, idx.cross0 + 10 * l, perms, hdr.taps > 0));
  }
  this.mhc = new Mhc(cact, idx.decMhc, L, this.lanes, W, quant);
  this.sites = hdr.engramSites; this.engrams = [];
  for (l = 0; l < this.sites.length; l++) this.engrams.push(new Engram(cact, idx.engram0 + 4 * l, hdr, quant));
  this.finalNorm = cact.fp16(idx.finalNorm);
  this.peGate = sigmoid(cact.fp16(idx.peGate)[0]);
  this.rope = new Rope(this.qk, hdr.ropeTheta);
  this.quant = quant; this.embScale = Math.sqrt(W);
  // scratch
  var od = this.H * this.vd;
  this.stream = new Float32Array(this.lanes * W); this.u = new Float32Array(W); this.x = new Float32Array(W);
  this.tmp = new Float32Array(W); this.y = new Float32Array(W);
  this.qv = new Float32Array(this.H * this.qk); this.gt = new Float32Array(od); this.att = new Float32Array(od);
  this.mst = this.mhc.state(); this.emb = new Float32Array(W);
  this.logit = new Float64Array(4096);
}

// Sinusoidal positions gated into the encoder output, then per-layer cross K / V.
Decoder.prototype.prepareCross = function (enc, T) {
  var W = this.W, half = W >> 1, H = this.H, qk = this.qk, vd = this.vd, t, h, l;
  var mem = new Float32Array(T * W), inc = -9.210340371976184 / (half - 1), g = this.peGate;
  for (t = 0; t < T; t++) {
    for (h = 0; h < half; h++) {
      var a = Math.exp(h * inc) * t;
      mem[t * W + h] = g * Math.sin(a) + enc[t * W + h];
      mem[t * W + half + h] = g * Math.cos(a) + enc[t * W + half + h];
    }
  }
  var cross = [], qd = H * qk, od = H * vd;
  for (l = 0; l < this.L; l++) cross.push({ K: new Float32Array(T * qd), V: new Float32Array(T * od) });
  for (t = 0; t < T; t++) {
    this.quant.set(mem, t * W, W);
    for (l = 0; l < this.L; l++) {
      var ly = this.layers[l], K = cross[l].K, V = cross[l].V;
      ly.ck.apply(this.quant, K, t * qd);
      ly.cv.apply(this.quant, V, t * od);
      for (h = 0; h < H; h++) {
        rmsNorm(K, t * qd + h * qk, qk, ly.ckNorm, K, t * qd + h * qk);
        fakeQuant8(K, t * qd + h * qk, qk);
        fakeQuant8(V, t * od + h * vd, vd);
      }
    }
  }
  return { T: T, layers: cross };
};

// A hypothesis: its tokens plus everything the decoder caches per position.
Decoder.prototype.newState = function () {
  var st = { tokens: [], K: [], V: [], raw: [], evals: [] }, l;
  for (l = 0; l < this.L; l++) { st.K.push([]); st.V.push([]); st.raw.push([]); }
  for (l = 0; l < this.engrams.length; l++) st.evals.push([]);
  return st;
};

Decoder.prototype.cloneState = function (s) {
  var st = { tokens: s.tokens.slice(), K: [], V: [], raw: [], evals: [] }, l;
  for (l = 0; l < this.L; l++) { st.K.push(s.K[l].slice()); st.V.push(s.V[l].slice()); st.raw.push(s.raw[l].slice()); }
  for (l = 0; l < s.evals.length; l++) st.evals.push(s.evals[l].slice());
  return st;
};

// Feeds one token and returns the logits for the next one (a fresh Float32Array).
// opts.crossAttn, when given, receives the per-layer cross-attention rows averaged over heads.
Decoder.prototype.step = function (st, token, cross, opts) {
  var W = this.W, n = this.lanes, H = this.H, KH = this.KH, qk = this.qk, vd = this.vd;
  var rep = H / KH, pos = st.tokens.length, quant = this.quant;
  var stream = this.stream, u = this.u, x = this.x, tmp = this.tmp, y = this.y;
  var qd = H * qk, kd = KH * qk, vdim = KH * vd, od = H * vd, T = cross.T;
  var c, i, l, hd, d, s2, scale = 1 / Math.sqrt(qk), logit = this.logit;
  if (logit.length < Math.max(T, pos + 1)) logit = this.logit = new Float64Array(Math.max(T, pos + 1) * 2);
  st.tokens.push(token);
  this.embedding.rowFloat(token, this.emb, 0);
  for (i = 0; i < n; i++) for (c = 0; c < W; c++) stream[i * W + c] = this.emb[c] * this.embScale;
  var eg = [];
  for (i = 0; i < this.engrams.length; i++) {
    var pr = this.engrams[i].project(st.tokens, pos);
    st.evals[i].push(pr.val);
    eg.push({ key: pr.key, val: this.engrams[i].tapped(st.evals[i], pos) });
  }
  for (l = 0; l < this.L; l++) {
    var ly = this.layers[l];
    this.mhc.begin(l, stream, 0, u, 0, this.mst);
    for (c = 0; c < W; c++) x[c] = u[c];
    var site = this.sites.indexOf(l);
    if (site >= 0) {
      var su = 0, sk = 0, dot = 0, key = eg[site].key, val = eg[site].val;
      for (c = 0; c < W; c++) { su += u[c] * u[c]; sk += key[c] * key[c]; dot += u[c] * key[c]; }
      dot /= Math.sqrt(su / W + 1e-6) * Math.sqrt(sk / W + 1e-6);
      var alpha = sigmoid(dot / Math.sqrt(W));
      for (c = 0; c < W; c++) x[c] += alpha * val[c];
    }
    // causal self attention with conv taps on Q, K, V
    rmsNorm(x, 0, W, ly.normIn, tmp, 0);
    quant.set(tmp, 0, W);
    var raw = new Float32Array(qd + kd + vdim);
    ly.q.apply(quant, raw, 0); ly.k.apply(quant, raw, qd); ly.v.apply(quant, raw, qd + kd);
    ly.gate.apply(quant, this.gt, 0);
    st.raw[l].push(raw);
    var qv = this.qv, Kt = new Float32Array(kd), Vt = new Float32Array(vdim);
    if (this.taps) {
      for (c = 0; c < qd; c++) qv[c] = 0;
      for (var j = 0; j < this.taps && j <= pos; j++) {
        var r = st.raw[l][pos - j];
        for (c = 0; c < qd; c++) qv[c] += ly.qT[j * qd + c] * r[c];
        for (c = 0; c < kd; c++) Kt[c] += ly.kT[j * kd + c] * r[qd + c];
        for (c = 0; c < vdim; c++) Vt[c] += ly.vT[j * vdim + c] * r[qd + kd + c];
      }
    } else {
      for (c = 0; c < qd; c++) qv[c] = raw[c];
      for (c = 0; c < kd; c++) Kt[c] = raw[qd + c];
      for (c = 0; c < vdim; c++) Vt[c] = raw[qd + kd + c];
    }
    for (hd = 0; hd < H; hd++) {
      rmsNorm(qv, hd * qk, qk, ly.qNorm, qv, hd * qk);
      this.rope.apply(qv, hd * qk, pos);
      fakeQuant8(qv, hd * qk, qk);
    }
    for (hd = 0; hd < KH; hd++) {
      rmsNorm(Kt, hd * qk, qk, ly.kNorm, Kt, hd * qk);
      this.rope.apply(Kt, hd * qk, pos);
      fakeQuant8(Kt, hd * qk, qk);
      fakeQuant8(Vt, hd * vd, vd);
    }
    st.K[l].push(Kt); st.V[l].push(Vt);
    var Ks = st.K[l], Vs = st.V[l], att = this.att, len = pos + 1;
    for (hd = 0; hd < H; hd++) {
      var kh = (hd / rep) | 0, qo = hd * qk, ko = kh * qk, vo = kh * vd, oo = hd * vd, m = -Infinity, sum = 0;
      for (s2 = 0; s2 < len; s2++) {
        var kr = Ks[s2], dt = 0;
        for (d = 0; d < qk; d++) dt += qv[qo + d] * kr[ko + d];
        logit[s2] = dt * scale; if (logit[s2] > m) m = logit[s2];
      }
      for (s2 = 0; s2 < len; s2++) { logit[s2] = Math.exp(logit[s2] - m); sum += logit[s2]; }
      for (d = 0; d < vd; d++) att[oo + d] = 0;
      for (s2 = 0; s2 < len; s2++) {
        var wgt = logit[s2] / sum, vr = Vs[s2];
        for (d = 0; d < vd; d++) att[oo + d] += wgt * vr[vo + d];
      }
    }
    for (c = 0; c < od; c++) att[c] *= sigmoid(this.gt[c]);
    quant.set(att, 0, od);
    ly.outProj.apply(quant, y, 0);
    rmsNorm(y, 0, W, ly.postAttn, y, 0);
    for (c = 0; c < W; c++) x[c] += ly.attnGate * y[c];
    // gated cross attention over the clip
    rmsNorm(x, 0, W, ly.crossNorm, tmp, 0);
    quant.set(tmp, 0, W);
    ly.cq.apply(quant, qv, 0);
    ly.cGate.apply(quant, this.gt, 0);
    var CK = cross.layers[l].K, CV = cross.layers[l].V, wantAttn = opts && opts.crossAttn, acc = null;
    if (wantAttn) acc = new Float32Array(T);
    for (hd = 0; hd < H; hd++) {
      rmsNorm(qv, hd * qk, qk, ly.cqNorm, qv, hd * qk);
      fakeQuant8(qv, hd * qk, qk);
      qo = hd * qk; oo = hd * vd; m = -Infinity; sum = 0;
      for (s2 = 0; s2 < T; s2++) {
        var kb = s2 * qd + qo; dt = 0;
        for (d = 0; d < qk; d++) dt += qv[qo + d] * CK[kb + d];
        logit[s2] = dt * scale; if (logit[s2] > m) m = logit[s2];
      }
      for (s2 = 0; s2 < T; s2++) { logit[s2] = Math.exp(logit[s2] - m); sum += logit[s2]; }
      for (d = 0; d < vd; d++) att[oo + d] = 0;
      for (s2 = 0; s2 < T; s2++) {
        wgt = logit[s2] / sum;
        if (acc) acc[s2] += wgt / H;
        var vb = s2 * od + oo;
        for (d = 0; d < vd; d++) att[oo + d] += wgt * CV[vb + d];
      }
    }
    if (acc) opts.crossAttn.push(acc);
    for (c = 0; c < od; c++) att[c] *= sigmoid(this.gt[c]);
    quant.set(att, 0, od);
    ly.cOut.apply(quant, y, 0);
    rmsNorm(y, 0, W, ly.postCross, y, 0);
    for (c = 0; c < W; c++) x[c] += ly.crossGate * y[c];
    // Hadamard MLP
    rmsNorm(x, 0, W, ly.preHada, tmp, 0);
    ly.hada.run(tmp, 0, y, 0);
    for (c = 0; c < W; c++) x[c] += y[c];
    this.mhc.finish(stream, 0, x, 0, u, 0, this.mst);
  }
  for (c = 0; c < W; c++) {
    var a = 0;
    for (i = 0; i < n; i++) a += stream[i * W + c];
    tmp[c] = a / n;
  }
  rmsNorm(tmp, 0, W, this.finalNorm, tmp, 0);
  quant.set(tmp, 0, W);
  var out = new Float32Array(this.embedding.out);
  this.embedding.apply(quant, out, 0);
  return out;
};

// -------------------------------------------------------------- tokenizer

var TK_UNKNOWN = 1, TK_CONTROL = 2, TK_BYTE = 4;

// The SentencePiece BPE dump attached to the archive (decode side only).
function Tokenizer(blob) {
  var dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  var n = dv.getUint32(0, true), off = 24, dec = typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8') : null;
  this.padId = dv.getUint32(4, true); this.eosId = dv.getUint32(8, true);
  this.bosId = dv.getUint32(12, true); this.unkId = dv.getUint32(16, true);
  this.addDummyPrefix = blob[20] !== 0;
  this.bytes = []; this.types = new Uint8Array(n); this.size = n;
  for (var i = 0; i < n; i++) {
    var type = blob[off + 4], len = dv.getUint16(off + 5, true);
    off += 7;
    var piece = blob.subarray(off, off + len);
    off += len;
    this.types[i] = type;
    if (type === TK_BYTE) {
      var s = '';
      for (var k = 0; k < len; k++) s += String.fromCharCode(piece[k]);
      this.bytes.push(new Uint8Array([parseInt(s.substr(3, 2), 16)]));
    } else {
      this.bytes.push(piece);
    }
  }
  this.dec = dec;
}

function utf8Decode(u8, dec) {
  if (dec) return dec.decode(u8);
  var s = '';
  for (var i = 0; i < u8.length; i++) s += '%' + (u8[i] < 16 ? '0' : '') + u8[i].toString(16);
  try { return decodeURIComponent(s); } catch (e) { return ''; }
}

Tokenizer.prototype.decode = function (ids) {
  var parts = [], total = 0, i;
  for (i = 0; i < ids.length; i++) {
    var id = ids[i];
    if (id < 0 || id >= this.size) continue;
    var t = this.types[id];
    if (t === TK_CONTROL || t === TK_UNKNOWN) continue;
    parts.push(this.bytes[id]); total += this.bytes[id].length;
  }
  var buf = new Uint8Array(total), o = 0;
  for (i = 0; i < parts.length; i++) { buf.set(parts[i], o); o += parts[i].length; }
  var text = utf8Decode(buf, this.dec).replace(/▁/g, ' ');
  if (this.addDummyPrefix && text.charAt(0) === ' ') text = text.substring(1);
  return text;
};

// ----------------------------------------------------------- beam search

var LANGUAGES = ['en', 'de', 'fr', 'es', 'it', 'nl', 'pl'];
var MAX_AUDIO_SAMPLES = 480000;   // one pass covers 30 s
var SILENCE_RANGE_DB = 5;

// If the tail of `t` repeats with a period of 1..8 tokens four times over, returns
// the length that keeps a single copy of the loop; otherwise -1.
function repetitionCut(t) {
  var n = t.length;
  for (var p = 1; p <= 8; p++) {
    if (n < 4 * p) return -1;
    var ok = true;
    for (var c = n - 4 * p; c < n - p; c++) if (t[c] !== t[c + p]) { ok = false; break; }
    if (ok) {
      var keep = n - 3 * p;
      return keep > 0 ? keep : -1;
    }
  }
  return -1;
}

// ------------------------------------------------------------------ model

function Model(arrayBuffer) {
  var cact = this.cact = new Cact(arrayBuffer), hdr = this.hdr = cact.header, i;
  var L = hdr.layers, perLayer = 20 + (hdr.taps ? 7 : 4);
  var p = 1 + perLayer * L;
  this.idx = { embedding: 0, decLayer0: 1, decPerLayer: perLayer, decMhc: p };
  p += 9;
  this.idx.perms = p; p += 2;
  this.idx.engram0 = p; p += 4 * hdr.engramSites.length;
  this.idx.finalNorm = p++;
  this.idx.manifest = p++;
  var mt = cact.tensors[this.idx.manifest];
  if (!mt || mt.dtype !== FP32 || mt.shape.length !== 1) {
    throw new Error('this .cact holds no Whistle audio manifest (is it a Needle text model?)');
  }
  var m = cact.fp32(this.idx.manifest);
  this.cfg = {
    textVocab: m[2], encLayers: m[3], encLanes: m[4], convKernel: m[5], stemWidth: m[6],
    stemStages: m[7], melBins: m[8], sampleRate: m[9], nFft: m[10], windowSamples: m[11],
    hopSamples: m[12]
  };
  this.idx.cross0 = p; p += 10 * L;
  this.idx.peGate = p++;
  this.idx.enc0 = p; p += Encoder.TENSORS_PER_LAYER * this.cfg.encLayers;
  this.idx.encMhc = p; p += 9;
  this.idx.encFinalNorm = p++;
  this.idx.stem = p; p += 2 * this.cfg.stemStages;
  this.idx.mel = p++;
  this.idx.tokenizer = p++;
  if (p !== cact.numTensors || cact.tensors[this.idx.tokenizer].dtype !== RAW) {
    throw new Error('unexpected tensor layout: ' + cact.numTensors + ' tensors, expected ' + p);
  }
  var p1 = cact.fp32(this.idx.perms), p2 = cact.fp32(this.idx.perms + 1);
  this.perms = [new Int32Array(p1.length), new Int32Array(p2.length)];
  for (i = 0; i < p1.length; i++) { this.perms[0][i] = p1[i]; this.perms[1][i] = p2[i]; }
  this.quant = new Quantizer(128, 4096);
  this.frontEnd = new FrontEnd(this.cfg, cact.fp32(this.idx.mel));
  this.stem = new Stem(cact, this.idx.stem, this.cfg, this.quant);
  this.encoder = new Encoder(cact, this.idx.enc0, this.cfg, hdr, this.perms, this.quant);
  this.decoder = new Decoder(cact, this.idx, hdr, this.perms, this.quant);
  this.tokenizer = new Tokenizer(cact.raw(this.idx.tokenizer));
  this.languages = LANGUAGES.slice(0, hdr.vocab - this.cfg.textVocab);
}

// The loudness range (dB) between the 99th and 10th percentile frame; below
// SILENCE_RANGE_DB the clip is silence or steady noise.
Model.prototype.loudnessRange = function (db) {
  if (db.length < 2) return 0;
  var s = Float32Array.from(db).sort(), f = Math.fround(db.length - 1);
  return s[Math.floor(Math.fround(f * Math.fround(0.99)))] - s[Math.floor(Math.fround(f * Math.fround(0.1)))];
};

// Beam search over one encoded clip. Returns { tokens, language, score }.
Model.prototype.decode = function (enc, frames, opts) {
  opts = opts || {};
  var dec = this.decoder, tk = this.tokenizer, textVocab = this.cfg.textVocab;
  var beamWidth = opts.beams || 5, perBeam = beamWidth + 1, maxLen = this.hdr.maxSeq - 1;
  var cross = dec.prepareCross(enc, frames), i, b;
  var st0 = dec.newState(), lg = dec.step(st0, tk.bosId, cross);
  var lang = -1;
  if (opts.language) {
    lang = this.languages.indexOf(opts.language);
    if (lang < 0) throw new Error('unknown language "' + opts.language + '"');
  } else {
    for (i = textVocab; i < lg.length; i++) if (lang < 0 || lg[i] > lg[textVocab + lang]) lang = i - textVocab;
  }
  var beams = [{ state: st0, tokens: [], score: 0 }];
  var best = null, bestScore = -1e30, finished = 0, steps = 0;
  while (beams.length && finished < beamWidth) {
    var cands = [];
    for (b = 0; b < beams.length; b++) {
      var bm = beams[b], input = bm.tokens.length ? bm.tokens[bm.tokens.length - 1] : textVocab + lang;
      lg = dec.step(bm.state, input, cross);
      var mx = -Infinity, sum = 0;
      for (i = 0; i < textVocab; i++) if (lg[i] > mx) mx = lg[i];
      for (i = 0; i < textVocab; i++) sum += Math.exp(lg[i] - mx);
      var base = bm.score - (Math.log(sum) + mx);
      // the perBeam best continuations of this beam
      var top = [];
      for (i = 0; i < textVocab; i++) {
        var v = lg[i];
        if (top.length === perBeam && v <= top[perBeam - 1].v) continue;
        var k = top.length < perBeam ? top.length : perBeam - 1;
        while (k > 0 && top[k - 1].v < v) { top[k] = top[k - 1]; k--; }
        top[k] = { v: v, token: i };
        if (top.length > perBeam) top.length = perBeam;
      }
      for (i = 0; i < top.length; i++) cands.push({ score: base + top[i].v, beam: b, token: top[i].token });
    }
    cands.sort(function (x, y) { return y.score - x.score; });
    var next = [], used = [];
    for (i = 0; i < cands.length && next.length < beamWidth; i++) {
      var c = cands[i], toks = beams[c.beam].tokens.slice(), cut = -1, done = c.token === tk.eosId;
      if (!done) {
        toks.push(c.token);
        cut = repetitionCut(toks);
        done = cut >= 0 || toks.length >= maxLen;
      }
      if (!done) {
        next.push({ parent: c.beam, tokens: toks, score: c.score });
        continue;
      }
      if (finished === beamWidth) continue;
      var norm = c.score / (toks.length || 1);
      if (!(norm <= bestScore)) {
        best = cut >= 0 ? toks.slice(0, cut) : toks;
        bestScore = norm;
      }
      finished++;
    }
    // every child but one works on its own copy of the parent's caches
    for (i = 0; i < next.length; i++) {
      var p = next[i].parent;
      next[i].state = used[p] ? dec.cloneState(beams[p].state) : beams[p].state;
      used[p] = true;
    }
    beams = next;
    steps++;
    if (opts.onStep) opts.onStep(steps, best ? tk.decode(best) : (beams.length ? tk.decode(beams[0].tokens) : ''));
  }
  return { tokens: best || [], language: this.languages[lang] || '', score: bestScore, steps: steps };
};

// Speech to text for 16 kHz mono float PCM in [-1, 1].
// opts: { language: 'en'|'de'|'fr'|'es'|'it'|'nl'|'pl' (default: detect), beams: 5,
//         onProgress(stage, fraction), onStep(step, partialText) }
Model.prototype.transcribe = function (pcm, opts) {
  opts = opts || {};
  var now = typeof performance !== 'undefined' ? function () { return performance.now(); } : Date.now;
  if (pcm.length > MAX_AUDIO_SAMPLES) return this.transcribeLong(pcm, opts);
  var t0 = now(), prog = opts.onProgress || function () {};
  prog('frontend', 0);
  var fe = this.frontEnd.run(pcm), range = this.loudnessRange(fe.db);
  if (range < SILENCE_RANGE_DB) {
    return { text: '', language: '', tokens: [], silent: true, frames: 0, timings: { total: now() - t0 } };
  }
  prog('stem', 0);
  var st = this.stem.run(fe.feat, fe.T), t1 = now();
  var enc = this.encoder.run(st.data, st.T, function (f) { prog('encoder', f); }), t2 = now();
  prog('decoder', 0);
  var r = this.decode(enc, st.T, opts), t3 = now();
  return {
    text: this.tokenizer.decode(r.tokens), language: r.language, tokens: r.tokens, silent: false,
    frames: st.T, score: r.score,
    timings: { frontend: t1 - t0, encoder: t2 - t1, decoder: t3 - t2, total: t3 - t0, steps: r.steps }
  };
};

// Longer audio: consecutive windows of up to 30 s, each transcribed on its own.
// A window that is not the last one ends at the quietest 100 ms of its final five
// seconds, so the cut falls in a pause rather than inside a word.
Model.prototype.transcribeLong = function (pcm, opts) {
  var parts = [], language = opts.language || '', tokens = [], off = 0, k;
  var timings = { frontend: 0, encoder: 0, decoder: 0, total: 0, steps: 0 };
  var search = 5 * this.cfg.sampleRate, block = this.cfg.sampleRate / 10;
  while (off < pcm.length) {
    var len = Math.min(MAX_AUDIO_SAMPLES, pcm.length - off);
    if (off + len < pcm.length) {
      var bestE = Infinity, bestAt = len;
      for (var s = len - search; s + block <= len; s += block >> 1) {
        var e = 0;
        for (var i = 0; i < block; i++) { var x = pcm[off + s + i]; e += x * x; }
        if (e < bestE) { bestE = e; bestAt = s + (block >> 1); }
      }
      len = bestAt;
    }
    var o = {};
    for (k in opts) o[k] = opts[k];
    if (language) o.language = language;
    if (opts.onProgress) {
      o.onProgress = (function (base, span, total) {
        return function (stage, f) { opts.onProgress(stage, f, (base + span * (stage === 'decoder' ? 0.6 : 0.3 * f)) / total); };
      })(off, len, pcm.length);
    }
    var r = this.transcribe(pcm.subarray(off, off + len), o);
    if (r.text) parts.push(r.text);
    if (!language && r.language) language = r.language;
    for (k = 0; k < r.tokens.length; k++) tokens.push(r.tokens[k]);
    for (k in timings) timings[k] += r.timings[k] || 0;
    if (opts.onWindow) opts.onWindow(parts.join(' '), (off + len) / pcm.length);
    off += len;
  }
  return { text: parts.join(' '), language: language, tokens: tokens, silent: parts.length === 0,
    timings: timings };
};

// The encoder output: one row of `width` floats per 80 ms frame.
Model.prototype.embed = function (pcm, onProgress) {
  var fe = this.frontEnd.run(pcm);
  var st = this.stem.run(fe.feat, fe.T);
  var out = this.encoder.run(st.data, st.T, onProgress);
  return { data: out, frames: st.T, width: this.hdr.width, db: fe.db, stem: st.data, mel: fe.feat };
};

return {
  Model: Model, Cact: Cact, FrontEnd: FrontEnd, Tokenizer: Tokenizer, LANGUAGES: LANGUAGES,
  _internal: { fwht: fwht, f16Table: f16Table, sigmoid: sigmoid, QMat: QMat, Quantizer: Quantizer,
    HadaMLP: HadaMLP, Mhc: Mhc, Rope: Rope, Stem: Stem, Encoder: Encoder, rmsNorm: rmsNorm,
    fakeQuant8: fakeQuant8 }
};
});
