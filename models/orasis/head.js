// Trainable classifier on top of the encoder: shrinkage LDA, one linear score per class. Plain ES5, no dependencies.
// Training is a single linear solve, so it runs on the device from a few examples per class. Same maths as runtime/rust.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.OrasisHead = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // In-place lower Cholesky factor of the n*n matrix a (row-major, Float64Array).
  function cholesky(a, n) {
    var i, j, k, s, ri, rj;
    for (i = 0; i < n; i++) {
      ri = i * n;
      for (j = 0; j <= i; j++) {
        rj = j * n;
        s = a[ri + j];
        for (k = 0; k < j; k++) s -= a[ri + k] * a[rj + k];
        if (i === j) a[ri + i] = Math.sqrt(s > 1e-12 ? s : 1e-12);
        else a[ri + j] = s / a[rj + j];
      }
    }
  }

  // Solves L L' x = b in place.
  function cholSolve(l, n, b) {
    var i, k, s;
    for (i = 0; i < n; i++) {
      s = b[i];
      for (k = 0; k < i; k++) s -= l[i * n + k] * b[k];
      b[i] = s / l[i * n + i];
    }
    for (i = n - 1; i >= 0; i--) {
      s = b[i];
      for (k = i + 1; k < n; k++) s -= l[k * n + i] * b[k];
      b[i] = s / l[i * n + i];
    }
  }

  // x: n*dim unit vectors, row-major. y: class index 0..classes-1 per row. shrink in (0, 1].
  // Every class needs at least one example.
  function fitLDA(x, y, classes, dim, shrink) {
    var n = y.length, c, i, j, k, row;
    if (n === 0 || x.length !== n * dim) throw new Error('bad training data');
    if (!(shrink > 0 && shrink <= 1)) throw new Error('shrink must be in (0, 1]');
    var mu = new Float64Array(classes * dim), count = new Float64Array(classes);
    for (i = 0; i < n; i++) {
      c = y[i];
      if (!(c >= 0 && c < classes)) throw new Error('bad label');
      count[c]++;
      row = i * dim;
      for (j = 0; j < dim; j++) mu[c * dim + j] += x[row + j];
    }
    for (c = 0; c < classes; c++) {
      if (!count[c]) throw new Error('class ' + c + ' has no examples');
      for (j = 0; j < dim; j++) mu[c * dim + j] /= count[c];
    }
    var r = new Float64Array(n * dim), trace = 0, v;
    for (i = 0; i < n; i++) {
      for (j = 0; j < dim; j++) {
        v = x[i * dim + j] - mu[y[i] * dim + j];
        r[i * dim + j] = v;
        trace += v * v;
      }
    }
    trace /= n;
    var a = shrink * trace / dim, bb = (1 - shrink) / n;
    if (!(a > 0)) a = 1e-6; // all examples of each class identical
    var W = new Float32Array(classes * dim), B = new Float32Array(classes), sol = new Float64Array(dim), t;
    if (n < dim) {
      // Woodbury: only an n*n system when there are fewer examples than dimensions.
      var ratio = bb / a, g = new Float64Array(n * n), tmp = new Float64Array(n);
      for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
          t = 0;
          for (k = 0; k < dim; k++) t += r[i * dim + k] * r[j * dim + k];
          g[i * n + j] = ratio * t + (i === j ? 1 : 0);
        }
      }
      cholesky(g, n);
      for (c = 0; c < classes; c++) {
        for (i = 0; i < n; i++) {
          t = 0;
          for (k = 0; k < dim; k++) t += r[i * dim + k] * mu[c * dim + k];
          tmp[i] = t;
        }
        cholSolve(g, n, tmp);
        for (k = 0; k < dim; k++) {
          t = 0;
          for (i = 0; i < n; i++) t += r[i * dim + k] * tmp[i];
          sol[k] = (mu[c * dim + k] - ratio * t) / a;
        }
        store(c);
      }
    } else {
      var s = new Float64Array(dim * dim);
      for (i = 0; i < n; i++) {
        row = i * dim;
        for (j = 0; j < dim; j++) {
          t = r[row + j];
          if (t === 0) continue;
          for (k = 0; k <= j; k++) s[j * dim + k] += t * r[row + k];
        }
      }
      for (j = 0; j < dim; j++) {
        for (k = 0; k <= j; k++) s[j * dim + k] *= bb;
        s[j * dim + j] += a;
      }
      cholesky(s, dim);
      for (c = 0; c < classes; c++) {
        for (k = 0; k < dim; k++) sol[k] = mu[c * dim + k];
        cholSolve(s, dim, sol);
        store(c);
      }
    }
    function store(cls) {
      var q, d = 0;
      for (q = 0; q < dim; q++) {
        W[cls * dim + q] = sol[q];
        d += sol[q] * mu[cls * dim + q];
      }
      B[cls] = -0.5 * d;
    }
    return { classes: classes, dim: dim, W: W, b: B };
  }

  // Raw score per class for a unit vector z.
  function scores(head, z) {
    var out = new Float64Array(head.classes), c, j, s, base;
    if (z.length !== head.dim) throw new Error('bad vector size');
    for (c = 0; c < head.classes; c++) {
      base = c * head.dim;
      s = head.b[c];
      for (j = 0; j < head.dim; j++) s += head.W[base + j] * z[j];
      out[c] = s;
    }
    return out;
  }

  // Best k {index, score, p}, p being the softmax of the raw scores.
  function top(head, z, k) {
    var s = scores(head, z), max = -Infinity, sum = 0, res = [], i;
    for (i = 0; i < s.length; i++) if (s[i] > max) max = s[i];
    for (i = 0; i < s.length; i++) sum += Math.exp(s[i] - max);
    for (i = 0; i < s.length; i++) res.push({ index: i, score: s[i], p: Math.exp(s[i] - max) / sum });
    res.sort(function (a, b) { return b.score - a.score; });
    return res.slice(0, k);
  }

  function toF16(v) {
    var f = new Float32Array(1), u = new Uint32Array(f.buffer);
    f[0] = v;
    var i = u[0], sign = (i >>> 16) & 0x8000, e = ((i >>> 23) & 255) - 112, m = i & 0x7fffff, r;
    if (e <= 0) {
      if (e < -10) return sign;
      m |= 0x800000;
      r = m >> (14 - e);
      if ((m >> (13 - e)) & 1) r++;
      return sign | r;
    }
    if (e >= 31) return sign | 0x7bff;
    r = (e << 10) | (m >> 13);
    if ((m >> 12) & 1) r++;
    return sign | (r > 0x7bff ? 0x7bff : r);
  }

  // File layout: "ORH1", u16 classes, u16 dim, then classes*(dim+1) little-endian fp16 values: each row's weights, then its bias.
  function save(head) {
    var c = head.classes, d = head.dim, u8 = new Uint8Array(8 + c * (d + 1) * 2), i, j, p = 8, v;
    u8[0] = 79; u8[1] = 82; u8[2] = 72; u8[3] = 49;
    u8[4] = c & 255; u8[5] = c >> 8; u8[6] = d & 255; u8[7] = d >> 8;
    for (i = 0; i < c; i++) {
      for (j = 0; j <= d; j++) {
        v = toF16(j < d ? head.W[i * d + j] : head.b[i]);
        u8[p++] = v & 255;
        u8[p++] = v >> 8;
      }
    }
    return u8;
  }

  function f16(v) {
    var s = (v & 0x8000) ? -1 : 1, e = (v >> 10) & 31, m = v & 1023;
    if (e === 0) return s * m * 5.960464477539063e-8;
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * Math.pow(2, e - 15);
  }

  function load(bytes) {
    var u8 = new Uint8Array(bytes);
    if (u8.length < 8 || u8[0] !== 79 || u8[1] !== 82 || u8[2] !== 72 || u8[3] !== 49) throw new Error('bad head file');
    var c = u8[4] | (u8[5] << 8), d = u8[6] | (u8[7] << 8);
    if (u8.length !== 8 + c * (d + 1) * 2) throw new Error('bad head size');
    var W = new Float32Array(c * d), b = new Float32Array(c), p = 8, i, j, v;
    for (i = 0; i < c; i++) {
      for (j = 0; j <= d; j++) {
        v = f16(u8[p] | (u8[p + 1] << 8));
        p += 2;
        if (j < d) W[i * d + j] = v;
        else b[i] = v;
      }
    }
    return { classes: c, dim: d, W: W, b: b };
  }

  return { fitLDA: fitLDA, scores: scores, top: top, save: save, load: load };
}));
