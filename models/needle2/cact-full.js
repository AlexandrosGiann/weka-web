/* Needle 2 .cact parser — Full-run alpha.
 * ES5-compatible style for old Firefox/Android.
 *
 * Header layout follows the public Needle 2 container:
 *   30 x 4 bytes = 120 bytes
 *   h[0] magic
 *   h[1] tensor count
 *   h[2] codebook float count
 *   h[3..28] model geometry
 *   bytes 116..119 rope_theta float32
 *
 * Tensor directory records are 44 bytes each.
 */
(function (global) {
  "use strict";

  var NEEDLE_TAG = 0x05E12A83;
  var HEADER_BYTES = 120;
  var RECORD_BYTES = 44;

  var DT_FP16 = 1;
  var DT_FP32 = 2;
  var DT_CQ   = 3;
  var DT_RAW  = 4;

  function u64AsNumber(dv, off) {
    var lo = dv.getUint32(off, true);
    var hi = dv.getUint32(off + 4, true);
    if (hi !== 0) {
      throw new Error("u64 too large for this old-JS runtime at byte " + off);
    }
    return lo;
  }

  function shapeProduct(shape, ndim) {
    var p = 1, i;
    if (ndim === 0) return 0;
    for (i = 0; i < ndim; i++) p *= shape[i];
    return p;
  }

  function parseRecord(dv, off, fileBytes, index) {
    var dtype, ndim, shape, dataOffset, nbytes, group, bits;
    if (off + RECORD_BYTES > fileBytes) {
      throw new Error("tensor directory OOB at #" + index);
    }

    dtype = dv.getUint8(off);
    ndim = dv.getUint8(off + 1);
    shape = [
      dv.getUint32(off + 4, true),
      dv.getUint32(off + 8, true),
      dv.getUint32(off + 12, true),
      dv.getUint32(off + 16, true)
    ];
    dataOffset = u64AsNumber(dv, off + 20);
    nbytes = u64AsNumber(dv, off + 28);
    group = dv.getUint32(off + 36, true);
    bits = dv.getUint32(off + 40, true);

    if (ndim > 4) throw new Error("bad ndim at tensor #" + index);
    if (dataOffset > fileBytes || nbytes > fileBytes ||
        dataOffset + nbytes > fileBytes) {
      throw new Error("tensor blob OOB at #" + index);
    }

    return {
      index: index,
      label: "",
      dtype: dtype,
      ndim: ndim,
      shape: shape,
      offset: dataOffset,
      nbytes: nbytes,
      group: group,
      bits: bits,
      elements: shapeProduct(shape, ndim)
    };
  }

  function geometryFromHeader(dv) {
    var g = {}, i;
    g.numTensors = dv.getUint32(4, true);
    g.codebookCount = dv.getUint32(8, true);
    g.kvWindow = dv.getUint32(12, true);
    g.kvBits = dv.getUint32(16, true);
    g.vocab = dv.getUint32(20, true);
    g.dModel = dv.getUint32(24, true);
    g.nHeads = dv.getUint32(28, true);
    g.nKV = dv.getUint32(32, true);
    g.nLayers = dv.getUint32(36, true);
    g.headDim = dv.getUint32(40, true);
    g.maxSeq = dv.getUint32(44, true);
    g.hadaN = dv.getUint32(48, true);
    g.lanes = dv.getUint32(52, true);
    g.slots = dv.getUint32(56, true);
    g.subDim = dv.getUint32(60, true);
    g.nTables = dv.getUint32(64, true);
    g.taps = dv.getUint32(68, true);
    g.dilation = dv.getUint32(72, true);
    g.nOrders = dv.getUint32(76, true);
    g.orders = [];
    for (i = 0; i < 4; i++) g.orders.push(dv.getUint32(80 + i * 4, true));
    g.nSites = dv.getUint32(96, true);
    g.sites = [];
    for (i = 0; i < 4; i++) g.sites.push(dv.getUint32(100 + i * 4, true));
    g.ropeTheta = dv.getFloat32(116, true);
    return g;
  }

  function labelCanonical(tensors, g) {
    var idx = 0, l, s, names, j;

    if (tensors.length === 0) return;
    tensors[idx++].label = "embedding";

    names = [
      "norm_in",
      "q_proj",
      "k_proj",
      "v_proj",
      "q_norm",
      "k_norm",
      "gate_proj",
      "out_proj",
      "post_norm",
      "attn_gate",
      "pre_hada",
      "d1",
      "d2",
      "d3"
    ];

    for (l = 0; l < g.nLayers; l++) {
      for (j = 0; j < names.length; j++) {
        if (idx >= tensors.length) return;
        tensors[idx++].label = "layer." + l + "." + names[j];
      }
    }

    names = [
      "mhc.a_pre", "mhc.a_post", "mhc.a_res",
      "mhc.b_pre", "mhc.b_post", "mhc.b_res",
      "mhc.phi_pre", "mhc.phi_post", "mhc.phi_res"
    ];
    for (j = 0; j < names.length && idx < tensors.length; j++) {
      tensors[idx++].label = names[j];
    }

    for (s = 0; s < g.nSites; s++) {
      names = ["tables", "key_proj", "value_proj", "taps"];
      for (j = 0; j < names.length && idx < tensors.length; j++) {
        tensors[idx++].label = "engram." + s + "." + names[j];
      }
    }

    if (idx < tensors.length) tensors[idx++].label = "final_norm";

    while (idx < tensors.length) {
      if (tensors[idx].dtype === DT_RAW) tensors[idx].label = "tokenizer.raw";
      else tensors[idx].label = "optional." + idx;
      idx++;
    }
  }

  function CactModel(buffer) {
    var i, recOff;

    if (!buffer || buffer.byteLength < HEADER_BYTES) {
      throw new Error("model file too small");
    }

    this.buffer = buffer;
    this.bytes = new Uint8Array(buffer);
    this.dv = new DataView(buffer);

    this.tag = this.dv.getUint32(0, true);
    if (this.tag !== NEEDLE_TAG) {
      throw new Error("bad NEEDLE_TAG: 0x" + this.tag.toString(16));
    }

    this.geometry = geometryFromHeader(this.dv);

    if (this.geometry.numTensors < 1 || this.geometry.numTensors > 10000) {
      throw new Error("implausible tensor count: " + this.geometry.numTensors);
    }
    if (this.geometry.codebookCount < 4 || this.geometry.codebookCount > 1024) {
      throw new Error("implausible codebook count: " + this.geometry.codebookCount);
    }

    this.codebook = new Float32Array(this.geometry.codebookCount);
    for (i = 0; i < this.geometry.codebookCount; i++) {
      this.codebook[i] = this.dv.getFloat32(HEADER_BYTES + i * 4, true);
    }

    this.directoryOffset = HEADER_BYTES + this.geometry.codebookCount * 4;
    this.tensors = [];
    recOff = this.directoryOffset;

    for (i = 0; i < this.geometry.numTensors; i++) {
      this.tensors.push(parseRecord(
        this.dv, recOff, buffer.byteLength, i
      ));
      recOff += RECORD_BYTES;
    }

    labelCanonical(this.tensors, this.geometry);
    this.validate();
  }

  CactModel.prototype.validate = function () {
    var i, t;
    var counts = {
      total: this.tensors.length,
      cq: 0, cq2: 0, cq3: 0, cq4: 0, cq5: 0,
      fp16: 0, fp32: 0, raw: 0, other: 0
    };

    for (i = 0; i < this.tensors.length; i++) {
      t = this.tensors[i];
      if (t.dtype === DT_CQ) {
        counts.cq++;
        if (t.bits === 2) counts.cq2++;
        else if (t.bits === 3) counts.cq3++;
        else if (t.bits === 4) counts.cq4++;
        else if (t.bits === 5) counts.cq5++;
      } else if (t.dtype === DT_FP16) counts.fp16++;
      else if (t.dtype === DT_FP32) counts.fp32++;
      else if (t.dtype === DT_RAW) counts.raw++;
      else counts.other++;
    }

    this.counts = counts;
  };

  CactModel.prototype.cqShapeCensus = function () {
    var map = {}, out = [], i, t, key, e, dims, j;

    for (i = 0; i < this.tensors.length; i++) {
      t = this.tensors[i];
      if (t.dtype !== DT_CQ || t.ndim !== 2) continue;

      dims = [];
      for (j = 0; j < t.ndim; j++) dims.push(t.shape[j]);

      key = "b" + t.bits + "|g" + t.group + "|" + dims.join("x");
      e = map[key];

      if (!e) {
        e = {
          bits: t.bits,
          group: t.group,
          shape: dims,
          count: 0,
          totalElements: 0,
          representativeIndex: t.index,
          representativeLabel: t.label,
          labels: []
        };
        map[key] = e;
        out.push(e);
      }

      e.count++;
      e.totalElements += t.elements;
      if (e.labels.length < 8) e.labels.push(t.label);
    }

    out.sort(function (a, b) {
      if (a.bits !== b.bits) return a.bits - b.bits;
      if (a.shape[1] !== b.shape[1]) return a.shape[1] - b.shape[1];
      return a.shape[0] - b.shape[0];
    });

    return out;
  };

  CactModel.prototype.cq2Representatives = function () {
    var c = this.cqShapeCensus(), out = [], i;
    for (i = 0; i < c.length; i++) {
      if (c[i].bits === 2) out.push({
        census: c[i],
        tensor: this.tensors[c[i].representativeIndex]
      });
    }
    return out;
  };

  CactModel.prototype.summary = function () {
    return {
      fileBytes: this.buffer.byteLength,
      tag: "0x" + this.tag.toString(16),
      geometry: this.geometry,
      counts: this.counts,
      cqShapes: this.cqShapeCensus()
    };
  };

  global.NeedleCact = {
    CactModel: CactModel,
    DT_FP16: DT_FP16,
    DT_FP32: DT_FP32,
    DT_CQ: DT_CQ,
    DT_RAW: DT_RAW
  };
}(this));
