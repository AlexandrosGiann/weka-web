/* Needle 2 tokenizer reader + lightweight SentencePiece-BPE encoder.
 *
 * The .cact RAW tokenizer block is a compact export of the SentencePiece
 * vocabulary.  This file intentionally avoids TextEncoder/TextDecoder so it
 * also works on old Firefox/Android builds.
 */
(function (global) {
  "use strict";

  function utf8Encode(s) {
    var out = [], i = 0, c, c2, cp;
    while (i < s.length) {
      c = s.charCodeAt(i++);
      if (c >= 0xD800 && c <= 0xDBFF && i < s.length) {
        c2 = s.charCodeAt(i);
        if (c2 >= 0xDC00 && c2 <= 0xDFFF) {
          i++;
          cp = 0x10000 + ((c - 0xD800) << 10) + (c2 - 0xDC00);
        } else cp = 0xFFFD;
      } else if (c >= 0xDC00 && c <= 0xDFFF) cp = 0xFFFD;
      else cp = c;

      if (cp <= 0x7F) out.push(cp);
      else if (cp <= 0x7FF) {
        out.push(0xC0 | (cp >>> 6), 0x80 | (cp & 63));
      } else if (cp <= 0xFFFF) {
        out.push(0xE0 | (cp >>> 12), 0x80 | ((cp >>> 6) & 63), 0x80 | (cp & 63));
      } else {
        out.push(0xF0 | (cp >>> 18), 0x80 | ((cp >>> 12) & 63),
                 0x80 | ((cp >>> 6) & 63), 0x80 | (cp & 63));
      }
    }
    return out;
  }

  function utf8Decode(bytes) {
    var s = "", i = 0, b0, b1, b2, b3, cp;
    while (i < bytes.length) {
      b0 = bytes[i++] & 255;
      if (b0 < 0x80) cp = b0;
      else if ((b0 & 0xE0) === 0xC0 && i < bytes.length) {
        b1 = bytes[i++] & 63;
        cp = ((b0 & 31) << 6) | b1;
      } else if ((b0 & 0xF0) === 0xE0 && i + 1 < bytes.length) {
        b1 = bytes[i++] & 63; b2 = bytes[i++] & 63;
        cp = ((b0 & 15) << 12) | (b1 << 6) | b2;
      } else if ((b0 & 0xF8) === 0xF0 && i + 2 < bytes.length) {
        b1 = bytes[i++] & 63; b2 = bytes[i++] & 63; b3 = bytes[i++] & 63;
        cp = ((b0 & 7) << 18) | (b1 << 12) | (b2 << 6) | b3;
      } else cp = 0xFFFD;

      if (cp <= 0xFFFF) s += String.fromCharCode(cp);
      else {
        cp -= 0x10000;
        s += String.fromCharCode(0xD800 | (cp >>> 10), 0xDC00 | (cp & 1023));
      }
    }
    return s;
  }

  function bytesKey(bytes, start, len) {
    var s = "$", i, end = start + len;
    for (i = start; i < end; i++) s += String.fromCharCode(bytes[i] & 255);
    return s;
  }

  function binaryToBytes(s) {
    var out = new Array(s.length), i;
    for (i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 255;
    return out;
  }

  function parseBytePiece(bin) {
    var t = "", i, n;
    for (i = 0; i < bin.length; i++) t += String.fromCharCode(bin.charCodeAt(i) & 255);
    if (t.length >= 6 && t.substr(0, 3) === "<0x" && t.charAt(t.length - 1) === ">") {
      n = parseInt(t.substr(3, 2), 16);
      if (!isNaN(n)) return n & 255;
    }
    return -1;
  }

  function findRaw(model) {
    var i, t;
    for (i = 0; i < model.tensors.length; i++) {
      t = model.tensors[i];
      if (t.label === "tokenizer.raw" || t.dtype === NeedleCact.DT_RAW) return t;
    }
    throw new Error("tokenizer.raw not found in .cact");
  }

  function Tokenizer(model) {
    var t = findRaw(model), dv = model.dv, p = t.offset, end = t.offset + t.nbytes;
    var n, i, score, type, len, bin, id, b;
    if (t.nbytes < 24) throw new Error("tokenizer.raw too small");

    this.nPieces = dv.getUint32(p, true);
    this.padId = dv.getUint32(p + 4, true);
    this.eosId = dv.getUint32(p + 8, true);
    this.bosId = dv.getUint32(p + 12, true);
    this.unkId = dv.getUint32(p + 16, true);
    this.addDummy = model.bytes[p + 20] !== 0;
    this.byteFallback = model.bytes[p + 21] !== 0;
    p += 24;

    n = this.nPieces;
    if (n < 1 || n > 100000) throw new Error("implausible tokenizer piece count: " + n);
    this.pieces = new Array(n);       /* binary UTF-8 strings */
    this.scores = new Float32Array(n);
    this.types = new Uint8Array(n);
    this.lookup = {};
    this.byteId = new Int32Array(256);
    for (i = 0; i < 256; i++) this.byteId[i] = -1;
    this.markers = [];

    for (i = 0; i < n; i++) {
      if (p + 7 > end) throw new Error("tokenizer piece header OOB at " + i);
      score = dv.getFloat32(p, true);
      type = model.bytes[p + 4];
      len = dv.getUint16(p + 5, true);
      p += 7;
      if (p + len > end) throw new Error("tokenizer piece OOB at " + i);
      bin = bytesKey(model.bytes, p, len).substr(1);
      p += len;
      this.pieces[i] = bin;
      this.scores[i] = score;
      this.types[i] = type;
      this.lookup["$" + bin] = i;
      if (type === 4) {
        b = parseBytePiece(bin);
        if (b >= 0) this.byteId[b] = i;
      }
      if (type === 3) this.markers.push(i);
    }

    var self = this;
    this.markers.sort(function (a, b2) {
      return self.pieces[b2].length - self.pieces[a].length;
    });
  }

  Tokenizer.prototype.pieceText = function (id) {
    if (id < 0 || id >= this.nPieces) return "<?>";
    return utf8Decode(binaryToBytes(this.pieces[id]));
  };

  Tokenizer.prototype._lookupBin = function (bin) {
    var x = this.lookup["$" + bin];
    return typeof x === "number" ? x : -1;
  };

  Tokenizer.prototype._utf8Symbols = function (bytes) {
    var syms = [], i = 0, b, len, bin, id, j, bid;
    while (i < bytes.length) {
      b = bytes[i];
      if (b < 0x80) len = 1;
      else if ((b & 0xE0) === 0xC0) len = 2;
      else if ((b & 0xF0) === 0xE0) len = 3;
      else if ((b & 0xF8) === 0xF0) len = 4;
      else len = 1;
      if (i + len > bytes.length) len = 1;
      bin = bytesKey(bytes, i, len).substr(1);
      id = this._lookupBin(bin);
      if (id >= 0 && this.types[id] !== 2) {
        syms.push({bin: bin, id: id});
      } else if (this.byteFallback) {
        for (j = 0; j < len; j++) {
          bid = this.byteId[bytes[i + j] & 255];
          if (bid >= 0) syms.push({bin: this.pieces[bid], id: bid, byteOnly: true});
          else syms.push({bin: bin, id: this.unkId});
        }
      } else syms.push({bin: bin, id: this.unkId});
      i += len;
    }
    return syms;
  };

  Tokenizer.prototype._encodePlain = function (text, addDummyHere) {
    var bytes = utf8Encode(text), norm = [], i, b;
    var ws = [0xE2, 0x96, 0x81]; /* U+2581 */
    var syms, best, bestScore, bestId, bestBin, pair, id, score, out;

    if (addDummyHere && this.addDummy) {
      norm.push(ws[0], ws[1], ws[2]);
    }
    for (i = 0; i < bytes.length; i++) {
      b = bytes[i];
      if (b === 0x20) norm.push(ws[0], ws[1], ws[2]);
      else norm.push(b);
    }

    syms = this._utf8Symbols(norm);
    if (!syms.length) return [];

    /* SentencePiece BPE: repeatedly merge the adjacent pair whose resulting
     * vocabulary piece has the highest model score.  Byte-fallback symbols
     * can still participate if their literal concatenation is a normal piece. */
    while (syms.length > 1) {
      best = -1; bestScore = -Infinity; bestId = -1; bestBin = "";
      for (i = 0; i + 1 < syms.length; i++) {
        pair = syms[i].bin + syms[i + 1].bin;
        id = this._lookupBin(pair);
        if (id < 0) continue;
        if (this.types[id] === 2 || this.types[id] === 4) continue;
        score = this.scores[id];
        if (score > bestScore) {
          bestScore = score; best = i; bestId = id; bestBin = pair;
        }
      }
      if (best < 0) break;
      syms.splice(best, 2, {bin: bestBin, id: bestId});
    }

    out = [];
    for (i = 0; i < syms.length; i++) out.push(syms[i].id);
    return out;
  };

  Tokenizer.prototype.encode = function (text) {
    var out = [], i = 0, m, mid, mb, pos, bestPos, bestId, plain, k;
    var textBytes = utf8Encode(text), bin = "", j;
    /* Work in a binary UTF-8 string so USER_DEFINED markers are matched on
     * exactly the bytes stored in the .cact blob. */
    for (j = 0; j < textBytes.length; j++) bin += String.fromCharCode(textBytes[j]);

    while (i < bin.length) {
      bestPos = -1; bestId = -1;
      for (m = 0; m < this.markers.length; m++) {
        mid = this.markers[m]; mb = this.pieces[mid];
        pos = bin.indexOf(mb, i);
        if (pos >= 0 && (bestPos < 0 || pos < bestPos ||
            (pos === bestPos && mb.length > this.pieces[bestId].length))) {
          bestPos = pos; bestId = mid;
        }
      }
      if (bestPos < 0) {
        plain = utf8Decode(binaryToBytes(bin.substring(i)));
        k = this._encodePlain(plain, out.length === 0);
        Array.prototype.push.apply(out, k);
        break;
      }
      if (bestPos > i) {
        plain = utf8Decode(binaryToBytes(bin.substring(i, bestPos)));
        k = this._encodePlain(plain, out.length === 0);
        Array.prototype.push.apply(out, k);
      }
      out.push(bestId);
      i = bestPos + this.pieces[bestId].length;
    }
    return out;
  };

  Tokenizer.prototype.decode = function (ids) {
    var bytes = [], i, id, type, bin, j, b, p;
    var ws0 = 0xE2, ws1 = 0x96, ws2 = 0x81;
    for (i = 0; i < ids.length; i++) {
      id = ids[i] | 0;
      if (id < 0 || id >= this.nPieces) continue;
      type = this.types[id];
      if (type === 2) continue; /* CONTROL */
      if (type === 4) {
        b = parseBytePiece(this.pieces[id]);
        if (b >= 0) bytes.push(b);
        continue;
      }
      bin = this.pieces[id];
      for (j = 0; j < bin.length; j++) bytes.push(bin.charCodeAt(j) & 255);
    }

    /* Convert SentencePiece metaspace U+2581 back to ASCII space. */
    var un = [];
    for (p = 0; p < bytes.length; p++) {
      if (p + 2 < bytes.length && bytes[p] === ws0 && bytes[p + 1] === ws1 && bytes[p + 2] === ws2) {
        un.push(0x20); p += 2;
      } else un.push(bytes[p]);
    }
    if (this.addDummy && un.length && un[0] === 0x20) un.shift();
    return utf8Decode(un);
  };

  Tokenizer.prototype.summary = function () {
    return {
      nPieces: this.nPieces, padId: this.padId, eosId: this.eosId,
      bosId: this.bosId, unkId: this.unkId,
      addDummy: this.addDummy, byteFallback: this.byteFallback,
      userDefinedPieces: this.markers.length
    };
  };

  global.NeedleTokenizer = {Tokenizer: Tokenizer, utf8Encode: utf8Encode, utf8Decode: utf8Decode};
}(this));
