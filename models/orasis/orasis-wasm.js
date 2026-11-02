// WebAssembly loader for the Rust runtime. Picks the SIMD build when the engine supports it.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.OrasisWasm = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  // (func (result v128) i32.const 0 i8x16.splat)
  var SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 8, 1, 6, 0, 65, 0, 253, 15, 11]);

  function simdSupported() {
    return typeof WebAssembly === 'object' && WebAssembly.validate(SIMD_PROBE);
  }

  // wasmBytes: pick orasis-simd.wasm when simdSupported(), else orasis.wasm.
  function create(wasmBytes, modelBytes) {
    return WebAssembly.instantiate(wasmBytes, {}).then(function (r) {
      var e = r.instance.exports;
      var m = new Uint8Array(modelBytes);
      var p = e.orasis_alloc(m.length);
      new Uint8Array(e.memory.buffer, p, m.length).set(m);
      var size = e.orasis_load(p, m.length);
      e.orasis_free(p, m.length);
      if (!size) throw new Error('invalid model file');
      var dim = new Uint16Array(m.buffer, m.byteOffset, 6)[5];
      var rgbLen = size * size * 3, rgbP = e.orasis_alloc(rgbLen), outP = e.orasis_alloc(dim * 4);
      return {
        size: size,
        dim: dim,
        encode: function (rgb) {
          if (rgb.length !== rgbLen) throw new Error('expected ' + rgbLen + ' bytes, got ' + rgb.length);
          // memory may have grown: views are recreated on each call
          new Uint8Array(e.memory.buffer, rgbP, rgbLen).set(rgb);
          if (!e.orasis_encode(rgbP, outP)) throw new Error('encode failed');
          return new Float32Array(new Float32Array(e.memory.buffer, outP, dim));
        }
      };
    });
  }

  return { create: create, simdSupported: simdSupported };
}));
