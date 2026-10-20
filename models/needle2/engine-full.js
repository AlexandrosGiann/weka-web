/* Needle 2 Pure-JS — full 27-layer core runner (alpha)
 *
 * Pure JavaScript / TypedArrays / DataView. No WASM, no WebGL, no framework.
 * Implements the complete Needle 2 base transformer core in the .cact file:
 *   - 27 layers, 4-lane mHC routing
 *   - CQ2/CQ3/CQ4 packed matrix-vector kernels
 *   - split-half RoPE
 *   - causal 8Q/4KV GQA
 *   - persistent int8 K/V cache, one scale per KV-head row
 *   - 256-token sliding attention window
 *   - Engram sites (hashed 2/3-grams + dilated value convolution)
 *   - Hadamard MLP
 *   - final norm + tied embedding logits
 *
 * This is the full MODEL CORE. It intentionally does not reproduce the
 * official outer agent/tool grammar/retrieval/confidence machinery.
 */
(function (global) {
  "use strict";

  var EPS = 1e-6;
  var ENGRAM_SEED = 0x9E3779B9;
  var ENGRAM_PRIME = 0x01000193;

  function fp16ToF32(h) {
    var s=(h>>>15)&1,e=(h>>>10)&31,f=h&1023,v;
    if(e===0) v=f===0?0:(f/1024)*Math.pow(2,-14);
    else if(e===31) v=f?NaN:Infinity;
    else v=(1+f/1024)*Math.pow(2,e-15);
    return s?-v:v;
  }

  function imul32(a,b) {
    if (Math.imul) return Math.imul(a,b);
    var ah=(a>>>16)&0xffff, al=a&0xffff;
    var bh=(b>>>16)&0xffff, bl=b&0xffff;
    return (al*bl + (((ah*bl + al*bh)&0xffff)<<16))|0;
  }

  function tensorCount(t) {
    var n=1,i;
    if(t.ndim===0) return t.nbytes/(t.dtype===NeedleCact.DT_FP16?2:4);
    for(i=0;i<t.ndim;i++) n*=t.shape[i];
    return n;
  }

  /*
   * PERF: fp16 -> f32 lookup table (256 KB, built once).
   * Every fp16 value is exactly representable in float32, so a table lookup
   * returns the very same number fp16ToF32() computes, without Math.pow or
   * DataView calls in the hot CQ loops.
   */
  var _f16Table=null;
  function f16Table() {
    var t=_f16Table,i;
    if(t)return t;
    t=new Float32Array(65536);
    for(i=0;i<65536;i++)t[i]=fp16ToF32(i);
    _f16Table=t;
    return t;
  }

  function readFP(model,t) {
    var n=tensorCount(t),o=new Float32Array(n),i;
    if(t.dtype===NeedleCact.DT_FP16) {
      for(i=0;i<n;i++) o[i]=fp16ToF32(model.dv.getUint16(t.offset+i*2,true));
    } else if(t.dtype===NeedleCact.DT_FP32) {
      for(i=0;i<n;i++) o[i]=model.dv.getFloat32(t.offset+i*4,true);
    } else throw new Error(t.label+" is not FP16/FP32");
    return o;
  }

  function fwhtInPlace(a,off,n) {
    var len,i,j,p,q,x,y;
    for(len=1;len<n;len<<=1) {
      for(i=0;i<n;i+=(len<<1)) {
        for(j=0;j<len;j++) {
          p=off+i+j;q=p+len;x=a[p];y=a[q];a[p]=x+y;a[q]=x-y;
        }
      }
    }
  }

  /* PERF: the descriptor depends only on the tensor header; cache it. */
  function cqDesc(t) {
    var d=t._cqDesc;
    if(!d) { d=cqDescBuild(t); t._cqDesc=d; }
    return d;
  }

  function cqDescBuild(t) {
    var input=t.shape[1],group=t.group,inPad=Math.ceil(input/group)*group;
    var nGroups=inPad/group,groupBits=group*t.bits,groupBytes,rowBytes;
    if(t.bits!==2&&t.bits!==3&&t.bits!==4)
      throw new Error("full runner supports CQ2/CQ3/CQ4; got CQ"+t.bits+" at "+t.label);
    if((groupBits&7)!==0)
      throw new Error("CQ group is not byte-aligned at "+t.label+": group="+group+" bits="+t.bits);
    groupBytes=groupBits>>>3;
    rowBytes=groupBytes*nGroups;
    return {out:t.shape[0],input:input,group:group,inPad:inPad,nGroups:nGroups,
      groupBytes:groupBytes,rowBytes:rowBytes,normsBase:t.offset+t.shape[0]*rowBytes};
  }

  /*
   * NativeQ A1
   * ----------
   * Official single-token Needle does NOT dequantize the activation before
   * CQ projection.  It keeps the FWHT-domain activation as int8, decodes
   * packed CQ codebook indices to an int8 weight LUT, accumulates int8*int8
   * into an integer dot, then applies:
   *
   *   activation_group_scale * codebook_scale * tensor_group_scale
   *
   * This scalar JS path keeps the original logical order; native byte
   * repacking/interleaving is only a SIMD layout detail and does not change
   * the mathematical dot product.
   */
  var _f32tmp=new Float32Array(1);
  var f32=Math.fround || function(x) {
    _f32tmp[0]=x;return _f32tmp[0];
  };
  function lroundfCompat(x) {
    return x>=0?Math.floor(x+0.5):Math.ceil(x-0.5);
  }
  function clampS8(q) {
    if(q>127)return 127;
    if(q<-127)return -127;
    return q;
  }

  function nativeWeightCodebooks(model) {
    var c=model._nativeWeightCodebooks;
    var bitsList=[2,3,4],starts=[0,4,12],counts=[4,8,16];
    var q,scales,maxAbs,si,bits,start,count,i,v,m,scale;
    if(c)return c;
    if(model.codebook.length<28)
      throw new Error("NativeQ A1c expects the 4+8+16 Needle codebook layout");

    /*
     * Exact native Engine::load layout recovered from ARM code:
     *   CQ2 -> codebook[0..3]   -> independent int8 LUT + scale
     *   CQ3 -> codebook[4..11]  -> independent int8 LUT + scale
     *   CQ4 -> codebook[12..27] -> independent int8 LUT + scale
     *
     * For each segment native computes scale=maxAbs/127 in float32 and then
     * stores lroundf(codebook[i]/scale) as one signed byte.  Do not use one
     * global 28-value scale: that was the remaining A1/A1b approximation.
     */
    q=new Int8Array(model.codebook.length);
    scales=new Float32Array(6);
    maxAbs=new Float32Array(6);

    for(si=0;si<bitsList.length;si++) {
      bits=bitsList[si];start=starts[si];count=counts[si];m=0;
      for(i=0;i<count;i++) {
        v=Math.abs(model.codebook[start+i]);
        if(v>m)m=v;
      }
      scale=m>0?f32(m/127):f32(1.0);
      scales[bits]=scale;
      maxAbs[bits]=m;
      for(i=0;i<count;i++) {
        /* Native load uses vdiv.f32 here, then lroundf, then strb. */
        q[start+i]=lroundfCompat(f32(model.codebook[start+i]/scale));
      }
    }

    /*
     * PERF: per-byte unpacked weight tables.  w2[b*4+j] is the CQ2 int8
     * weight for 2-bit field j of byte b; w4[b*2+j] the CQ4 weight for
     * nibble j.  Pure re-indexing of q[], so integer dots are unchanged.
     */
    var w2=new Int8Array(1024),w4=new Int8Array(512),b;
    for(b=0;b<256;b++) {
      w2[b*4]=q[b&3];w2[b*4+1]=q[(b>>>2)&3];w2[b*4+2]=q[(b>>>4)&3];w2[b*4+3]=q[(b>>>6)&3];
      w4[b*2]=q[12+(b&15)];w4[b*2+1]=q[12+((b>>>4)&15)];
    }

    c={q:q,scales:scales,maxAbs:maxAbs,w2:w2,w4:w4};
    model._nativeWeightCodebooks=c;
    return c;
  }

  /*
   * NativeQ A1g — exact activation quantizer geometry.
   *
   * Native Engine::quantize_activation reads Engine+0xc8.  Direct runtime
   * inspection on Needle 2 proved that field is 128.  It is NOT headDim.
   *
   * For each 128-value block native does:
   *   1. float32 FWHT
   *   2. multiply by float32 1/sqrt(128)
   *   3. scale = maxAbs/127 (or 1 for an all-zero block)
   *   4. int8 q = lroundf(value / scale)
   *   5. CQ2 storage transpose: dst[(i&3)*32 + (i>>2)] = q[i]
   *
   * The scalar JS projection keeps q in logical order because the transpose
   * is a SIMD storage layout; the mathematical dot product is unchanged.
   */
  var NATIVE_ACT_QBLOCK=128;

  function prepareCQInput(x,d) {
    var block=NATIVE_ACT_QBLOCK;
    if((d.inPad%block)!==0) throw new Error("native activation qblock does not divide inPad");
    if((d.group%block)!==0 && (block%d.group)!==0)
      throw new Error("native activation qblock incompatible with CQ group");

    var tmp=new Float32Array(d.inPad);
    var q=new Int8Array(d.inPad);
    var nBlocks=(d.inPad/block)|0;
    var scales=new Float32Array(nBlocks);
    var i,g,k,v,m,scale,inv,qi;
    var norm=f32(1/Math.sqrt(block));

    for(i=0;i<d.inPad;i++)tmp[i]=i<x.length?x[i]:0;

    for(g=0;g<nBlocks;g++) {
      var off=g*block;
      fwhtInPlace(tmp,off,block);

      m=0;
      for(k=0;k<block;k++) {
        v=f32(tmp[off+k]*norm);
        tmp[off+k]=v;
        v=Math.abs(v);
        if(v>m)m=v;
      }

      scale=m>0?f32(m/127):f32(1.0);
      scales[g]=scale;
      inv=f32(1/scale);

      for(k=0;k<block;k++) {
        qi=clampS8(lroundfCompat(f32(tmp[off+k]*inv)));
        q[off+k]=qi;
      }
    }

    return {
      q:q,scales:scales,inPad:d.inPad,group:d.group,nGroups:d.nGroups,
      block:block,nBlocks:nBlocks
    };
  }

  /*
   * A1e diagnostic:
   * native matmul_row_sub4 does NOT index CQ2 LUT with raw 2-bit code.
   * It maps 0->1, 1->2, 2->3, 3->0 before the LUT lookup.
   */
  /* A1e proved direct CQ2 indexing is correct for wbits==2. */
  var CQ2_CODE_MODE=0;

  function cq2CodeIndex(raw) {
    return raw;
  }

  function nativeIntDotCQ2Scalar(u8,bp,q,xoff,n,cbq) {
    var k,b,acc=0;
    for(k=0;k<n;k+=4) {
      b=u8[bp++];
      acc += q[xoff+k]   * cbq[cq2CodeIndex(b&3)];
      acc += q[xoff+k+1] * cbq[cq2CodeIndex((b>>>2)&3)];
      acc += q[xoff+k+2] * cbq[cq2CodeIndex((b>>>4)&3)];
      acc += q[xoff+k+3] * cbq[cq2CodeIndex((b>>>6)&3)];
    }
    return acc;
  }


  function nativeIntDotCQ2Neon(u8,bp,q,xoff,n,cbq) {
    var bytes=n>>>2,base,l,k0,k1,b0,b1,s01,s23,acc=0;
    if(bytes!==32) return nativeIntDotCQ2Scalar(u8,bp,q,xoff,n,cbq);
    for(base=0;base<32;base+=16) {
      for(l=0;l<8;l++) {
        k0=base+l; k1=k0+8;
        b0=u8[bp+k0]; b1=u8[bp+k1];
        s01 =
          q[xoff+4*k0]   * cbq[cq2CodeIndex(b0&3)] +
          q[xoff+4*k0+1] * cbq[cq2CodeIndex((b0>>>2)&3)] +
          q[xoff+4*k1]   * cbq[cq2CodeIndex(b1&3)] +
          q[xoff+4*k1+1] * cbq[cq2CodeIndex((b1>>>2)&3)];
        s23 =
          q[xoff+4*k0+2] * cbq[cq2CodeIndex((b0>>>4)&3)] +
          q[xoff+4*k0+3] * cbq[cq2CodeIndex((b0>>>6)&3)] +
          q[xoff+4*k1+2] * cbq[cq2CodeIndex((b1>>>4)&3)] +
          q[xoff+4*k1+3] * cbq[cq2CodeIndex((b1>>>6)&3)];
        s01=(s01<<16)>>16;
        s23=(s23<<16)>>16;
        acc+=s01+s23;
      }
    }
    return acc;
  }

  function cq2MaskBit(label) {
    if(!label)return 0;
    if(/\.q_proj$/.test(label))return 1;
    if(/\.k_proj$/.test(label))return 2;
    if(/\.v_proj$/.test(label))return 4;
    if(/\.gate_proj$/.test(label))return 8;
    if(/\.out_proj$/.test(label))return 16;
    return 0;
  }

  function nativeIntDotCQ2Dispatch(label,u8,bp,q,xoff,n,cbq) {
    if(/^engram\\.[01]\\.(key_proj|value_proj)$/.test(label))
      return nativeIntDotCQ2Neon(u8,bp,q,xoff,n,cbq);

    /* Keep the A54 diagnostic state, add exactly ONE A55 candidate. */
    if(label==="layer.8.out_proj" ||
       label==="layer.6.out_proj" ||
       label==="layer.11.out_proj")
      return nativeIntDotCQ2Neon(u8,bp,q,xoff,n,cbq);

    var mask=(global.NEEDLE_CQ2_NEON_MASK|0),bit=cq2MaskBit(label);
    if(bit && (mask&bit))
      return nativeIntDotCQ2Neon(u8,bp,q,xoff,n,cbq);
    return nativeIntDotCQ2Scalar(u8,bp,q,xoff,n,cbq);
  }

  function nativeIntDotCQ2(u8,bp,q,xoff,n,cbq) {
    return nativeIntDotCQ2Scalar(u8,bp,q,xoff,n,cbq);
  }

  function nativeIntDotCQ3(u8,bp,q,xoff,n,cbq) {
    var k,j,w,acc=0;
    for(k=0;k<n;k+=8) {
      w=u8[bp] | (u8[bp+1]<<8) | (u8[bp+2]<<16);
      bp+=3;
      for(j=0;j<8;j++)
        acc += q[xoff+k+j] * cbq[4+((w>>>(3*j))&7)];
    }
    return acc;
  }

  function nativeIntDotCQ4(u8,bp,q,xoff,n,cbq) {
    var k,b,acc=0;
    for(k=0;k<n;k+=2) {
      b=u8[bp++];
      acc += q[xoff+k]   * cbq[12+(b&15)];
      acc += q[xoff+k+1] * cbq[12+((b>>>4)&15)];
    }
    return acc;
  }

  /*
   * PERF fast kernels.  Same integer dot products as the reference kernels
   * above (int8*int8 sums are exact, so only the indexing changes); the
   * weights come from the per-byte w2/w4 tables built with the codebooks.
   */
  /* Bit-exact twin of nativeIntDotCQ2Neon for the 128-wide (32 byte) case,
   * including the int16 wrap of the s01/s23 lane sums. */
  function fastDotCQ2Neon128(u8,bp,q,x,w2) {
    var base,l,k0,o0,o1,x0,x1,s01,s23,acc=0;
    for(base=0;base<32;base+=16) {
      for(l=0;l<8;l++) {
        k0=base+l;
        o0=u8[bp+k0]<<2; o1=u8[bp+k0+8]<<2;
        x0=x+4*k0; x1=x0+32;
        s01=q[x0]*w2[o0]+q[x0+1]*w2[o0+1]+q[x1]*w2[o1]+q[x1+1]*w2[o1+1];
        s23=q[x0+2]*w2[o0+2]+q[x0+3]*w2[o0+3]+q[x1+2]*w2[o1+2]+q[x1+3]*w2[o1+3];
        acc=(acc+((s01<<16)>>16)+((s23<<16)>>16))|0;
      }
    }
    return acc;
  }

  /*
   * PERF: CQ2 nibble tables (LUT-GEMV).  For every weight byte position p
   * (4 activations) build two 16-entry tables of partial integer dots:
   *   t[p*32+n]    = q[4p]  *w(n&3) + q[4p+1]*w(n>>2)   (low nibble)
   *   t[p*32+16+n] = q[4p+2]*w(n&3) + q[4p+3]*w(n>>2)   (high nibble)
   * A row then needs 2 table loads per byte instead of 4 multiply-adds.
   * Built once per prepared activation and shared by all CQ2 tensors that
   * reuse it (q/k/v/gate, engram key/value).  |entry| <= 2*127*127, so
   * Int16 is enough and the table (inPad*8 bytes) stays in L1 cache.
   */
  function cq2NibbleTable(prepared,cbq) {
    var t=prepared._cq2Nib,q,n,p,x,o,a0,a1,i,j,c0,c1,c2,c3;
    if(t)return t;
    q=prepared.q;n=prepared.inPad;
    t=new Int16Array(n*8);
    c0=cbq[0];c1=cbq[1];c2=cbq[2];c3=cbq[3];
    for(p=0,x=0,o=0;x<n;p++,x+=4,o+=32) {
      for(j=0;j<2;j++) {
        a0=q[x+2*j];a1=q[x+2*j+1];
        var m0=a0*c0,m1=a0*c1,m2=a0*c2,m3=a0*c3;
        for(i=0;i<4;i++) {
          var hb=a1*cbq[i],b=o+16*j+(i<<2);
          t[b]=m0+hb;t[b+1]=m1+hb;t[b+2]=m2+hb;t[b+3]=m3+hb;
        }
      }
    }
    prepared._cq2Nib=t;
    return t;
  }

  function lutDotCQ2(u8,bp,t,x,n) {
    var e=bp+(n>>>2),o=x<<3,b,acc=0;
    for(;bp<e;bp++,o+=32) {
      b=u8[bp];
      acc=(acc+t[o+(b&15)]+t[o+16+(b>>>4)])|0;
    }
    return acc;
  }

  function fastDotCQ4(u8,bp,q,x,n,w4) {
    var e=x+n,o,acc=0;
    for(;x<e;x+=2) {
      o=u8[bp++]<<1;
      acc=(acc+q[x]*w4[o]+q[x+1]*w4[o+1])|0;
    }
    return acc;
  }

  /*
   * PERF: resolve nativeIntDotCQ2Dispatch's label routing once per tensor
   * instead of running regexes for every 128-wide sub-block.  The routing
   * itself is unchanged (including the engram regex exactly as written
   * above); only NEEDLE_CQ2_NEON_MASK is still read per call so it can be
   * toggled at runtime.
   */
  function cq2UsesNeon(t) {
    var label=t.label;
    if(t._cq2Fixed===undefined) {
      t._cq2Fixed=
        /^engram\\.[01]\\.(key_proj|value_proj)$/.test(label) ||
        label==="layer.8.out_proj" ||
        label==="layer.6.out_proj" ||
        label==="layer.11.out_proj";
      t._cq2Bit=cq2MaskBit(label);
    }
    if(t._cq2Fixed)return true;
    return !!(t._cq2Bit && ((global.NEEDLE_CQ2_NEON_MASK|0)&t._cq2Bit));
  }

  function cqMatvecRows(model,t,prepared,rowStart,rowCount,lut2,lut4) {
    var d=cqDesc(t),y=new Float32Array(rowCount),u8=model.bytes;
    var wc=nativeWeightCodebooks(model),cbq=wc.q,w2=wc.w2,w4=wc.w4,cbScale=wc.scales[t.bits];
    var F16=f16Table(),bits=t.bits;
    var r,g,sb,bp,wbp,np,xoff,dot,actScale,wScale,term,acc;
    var stop=rowStart+rowCount;

    if(rowStart<0||stop>d.out)throw new Error("CQ row slice OOB: "+t.label);
    if(!prepared || !prepared.q || !prepared.scales)
      throw new Error("NativeQ A1c expected quantized activation for "+t.label);
    if(prepared.inPad!==d.inPad || prepared.group!==d.group)
      throw new Error("NativeQ A1g activation geometry mismatch for "+t.label);

    var ab=prepared.block||d.group;
    var subPerGroup=(d.group/ab)|0;
    var subBytes=(t.bits*ab+7)>>3;
    if(subPerGroup<1 || subPerGroup*ab!==d.group)
      throw new Error("A1g requires activation block to divide tensor group");
    if(bits!==2&&bits!==3&&bits!==4)
      throw new Error("NativeQ A1g unsupported CQ"+bits+" at "+t.label);

    var xq=prepared.q,aScales=prepared.scales;
    var nGroups=d.nGroups,group=d.group,groupBytes=d.groupBytes,rowBytes=d.rowBytes;
    var neon=bits===2 && ab===128 && cq2UsesNeon(t);
    var nib=(bits===2 && !neon)?cq2NibbleTable(prepared,cbq):null;

    for(r=rowStart;r<stop;r++) {
      acc=0;
      bp=t.offset+r*rowBytes;
      np=d.normsBase+r*nGroups*2;

      for(g=0;g<nGroups;g++) {
        wScale=F16[u8[np]|(u8[np+1]<<8)];
        np+=2;

        for(sb=0;sb<subPerGroup;sb++) {
          xoff=g*group+sb*ab;
          wbp=bp+sb*subBytes;

          if(bits===2)
            dot=neon?fastDotCQ2Neon128(u8,wbp,xq,xoff,w2):lutDotCQ2(u8,wbp,nib,xoff,ab);
          else if(bits===4)
            dot=fastDotCQ4(u8,wbp,xq,xoff,ab,w4);
          else
            dot=nativeIntDotCQ3(u8,wbp,xq,xoff,ab,cbq);

          actScale=aScales[g*subPerGroup+sb];

          /*
           * Native scalar order:
           * int32 -> float -> *activationScale -> *codebookScale
           * -> *weight/group scale -> accumulate.
           */
          term=f32(dot);
          term=f32(term*actScale);
          term=f32(term*cbScale);
          term=f32(term*wScale);
          acc=f32(acc+term);
        }

        bp += groupBytes;
      }
      y[r-rowStart]=acc;
    }
    return y;
  }

  function cqApply(model,t,x,lut2,lut4) {
    return cqMatvecRows(model,t,prepareCQInput(x,cqDesc(t)),0,t.shape[0],lut2,lut4);
  }

  function cqRow(model,t,row) {
    var d=cqDesc(t),out=new Float32Array(t.shape[1]),tmp=new Float32Array(d.group);
    var inv=1/Math.sqrt(d.group),g,k,j,b,bp,w,norm,dst,np,F16=f16Table(),u8=model.bytes;
    if(row<0||row>=d.out) throw new Error("CQ row OOB "+t.label+" row="+row);
    for(g=0;g<d.nGroups;g++) {
      np=d.normsBase+(row*d.nGroups+g)*2;
      norm=F16[u8[np]|(u8[np+1]<<8)];
      bp=t.offset+row*d.rowBytes+g*d.groupBytes;
      if(t.bits===4) {
        for(k=0;k<d.group;k+=2) { b=model.bytes[bp++];tmp[k]=model.codebook[12+(b&15)]*norm;tmp[k+1]=model.codebook[12+((b>>>4)&15)]*norm; }
      } else if(t.bits===3) {
        for(k=0;k<d.group;k+=8) {
          w=model.bytes[bp] | (model.bytes[bp+1]<<8) | (model.bytes[bp+2]<<16);bp+=3;
          for(j=0;j<8;j++)tmp[k+j]=model.codebook[4+((w>>>(3*j))&7)]*norm;
        }
      } else if(t.bits===2) {
        for(k=0;k<d.group;k+=4) { b=model.bytes[bp++];tmp[k]=model.codebook[b&3]*norm;tmp[k+1]=model.codebook[(b>>>2)&3]*norm;tmp[k+2]=model.codebook[(b>>>4)&3]*norm;tmp[k+3]=model.codebook[(b>>>6)&3]*norm; }
      } else {
        throw new Error("CQ row unsupported CQ"+t.bits+" at "+t.label);
      }
      fwhtInPlace(tmp,0,d.group);
      for(k=0;k<d.group;k++) { dst=g*d.group+k;if(dst<out.length) out[dst]=tmp[k]*inv; }
    }
    return out;
  }

  function rmsUnit(x) {
    var o=new Float32Array(x.length),i,s=0,inv;
    for(i=0;i<x.length;i++) s+=x[i]*x[i];
    inv=1/Math.sqrt(s/x.length+EPS);for(i=0;i<x.length;i++) o[i]=x[i]*inv;return o;
  }

  function zcRmsNorm(x,scale) {
    var o=new Float32Array(x.length),i,s=0,inv;
    for(i=0;i<x.length;i++) s+=x[i]*x[i];
    inv=1/Math.sqrt(s/x.length+EPS);for(i=0;i<x.length;i++) o[i]=(1+scale[i])*x[i]*inv;return o;
  }

  function zcRmsNormHeads(x,scale,hd) {
    var o=new Float32Array(x.length),h,i,off,s,inv,heads=x.length/hd;
    for(h=0;h<heads;h++) { off=h*hd;s=0;for(i=0;i<hd;i++)s+=x[off+i]*x[off+i];
      inv=1/Math.sqrt(s/hd+EPS);for(i=0;i<hd;i++)o[off+i]=(1+scale[i])*x[off+i]*inv; }
    return o;
  }

  function sigmoid(x) { var z;if(x>=0){z=Math.exp(-x);return 1/(1+z);}z=Math.exp(x);return z/(1+z); }
  function silu(x) { return x*sigmoid(x); }

  function hadamardMLP(x,d1,d2,d3) {
    var n=d1.length,z=new Float32Array(n),i,inv=1/Math.sqrt(n);
    for(i=0;i<n;i++) z[i]=d1[i]*(i<x.length?x[i]:0);
    fwhtInPlace(z,0,n);for(i=0;i<n;i++)z[i]*=inv;
    for(i=0;i<n;i++)z[i]=silu(d2[i]*z[i]);
    fwhtInPlace(z,0,n);for(i=0;i<n;i++)z[i]*=inv;
    var out=new Float32Array(x.length);for(i=0;i<x.length;i++)out[i]=d3[i]*z[i];return out;
  }

  function logsumexp4(a,off,stride) {
    var i,v,m=-Infinity,s=0;for(i=0;i<4;i++){v=a[off+i*stride];if(v>m)m=v;}
    for(i=0;i<4;i++)s+=Math.exp(a[off+i*stride]-m);return m+Math.log(s);
  }

  function sinkhorn4(logits) {
    var a=new Float32Array(logits),it,r,c,lse;
    for(it=0;it<20;it++) {
      for(r=0;r<4;r++){lse=logsumexp4(a,r*4,1);for(c=0;c<4;c++)a[r*4+c]-=lse;}
      for(c=0;c<4;c++){lse=logsumexp4(a,c,4);for(r=0;r<4;r++)a[r*4+c]-=lse;}
    }
    for(r=0;r<16;r++)a[r]=Math.exp(a[r]);return a;
  }

  function concatLanes(lanes) {
    var C=lanes[0].length,o=new Float32Array(4*C),l,i;
    for(l=0;l<4;l++)for(i=0;i<C;i++)o[l*C+i]=lanes[l][i];return o;
  }

  function repeatEmbedding(e) {
    var a=[],l,v;for(l=0;l<4;l++){v=new Float32Array(e.length);v.set(e);a.push(v);}return a;
  }

  function weightedLanes(lanes,w) {
    var C=lanes[0].length,o=new Float32Array(C),i,l,s;
    for(i=0;i<C;i++){s=0;for(l=0;l<4;l++)s+=w[l]*lanes[l][i];o[i]=s;}return o;
  }

  /*
   * PERF: cos/sin of every RoPE angle for one position.  Same expressions as
   * the original per-element computation, evaluated once per token instead
   * of once per head per layer.  Float64Array keeps the doubles unrounded.
   */
  function ropeTable(hd,pos,theta) {
    var half=hd>>1,c=new Float64Array(half),s=new Float64Array(half),j,freq,angle;
    for(j=0;j<half;j++){freq=1/Math.pow(theta,(2*j)/hd);angle=pos*freq;c[j]=Math.cos(angle);s[j]=Math.sin(angle);}
    return {hd:hd,pos:pos,theta:theta,c:c,s:s};
  }

  function applyRoPEInPlace(x,heads,hd,pos,theta,rt) {
    var h,j,off,half=hd>>1,C,S,c,s,a,b;if(pos===0)return;
    if(!rt||rt.hd!==hd||rt.pos!==pos||rt.theta!==theta)rt=ropeTable(hd,pos,theta);
    C=rt.c;S=rt.s;
    for(h=0;h<heads;h++){off=h*hd;for(j=0;j<half;j++){c=C[j];s=S[j];
      a=x[off+j];b=x[off+half+j];x[off+j]=a*c-b*s;x[off+half+j]=b*c+a*s;}}
  }

  function roundSymmetric(x) { return x>=0?Math.floor(x+0.5):Math.ceil(x-0.5); }

  function LayerKV(nKV,hd,cap) {
    this.nKV=nKV;this.hd=hd;this.cap=cap;
    this.k=new Int8Array(nKV*cap*hd);this.v=new Int8Array(nKV*cap*hd);
    this.ks=new Float32Array(nKV*cap);this.vs=new Float32Array(nKV*cap);
    this.slotPos=new Int32Array(cap);this.reset();
  }
  LayerKV.prototype.reset=function(){var i;for(i=0;i<this.cap;i++)this.slotPos[i]=-1;};
  LayerKV.prototype._base=function(h,pos){return (h*this.cap+(pos%this.cap))*this.hd;};
  LayerKV.prototype._si=function(h,pos){return h*this.cap+(pos%this.cap);};
  function quantizeRowInto(src,srcOff,dst,dstOff,n) {
    var i,m=0,a,scale,q;for(i=0;i<n;i++){a=Math.abs(src[srcOff+i]);if(a>m)m=a;}
    if(m<1e-30){for(i=0;i<n;i++)dst[dstOff+i]=0;return 0;}
    scale=m/127;for(i=0;i<n;i++){q=roundSymmetric(src[srcOff+i]/scale);if(q>127)q=127;if(q<-127)q=-127;dst[dstOff+i]=q;}return scale;
  }
  LayerKV.prototype.store=function(pos,k,v){var h,off,base,si,slot=pos%this.cap;
    for(h=0;h<this.nKV;h++){off=h*this.hd;base=this._base(h,pos);si=this._si(h,pos);this.ks[si]=quantizeRowInto(k,off,this.k,base,this.hd);this.vs[si]=quantizeRowInto(v,off,this.v,base,this.hd);}this.slotPos[slot]=pos;};
  LayerKV.prototype.dotK=function(q,qOff,h,pos){var slot=pos%this.cap;if(this.slotPos[slot]!==pos)throw new Error("KV miss at pos "+pos);var base=this._base(h,pos),si=this._si(h,pos),i,s=0;for(i=0;i<this.hd;i++)s+=q[qOff+i]*this.k[base+i];return s*this.ks[si];};
  LayerKV.prototype.addV=function(out,outOff,h,pos,w){var base=this._base(h,pos),si=this._si(h,pos),scale=this.vs[si]*w,i;for(i=0;i<this.hd;i++)out[outOff+i]+=this.v[base+i]*scale;};

  /*
   * PERF: LayerKV.dotK/addV inlined.  Arithmetic and its order are exactly
   * those of the methods: s=sum(q*k) -> *kScale -> *scale, and
   * out += v*(vScale*w).
   */
  function attentionFromCache(q,cache,pos,nHeads,nKV,hd) {
    var out=new Float32Array(nHeads*hd),start=Math.max(0,pos-cache.cap+1),n=pos-start+1;
    var h,kvh,t,j,m,s,e,scale=1/Math.sqrt(hd),scores=new Float32Array(n),w;
    var K=cache.k,V=cache.v,KS=cache.ks,VS=cache.vs,slotPos=cache.slotPos,cap=cache.cap;
    var i,qo,si,base,slot,vsc;
    for(j=0;j<n;j++){t=start+j;if(slotPos[t%cap]!==t)throw new Error("KV miss at pos "+t);}
    for(h=0;h<nHeads;h++) {
      kvh=Math.floor(h/(nHeads/nKV));m=-Infinity;qo=h*hd;
      for(j=0;j<n;j++){
        slot=(start+j)%cap;si=kvh*cap+slot;base=si*hd;
        s=0;for(i=0;i<hd;i++)s+=q[qo+i]*K[base+i];
        s=s*KS[si]*scale;scores[j]=s;if(s>m)m=s;
      }
      s=0;for(j=0;j<n;j++){e=Math.exp(scores[j]-m);scores[j]=e;s+=e;}
      for(j=0;j<n;j++){
        w=scores[j]/s;
        slot=(start+j)%cap;si=kvh*cap+slot;base=si*hd;vsc=VS[si]*w;
        for(i=0;i<hd;i++)out[qo+i]+=V[base+i]*vsc;
      }
    }
    return out;
  }

  function EngramRing(depth,C) {
    if((C&31)!==0)throw new Error("native Engram vraw cache expects C multiple of 32");
    this.depth=depth;this.C=C;this.blocks=C>>>5;
    this.data=new Int8Array(depth*C);
    this.scale=new Float32Array(depth*this.blocks);
    this.pos=new Int32Array(depth);
    this.reset();
  }
  EngramRing.prototype.reset=function(){
    var i;
    for(i=0;i<this.depth;i++)this.pos[i]=-1;
  };
  function nativeLroundf(x){
    return x>=0?Math.floor(x+0.5):Math.ceil(x-0.5);
  }
  EngramRing.prototype.store=function(pos,v){
    var slot=pos%this.depth,base=slot*this.C,sbase=slot*this.blocks;
    var b,i,j,m,sc,inv,q;
    for(b=0;b<this.blocks;b++){
      j=b<<5;m=0;
      for(i=0;i<32;i++){
        var ax=Math.abs(v[j+i]);
        if(ax>m)m=ax;
      }
      sc=m>0?Math.fround(m/127):Math.fround(1);
      inv=Math.fround(1/sc);
      this.scale[sbase+b]=sc;
      for(i=0;i<32;i++){
        q=nativeLroundf(Math.fround(v[j+i]*inv));
        this.data[base+j+i]=(q<<24)>>24;
      }
    }
    this.pos[slot]=pos;
  };
  EngramRing.prototype.addScaled=function(pos,taps,tapOff,out){
    var slot=pos%this.depth,i,b,j,base,sbase,sc;
    if(this.pos[slot]!==pos)return;
    base=slot*this.C;sbase=slot*this.blocks;
    for(b=0;b<this.blocks;b++){
      j=b<<5;sc=this.scale[sbase+b];
      for(i=0;i<32;i++)
        out[j+i]+=taps[tapOff+j+i]*(this.data[base+j+i]*sc);
    }
  };

  function engramHash(history,pos,order,tableIndex,slots) {
    var seed=imul32(ENGRAM_SEED,tableIndex+1)>>>0,acc=seed,j,tok;
    for(j=0;j<order;j++) {
      tok=history[pos-j]>>>0;
      acc=imul32((acc^tok)>>>0,ENGRAM_PRIME)>>>0;
    }
    acc=(acc^(acc>>>15))>>>0;
    return acc%slots;
  }

  function argmax(a) { var i,b=0,v=-Infinity;for(i=0;i<a.length;i++){if(a[i]>v){v=a[i];b=i;}}return b; }
  function topK(a,k) {
    var ids=[],vals=[],i,j,v,p;
    for(i=0;i<a.length;i++){v=a[i];p=ids.length;for(j=0;j<ids.length;j++)if(v>vals[j]){p=j;break;}if(p<k){ids.splice(p,0,i);vals.splice(p,0,v);if(ids.length>k){ids.pop();vals.pop();}}}
    var out=[];for(i=0;i<ids.length;i++)out.push({id:ids[i],logit:vals[i]});return out;
  }

  function Engine(model) {
    var i,l,s,g=model.geometry,self=this;
    this.model=model;this.g=g;this.tmap={};
    for(i=0;i<model.tensors.length;i++)this.tmap[model.tensors[i].label]=model.tensors[i];
    function T(name){var t=self.tmap[name];if(!t)throw new Error("tensor not found: "+name);return t;}
    function F(name){return readFP(model,T(name));}

    if(g.dModel!==512||g.nHeads!==8||g.nKV!==4||g.headDim!==64||g.lanes!==4||g.nLayers!==27)
      throw new Error("This alpha expects Needle 2 base geometry (512/8/4/64/4/27)");
    if(g.kvBits!==8)throw new Error("This alpha expects int8 KV cache");

    /*
     * NativeQ A1c uses the native 4+8+16 segmented int8 codebook LUTs.
     * The old Float32 CQ lookup tables remain intentionally disabled.
     */
    this.lut2=null;this.lut4=null;this.embedding=T("embedding");
    this.layers=[];
    for(l=0;l<g.nLayers;l++)this.layers.push({
      normIn:F("layer."+l+".norm_in"),q:T("layer."+l+".q_proj"),k:T("layer."+l+".k_proj"),v:T("layer."+l+".v_proj"),
      qNorm:F("layer."+l+".q_norm"),kNorm:F("layer."+l+".k_norm"),gate:T("layer."+l+".gate_proj"),out:T("layer."+l+".out_proj"),
      postNorm:F("layer."+l+".post_norm"),attnGate:F("layer."+l+".attn_gate")[0],preHada:F("layer."+l+".pre_hada"),
      d1:F("layer."+l+".d1"),d2:F("layer."+l+".d2"),d3:F("layer."+l+".d3")
    });
    this.mhc={aPre:F("mhc.a_pre"),aPost:F("mhc.a_post"),aRes:F("mhc.a_res"),
      bPre:F("mhc.b_pre"),bPost:F("mhc.b_post"),bRes:F("mhc.b_res"),
      phiPre:T("mhc.phi_pre"),phiPost:T("mhc.phi_post"),phiRes:T("mhc.phi_res")};
    this.engrams=[];this.siteByLayer={};
    for(s=0;s<g.nSites;s++) {
      var es={layer:g.sites[s],tables:T("engram."+s+".tables"),key:T("engram."+s+".key_proj"),value:T("engram."+s+".value_proj"),taps:F("engram."+s+".taps")};
      this.engrams.push(es);this.siteByLayer["$"+es.layer]=s;
    }
    this.finalNorm=F("final_norm");
    /* A1Z_PUREJS_CONFIDENCE_V1
     *
     * Model-derived Pure-JS confidence head.
     * This is intentionally NOT advertised as native-ARM calibrated.
     *
     * optional.401 : [8,512] learned probes
     * optional.402 : [1,4096] projection
     * optional.403 : [1] bias
     */
    this.confProbes=F("optional.401");
    this.confProj=F("optional.402");
    this.confBias=F("optional.403")[0];

    if(this.confProbes.length!==4096)
      throw new Error(
        "confidence probes expected 4096 values, got "+
        this.confProbes.length
      );

    if(this.confProj.length!==4096)
      throw new Error(
        "confidence projection expected 4096 values, got "+
        this.confProj.length
      );

    this._confMax=new Float32Array(8);
    this._confDenom=new Float32Array(8);
    this._confSum=new Float32Array(4096);
    this._confTmp=new Float32Array(g.dModel);
    this._confCellCount=0;
    this._confActive=false;
    this.tokenizer=new NeedleTokenizer.Tokenizer(model);
    this.caches=[];for(l=0;l<g.nLayers;l++)this.caches.push(new LayerKV(g.nKV,g.headDim,g.kvWindow));
    this.engramDepth=1+(g.taps-1)*g.dilation;
    this.engramRings=[];for(s=0;s<g.nSites;s++)this.engramRings.push(new EngramRing(this.engramDepth,g.dModel));
    this.history=new Int32Array(g.maxSeq);
    this.position=0;
  }

  Engine.prototype.reset=function(){var i;this.position=0;for(i=0;i<this.caches.length;i++)this.caches[i].reset();for(i=0;i<this.engramRings.length;i++)this.engramRings[i].reset();this.confidenceReset();};

  Engine.prototype.confidenceReset=function() {
    var p,i;

    this._confCellCount=0;
    this._confActive=true;

    for(p=0;p<8;p++) {
      this._confMax[p]=-Infinity;
      this._confDenom[p]=0;
    }

    for(i=0;i<4096;i++)
      this._confSum[i]=0;
  };

  Engine.prototype._confidenceAdd=function(x,tokenId) {
    if(!this._confActive)
      return;

    /*
     * Match the model's pad semantics: token id 0 does not
     * contribute to the confidence pool.
     */
    if((tokenId|0)===0)
      return;

    var P=8;
    var D=512;
    var invSqrtD=f32(1/Math.sqrt(D));

    var p,i,base;
    var dot,score,oldMax,scale,w;

    for(p=0;p<P;p++) {
      base=p*D;
      dot=f32(0);

      for(i=0;i<D;i++) {
        dot=f32(
          dot+
          f32(
            x[i]*
            this.confProbes[base+i]
          )
        );
      }

      score=f32(
        dot*invSqrtD
      );

      oldMax=this._confMax[p];

      if(score>oldMax) {
        if(oldMax===-Infinity)
          scale=f32(0);
        else
          scale=f32(
            Math.exp(
              f32(oldMax-score)
            )
          );

        for(i=0;i<D;i++)
          this._confSum[base+i]=f32(
            this._confSum[base+i]*
            scale
          );

        this._confDenom[p]=f32(
          this._confDenom[p]*
          scale
        );

        this._confMax[p]=score;
      }

      w=f32(
        Math.exp(
          f32(
            score-
            this._confMax[p]
          )
        )
      );

      this._confDenom[p]=f32(
        this._confDenom[p]+w
      );

      for(i=0;i<D;i++) {
        this._confSum[base+i]=f32(
          this._confSum[base+i]+
          f32(w*x[i])
        );
      }
    }

    this._confCellCount++;
  };

  Engine.prototype.confidenceValue=function() {
    if(!this._confActive ||
       this._confCellCount===0)
      return null;

    var P=8;
    var D=512;

    var pooled=
      new Float32Array(P*D);

    var p,i,base,d;

    for(p=0;p<P;p++) {
      base=p*D;
      d=this._confDenom[p];

      for(i=0;i<D;i++) {
        pooled[base+i]=
          d>0 ?
          f32(
            this._confSum[base+i]/d
          ) :
          f32(0);
      }
    }

    var z=f32(0);

    for(i=0;i<4096;i++) {
      z=f32(
        z+
        f32(
          pooled[i]*
          this.confProj[i]
        )
      );
    }

    z=f32(
      z+
      this.confBias
    );

    var q,c;

    if(z>=0) {
      q=f32(
        Math.exp(
          f32(-z)
        )
      );

      c=f32(
        1/f32(1+q)
      );
    } else {
      q=f32(
        Math.exp(z)
      );

      c=f32(
        q/f32(1+q)
      );
    }

    return c;
  };


  Engine.prototype.nativeCodebookInfo=function(){
    var c=nativeWeightCodebooks(this.model),q=c.q;
    function arr(a,b){var o=[],i;for(i=a;i<b;i++)o.push(q[i]);return o;}
    return {
      cq2:{scale:c.scales[2],maxAbs:c.maxAbs[2],lut:arr(0,4)},
      cq3:{scale:c.scales[3],maxAbs:c.maxAbs[3],lut:arr(4,12)},
      cq4:{scale:c.scales[4],maxAbs:c.maxAbs[4],lut:arr(12,28)}
    };
  };

  Engine.prototype.memoryInfo=function(){var i,kv=0,er=0;for(i=0;i<this.caches.length;i++)kv+=this.caches[i].k.byteLength+this.caches[i].v.byteLength+this.caches[i].ks.byteLength+this.caches[i].vs.byteLength+this.caches[i].slotPos.byteLength;for(i=0;i<this.engramRings.length;i++)er+=this.engramRings[i].data.byteLength+this.engramRings[i].pos.byteLength;return {modelBytes:this.model.buffer.byteLength,kvBytes:kv,engramRingBytes:er};};

  Engine.prototype._engramPair=function(siteIndex,pos) {
    var g=this.g,site=this.engrams[siteIndex],ring=this.engramRings[siteIndex];
    var e=new Float32Array(g.dModel),heads=g.nTables/g.nOrders,ti,oi,order,rowIndex,row,j,i;
    for(ti=0;ti<g.nTables;ti++) {
      oi=Math.floor(ti/heads);order=g.orders[oi];
      if(pos>=order-1) {
        rowIndex=ti*g.slots+engramHash(this.history,pos,order,ti,g.slots);
        row=cqRow(this.model,site.tables,rowIndex);
        e.set(row,ti*g.subDim);
      }
    }
    var prep=prepareCQInput(e,cqDesc(site.key));
    var ek=cqMatvecRows(this.model,site.key,prep,0,g.dModel,this.lut2,this.lut4);
    var rawV=cqMatvecRows(this.model,site.value,prep,0,g.dModel,this.lut2,this.lut4);
    ring.store(pos,rawV);
    var ev=new Float32Array(g.dModel),p;
    for(j=0;j<g.taps;j++){p=pos-j*g.dilation;if(p>=0)ring.addScaled(p,site.taps,j*g.dModel,ev);}
    return {k:ek,v:ev};
  };

  Engine.prototype._layerBlock=function(u,layerIndex,pos,engramPairs) {
    var g=this.g,L=this.layers[layerIndex],x=u,i,siteIndex=this.siteByLayer["$"+layerIndex],dot=0,alpha;
    if(typeof siteIndex==="number") {
      var pair=engramPairs[siteIndex],xu=rmsUnit(u),ku=rmsUnit(pair.k);
      for(i=0;i<g.dModel;i++)dot+=xu[i]*ku[i];
      alpha=sigmoid(dot/Math.sqrt(g.dModel));x=new Float32Array(g.dModel);for(i=0;i<g.dModel;i++)x[i]=u[i]+alpha*pair.v[i];
    }
    var skip=x,norm=zcRmsNorm(x,L.normIn),prep=prepareCQInput(norm,cqDesc(L.q));
    var q=cqMatvecRows(this.model,L.q,prep,0,g.dModel,this.lut2,this.lut4);
    var k=cqMatvecRows(this.model,L.k,prep,0,g.nKV*g.headDim,this.lut2,this.lut4);
    var v=cqMatvecRows(this.model,L.v,prep,0,g.nKV*g.headDim,this.lut2,this.lut4);
    var gate=cqMatvecRows(this.model,L.gate,prep,0,g.dModel,this.lut2,this.lut4);
    q=zcRmsNormHeads(q,L.qNorm,g.headDim);k=zcRmsNormHeads(k,L.kNorm,g.headDim);
    applyRoPEInPlace(q,g.nHeads,g.headDim,pos,g.ropeTheta,this._rope);applyRoPEInPlace(k,g.nKV,g.headDim,pos,g.ropeTheta,this._rope);
    this.caches[layerIndex].store(pos,k,v);
    var att=attentionFromCache(q,this.caches[layerIndex],pos,g.nHeads,g.nKV,g.headDim);
    for(i=0;i<g.dModel;i++)att[i]*=sigmoid(gate[i]);
    var proj=cqApply(this.model,L.out,att,this.lut2,this.lut4),post=zcRmsNorm(proj,L.postNorm);
    var after=new Float32Array(g.dModel),ag=sigmoid(L.attnGate);for(i=0;i<g.dModel;i++)after[i]=skip[i]+ag*post[i];
    var preh=zcRmsNorm(after,L.preHada),mlp=hadamardMLP(preh,L.d1,L.d2,L.d3),out=new Float32Array(g.dModel);
    for(i=0;i<g.dModel;i++)out[i]=after[i]+mlp[i];return out;
  };

  Engine.prototype.forwardToken=function(tokenId,wantLogits,progress) {
    var g=this.g,pos=this.position,t0=Date.now(),i,l,j,s,owner,preRaw,postRaw,resRaw,hpre,hpost,hlog,hres,u,block,y,newLanes,laneOut,nx,prepPhi;
    if(pos>=g.maxSeq)throw new Error("max sequence reached: "+g.maxSeq);
    tokenId=tokenId|0;if(tokenId<0||tokenId>=g.vocab)throw new Error("token id OOB: "+tokenId);
    this.history[pos]=tokenId;
    if(pos>0 && (!this._rope||this._rope.pos!==pos))this._rope=ropeTable(g.headDim,pos,g.ropeTheta);

    var engramPairs=[],es;
    for(es=0;es<g.nSites;es++)engramPairs[es]=this._engramPair(es,pos);

    var emb=cqRow(this.model,this.embedding,tokenId),scale=Math.sqrt(g.dModel);
    for(i=0;i<g.dModel;i++)emb[i]*=scale;
    var lanes=repeatEmbedding(emb),mhc=this.mhc;
    if(this._confActive)
      this._confidenceAdd(
        emb,
        tokenId
      );

    for(l=0;l<g.nLayers;l++) {
      nx=rmsUnit(concatLanes(lanes));prepPhi=prepareCQInput(nx,cqDesc(mhc.phiPre));
      preRaw=cqMatvecRows(this.model,mhc.phiPre,prepPhi,l*4,4,this.lut2,this.lut4);
      postRaw=cqMatvecRows(this.model,mhc.phiPost,prepPhi,l*4,4,this.lut2,this.lut4);
      resRaw=cqMatvecRows(this.model,mhc.phiRes,prepPhi,l*16,16,this.lut2,this.lut4);
      owner=l&3;hpre=new Float32Array(4);hpost=new Float32Array(4);
      for(i=0;i<4;i++) {
        hpre[i]=sigmoid(mhc.aPre[l]*preRaw[i]+mhc.bPre[l*4+i]+(i===owner?4:-4));
        hpost[i]=2*sigmoid(mhc.aPost[l]*postRaw[i]+mhc.bPost[l*4+i]+(i===owner?0:-4));
      }
      hlog=new Float32Array(16);for(i=0;i<16;i++)hlog[i]=mhc.aRes[l]*resRaw[i]+mhc.bRes[l*16+i];hres=sinkhorn4(hlog);
      u=weightedLanes(lanes,hpre);block=this._layerBlock(u,l,pos,engramPairs);y=new Float32Array(g.dModel);for(i=0;i<g.dModel;i++)y[i]=block[i]-u[i];
      newLanes=[];
      for(i=0;i<4;i++) { laneOut=new Float32Array(g.dModel);for(j=0;j<g.dModel;j++){s=0;for(var k2=0;k2<4;k2++)s+=hres[i*4+k2]*lanes[k2][j];laneOut[j]=s+hpost[i]*y[j];}newLanes.push(laneOut); }
      lanes=newLanes;
      if(this._confActive) {
        var _cm=this._confTmp;
        var _ci;

        for(_ci=0;_ci<g.dModel;_ci++) {
          _cm[_ci]=f32(
            0.25*(
              lanes[0][_ci]+
              lanes[1][_ci]+
              lanes[2][_ci]+
              lanes[3][_ci]
            )
          );
        }

        this._confidenceAdd(
          _cm,
          tokenId
        );
      }
      if(progress && (l===0||l===2||l===7||l===15||l===23||l===26))progress({layer:l,position:pos,elapsedMs:Date.now()-t0});
    }

    var mean=new Float32Array(g.dModel);for(i=0;i<g.dModel;i++)mean[i]=0.25*(lanes[0][i]+lanes[1][i]+lanes[2][i]+lanes[3][i]);
    var hidden=zcRmsNorm(mean,this.finalNorm),logits=null;
    if(wantLogits)logits=cqApply(this.model,this.embedding,hidden,this.lut2,this.lut4);
    this.position=pos+1;
    return {tokenId:tokenId,position:pos,elapsedMs:Date.now()-t0,hidden:hidden,logits:logits};
  };

  Engine.prototype.argmax=function(logits){return argmax(logits);};
  Engine.prototype.topK=function(logits,k){return topK(logits,k||5);};


  Engine.prototype.debugLayer0Token=function(tokenId) {
    var g=this.g,pos=0,i,j,s,owner=0,k2;
    this.reset();
    this.history[0]=tokenId|0;

    var emb=cqRow(this.model,this.embedding,tokenId|0),scale=Math.sqrt(g.dModel);
    for(i=0;i<g.dModel;i++)emb[i]*=scale;
    var lanes=repeatEmbedding(emb),mhc=this.mhc,l=0;

    var nx=rmsUnit(concatLanes(lanes));
    var prepPhi=prepareCQInput(nx,cqDesc(mhc.phiPre));
    var preRaw=cqMatvecRows(this.model,mhc.phiPre,prepPhi,0,4,this.lut2,this.lut4);
    var postRaw=cqMatvecRows(this.model,mhc.phiPost,prepPhi,0,4,this.lut2,this.lut4);
    var resRaw=cqMatvecRows(this.model,mhc.phiRes,prepPhi,0,16,this.lut2,this.lut4);

    var hpre=new Float32Array(4),hpost=new Float32Array(4);
    for(i=0;i<4;i++) {
      hpre[i]=sigmoid(mhc.aPre[0]*preRaw[i]+mhc.bPre[i]+(i===owner?4:-4));
      hpost[i]=2*sigmoid(mhc.aPost[0]*postRaw[i]+mhc.bPost[i]+(i===owner?0:-4));
    }
    var hlog=new Float32Array(16);
    for(i=0;i<16;i++)hlog[i]=mhc.aRes[0]*resRaw[i]+mhc.bRes[i];
    var hres=sinkhorn4(hlog);
    var u=weightedLanes(lanes,hpre);
    var block=this._layerBlock(u,0,pos,[]);
    var y=new Float32Array(g.dModel);
    for(i=0;i<g.dModel;i++)y[i]=block[i]-u[i];

    var newLanes=[];
    for(i=0;i<4;i++) {
      var laneOut=new Float32Array(g.dModel);
      for(j=0;j<g.dModel;j++) {
        s=0;
        for(k2=0;k2<4;k2++)s+=hres[i*4+k2]*lanes[k2][j];
        laneOut[j]=s+hpost[i]*y[j];
      }
      newLanes.push(laneOut);
    }

    var mean=new Float32Array(g.dModel),sum=0,absSum=0,maxAbs=0,v;
    for(i=0;i<g.dModel;i++) {
      v=0.25*(newLanes[0][i]+newLanes[1][i]+newLanes[2][i]+newLanes[3][i]);
      mean[i]=v; sum+=v; absSum+=Math.abs(v); if(Math.abs(v)>maxAbs)maxAbs=Math.abs(v);
    }
    var out={first8:Array.prototype.slice.call(mean,0,8),sum:sum,absSum:absSum,maxAbs:maxAbs};
    this.reset();
    return out;
  };



  Engine.prototype.debugLayer0StagesA1S=function(tokenId) {
    var g=this.g,pos=0,i,j,s,owner=0,k2,h,si,base;
    var C={};

    function ck(x) {
      var i,v,sum=0,absSum=0,maxAbs=0,a,first8=[];
      for(i=0;i<x.length;i++) {
        v=Number(x[i]);
        sum+=v;
        a=Math.abs(v);
        absSum+=a;
        if(a>maxAbs)maxAbs=a;
        if(i<8)first8.push(v);
      }
      return {
        length:x.length,
        sum:sum,
        absSum:absSum,
        maxAbs:maxAbs,
        first8:first8
      };
    }

    this.reset();
    this.history[0]=tokenId|0;

    var emb=cqRow(this.model,this.embedding,tokenId|0);
    var scale=Math.sqrt(g.dModel);
    for(i=0;i<g.dModel;i++)emb[i]*=scale;
    C.embeddingScaled=ck(emb);

    var lanes=repeatEmbedding(emb),mhc=this.mhc,l=0;

    var nx=rmsUnit(concatLanes(lanes));
    C.mhcInputRms=ck(nx);

    var prepPhi=prepareCQInput(nx,cqDesc(mhc.phiPre));

    var preRaw=cqMatvecRows(
      this.model,mhc.phiPre,prepPhi,0,4,this.lut2,this.lut4
    );
    var postRaw=cqMatvecRows(
      this.model,mhc.phiPost,prepPhi,0,4,this.lut2,this.lut4
    );
    var resRaw=cqMatvecRows(
      this.model,mhc.phiRes,prepPhi,0,16,this.lut2,this.lut4
    );

    C.phiPreRaw=ck(preRaw);
    C.phiPostRaw=ck(postRaw);
    C.phiResRaw=ck(resRaw);

    var hpre=new Float32Array(4);
    var hpost=new Float32Array(4);

    for(i=0;i<4;i++) {
      hpre[i]=sigmoid(
        mhc.aPre[0]*preRaw[i]+mhc.bPre[i]+
        (i===owner?4:-4)
      );
      hpost[i]=2*sigmoid(
        mhc.aPost[0]*postRaw[i]+mhc.bPost[i]+
        (i===owner?0:-4)
      );
    }

    C.hpre=ck(hpre);
    C.hpost=ck(hpost);

    var hlog=new Float32Array(16);
    for(i=0;i<16;i++)
      hlog[i]=mhc.aRes[0]*resRaw[i]+mhc.bRes[i];

    var hres=sinkhorn4(hlog);
    C.hres=ck(hres);

    var u=weightedLanes(lanes,hpre);
    C.u=ck(u);

    if(typeof this.siteByLayer["$0"]==="number")
      throw new Error(
        "A1S assumption failed: layer0 has an engram site"
      );

    var L=this.layers[0];
    var skip=u;

    var norm=zcRmsNorm(u,L.normIn);
    C.normIn=ck(norm);

    /*
     * Match _layerBlock exactly:
     * one activation preparation based on q geometry,
     * reused by q/k/v/gate.
     */
    var prep=prepareCQInput(norm,cqDesc(L.q));

    var qRaw=cqMatvecRows(
      this.model,L.q,prep,0,g.dModel,this.lut2,this.lut4
    );
    var kRaw=cqMatvecRows(
      this.model,L.k,prep,0,g.nKV*g.headDim,
      this.lut2,this.lut4
    );
    var v=cqMatvecRows(
      this.model,L.v,prep,0,g.nKV*g.headDim,
      this.lut2,this.lut4
    );
    var gate=cqMatvecRows(
      this.model,L.gate,prep,0,g.dModel,
      this.lut2,this.lut4
    );

    C.qRaw=ck(qRaw);
    C.kRaw=ck(kRaw);
    C.vRaw=ck(v);
    C.gateRaw=ck(gate);

    var q=zcRmsNormHeads(qRaw,L.qNorm,g.headDim);
    var k=zcRmsNormHeads(kRaw,L.kNorm,g.headDim);

    C.qNorm=ck(q);
    C.kNorm=ck(k);

    applyRoPEInPlace(
      q,g.nHeads,g.headDim,pos,g.ropeTheta
    );
    applyRoPEInPlace(
      k,g.nKV,g.headDim,pos,g.ropeTheta
    );

    C.qRope=ck(q);
    C.kRope=ck(k);

    var cache=this.caches[0];
    cache.store(pos,k,v);

    var ks=new Float32Array(g.nKV);
    var vs=new Float32Array(g.nKV);
    var vdq=new Float32Array(g.nKV*g.headDim);

    for(h=0;h<g.nKV;h++) {
      si=cache._si(h,pos);
      base=cache._base(h,pos);

      ks[h]=cache.ks[si];
      vs[h]=cache.vs[si];

      for(i=0;i<g.headDim;i++)
        vdq[h*g.headDim+i]=
          cache.v[base+i]*cache.vs[si];
    }

    C.kvKScales=ck(ks);
    C.kvVScales=ck(vs);
    C.vCacheDequant=ck(vdq);

    var att=attentionFromCache(
      q,cache,pos,g.nHeads,g.nKV,g.headDim
    );

    /*
     * pos=0 => exactly one KV position per head,
     * therefore every softmax weight is exactly 1.
     */
    var aw=new Float32Array(g.nHeads);
    for(i=0;i<g.nHeads;i++)aw[i]=1;

    C.attentionWeights=ck(aw);
    C.attentionBeforeGateInt8=ck(att);

    /*
     * _layerBlock gates attention in-place.
     * Work on a copy only so we can retain the pre-gate checkpoint.
     */
    var gated=new Float32Array(att);
    for(i=0;i<g.dModel;i++)
      gated[i]*=sigmoid(gate[i]);

    C.attentionAfterGate=ck(gated);

    var proj=cqApply(
      this.model,L.out,gated,this.lut2,this.lut4
    );
    C.attentionProjected=ck(proj);

    var post=zcRmsNorm(proj,L.postNorm);
    C.postNorm=ck(post);

    var after=new Float32Array(g.dModel);
    var ag=sigmoid(L.attnGate);

    for(i=0;i<g.dModel;i++)
      after[i]=skip[i]+ag*post[i];

    C.afterAttention=ck(after);

    var preh=zcRmsNorm(after,L.preHada);
    C.preHada=ck(preh);

    var mlp=hadamardMLP(
      preh,L.d1,L.d2,L.d3
    );
    C.mlp=ck(mlp);

    var block=new Float32Array(g.dModel);
    var y=new Float32Array(g.dModel);

    for(i=0;i<g.dModel;i++) {
      block[i]=after[i]+mlp[i];
      y[i]=block[i]-u[i];
    }

    C.block=ck(block);
    C.y=ck(y);

    var newLanes=[];

    for(i=0;i<4;i++) {
      var laneOut=new Float32Array(g.dModel);

      for(j=0;j<g.dModel;j++) {
        s=0;
        for(k2=0;k2<4;k2++)
          s+=hres[i*4+k2]*lanes[k2][j];

        laneOut[j]=s+hpost[i]*y[j];
      }

      newLanes.push(laneOut);
    }

    var mean=new Float32Array(g.dModel);

    for(i=0;i<g.dModel;i++)
      mean[i]=0.25*(
        newLanes[0][i]+
        newLanes[1][i]+
        newLanes[2][i]+
        newLanes[3][i]
      );

    C.laneMean=ck(mean);

    this.reset();
    return C;
  };

  Engine.prototype.debugCQGeometry=function() {
    var L=this.layers[0];
    function x(t){return {label:t.label,bits:t.bits,group:t.group,shape:[t.shape[0],t.shape[1]]};}
    return {
      headDim:this.g.headDim,
      q:x(L.q), k:x(L.k), v:x(L.v), gate:x(L.gate), out:x(L.out),
      phiPre:x(this.mhc.phiPre), phiPost:x(this.mhc.phiPost), phiRes:x(this.mhc.phiRes)
    };
  };

  Engine.prototype.debugA1HRowProbe=function() {
    var t=this.layers[0].q,d=cqDesc(t),x=new Float32Array(512),i,g,k,b,bp;
    if(t.bits!==2 || d.group!==128 || d.input!==512)
      throw new Error("A1H expects layer0 q_proj CQ2 group128 512x512");

    /* Same exact deterministic binary-fraction vector used by the native
     * quant dump / row probe.  This intentionally bypasses mHC and the rest
     * of the transformer so only quantize_activation + CQ2 row math remain.
     */
    for(i=0;i<512;i++)x[i]=((((i*37+11)%257)-128)/32);

    var p=prepareCQInput(x,d),wc=nativeWeightCodebooks(this.model);
    var cbq=wc.q,cbScale=wc.scales[2],u8=this.model.bytes,dv=this.model.dv;
    var groups=[],acc=f32(0),row=0;

    for(g=0;g<d.nGroups;g++) {
      bp=t.offset+row*d.rowBytes+g*d.groupBytes;
      var xoff=g*d.group;
      var dot=nativeIntDotCQ2(u8,bp,p.q,xoff,d.group,cbq);
      var actScale=p.scales[g];
      var wScale=fp16ToF32(dv.getUint16(d.normsBase+(row*d.nGroups+g)*2,true));
      var fDot=f32(dot);
      var afterAct=f32(fDot*actScale);
      var afterCb=f32(afterAct*cbScale);
      var term=f32(afterCb*wScale);
      acc=f32(acc+term);

      var packedFirst16=[],weightBytesFirst8=[],weightInt8First16=[];
      for(i=0;i<16;i++)packedFirst16.push(p.q[xoff+i*4]);
      for(i=0;i<8;i++)weightBytesFirst8.push(u8[bp+i]);
      for(i=0;i<4;i++) {
        b=u8[bp+i];
        weightInt8First16.push(cbq[b&3],cbq[(b>>>2)&3],cbq[(b>>>4)&3],cbq[(b>>>6)&3]);
      }
      groups.push({
        group:g,actScale:actScale,wScale:wScale,intDot:dot,
        fDot:fDot,afterAct:afterAct,afterCb:afterCb,term:term,running:acc,
        packedFirst16:packedFirst16,weightBytesFirst8:weightBytesFirst8,
        weightInt8First16:weightInt8First16
      });
    }

    var full=cqMatvecRows(this.model,t,p,0,1,this.lut2,this.lut4)[0];
    return {
      tensor:t.label,bits:t.bits,group:d.group,nGroups:d.nGroups,
      rowBytes:d.rowBytes,offset:t.offset,normsBase:d.normsBase,
      codebookScale:cbScale,codebookLUT:Array.prototype.slice.call(cbq,0,4),
      actScales:Array.prototype.slice.call(p.scales,0),
      groups:groups,full:full,groupAccum:acc,
      inputFirst16:Array.prototype.slice.call(x,0,16)
    };
  };


  function a1iF32HashHex(vec) {
    var ab=new ArrayBuffer(4),dv=new DataView(ab),h=2166136261>>>0,i,b;
    for(i=0;i<vec.length;i++) {
      dv.setFloat32(0,vec[i],true);
      for(b=0;b<4;b++) h=Math.imul((h^dv.getUint8(b))>>>0,16777619)>>>0;
    }
    var z=(h>>>0).toString(16); while(z.length<8)z="0"+z;
    return z;
  }

  Engine.prototype.debugA1IFullQProj=function() {
    var t=this.layers[0].q,d=cqDesc(t),x=new Float32Array(512),i;
    if(t.bits!==2 || d.group!==128 || d.input!==512 || t.shape[0]!==512)
      throw new Error("A1I expects layer0 q_proj CQ2 group128 512x512");
    for(i=0;i<512;i++)x[i]=((((i*37+11)%257)-128)/32);
    var p=prepareCQInput(x,d);
    var y=cqMatvecRows(this.model,t,p,0,512,this.lut2,this.lut4);
    var sumF32=f32(0),sumDouble=0,absSum=0,maxAbs=0,maxRow=0,v,a;
    for(i=0;i<y.length;i++) {
      v=y[i]; sumF32=f32(sumF32+v); sumDouble+=v; a=Math.abs(v); absSum+=a;
      if(a>maxAbs){maxAbs=a;maxRow=i;}
    }
    var sel=[0,1,2,3,7,31,63,127,128,255,256,511],selected=[];
    for(i=0;i<sel.length;i++)selected.push([sel[i],y[sel[i]]]);
    return {
      tensor:t.label,bits:t.bits,group:d.group,nGroups:d.nGroups,rows:y.length,
      hash:a1iF32HashHex(y),sumF32:sumF32,sumDouble:sumDouble,absSum:absSum,
      maxAbs:maxAbs,maxRow:maxRow,
      first8:Array.prototype.slice.call(y,0,8),
      last8:Array.prototype.slice.call(y,y.length-8),selected:selected,
      row0:y[0]
    };
  };

  Engine.prototype.debugNativeQuantSelfTest=function() {
    var x=new Float32Array(512),i,b,p;
    for(i=0;i<512;i++) x[i]=((((i*37+11)%257)-128)/32);
    p=prepareCQInput(x,{inPad:512,group:128,nGroups:4});
    var packed=new Int8Array(512);
    for(b=0;b<4;b++) {
      for(i=0;i<128;i++)
        packed[b*128+((i&3)*32+(i>>>2))]=p.q[b*128+i];
    }
    var expectedScales=new Float32Array([0.07266373932, 0.04471615329, 0.07825326174, 0.05589519069]);
    var expectedPacked=new Int8Array([-11, -18, 13, -10, -4, -49, -10, -10, 1, 68, -10, -10, -10, -49, -10, -49, 13, -10, -10, -10, -10, 29, -10, 29, -49, -10, -10, -49, -10, -88, 29, 29, -2, -29, 10, 49, 10, 10, 10, 10, -29, -29, 10, -68, 10, -29, -29, 10, 10, 10, 49, -29, -29, 10, 10, -29, 10, -29, 10, -29, 10, 10, 10, 10, 6, -49, -10, -10, 29, -10, -10, 29, -10, -49, -10, -49, 29, -127, 29, 29, -10, 29, -10, 29, 29, 29, -49, -49, -10, -88, 29, 29, -10, 29, -49, 68, 10, 10, 10, 10, -29, 10, -29, 10, 10, -29, -29, 10, 10, 10, -29, 49, -29, 10, 10, -29, -29, -29, -68, 10, 10, 10, 10, 10, -29, 88, 49, 10, 11, 50, 5, 0, -23, 63, -32, 0, 50, -63, 32, 63, 32, 127, -32, 63, 5, 0, -32, 0, -32, -63, 32, 0, -32, 63, -32, 63, -32, 127, 32, -63, -19, 32, 63, -32, 0, -32, 0, -32, 0, 95, -63, 32, 0, 32, 0, 32, 0, -32, -63, -95, 0, 32, 0, 32, 0, 32, -63, -32, 0, -32, 0, -32, -7, 63, -32, 0, -32, -63, -32, -63, 32, 127, -32, 63, -32, 127, -32, -127, -32, -63, 32, 0, 32, -63, -95, 63, -32, 127, 32, -63, -95, -127, 32, 0, 0, -32, 0, -32, -63, -32, 0, 32, 0, 32, 0, 32, -63, -95, 0, -32, 0, 32, 0, 32, -63, 32, 0, 95, 0, -32, 0, -32, 63, -32, 127, 32, -5, -44, 3, 0, 5, -18, 0, 18, -8, 0, 18, -36, 0, -91, 0, -18, 21, 18, 0, 18, -18, 36, 18, 0, -36, -54, 0, -18, 18, -36, 18, 36, 7, 0, -18, -36, 0, 18, 0, 18, 18, -36, 18, 36, 0, -18, 36, 18, 0, 18, -36, 54, 18, 0, 18, 0, 0, -18, 0, 54, 18, 36, -18, 0, 14, -18, 0, 18, -18, 36, -18, 36, 0, -91, 0, -18, 18, -36, -18, 73, -18, 36, 18, 0, -36, -18, -36, -18, 18, -36, 18, 36, 0, 127, 36, 18, 0, 18, 0, 18, -18, 36, 18, 0, 0, -18, 36, 18, -18, 73, -18, 0, 18, 0, 18, 0, -36, -18, 73, 18, 18, 36, -18, 0, 36, 18, 36, -54, 3, 65, 4, 0, -18, 0, 0, -25, 40, 51, 0, 25, 0, 127, -25, 0, 4, -51, 0, -25, 0, -25, 25, 0, -51, 76, -25, 0, -25, 0, 0, -76, 10, 0, -51, 25, 0, -25, 25, 0, 0, 25, 25, -51, -25, 0, 0, -76, 0, -25, 76, 51, 25, 0, 0, -25, -25, 0, 51, -25, 0, -76, -25, 0, -5, 0, 0, -25, -51, -76, 25, 0, 0, 127, -25, 0, -76, -51, 0, -76, 0, -25, 25, 0, -25, 51, 0, 76, -25, 0, 0, -76, 51, -127, 76, 0, 0, -25, 25, 0, 25, 0, 51, 25, -25, 0, 0, -76, 0, -76, 25, 51, 25, 0, 0, -25, 0, 76, 76, -51, 0, -76, -25, 0, 76, 0, -51, -25]);
    var expectedScratch=new Int8Array([3, 10, -5, 0, 65, 0, 0, -25, 4, -51, 0, 25, 0, 25, -25, 0, -18, 0, -51, 25, 0, -25, -76, 0, 0, 25, 25, 51, -25, 0, 0, 25, 40, 0, 0, -25, 51, 25, 127, 0, 0, 25, -25, 0, 25, -51, 0, -76, 0, -25, -76, 0, 127, 0, -51, -76, -25, 0, 0, 25, 0, -76, -76, 51, 4, 0, 0, 25, -51, -25, -25, 0, 0, 76, 25, 0, -25, 51, 0, -25, 0, 25, -25, 0, -25, 0, 51, 76, 25, 0, 0, 76, 0, -25, 76, -51, -51, -25, -25, 0, 76, 0, 0, -76, -25, 51, 0, -25, 0, -25, -76, 0, -25, 0, 51, 76, 0, -76, -127, 0, 0, -25, 76, -51, -76, 0, 0, -25]);
    var scaleBitMismatch=0,packedMismatch=0,scratchMismatch=0;
    var aa=new ArrayBuffer(4),fa=new Float32Array(aa),ua=new Uint32Array(aa);
    var bb=new ArrayBuffer(4),fb=new Float32Array(bb),ub=new Uint32Array(bb);
    for(i=0;i<4;i++) {
      fa[0]=p.scales[i];fb[0]=expectedScales[i];
      if(ua[0]!==ub[0])scaleBitMismatch++;
    }
    for(i=0;i<512;i++)if(packed[i]!==expectedPacked[i])packedMismatch++;
    for(i=0;i<128;i++)if(p.q[384+i]!==expectedScratch[i])scratchMismatch++;
    return {
      block:p.block,
      scales:Array.prototype.slice.call(p.scales,0),
      expectedScales:Array.prototype.slice.call(expectedScales,0),
      scaleBitMismatch:scaleBitMismatch,
      packedMismatch:packedMismatch,
      scratchMismatch:scratchMismatch,
      packedFirst32:Array.prototype.slice.call(packed,0,32),
      expectedPackedFirst32:Array.prototype.slice.call(expectedPacked,0,32),
      scratchFirst32:Array.prototype.slice.call(p.q,384,416),
      expectedScratchFirst32:Array.prototype.slice.call(expectedScratch,0,32),
      pass:(scaleBitMismatch===0&&packedMismatch===0&&scratchMismatch===0)
    };
  };

  global.NeedleFull={Engine:Engine,nativeActivationQuant:true,nativeIntegerProjection:true,quantExactProbe:true,rowProbe:true,fullQProjProbe:true,build:"nativeq-a1s-stage-probe"};

}(this));
