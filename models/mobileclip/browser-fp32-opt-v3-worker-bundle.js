(function () {
"use strict";
var modules = {};
var deps = {};
modules[0] = function (module, exports, require) {
"use strict";

var FP32 = require("./mobileclip-fp32-executor-opt-v3.js");
var Kernels = require("./mobileclip-fp32-kernels-opt-v3.js");
var Pool = require("./mobileclip-fp32-pool-v3.js");

/*
 * This bundle runs in two roles, chosen by the worker URL:
 *
 *   (default)     coordinator: loads the model and runs the graph
 *   ?role=helper  helper: computes kernel slices for the coordinator
 *
 * The page creates every worker and relays coordinator <-> helper
 * messages, because old browsers cannot nest workers.
 */
var IS_HELPER = /[?&]role=helper(&|$)/.test(
    String(self.location && self.location.search)
);

/* Asset URLs are unchanged from v2 so cached downloads are reused. */

var PLAN_URL = "compiled-fp32-v1/plan.json?v=fp32-opt-v2";
var WEIGHTS_PARTS = [
    "compiled-fp32-v1/weights.part00.bin?v=fp32-opt-v2-manual",
    "compiled-fp32-v1/weights.part01.bin?v=fp32-opt-v2-manual",
    "compiled-fp32-v1/weights.part02.bin?v=fp32-opt-v2-manual"
];
var WEIGHTS_TOTAL_BYTES = 45425796;
var LABELS_URL = "text-bank/labels.json?v=fp32-opt-v2";
var EMBEDDINGS_URL = "text-bank/embeddings.f32?v=fp32-opt-v2";

var plan = null;
var weightsBuffer = null;
var labels = null;
var textBank = null;
var textDim = 512;
var executor = null;
var ready = false;
var busy = false;

var pool = Pool.createPool(function (helper, task, transfer) {
    postWithTransfer(
        {type: "pool-task", to: helper, task: task},
        transfer
    );
});

function postWithTransfer(message, transfer) {
    try {
        self.postMessage(message, transfer);
    } catch (e) {
        /* No transferable support: fall back to a structured copy. */
        self.postMessage(message);
    }
}

function postStatus(text) {
    self.postMessage({
        type: "status",
        text: text
    });
}

function postLoadProgress(phase, loaded, total, startPct, endPct) {
    var pct = startPct;

    if (total > 0) {
        pct = startPct +
            (endPct - startPct) * loaded / total;
    }

    if (pct < startPct) {
        pct = startPct;
    }
    if (pct > endPct) {
        pct = endPct;
    }

    self.postMessage({
        type: "load-progress",
        phase: phase,
        loaded: loaded,
        total: total,
        percent: pct
    });
}

function xhrText(url, phase, startPct, endPct, done) {
    var xhr = new XMLHttpRequest();

    xhr.open("GET", url, true);

    xhr.onprogress = function (e) {
        postLoadProgress(
            phase,
            e.loaded || 0,
            e.lengthComputable ? e.total : 0,
            startPct,
            endPct
        );
    };

    xhr.onerror = function () {
        done(new Error("XHR failed: " + url));
    };

    xhr.onload = function () {
        if (xhr.status < 200 || xhr.status >= 300) {
            done(new Error(
                "HTTP " + xhr.status + " for " + url
            ));
            return;
        }

        postLoadProgress(
            phase,
            1,
            1,
            startPct,
            endPct
        );

        done(null, xhr.responseText);
    };

    xhr.send(null);
}

function xhrArrayBuffer(url, phase, startPct, endPct, done) {
    var xhr = new XMLHttpRequest();

    xhr.open("GET", url, true);

    try {
        xhr.responseType = "arraybuffer";
    } catch (e) {
        done(new Error(
            "ArrayBuffer XHR unsupported for " + url +
            ": " + e.message
        ));
        return;
    }

    xhr.onprogress = function (e) {
        postLoadProgress(
            phase,
            e.loaded || 0,
            e.lengthComputable ? e.total : 0,
            startPct,
            endPct
        );
    };

    xhr.onerror = function () {
        done(new Error("XHR failed: " + url));
    };

    xhr.onload = function () {
        var buf;

        if (xhr.status < 200 || xhr.status >= 300) {
            done(new Error(
                "HTTP " + xhr.status + " for " + url
            ));
            return;
        }

        buf = xhr.response;

        if (!buf || typeof buf.byteLength !== "number") {
            done(new Error(
                "No ArrayBuffer response for " + url
            ));
            return;
        }

        postLoadProgress(
            phase,
            1,
            1,
            startPct,
            endPct
        );

        done(null, buf);
    };

    xhr.send(null);
}


function loadWeightParts(done) {
    var combined = new Uint8Array(WEIGHTS_TOTAL_BYTES);
    var offset = 0;
    var index = 0;

    function next() {
        var startPct;
        var endPct;

        if (index >= WEIGHTS_PARTS.length) {
            if (offset !== WEIGHTS_TOTAL_BYTES) {
                done(new Error(
                    "weight-part total mismatch: loaded=" +
                    offset + " expected=" + WEIGHTS_TOTAL_BYTES
                ));
                return;
            }

            postLoadProgress(
                "FP32 weights",
                1,
                1,
                3,
                90
            );

            done(null, combined.buffer);
            return;
        }

        startPct = 3 +
            (87 * index / WEIGHTS_PARTS.length);
        endPct = 3 +
            (87 * (index + 1) / WEIGHTS_PARTS.length);

        xhrArrayBuffer(
            WEIGHTS_PARTS[index],
            "FP32 weights part " +
                (index + 1) + "/" + WEIGHTS_PARTS.length,
            startPct,
            endPct,
            function (err, buf) {
                var part;

                if (err) {
                    done(err);
                    return;
                }

                part = new Uint8Array(buf);

                if (offset + part.length > combined.length) {
                    done(new Error(
                        "weight parts exceed expected total bytes"
                    ));
                    return;
                }

                combined.set(part, offset);
                offset += part.length;
                index += 1;

                part = null;
                buf = null;

                next();
            }
        );
    }

    next();
}

function normalizeTextBankInPlace(bank, count, dim) {
    var i;
    var j;
    var off;
    var v;
    var ss;
    var norm;

    for (i = 0; i < count; ++i) {
        off = i * dim;
        ss = 0.0;

        for (j = 0; j < dim; ++j) {
            v = bank[off + j];
            ss += v * v;
        }

        norm = Math.sqrt(ss);

        if (!(norm > 0.0) || !isFinite(norm)) {
            throw new Error(
                "invalid text embedding norm at label " + i
            );
        }

        for (j = 0; j < dim; ++j) {
            bank[off + j] /= norm;
        }
    }
}

function loadAll() {
    postStatus("Loading FP32 plan...");

    xhrText(
        PLAN_URL,
        "FP32 plan",
        0,
        3,
        function (err, text) {
            if (err) {
                fail(err);
                return;
            }

            try {
                plan = JSON.parse(text);
            } catch (e) {
                fail(new Error(
                    "plan JSON parse failed: " + e.message
                ));
                return;
            }

            postStatus("Loading FP32 weights in 3 GitHub-safe parts (~43.3 MiB)...");

            loadWeightParts(
                function (err2, buf) {
                    if (err2) {
                        fail(err2);
                        return;
                    }

                    weightsBuffer = buf;

                    postStatus("Loading text labels...");

                    xhrText(
                        LABELS_URL,
                        "text labels",
                        90,
                        92,
                        function (err3, labelText) {
                            var meta;

                            if (err3) {
                                fail(err3);
                                return;
                            }

                            try {
                                meta = JSON.parse(labelText);
                                labels = meta.labels;
                                textDim = meta.dim || 512;
                            } catch (e2) {
                                fail(new Error(
                                    "labels JSON parse failed: " +
                                    e2.message
                                ));
                                return;
                            }

                            if (!labels || !labels.length) {
                                fail(new Error(
                                    "labels array missing/empty"
                                ));
                                return;
                            }

                            postStatus(
                                "Loading text embeddings (~2.5 MiB)..."
                            );

                            xhrArrayBuffer(
                                EMBEDDINGS_URL,
                                "text embeddings",
                                92,
                                100,
                                function (err4, embBuf) {
                                    if (err4) {
                                        fail(err4);
                                        return;
                                    }

                                    try {
                                        textBank =
                                            new Float32Array(embBuf);

                                        if (
                                            textBank.length !==
                                            labels.length * textDim
                                        ) {
                                            throw new Error(
                                                "text bank size mismatch: " +
                                                textBank.length +
                                                " floats vs expected " +
                                                (labels.length * textDim)
                                            );
                                        }

                                        postStatus(
                                            "Normalizing text bank..."
                                        );

                                        normalizeTextBankInPlace(
                                            textBank,
                                            labels.length,
                                            textDim
                                        );

                                        executor =
                                            FP32.createExecutor(
                                                plan,
                                                weightsBuffer
                                            );

                                        ready = true;

                                        self.postMessage({
                                            type: "ready",
                                            nodes: plan.nodes.length,
                                            labels: labels.length,
                                            weightsBytes:
                                                weightsBuffer.byteLength,
                                            embeddingBytes:
                                                embBuf.byteLength
                                        });
                                    } catch (e3) {
                                        fail(e3);
                                    }
                                }
                            );
                        }
                    );
                }
            );
        }
    );
}

function arrayCopyF32(a) {
    var out = new Array(a.length);
    var i;
    for (i = 0; i < a.length; ++i) {
        out[i] = a[i];
    }
    return out;
}

function classify(imageEmbedding, topK) {
    var dim = textDim;
    var count = labels.length;
    var image = new Float32Array(dim);
    var ss = 0.0;
    var norm;
    var i;
    var j;
    var off;
    var v;
    var score;
    var top = [];
    var pos;

    if (imageEmbedding.length !== dim) {
        throw new Error(
            "image embedding dim mismatch: " +
            imageEmbedding.length + " vs " + dim
        );
    }

    for (j = 0; j < dim; ++j) {
        v = imageEmbedding[j];
        ss += v * v;
    }

    norm = Math.sqrt(ss);

    if (!(norm > 0.0) || !isFinite(norm)) {
        throw new Error("invalid image embedding norm");
    }

    for (j = 0; j < dim; ++j) {
        image[j] = imageEmbedding[j] / norm;
    }

    for (i = 0; i < count; ++i) {
        off = i * dim;
        score = 0.0;

        for (j = 0; j < dim; ++j) {
            score += image[j] * textBank[off + j];
        }

        pos = top.length;

        while (pos > 0 && score > top[pos - 1].score) {
            pos -= 1;
        }

        if (pos < topK) {
            top.splice(pos, 0, {
                label: labels[i],
                score: score,
                index: i
            });

            if (top.length > topK) {
                top.pop();
            }
        }
    }

    return {
        top: top,
        rawNorm: norm
    };
}

function runImage(inputBuffer) {
    var input;
    var tensor;
    var infStart;

    if (!ready) {
        throw new Error("model is not ready");
    }

    if (busy) {
        throw new Error("worker is already running");
    }

    input = new Float32Array(inputBuffer);

    if (input.length !== 3 * 256 * 256) {
        throw new Error(
            "input length mismatch: " + input.length
        );
    }

    tensor = new FP32.Tensor(
        input,
        [1, 3, 256, 256],
        "f32"
    );

    busy = true;
    infStart = Date.now();

    executor.runAsync(
        tensor,
        {
            pool: pool,
            progressEvery: 8,
            onProgress: function (done, total) {
                self.postMessage({
                    type: "infer-progress",
                    done: done,
                    total: total,
                    percent: 100 * done / total
                });
            }
        },
        function (err, outputs) {
            busy = false;

            if (err) {
                fail(err);
                return;
            }

            try {
                finishImage(outputs, Date.now() - infStart);
            } catch (e) {
                fail(e);
            }
        }
    );
}

function finishImage(outputs, infMs) {
    var embedding;
    var classStart;
    var classMs;
    var cls;
    var fullEmbedding;

    if (
        !outputs.image_embeds ||
        !outputs.image_embeds.data
    ) {
        throw new Error("image_embeds missing");
    }

    embedding = outputs.image_embeds.data;

    classStart = Date.now();
    cls = classify(embedding, 10);
    classMs = Date.now() - classStart;

    fullEmbedding = arrayCopyF32(embedding);

    self.postMessage({
        type: "result",
        inferenceMs: infMs,
        classifyMs: classMs,
        threads: pool.helpers().length + 1,
        embedding: fullEmbedding,
        first16: fullEmbedding.slice(0, 16),
        rawNorm: cls.rawNorm,
        top: cls.top,
        profile: outputs.__profile
    });
}

function runHelperSlice(t) {
    var r = Kernels.runSlice({
        kind: t.kind,
        x: new Float32Array(t.x),
        w: new Float32Array(t.w),
        b: new Float32Array(t.b),
        xShape: t.xShape,
        wShape: t.wShape,
        attrs: t.attrs
    });

    postWithTransfer(
        {type: "slice-result", id: t.id, out: r.data.buffer},
        [r.data.buffer]
    );
}

function fail(err) {
    self.postMessage({
        type: "error",
        text: (err && err.stack) ?
            err.stack :
            String(err)
    });
}

self.onmessage = function (ev) {
    var m = ev.data || {};

    try {
        if (m.type === "slice") {
            runHelperSlice(m);
        } else if (m.type === "run") {
            runImage(m.inputBuffer);
        } else if (m.type === "pool-helper") {
            pool.setAlive(m.helper, m.alive);
        } else if (m.type === "pool-result") {
            pool.onResult(m.id, m.out);
        }
    } catch (e) {
        if (IS_HELPER) {
            self.postMessage({
                type: "slice-error",
                id: m.id,
                text: (e && e.stack) ? e.stack : String(e)
            });
        } else {
            /* A rejected "run" leaves the run in progress alone. */
            if (m.type !== "run") {
                busy = false;
            }
            fail(e);
        }
    }
};

if (IS_HELPER) {
    self.postMessage({type: "helper-ready"});
} else {
    loadAll();
}

};
deps[0] = {'./mobileclip-fp32-executor-opt-v3.js':1,'./mobileclip-fp32-kernels-opt-v3.js':3,'./mobileclip-fp32-pool-v3.js':4};
modules[1] = function (module, exports, require) {
"use strict";

var K = require("./mobileclip-fp32-kernels-v1.js");
var O = require("./mobileclip-fp32-kernels-opt-v3.js");

/*
 * owned: the tensor is the only holder of its data buffer, so its last
 * consumer may overwrite the buffer instead of allocating a new one.
 */
/*
 * Smallest kernel call, in multiply-adds, worth splitting across the
 * worker pool; below this the message round trip costs more than it
 * saves.
 */
var PARALLEL_MIN_WORK = 3000000;

/* A GELU element costs roughly this many multiply-adds. */
var GELU_WORK_PER_ELEMENT = 8;

/* Give up on helper workers that stay silent this long (ms). */
var PARALLEL_TIMEOUT_MS = 20000;

function Tensor(data, shape, dtype, owned) {
    this.data = data;
    this.shape = shape.slice(0);
    this.dtype = dtype || "f32";
    this.owned = owned === true;
}

function product(shape, begin, end) {
    return K.product(shape, begin, end);
}

function makeTyped(dtype, n) {
    if (dtype === "f32") {
        return new Float32Array(n);
    }
    if (dtype === "i32") {
        return new Int32Array(n);
    }
    if (dtype === "u8" || dtype === "bool") {
        return new Uint8Array(n);
    }
    throw new Error("unsupported dtype: " + dtype);
}

function unaryFloat(x, fn) {
    var out = new Float32Array(x.data.length);
    var i;
    for (i = 0; i < out.length; ++i) {
        out[i] = fn(x.data[i]);
    }
    return new Tensor(out, x.shape, "f32");
}

function sameShape(a, b) {
    var i;

    if (a.length !== b.length) {
        return false;
    }

    for (i = 0; i < a.length; ++i) {
        if (a[i] !== b[i]) {
            return false;
        }
    }

    return true;
}

/*
 * If `small` broadcasts against `big` as [outer][C][inner] x [C]
 * (scalar, per-channel or trailing-dimension vector), return that
 * split; otherwise null.
 */
function midBroadcast(big, small) {
    var rank = big.shape.length;
    var lead = rank - small.shape.length;
    var at = -1;
    var d;
    var sd;

    if (lead < 0) {
        return null;
    }

    for (d = 0; d < small.shape.length; ++d) {
        sd = small.shape[d];

        if (sd === 1) {
            continue;
        }
        if (at >= 0 || sd !== big.shape[lead + d]) {
            return null;
        }
        at = lead + d;
    }

    if (at < 0) {
        return {outer: big.data.length, C: 1, inner: 1};
    }

    return {
        outer: product(big.shape, 0, at),
        C: big.shape[at],
        inner: product(big.shape, at + 1)
    };
}

/*
 * Add / Mul. aFree / bFree say that the operand's buffer may be
 * overwritten with the result.
 */
function opAddMul(a, b, isMul, aFree, bFree) {
    var big;
    var small;
    var bigFree;
    var mid;
    var out;
    var r;

    if (a.dtype === "f32" && b.dtype === "f32") {
        if (sameShape(a.shape, b.shape)) {
            out = aFree ? a.data :
                (bFree ? b.data : new Float32Array(a.data.length));

            if (isMul) {
                O.mulSame(a.data, b.data, out, out.length);
            } else {
                O.addSame(a.data, b.data, out, out.length);
            }

            return new Tensor(out, a.shape, "f32", true);
        }

        if (a.data.length >= b.data.length) {
            big = a;
            small = b;
            bigFree = aFree;
        } else {
            big = b;
            small = a;
            bigFree = bFree;
        }

        mid = midBroadcast(big, small);

        if (mid) {
            out = bigFree ? big.data :
                new Float32Array(big.data.length);

            if (isMul) {
                O.mulMid(
                    big.data, small.data,
                    mid.outer, mid.C, mid.inner, out
                );
            } else {
                O.addMid(
                    big.data, small.data,
                    mid.outer, mid.C, mid.inner, out
                );
            }

            return new Tensor(out, big.shape, "f32", true);
        }
    }

    r = K.binaryF32(
        a.data, a.shape, b.data, b.shape,
        isMul ?
            function (x, y) { return x * y; } :
            function (x, y) { return x + y; }
    );
    return new Tensor(r.data, r.shape, "f32", true);
}

function opDiv(a, b) {
    var r = K.binaryF32(
        a.data, a.shape, b.data, b.shape,
        function (x, y) { return x / y; }
    );
    return new Tensor(r.data, r.shape, "f32");
}

function opRelu(x) {
    return unaryFloat(x, function (v) {
        return v > 0.0 ? v : 0.0;
    });
}

function opErf(x) {
    return unaryFloat(x, K.erfApprox);
}


function geluChainAt(nodes, i) {
    var n0;
    var n1;
    var n2;
    var n3;
    var n4;

    if (i < 0 || i + 4 >= nodes.length) {
        return null;
    }

    n0 = nodes[i + 0];
    n1 = nodes[i + 1];
    n2 = nodes[i + 2];
    n3 = nodes[i + 3];
    n4 = nodes[i + 4];

    if (
        n0.op !== "Div" ||
        n1.op !== "Erf" ||
        n2.op !== "Add" ||
        n3.op !== "Mul" ||
        n4.op !== "Mul"
    ) {
        return null;
    }

    if (
        !n0.outputs || n0.outputs.length !== 1 ||
        !n1.outputs || n1.outputs.length !== 1 ||
        !n2.outputs || n2.outputs.length !== 1 ||
        !n3.outputs || n3.outputs.length !== 1 ||
        !n4.outputs || n4.outputs.length !== 1
    ) {
        return null;
    }

    if (
        !n1.inputs ||
        n1.inputs.indexOf(n0.outputs[0]) < 0
    ) {
        return null;
    }

    if (
        !n2.inputs ||
        n2.inputs.indexOf(n1.outputs[0]) < 0
    ) {
        return null;
    }

    if (
        !n3.inputs ||
        n3.inputs.indexOf(n2.outputs[0]) < 0 ||
        n3.inputs.indexOf(n0.inputs[0]) < 0
    ) {
        return null;
    }

    if (
        !n4.inputs ||
        n4.inputs.indexOf(n3.outputs[0]) < 0
    ) {
        return null;
    }

    return {
        n0: n0,
        n1: n1,
        n2: n2,
        n3: n3,
        n4: n4
    };
}

function otherInputName(node, producedName) {
    var i;

    for (i = 0; i < node.inputs.length; ++i) {
        if (
            node.inputs[i] &&
            node.inputs[i] !== producedName
        ) {
            return node.inputs[i];
        }
    }

    return null;
}

function opSigmoid(x) {
    return unaryFloat(x, K.sigmoid);
}

function opShape(x) {
    var out = new Int32Array(x.shape.length);
    var i;
    for (i = 0; i < x.shape.length; ++i) {
        out[i] = x.shape[i];
    }
    return new Tensor(out, [x.shape.length], "i32");
}

/*
 * Reshape, Unsqueeze and Squeeze return a view on the input buffer; the
 * executor clears `owned` on the input before calling them.
 */
function opReshape(x, shapeTensor, allowzero) {
    var spec = shapeTensor.data;
    var outShape = new Array(spec.length);
    var inferAt = -1;
    var known = 1;
    var total = x.data.length;
    var i;
    var v;

    for (i = 0; i < spec.length; ++i) {
        v = spec[i];

        if (v === 0 && !allowzero) {
            if (i >= x.shape.length) {
                throw new Error("Reshape zero index exceeds input rank");
            }
            v = x.shape[i];
        }

        if (v === -1) {
            if (inferAt >= 0) {
                throw new Error("Reshape has multiple -1");
            }
            inferAt = i;
            outShape[i] = -1;
        } else {
            outShape[i] = v;
            known *= v;
        }
    }

    if (inferAt >= 0) {
        outShape[inferAt] = total / known;
    }

    if (product(outShape) !== total) {
        throw new Error(
            "Reshape size mismatch: " +
            JSON.stringify(x.shape) + " -> " +
            JSON.stringify(outShape)
        );
    }

    return new Tensor(x.data, outShape, x.dtype);
}

function opTranspose(x, perm) {
    var rank = x.shape.length;
    var p = perm || [];
    var out = makeTyped(x.dtype, x.data.length);
    var i;

    if (p.length === 0) {
        for (i = 0; i < rank; ++i) {
            p[i] = rank - 1 - i;
        }
    }

    return new Tensor(
        out,
        O.transpose(x.data, x.shape, p, out),
        x.dtype,
        true
    );
}

function opUnsqueeze(x, axes) {
    var rank = x.shape.length + axes.length;
    var marks = {};
    var outShape = [];
    var i;
    var ax;
    var src = 0;

    for (i = 0; i < axes.length; ++i) {
        ax = axes[i];
        if (ax < 0) {
            ax += rank;
        }
        marks[ax] = true;
    }

    for (i = 0; i < rank; ++i) {
        if (marks[i]) {
            outShape.push(1);
        } else {
            outShape.push(x.shape[src++]);
        }
    }

    return new Tensor(x.data, outShape, x.dtype);
}

function opSqueeze(x, axes) {
    var remove = {};
    var outShape = [];
    var i;
    var ax;

    if (!axes || axes.length === 0) {
        for (i = 0; i < x.shape.length; ++i) {
            if (x.shape[i] !== 1) {
                outShape.push(x.shape[i]);
            }
        }
    } else {
        for (i = 0; i < axes.length; ++i) {
            ax = K.normalizeAxis(axes[i], x.shape.length);
            if (x.shape[ax] !== 1) {
                throw new Error("Squeeze axis dimension != 1");
            }
            remove[ax] = true;
        }
        for (i = 0; i < x.shape.length; ++i) {
            if (!remove[i]) {
                outShape.push(x.shape[i]);
            }
        }
    }

    return new Tensor(x.data, outShape, x.dtype);
}

function opConcat(xs, axis) {
    var rank = xs[0].shape.length;
    var ax = K.normalizeAxis(axis, rank);
    var outShape = xs[0].shape.slice(0);
    var outer = product(outShape, 0, ax);
    var inner = product(outShape, ax + 1);
    var totalAxis = 0;
    var i;
    var o;
    var j;
    var srcChunk;
    var dst;
    var out;

    for (i = 0; i < xs.length; ++i) {
        totalAxis += xs[i].shape[ax];
    }
    outShape[ax] = totalAxis;
    out = makeTyped(xs[0].dtype, product(outShape));

    dst = 0;
    for (o = 0; o < outer; ++o) {
        for (i = 0; i < xs.length; ++i) {
            srcChunk = xs[i].shape[ax] * inner;
            for (j = 0; j < srcChunk; ++j) {
                out[dst++] = xs[i].data[o * srcChunk + j];
            }
        }
    }

    return new Tensor(out, outShape, xs[0].dtype);
}

function opGather(data, indices, axis) {
    var rank = data.shape.length;
    var ax = K.normalizeAxis(axis, rank);
    var before = product(data.shape, 0, ax);
    var axisDim = data.shape[ax];
    var after = product(data.shape, ax + 1);
    var outShape = data.shape.slice(0, ax)
        .concat(indices.shape)
        .concat(data.shape.slice(ax + 1));
    var out = makeTyped(data.dtype, product(outShape));
    var b;
    var j;
    var k;
    var idx;
    var dst = 0;
    var src;

    for (b = 0; b < before; ++b) {
        for (j = 0; j < indices.data.length; ++j) {
            idx = indices.data[j];
            if (idx < 0) {
                idx += axisDim;
            }
            if (idx < 0 || idx >= axisDim) {
                throw new Error("Gather index out of range");
            }
            src = (b * axisDim + idx) * after;
            for (k = 0; k < after; ++k) {
                out[dst++] = data.data[src + k];
            }
        }
    }

    return new Tensor(out, outShape, data.dtype);
}

function opSlice(data, startsT, endsT, axesT, stepsT) {
    var rank = data.shape.length;
    var starts = startsT.data;
    var ends = endsT.data;
    var axes = axesT ? axesT.data : null;
    var steps = stepsT ? stepsT.data : null;
    var begin = new Array(rank);
    var end = new Array(rank);
    var step = new Array(rank);
    var outShape = new Array(rank);
    var inS = K.computeStrides(data.shape);
    var outS;
    var i;
    var d;
    var ax;
    var s;
    var e;
    var st;
    var dim;
    var out;
    var t;
    var coord;
    var src;

    for (i = 0; i < rank; ++i) {
        begin[i] = 0;
        end[i] = data.shape[i];
        step[i] = 1;
    }

    for (i = 0; i < starts.length; ++i) {
        ax = axes ? axes[i] : i;
        ax = K.normalizeAxis(ax, rank);
        dim = data.shape[ax];
        st = steps ? steps[i] : 1;
        if (st === 0) {
            throw new Error("Slice step=0");
        }

        s = starts[i];
        e = ends[i];

        if (s < 0) {
            s += dim;
        }
        if (e < 0) {
            e += dim;
        }

        if (st > 0) {
            s = Math.max(0, Math.min(dim, s));
            e = Math.max(0, Math.min(dim, e));
        } else {
            s = Math.max(-1, Math.min(dim - 1, s));
            e = Math.max(-1, Math.min(dim - 1, e));
        }

        begin[ax] = s;
        end[ax] = e;
        step[ax] = st;
    }

    for (d = 0; d < rank; ++d) {
        if (step[d] > 0) {
            outShape[d] = Math.max(
                0,
                Math.ceil((end[d] - begin[d]) / step[d])
            );
        } else {
            outShape[d] = Math.max(
                0,
                Math.ceil((begin[d] - end[d]) / (-step[d]))
            );
        }
    }

    out = makeTyped(data.dtype, product(outShape));
    outS = K.computeStrides(outShape);

    for (i = 0; i < out.length; ++i) {
        t = i;
        src = 0;
        for (d = 0; d < rank; ++d) {
            coord = Math.floor(t / outS[d]);
            t -= coord * outS[d];
            src += (begin[d] + coord * step[d]) * inS[d];
        }
        out[i] = data.data[src];
    }

    return new Tensor(out, outShape, data.dtype);
}

function opSplit(x, axis, split, outputCount) {
    var ax = K.normalizeAxis(axis, x.shape.length);
    var sizes;
    var inner = product(x.shape, ax + 1);
    var outer = product(x.shape, 0, ax);
    var axisDim = x.shape[ax];
    var outs = [];
    var i;
    var o;
    var j;
    var k;
    var offset;
    var src;
    var dst;
    var shape;
    var arr;

    if (split && split.length) {
        sizes = split.slice(0);
    } else {
        if (axisDim % outputCount !== 0) {
            throw new Error("Split not even");
        }
        sizes = [];
        for (i = 0; i < outputCount; ++i) {
            sizes.push(axisDim / outputCount);
        }
    }

    for (i = 0; i < sizes.length; ++i) {
        shape = x.shape.slice(0);
        shape[ax] = sizes[i];
        outs.push(new Tensor(
            makeTyped(x.dtype, product(shape)),
            shape,
            x.dtype
        ));
    }

    for (o = 0; o < outer; ++o) {
        offset = 0;
        for (i = 0; i < sizes.length; ++i) {
            arr = outs[i].data;
            src = (o * axisDim + offset) * inner;
            dst = o * sizes[i] * inner;
            for (j = 0; j < sizes[i] * inner; ++j) {
                arr[dst + j] = x.data[src + j];
            }
            offset += sizes[i];
        }
    }

    return outs;
}

function opReduceMean(x, axes, keepdims) {
    var rank = x.shape.length;
    var reduce = {};
    var outShape = [];
    var outShapeKeep = new Array(rank);
    var out;
    var counts;
    var inS = K.computeStrides(x.shape);
    var outS;
    var i;
    var d;
    var ax;
    var t;
    var coord;
    var oi;

    if (!axes || axes.length === 0) {
        axes = [];
        for (i = 0; i < rank; ++i) {
            axes.push(i);
        }
    }

    for (i = 0; i < axes.length; ++i) {
        ax = K.normalizeAxis(axes[i], rank);
        reduce[ax] = true;
    }

    for (d = 0; d < rank; ++d) {
        if (reduce[d]) {
            outShapeKeep[d] = 1;
            if (keepdims) {
                outShape.push(1);
            }
        } else {
            outShapeKeep[d] = x.shape[d];
            outShape.push(x.shape[d]);
        }
    }

    out = new Float32Array(product(outShape));

    for (d = rank - 1; d >= 0 && reduce[d]; --d) {
    }
    for (i = 0; i <= d && !reduce[i]; ++i) {
    }
    if (i > d) {
        /* Reduced axes are exactly the trailing ones. */
        O.reduceMeanTrailing(
            x.data,
            out.length,
            product(x.shape, d + 1),
            out
        );
        return new Tensor(out, outShape, "f32", true);
    }

    counts = new Int32Array(out.length);
    outS = K.computeStrides(outShapeKeep);

    for (i = 0; i < x.data.length; ++i) {
        t = i;
        oi = 0;
        for (d = 0; d < rank; ++d) {
            coord = Math.floor(t / inS[d]);
            t -= coord * inS[d];
            if (!reduce[d]) {
                oi += coord * outS[d];
            }
        }
        out[oi] += x.data[i];
        counts[oi] += 1;
    }

    for (i = 0; i < out.length; ++i) {
        out[i] /= counts[i];
    }

    return new Tensor(out, outShape, "f32");
}

function opSoftmax(x, axis) {
    var ax = K.normalizeAxis(axis, x.shape.length);
    var outer = product(x.shape, 0, ax);
    var dim = x.shape[ax];
    var inner = product(x.shape, ax + 1);
    var out = new Float32Array(x.data.length);
    var o;
    var k;
    var j;
    var base;
    var idx;
    var maxv;
    var sum;
    var v;

    for (o = 0; o < outer; ++o) {
        for (j = 0; j < inner; ++j) {
            base = o * dim * inner + j;
            maxv = -Infinity;

            for (k = 0; k < dim; ++k) {
                v = x.data[base + k * inner];
                if (v > maxv) {
                    maxv = v;
                }
            }

            sum = 0.0;
            for (k = 0; k < dim; ++k) {
                idx = base + k * inner;
                v = Math.exp(x.data[idx] - maxv);
                out[idx] = v;
                sum += v;
            }

            for (k = 0; k < dim; ++k) {
                idx = base + k * inner;
                out[idx] /= sum;
            }
        }
    }

    return new Tensor(out, x.shape, "f32");
}

function opBatchNormalization(x, scale, bias, mean, variance, epsilon) {
    var C = x.shape[1];
    var spatial = product(x.shape, 2);
    var N = x.shape[0];
    var out = new Float32Array(x.data.length);
    var n;
    var c;
    var p;
    var base;
    var mul;
    var add;

    for (c = 0; c < C; ++c) {
        mul = scale.data[c] /
            Math.sqrt(variance.data[c] + epsilon);
        add = bias.data[c] - mean.data[c] * mul;

        for (n = 0; n < N; ++n) {
            base = (n * C + c) * spatial;
            for (p = 0; p < spatial; ++p) {
                out[base + p] = x.data[base + p] * mul + add;
            }
        }
    }

    return new Tensor(out, x.shape, "f32");
}

function opPadConstant(x, padsT, constantT) {
    var rank = x.shape.length;
    var pads = padsT.data;
    var before = new Array(rank);
    var after = new Array(rank);
    var outShape = new Array(rank);
    var inS = K.computeStrides(x.shape);
    var outS;
    var out;
    var value = constantT ? constantT.data[0] : 0.0;
    var i;
    var d;
    var t;
    var coord;
    var dst;

    if (pads.length !== rank * 2) {
        throw new Error("Pad pads length mismatch");
    }

    for (d = 0; d < rank; ++d) {
        before[d] = pads[d];
        after[d] = pads[d + rank];
        if (before[d] < 0 || after[d] < 0) {
            throw new Error("negative Pad unsupported");
        }
        outShape[d] = x.shape[d] + before[d] + after[d];
    }

    out = new Float32Array(product(outShape));
    if (value !== 0.0) {
        for (i = 0; i < out.length; ++i) {
            out[i] = value;
        }
    }
    outS = K.computeStrides(outShape);

    for (i = 0; i < x.data.length; ++i) {
        t = i;
        dst = 0;
        for (d = 0; d < rank; ++d) {
            coord = Math.floor(t / inS[d]);
            t -= coord * inS[d];
            dst += (coord + before[d]) * outS[d];
        }
        out[dst] = x.data[i];
    }

    return new Tensor(out, outShape, "f32");
}

function opAveragePoolNCHW(x, attrs) {
    var N = x.shape[0];
    var C = x.shape[1];
    var H = x.shape[2];
    var W = x.shape[3];
    var kernel = attrs.kernel_shape;
    var strides = attrs.strides || kernel;
    var pads = attrs.pads || [0, 0, 0, 0];
    var KH = kernel[0];
    var KW = kernel[1];
    var sh = strides[0];
    var sw = strides[1];
    var pt = pads[0];
    var pl = pads[1];
    var pb = pads[2];
    var pr = pads[3];
    var OH = Math.floor((H + pt + pb - KH) / sh) + 1;
    var OW = Math.floor((W + pl + pr - KW) / sw) + 1;
    var out = new Float32Array(N * C * OH * OW);
    var n;
    var c;
    var oy;
    var ox;
    var ky;
    var kx;
    var iy;
    var ix;
    var sum;
    var count;
    var xo;
    var yo;

    if (attrs.ceil_mode) {
        throw new Error("AveragePool ceil_mode unsupported");
    }

    for (n = 0; n < N; ++n) {
        for (c = 0; c < C; ++c) {
            for (oy = 0; oy < OH; ++oy) {
                for (ox = 0; ox < OW; ++ox) {
                    sum = 0.0;
                    count = 0;

                    for (ky = 0; ky < KH; ++ky) {
                        iy = oy * sh - pt + ky;
                        if (iy < 0 || iy >= H) {
                            continue;
                        }
                        for (kx = 0; kx < KW; ++kx) {
                            ix = ox * sw - pl + kx;
                            if (ix < 0 || ix >= W) {
                                continue;
                            }
                            xo = ((n * C + c) * H + iy) * W + ix;
                            sum += x.data[xo];
                            count += 1;
                        }
                    }

                    yo = ((n * C + c) * OH + oy) * OW + ox;
                    out[yo] = sum / count;
                }
            }
        }
    }

    return new Tensor(out, [N, C, OH, OW], "f32");
}

function createExecutor(plan, weightsBuffer) {
    var weights = weightsBuffer;
    var staticCache = {};

    if (weightsBuffer instanceof Uint8Array) {
        weights = weightsBuffer.buffer.slice(
            weightsBuffer.byteOffset,
            weightsBuffer.byteOffset + weightsBuffer.byteLength
        );
    }

    if (!(weights instanceof ArrayBuffer)) {
        throw new Error("weights must be ArrayBuffer or Uint8Array");
    }

    function staticTensor(name) {
        var info;
        var data;

        if (!name) {
            return null;
        }
        if (staticCache[name]) {
            return staticCache[name];
        }

        info = plan.tensors[name];
        if (!info) {
            return null;
        }

        if (info.dtype === "f32") {
            data = new Float32Array(
                weights,
                info.offset,
                info.nbytes / 4
            );
        } else if (info.dtype === "i32") {
            data = new Int32Array(
                weights,
                info.offset,
                info.nbytes / 4
            );
        } else if (info.dtype === "u8" || info.dtype === "bool") {
            data = new Uint8Array(
                weights,
                info.offset,
                info.nbytes
            );
        } else {
            throw new Error(
                "unsupported static dtype " + info.dtype +
                " tensor=" + name
            );
        }

        staticCache[name] = new Tensor(
            data,
            info.shape,
            info.dtype
        );
        return staticCache[name];
    }

    var baseUses = {};
    var geluAt = {};
    var geluSkip = {};

    (function analyse() {
        var ni;
        var j;
        var name;
        var gc;

        for (ni = 0; ni < plan.nodes.length; ++ni) {
            for (j = 0; j < plan.nodes[ni].inputs.length; ++j) {
                name = plan.nodes[ni].inputs[j];
                if (name) {
                    baseUses[name] = (baseUses[name] || 0) + 1;
                }
            }
        }

        for (ni = 0; ni + 4 < plan.nodes.length; ++ni) {
            gc = geluChainAt(plan.nodes, ni);

            /*
             * Fuse only when nothing outside the chain reads the
             * intermediates; any other chain runs node by node.
             */
            if (
                gc &&
                baseUses[gc.n0.outputs[0]] === 1 &&
                baseUses[gc.n1.outputs[0]] === 1 &&
                baseUses[gc.n2.outputs[0]] === 1 &&
                baseUses[gc.n3.outputs[0]] === 1
            ) {
                gc.addName = otherInputName(gc.n2, gc.n1.outputs[0]);
                gc.mulName = otherInputName(gc.n4, gc.n3.outputs[0]);
                geluAt[ni] = gc;
                geluSkip[ni + 1] = true;
                geluSkip[ni + 2] = true;
                geluSkip[ni + 3] = true;
                geluSkip[ni + 4] = true;
                ni += 4;
            }
        }
    }());

    /*
     * Runs the graph. Without `done` the call is synchronous and returns
     * the outputs. With `done(err, outputs)` it may hand kernel slices
     * to options.pool and continue when they come back.
     */
    function execute(inputOrFeeds, options, done) {
        var env = {};
        var uses = {};
        var keep = {};
        var outputNames = {};
        var inputName = plan.inputs[0].name;
        var inputTensor;
        var ni;
        var node;
        var op;
        var attrs;
        var ins;
        var i;
        var j;
        var name;
        var t0;
        var elapsed;
        var opTimes = {};
        var opCalls = {};
        var result;
        var r;
        var outs;
        var wInfo;
        var fast;
        var gc;
        var xT;
        var divT;
        var addT;
        var mulT;
        var fusedData;
        var par = null;

        options = options || {};

        if (
            inputOrFeeds &&
            inputOrFeeds.data &&
            inputOrFeeds.shape
        ) {
            inputTensor = inputOrFeeds;
        } else if (
            inputOrFeeds &&
            inputOrFeeds[inputName]
        ) {
            inputTensor = inputOrFeeds[inputName];
        } else {
            throw new Error("input tensor not supplied: " + inputName);
        }

        env[inputName] = inputTensor;

        for (name in baseUses) {
            if (baseUses.hasOwnProperty(name)) {
                uses[name] = baseUses[name];
            }
        }

        for (i = 0; i < plan.outputs.length; ++i) {
            keep[plan.outputs[i].name] = true;
            outputNames[plan.outputs[i].name] = true;
        }

        function getTensor(tname) {
            var s;
            if (!tname) {
                return null;
            }
            if (env[tname]) {
                return env[tname];
            }
            s = staticTensor(tname);
            if (s) {
                return s;
            }
            throw new Error("tensor not found: " + tname);
        }

        /*
         * True when node input j is a temporary whose only remaining
         * reader is the current node, so its buffer can be reused.
         */
        function reusable(j) {
            return ins[j].owned && uses[node.inputs[j]] === 1;
        }

        function reportProgress() {
            if (
                options.progressEvery &&
                (
                    (ni + 1) % options.progressEvery === 0 ||
                    ni + 1 === plan.nodes.length
                )
            ) {
                if (options.onProgress) {
                    options.onProgress(ni + 1, plan.nodes.length);
                } else if (
                    typeof console !== "undefined" &&
                    console.log
                ) {
                    console.log(
                        "progress:",
                        (ni + 1) + "/" + plan.nodes.length
                    );
                }
            }
        }

        function consume(tname) {
            if (!tname || plan.tensors[tname]) {
                return;
            }
            uses[tname] -= 1;
            if (uses[tname] <= 0 && !keep[tname]) {
                delete env[tname];
            }
        }

        function afterNode() {
            elapsed = Date.now() - t0;
            opTimes[op] = (opTimes[op] || 0) + elapsed;
            opCalls[op] = (opCalls[op] || 0) + 1;

            for (j = 0; j < node.inputs.length; ++j) {
                consume(node.inputs[j]);
            }

            reportProgress();
        }

        /*
         * Split the current node's kernel call by output rows: the
         * helpers each get `per` rows, this thread computes the rest
         * while they work. makeSlice(r0, r1) describes rows [r0, r1)
         * for O.runSlice. Returns false if the call should run here.
         */
        function startParallel(
            rows, quantum, work, makeSlice, outShapeOf, outName
        ) {
            var pool = options.pool;
            var helpers = (done && pool) ? pool.helpers() : [];
            var per;
            var localRows;
            var local;
            var k;

            if (helpers.length < 1 || work < PARALLEL_MIN_WORK) {
                return false;
            }

            per = Math.floor(
                rows / (helpers.length + 1) / quantum
            ) * quantum;

            if (per < quantum) {
                return false;
            }

            localRows = rows - per * helpers.length;

            par = {
                out: null,
                rowSize: 0,
                left: 0,
                retry: [],
                timer: null,
                waited: false,
                outShape: null,
                outName: outName || node.outputs[0]
            };

            for (k = 0; k < helpers.length; ++k) {
                submitSlice(
                    pool,
                    helpers[k],
                    makeSlice,
                    localRows + k * per,
                    localRows + (k + 1) * per
                );
            }

            local = O.runSlice(makeSlice(0, localRows));
            par.rowSize = local.data.length / localRows;
            par.out = new Float32Array(rows * par.rowSize);
            par.out.set(local.data, 0);
            par.outShape = outShapeOf(local.shape);

            /* Slices that could not be posted at all. */
            for (k = 0; k < par.retry.length; ++k) {
                par.out.set(
                    O.runSlice(
                        makeSlice(par.retry[k][0], par.retry[k][1])
                    ).data,
                    par.retry[k][0] * par.rowSize
                );
            }

            if (par.left > 0) {
                par.timer = setTimeout(function () {
                    pool.failPending();
                }, PARALLEL_TIMEOUT_MS);
            } else {
                finishParallel();
            }

            return true;
        }

        function submitSlice(pool, helper, makeSlice, r0, r1) {
            var state = par;
            var slice = makeSlice(r0, r1);
            var ok = pool.submit(
                helper,
                slice,
                function (data) {
                    state.out.set(data, r0 * state.rowSize);
                    sliceDone(state);
                },
                function () {
                    /* Helper died: compute its rows here instead. */
                    state.out.set(
                        O.runSlice(slice).data,
                        r0 * state.rowSize
                    );
                    sliceDone(state);
                }
            );

            if (ok) {
                state.left += 1;
            } else {
                state.retry.push([r0, r1]);
            }
        }

        function sliceDone(state) {
            state.left -= 1;

            if (state.left === 0 && state === par) {
                try {
                    finishParallel();
                } catch (e) {
                    done(e);
                }
            }
        }

        function finishParallel() {
            var state = par;

            par = null;

            if (state.timer !== null) {
                clearTimeout(state.timer);
            }

            env[state.outName] =
                new Tensor(state.out, state.outShape, "f32", true);

            afterNode();

            /*
             * If nothing had to be waited for we are still inside
             * the loop of resume(), which moves on by itself.
             */
            if (state.waited) {
                ni += 1;
                resume();
            }
        }

        function convSlicer(kind, x, w, b, xShape, wShape, wRow) {
            var C = xShape[1];
            var image = xShape[2] * xShape[3];
            var mult = wShape[0] / C;

            if (kind === "depthwise") {
                return function (r0, r1) {
                    return {
                        kind: kind,
                        x: x.subarray(r0 * image, r1 * image),
                        w: w.subarray(
                            r0 * mult * wRow,
                            r1 * mult * wRow
                        ),
                        b: b.subarray(r0 * mult, r1 * mult),
                        xShape: [1, r1 - r0, xShape[2], xShape[3]],
                        wShape: [
                            (r1 - r0) * mult, 1, wShape[2], wShape[3]
                        ],
                        attrs: {
                            pads: attrs.pads,
                            strides: attrs.strides,
                            dilations: attrs.dilations,
                            group: r1 - r0
                        }
                    };
                };
            }

            return function (r0, r1) {
                return {
                    kind: kind,
                    x: x,
                    w: w.subarray(r0 * wRow, r1 * wRow),
                    b: b.subarray(r0, r1),
                    xShape: xShape,
                    wShape: [r1 - r0, wShape[1], wShape[2], wShape[3]],
                    attrs: attrs
                };
            };
        }

        function tryParallelConv(kind, xT, wT, bT, wShape) {
            var xs = xT.shape;
            var st = attrs.strides || [1, 1];
            var wRow = wShape[1] * wShape[2] * wShape[3];
            var work = wShape[0] * wRow * xs[2] * xs[3] /
                (st[0] * st[1]);

            if (xs[0] !== 1) {
                return false;
            }

            if (kind === "depthwise") {
                if (
                    wShape[1] !== 1 ||
                    (attrs.group || 1) !== xs[1] ||
                    (wShape[0] % xs[1]) !== 0
                ) {
                    return false;
                }

                return startParallel(
                    xs[1], 1, work,
                    convSlicer(
                        kind, xT.data, wT.data, bT.data,
                        xs, wShape, wRow
                    ),
                    convShapeOf(wShape[0])
                );
            }

            if ((wShape[0] % 4) !== 0 || (attrs.group || 1) !== 1) {
                return false;
            }

            return startParallel(
                wShape[0], 4, work,
                convSlicer(
                    kind, xT.data, wT.data, bT.data,
                    xs, wShape, wRow
                ),
                convShapeOf(wShape[0])
            );
        }

        function convShapeOf(M) {
            return function (localShape) {
                return [1, M, localShape[2], localShape[3]];
            };
        }

        /* A is [1..., M, K] with a single matrix, B is [K, N]. */
        function tryParallelMatMul(aT, bT) {
            var ar = aT.shape.length;
            var M = aT.shape[ar - 2];
            var Kd = aT.shape[ar - 1];
            var a = aT.data;

            if (
                bT.shape.length !== 2 ||
                ar < 2 ||
                a.length !== M * Kd ||
                bT.shape[0] !== Kd ||
                (M % 4) !== 0
            ) {
                return false;
            }

            return startParallel(
                M, 4, M * Kd * bT.shape[1],
                function (r0, r1) {
                    return {
                        kind: "matmul",
                        x: bT.data,
                        w: a.subarray(r0 * Kd, r1 * Kd),
                        b: a.subarray(0, 0),
                        xShape: bT.shape,
                        wShape: [r1 - r0, Kd],
                        attrs: null
                    };
                },
                function (localShape) {
                    return aT.shape.slice(0, ar - 1).concat(
                        [localShape[localShape.length - 1]]
                    );
                }
            );
        }

        /* Rows are the leading dimensions, split in whole planes. */
        function tryParallelGelu(xT, chain, divT, addT, mulT) {
            var rank = xT.shape.length;
            var rowSize = rank < 2 ? 1 :
                xT.shape[rank - 2] * xT.shape[rank - 1];
            var rows = xT.data.length / rowSize;
            var x = xT.data;
            var scalars = {
                div: divT.data[0],
                add: addT.data[0],
                mul: mulT.data[0]
            };
            var shape = xT.shape;

            return startParallel(
                rows, 1, x.length * GELU_WORK_PER_ELEMENT,
                function (r0, r1) {
                    return {
                        kind: "gelu",
                        x: x.subarray(r0 * rowSize, r1 * rowSize),
                        w: x.subarray(0, 0),
                        b: x.subarray(0, 0),
                        xShape: [r1 - r0, rowSize],
                        wShape: null,
                        attrs: scalars
                    };
                },
                function () {
                    return shape;
                },
                chain.n4.outputs[0]
            );
        }

        function resume() {
            try {
                for (; ni < plan.nodes.length; ++ni) {
                    if (stepNode()) {
                        par.waited = true;
                        return undefined;
                    }
                }
            } catch (e) {
                if (done) {
                    done(e);
                    return undefined;
                }
                throw e;
            }

            return finish();
        }

        /* Runs node ni. Returns true if it is waiting for the pool. */
        function stepNode() {
            node = plan.nodes[ni];
            op = node.op;
            attrs = node.attrs || {};
            ins = [];

            if (geluSkip[ni]) {
                /*
                 * Continuation of a fused GELU chain: the output was
                 * produced at the Div node and the inputs were
                 * released there.
                 */
                reportProgress();
                return false;
            }

            for (j = 0; j < node.inputs.length; ++j) {
                ins.push(
                    node.inputs[j] ? getTensor(node.inputs[j]) : null
                );
            }

            t0 = Date.now();

            try {
                if (geluAt[ni]) {
                    gc = geluAt[ni];
                    xT = ins[0];
                    divT = ins[1];
                    addT = getTensor(gc.addName);
                    mulT = getTensor(gc.mulName);

                    if (
                        xT.dtype !== "f32" ||
                        divT.data.length !== 1 ||
                        addT.data.length !== 1 ||
                        mulT.data.length !== 1
                    ) {
                        throw new Error(
                            "unexpected GELU scalar/input contract " +
                            "at node " + ni
                        );
                    }

                    op = "Gelu";

                    if (tryParallelGelu(xT, gc, divT, addT, mulT)) {
                        /* The skipped first Mul also read x. */
                        consume(node.inputs[0]);
                        return par !== null;
                    }

                    /*
                     * x is read by the Div and by the first Mul of the
                     * chain. If those are the only readers left, apply
                     * GELU in place.
                     */
                    fusedData =
                        (xT.owned && uses[node.inputs[0]] === 2) ?
                            xT.data :
                            new Float32Array(xT.data.length);

                    O.geluErfLut(
                        xT.data,
                        fusedData,
                        fusedData.length,
                        divT.data[0],
                        addT.data[0],
                        mulT.data[0]
                    );

                    env[gc.n4.outputs[0]] =
                        new Tensor(fusedData, xT.shape, "f32", true);

                    /* The skipped first Mul was the other reader of x. */
                    consume(node.inputs[0]);
                }

                else if (op === "Conv") {
                    wInfo = plan.tensors[node.inputs[1]];
                    fast = node.fast || {};

                    if (!ins[0] || !ins[1] || !ins[2]) {
                        throw new Error("Conv requires x,w,b");
                    }

                    if (
                        tryParallelConv(
                            fast.kind === "pointwise" ? "pointwise" :
                                ((fast.kind === "depthwise" ||
                                  fast.kind === "grouped") ?
                                    "depthwise" : "dense"),
                            ins[0], ins[1], ins[2], wInfo.shape
                        )
                    ) {
                        return par !== null;
                    }

                    if (fast.kind === "pointwise") {
                        r = O.convPointwiseNCHW(
                            ins[0].data,
                            ins[1].data,
                            ins[2].data,
                            ins[0].shape,
                            wInfo.shape
                        );
                    } else if (
                        fast.kind === "depthwise" ||
                        fast.kind === "grouped"
                    ) {
                        r = O.convDepthwiseNCHW(
                            ins[0].data,
                            ins[1].data,
                            ins[2].data,
                            ins[0].shape,
                            wInfo.shape,
                            attrs
                        );
                    } else {
                        r = O.convDenseNCHW(
                            ins[0].data,
                            ins[1].data,
                            ins[2].data,
                            ins[0].shape,
                            wInfo.shape,
                            attrs
                        );
                    }

                    env[node.outputs[0]] =
                        new Tensor(r.data, r.shape, "f32", true);
                }

                else if (op === "MatMul") {
                    if (tryParallelMatMul(ins[0], ins[1])) {
                        return par !== null;
                    }

                    r = O.matMul(
                        ins[0].data,
                        ins[0].shape,
                        ins[1].data,
                        ins[1].shape
                    );
                    env[node.outputs[0]] =
                        new Tensor(r.data, r.shape, "f32", true);
                }

                else if (op === "Add") {
                    env[node.outputs[0]] = opAddMul(
                        ins[0], ins[1], false,
                        reusable(0), reusable(1)
                    );
                }

                else if (op === "Mul") {
                    env[node.outputs[0]] = opAddMul(
                        ins[0], ins[1], true,
                        reusable(0), reusable(1)
                    );
                }

                else if (op === "Div") {
                    env[node.outputs[0]] = opDiv(ins[0], ins[1]);
                }

                else if (op === "Relu") {
                    env[node.outputs[0]] = opRelu(ins[0]);
                }

                else if (op === "Erf") {
                    env[node.outputs[0]] = opErf(ins[0]);
                }

                else if (op === "Sigmoid") {
                    env[node.outputs[0]] = opSigmoid(ins[0]);
                }

                else if (op === "Shape") {
                    env[node.outputs[0]] = opShape(ins[0]);
                }

                else if (op === "Reshape") {
                    ins[0].owned = false;
                    env[node.outputs[0]] = opReshape(
                        ins[0],
                        ins[1],
                        attrs.allowzero || 0
                    );
                }

                else if (op === "Transpose") {
                    env[node.outputs[0]] =
                        opTranspose(ins[0], attrs.perm);
                }

                else if (op === "Unsqueeze") {
                    ins[0].owned = false;
                    env[node.outputs[0]] =
                        opUnsqueeze(ins[0], attrs.axes || []);
                }

                else if (op === "Squeeze") {
                    ins[0].owned = false;
                    env[node.outputs[0]] =
                        opSqueeze(ins[0], attrs.axes || []);
                }

                else if (op === "Concat") {
                    env[node.outputs[0]] =
                        opConcat(ins, attrs.axis);
                }

                else if (op === "Gather") {
                    env[node.outputs[0]] =
                        opGather(ins[0], ins[1], attrs.axis || 0);
                }

                else if (op === "Slice") {
                    env[node.outputs[0]] = opSlice(
                        ins[0],
                        ins[1],
                        ins[2],
                        ins[3],
                        ins[4]
                    );
                }

                else if (op === "Split") {
                    outs = opSplit(
                        ins[0],
                        attrs.axis || 0,
                        attrs.split || null,
                        node.outputs.length
                    );
                    for (i = 0; i < node.outputs.length; ++i) {
                        env[node.outputs[i]] = outs[i];
                    }
                }

                else if (op === "ReduceMean") {
                    env[node.outputs[0]] = opReduceMean(
                        ins[0],
                        attrs.axes || [],
                        attrs.keepdims === undefined ? 1 : attrs.keepdims
                    );
                }

                else if (op === "Softmax") {
                    env[node.outputs[0]] =
                        opSoftmax(
                            ins[0],
                            attrs.axis === undefined ? -1 : attrs.axis
                        );
                }

                else if (op === "BatchNormalization") {
                    env[node.outputs[0]] = opBatchNormalization(
                        ins[0],
                        ins[1],
                        ins[2],
                        ins[3],
                        ins[4],
                        attrs.epsilon === undefined ?
                            1e-5 : attrs.epsilon
                    );
                }

                else if (op === "Pad") {
                    if (
                        attrs.mode !== undefined &&
                        attrs.mode !== "constant"
                    ) {
                        throw new Error(
                            "Pad mode unsupported: " + attrs.mode
                        );
                    }
                    env[node.outputs[0]] =
                        opPadConstant(ins[0], ins[1], ins[2]);
                }

                else if (op === "AveragePool") {
                    env[node.outputs[0]] =
                        opAveragePoolNCHW(ins[0], attrs);
                }

                else {
                    throw new Error("unsupported operator: " + op);
                }
            }
            catch (e) {
                e.message =
                    "node " + ni + "/" + plan.nodes.length +
                    " op=" + op +
                    " name=" + node.name +
                    ": " + e.message;
                throw e;
            }

            afterNode();
            return false;
        }

        function finish() {
            result = {};
            for (name in outputNames) {
                if (outputNames.hasOwnProperty(name)) {
                    result[name] = env[name];
                }
            }

            result.__profile = {
                times: opTimes,
                calls: opCalls
            };

            if (done) {
                done(null, result);
            }

            return result;
        }

        ni = 0;
        return resume();
    }

    function run(inputOrFeeds, options) {
        return execute(inputOrFeeds, options, null);
    }

    function runAsync(inputOrFeeds, options, done) {
        try {
            execute(inputOrFeeds, options, done);
        } catch (e) {
            done(e);
        }
    }

    return {
        run: run,
        runAsync: runAsync,
        Tensor: Tensor
    };
}

module.exports = {
    createExecutor: createExecutor,
    Tensor: Tensor
};

};
deps[1] = {'./mobileclip-fp32-kernels-v1.js':2,'./mobileclip-fp32-kernels-opt-v3.js':3};
modules[2] = function (module, exports, require) {
"use strict";

/*
 * MobileCLIP-S0 FP32 pure-JavaScript reference kernels v1.
 *
 * Deliberately conservative / ES5-style:
 *   - no WASM
 *   - no WebGL/WebGPU
 *   - no BigInt
 *   - no async/await
 *   - no ES modules
 *
 * Correctness-first. Optimization comes only after the FP32 path
 * is numerically and semantically validated.
 */

function product(shape, begin, end) {
    var r = 1;
    var i;
    begin = begin === undefined ? 0 : begin;
    end = end === undefined ? shape.length : end;
    for (i = begin; i < end; ++i) {
        r *= shape[i];
    }
    return r;
}

function computeStrides(shape) {
    var s = new Array(shape.length);
    var p = 1;
    var i;
    for (i = shape.length - 1; i >= 0; --i) {
        s[i] = p;
        p *= shape[i];
    }
    return s;
}

function normalizeAxis(axis, rank) {
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        throw new Error("axis out of range: " + axis + " rank=" + rank);
    }
    return axis;
}

function sameShape(a, b) {
    var i;
    if (a.length !== b.length) {
        return false;
    }
    for (i = 0; i < a.length; ++i) {
        if (a[i] !== b[i]) {
            return false;
        }
    }
    return true;
}

function broadcastShape(a, b) {
    var rank = Math.max(a.length, b.length);
    var out = new Array(rank);
    var i;
    var av;
    var bv;
    for (i = 0; i < rank; ++i) {
        av = a[a.length - rank + i];
        bv = b[b.length - rank + i];
        av = av === undefined ? 1 : av;
        bv = bv === undefined ? 1 : bv;
        if (av !== bv && av !== 1 && bv !== 1) {
            throw new Error(
                "broadcast mismatch: " +
                JSON.stringify(a) + " vs " + JSON.stringify(b)
            );
        }
        out[i] = Math.max(av, bv);
    }
    return out;
}

function binaryF32(a, aShape, b, bShape, fn) {
    var outShape = broadcastShape(aShape, bShape);
    var out = new Float32Array(product(outShape));
    var rank = outShape.length;
    var aPad = new Array(rank);
    var bPad = new Array(rank);
    var outS = computeStrides(outShape);
    var aS;
    var bS;
    var i;
    var d;
    var t;
    var coord;
    var ai;
    var bi;

    for (i = 0; i < rank; ++i) {
        aPad[i] = aShape[aShape.length - rank + i];
        bPad[i] = bShape[bShape.length - rank + i];
        if (aPad[i] === undefined) {
            aPad[i] = 1;
        }
        if (bPad[i] === undefined) {
            bPad[i] = 1;
        }
    }
    aS = computeStrides(aPad);
    bS = computeStrides(bPad);

    if (sameShape(aShape, bShape)) {
        for (i = 0; i < out.length; ++i) {
            out[i] = fn(a[i], b[i]);
        }
        return {data: out, shape: outShape};
    }

    if (a.length === 1) {
        for (i = 0; i < out.length; ++i) {
            out[i] = fn(a[0], b[i]);
        }
        return {data: out, shape: outShape};
    }

    if (b.length === 1) {
        for (i = 0; i < out.length; ++i) {
            out[i] = fn(a[i], b[0]);
        }
        return {data: out, shape: outShape};
    }

    for (i = 0; i < out.length; ++i) {
        t = i;
        ai = 0;
        bi = 0;
        for (d = 0; d < rank; ++d) {
            coord = Math.floor(t / outS[d]);
            t -= coord * outS[d];
            if (aPad[d] !== 1) {
                ai += coord * aS[d];
            }
            if (bPad[d] !== 1) {
                bi += coord * bS[d];
            }
        }
        out[i] = fn(a[ai], b[bi]);
    }

    return {data: out, shape: outShape};
}

/*
 * High-accuracy erf approximation from Numerical Recipes.
 * Absolute error is ~1e-7 over the range relevant to GELU.
 */
function erfApprox(x) {
    var sign = x < 0 ? -1.0 : 1.0;
    var a = Math.abs(x);
    var t = 1.0 / (1.0 + 0.5 * a);
    var p = 0.17087277;
    p = -0.82215223 + t * p;
    p = 1.48851587 + t * p;
    p = -1.13520398 + t * p;
    p = 0.27886807 + t * p;
    p = -0.18628806 + t * p;
    p = 0.09678418 + t * p;
    p = 0.37409196 + t * p;
    p = 1.00002368 + t * p;
    var tau = t * Math.exp(-a * a - 1.26551223 + t * p);
    return sign * (1.0 - tau);
}

function sigmoid(x) {
    if (x >= 0) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    var e = Math.exp(x);
    return e / (1.0 + e);
}

function convPointwiseNCHW(x, w, bias, xShape, wShape) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var P = H * W;
    var out = new Float32Array(N * M * P);
    var n;
    var m;
    var p;
    var c;
    var m0;
    var yBase;
    var xBase;
    var w0;
    var w1;
    var w2;
    var w3;
    var o0;
    var o1;
    var o2;
    var o3;
    var a0;
    var a1;
    var a2;
    var a3;
    var xv;

    for (n = 0; n < N; ++n) {
        xBase = n * C * P;
        yBase = n * M * P;

        for (m0 = 0; m0 + 3 < M; m0 += 4) {
            w0 = m0 * C;
            w1 = (m0 + 1) * C;
            w2 = (m0 + 2) * C;
            w3 = (m0 + 3) * C;

            o0 = yBase + m0 * P;
            o1 = o0 + P;
            o2 = o1 + P;
            o3 = o2 + P;

            for (p = 0; p < P; ++p) {
                a0 = bias[m0];
                a1 = bias[m0 + 1];
                a2 = bias[m0 + 2];
                a3 = bias[m0 + 3];

                for (c = 0; c < C; ++c) {
                    xv = x[xBase + c * P + p];
                    a0 += xv * w[w0 + c];
                    a1 += xv * w[w1 + c];
                    a2 += xv * w[w2 + c];
                    a3 += xv * w[w3 + c];
                }

                out[o0 + p] = a0;
                out[o1 + p] = a1;
                out[o2 + p] = a2;
                out[o3 + p] = a3;
            }
        }

        for (m = m0; m < M; ++m) {
            o0 = yBase + m * P;
            w0 = m * C;
            for (p = 0; p < P; ++p) {
                a0 = bias[m];
                for (c = 0; c < C; ++c) {
                    a0 += x[xBase + c * P + p] * w[w0 + c];
                }
                out[o0 + p] = a0;
            }
        }
    }

    return {
        data: out,
        shape: [N, M, H, W]
    };
}

function convDepthwiseNCHW(x, w, bias, xShape, wShape, attrs) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var KH = wShape[2];
    var KW = wShape[3];
    var pads = attrs.pads || [0, 0, 0, 0];
    var strides = attrs.strides || [1, 1];
    var dilations = attrs.dilations || [1, 1];
    var pt = pads[0];
    var pl = pads[1];
    var pb = pads[2];
    var pr = pads[3];
    var sh = strides[0];
    var sw = strides[1];
    var dh = dilations[0];
    var dw = dilations[1];
    var OH = Math.floor((H + pt + pb - dh * (KH - 1) - 1) / sh) + 1;
    var OW = Math.floor((W + pl + pr - dw * (KW - 1) - 1) / sw) + 1;
    var out = new Float32Array(N * M * OH * OW);
    var image = H * W;
    var outImage = OH * OW;
    var n;
    var c;
    var oy;
    var ox;
    var ky;
    var kx;
    var iy;
    var ix;
    var a;
    var xb;
    var yb;
    var wb;

    if (M !== C || attrs.group !== C) {
        throw new Error("depthwise contract mismatch");
    }

    for (n = 0; n < N; ++n) {
        for (c = 0; c < C; ++c) {
            xb = (n * C + c) * image;
            yb = (n * C + c) * outImage;
            wb = c * KH * KW;

            for (oy = 0; oy < OH; ++oy) {
                for (ox = 0; ox < OW; ++ox) {
                    a = bias[c];
                    for (ky = 0; ky < KH; ++ky) {
                        iy = oy * sh - pt + ky * dh;
                        if (iy < 0 || iy >= H) {
                            continue;
                        }
                        for (kx = 0; kx < KW; ++kx) {
                            ix = ox * sw - pl + kx * dw;
                            if (ix < 0 || ix >= W) {
                                continue;
                            }
                            a += x[xb + iy * W + ix] *
                                 w[wb + ky * KW + kx];
                        }
                    }
                    out[yb + oy * OW + ox] = a;
                }
            }
        }
    }

    return {
        data: out,
        shape: [N, M, OH, OW]
    };
}

function convGroupedNCHW(x, w, bias, xShape, wShape, attrs) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var Cg = wShape[1];
    var KH = wShape[2];
    var KW = wShape[3];
    var group = attrs.group || 1;
    var Mg;
    var pads = attrs.pads || [0, 0, 0, 0];
    var strides = attrs.strides || [1, 1];
    var dilations = attrs.dilations || [1, 1];
    var pt = pads[0];
    var pl = pads[1];
    var pb = pads[2];
    var pr = pads[3];
    var sh = strides[0];
    var sw = strides[1];
    var dh = dilations[0];
    var dw = dilations[1];
    var OH = Math.floor((H + pt + pb - dh * (KH - 1) - 1) / sh) + 1;
    var OW = Math.floor((W + pl + pr - dw * (KW - 1) - 1) / sw) + 1;
    var out = new Float32Array(N * M * OH * OW);
    var n;
    var g;
    var om;
    var ic;
    var oy;
    var ox;
    var ky;
    var kx;
    var iy;
    var ix;
    var oc;
    var inC;
    var a;
    var xo;
    var wo;
    var yo;

    if (C !== Cg * group || M % group !== 0) {
        throw new Error(
            "grouped Conv contract mismatch C=" + C +
            " Cg=" + Cg + " M=" + M + " group=" + group
        );
    }

    Mg = M / group;

    for (n = 0; n < N; ++n) {
        for (g = 0; g < group; ++g) {
            for (om = 0; om < Mg; ++om) {
                oc = g * Mg + om;
                for (oy = 0; oy < OH; ++oy) {
                    for (ox = 0; ox < OW; ++ox) {
                        a = bias[oc];

                        for (ic = 0; ic < Cg; ++ic) {
                            inC = g * Cg + ic;

                            for (ky = 0; ky < KH; ++ky) {
                                iy = oy * sh - pt + ky * dh;
                                if (iy < 0 || iy >= H) {
                                    continue;
                                }

                                for (kx = 0; kx < KW; ++kx) {
                                    ix = ox * sw - pl + kx * dw;
                                    if (ix < 0 || ix >= W) {
                                        continue;
                                    }

                                    xo = ((n * C + inC) * H + iy) * W + ix;
                                    wo = (((oc * Cg + ic) * KH + ky) * KW + kx);
                                    a += x[xo] * w[wo];
                                }
                            }
                        }

                        yo = ((n * M + oc) * OH + oy) * OW + ox;
                        out[yo] = a;
                    }
                }
            }
        }
    }

    return {
        data: out,
        shape: [N, M, OH, OW]
    };
}

function matMul(a, aShape, b, bShape) {
    var ar = aShape.length;
    var br = bShape.length;
    var M;
    var K;
    var K2;
    var N;
    var aBatch;
    var bBatch;
    var batchShape;
    var batchCount;
    var aPad;
    var bPad;
    var batchRank;
    var aBatchShape;
    var bBatchShape;
    var out;
    var outShape;
    var aBatchS;
    var bBatchS;
    var outBatchS;
    var bi;
    var d;
    var t;
    var coord;
    var aBatchIndex;
    var bBatchIndex;
    var aBase;
    var bBase;
    var outBase;
    var m;
    var n;
    var k;
    var sum;

    if (ar < 2 || br < 2) {
        throw new Error("MatMul rank<2 not implemented");
    }

    M = aShape[ar - 2];
    K = aShape[ar - 1];
    K2 = bShape[br - 2];
    N = bShape[br - 1];

    if (K !== K2) {
        throw new Error("MatMul K mismatch: " + K + " vs " + K2);
    }

    aBatch = aShape.slice(0, ar - 2);
    bBatch = bShape.slice(0, br - 2);
    batchShape = broadcastShape(aBatch, bBatch);
    batchRank = batchShape.length;
    batchCount = product(batchShape);

    aBatchShape = new Array(batchRank);
    bBatchShape = new Array(batchRank);

    for (d = 0; d < batchRank; ++d) {
        aBatchShape[d] = aBatch[aBatch.length - batchRank + d];
        bBatchShape[d] = bBatch[bBatch.length - batchRank + d];
        if (aBatchShape[d] === undefined) {
            aBatchShape[d] = 1;
        }
        if (bBatchShape[d] === undefined) {
            bBatchShape[d] = 1;
        }
    }

    aBatchS = computeStrides(aBatchShape);
    bBatchS = computeStrides(bBatchShape);
    outBatchS = computeStrides(batchShape);

    outShape = batchShape.concat([M, N]);
    out = new Float32Array(batchCount * M * N);

    for (bi = 0; bi < batchCount; ++bi) {
        t = bi;
        aBatchIndex = 0;
        bBatchIndex = 0;

        for (d = 0; d < batchRank; ++d) {
            coord = Math.floor(t / outBatchS[d]);
            t -= coord * outBatchS[d];

            if (aBatchShape[d] !== 1) {
                aBatchIndex += coord * aBatchS[d];
            }
            if (bBatchShape[d] !== 1) {
                bBatchIndex += coord * bBatchS[d];
            }
        }

        aBase = aBatchIndex * M * K;
        bBase = bBatchIndex * K * N;
        outBase = bi * M * N;

        for (m = 0; m < M; ++m) {
            for (n = 0; n < N; ++n) {
                sum = 0.0;
                for (k = 0; k < K; ++k) {
                    sum += a[aBase + m * K + k] *
                           b[bBase + k * N + n];
                }
                out[outBase + m * N + n] = sum;
            }
        }
    }

    return {
        data: out,
        shape: outShape
    };
}

module.exports = {
    product: product,
    computeStrides: computeStrides,
    normalizeAxis: normalizeAxis,
    broadcastShape: broadcastShape,
    binaryF32: binaryF32,
    erfApprox: erfApprox,
    sigmoid: sigmoid,
    convPointwiseNCHW: convPointwiseNCHW,
    convDepthwiseNCHW: convDepthwiseNCHW,
    convGroupedNCHW: convGroupedNCHW,
    matMul: matMul
};

};
deps[2] = {};
modules[3] = function (module, exports, require) {
"use strict";

/*
 * MobileCLIP-S0 FP32 pure-JavaScript optimized kernels v3.
 *
 * Same constraints as v1/v2: ES5 style, TypedArrays only, no WASM.
 *
 * What changed relative to v2:
 *
 *   - Every dense product (pointwise Conv, the initial 3x3 Conv, MatMul)
 *     goes through one packed GEMM micro-kernel. The input is copied
 *     ("packed") into a small pixel-major scratch block first, so the
 *     inner loop only walks contiguous memory instead of striding across
 *     a whole NCHW tensor for every output tile.
 *   - The micro-kernel is 4 output channels x 4 pixels: 16 accumulators
 *     plus 8 operands stay inside the double registers of a VFPv3-D32
 *     ARM core, where the v2 4x8 kernel needed 44 live values.
 *   - Depthwise/grouped Conv runs on a zero-padded copy of each plane,
 *     which removes every per-tap bounds branch.
 *   - GELU uses an interpolated erf table instead of exp + two rational
 *     polynomials + a division per element.
 *
 * The GEMM and depthwise kernels keep the v2 summation order, so their
 * results are bit-identical to v2. Only the GELU table changes values,
 * by about one float32 ulp of erf.
 */

var Base = require("./mobileclip-fp32-kernels-v1.js");

/*
 * Bytes of packed input one GEMM chunk may occupy. The chunk is re-read
 * once per block of 4 output channels, so it should stay inside L2.
 */
var PACK_CHUNK_BYTES = 98304;

var scratchSlots = {};

function scratch(slot, n) {
    var b = scratchSlots[slot];

    if (!b || b.length < n) {
        b = new Float32Array(n);
        scratchSlots[slot] = b;
    }

    return b;
}


/*
 * GEMM micro-kernels.
 *
 * xt holds G groups of pB pixels, packed as xt[g][c][t]. For every block
 * of 4 weight rows the whole packed chunk is streamed once:
 *
 *   out[oOff + (g * pB + t) * oSP + m * oSM] =
 *       bias[m] + sum_c xt[g][c][t] * w[wOff + m * C + c]
 *
 * Rows beyond the last full block of 4 are left untouched.
 */
function gemm4x4(xt, G, C, w, wOff, M, bias, out, oOff, oSP, oSM) {
    var m0, g, c, xi, o;
    var wo0, b0, w0, wo1, b1, w1, wo2, b2;
    var w2, wo3, b3, w3, x0, x1, x2, x3;
    var a0_0, a0_1, a0_2, a0_3, a1_0, a1_1, a1_2, a1_3;
    var a2_0, a2_1, a2_2, a2_3, a3_0, a3_1, a3_2, a3_3;

    for (m0 = 0; m0 + 3 < M; m0 += 4) {
        wo0 = wOff + (m0 + 0) * C;
        b0 = bias[m0 + 0];
        wo1 = wOff + (m0 + 1) * C;
        b1 = bias[m0 + 1];
        wo2 = wOff + (m0 + 2) * C;
        b2 = bias[m0 + 2];
        wo3 = wOff + (m0 + 3) * C;
        b3 = bias[m0 + 3];
        xi = 0;

        for (g = 0; g < G; ++g) {
            a0_0 = b0;
            a0_1 = b0;
            a0_2 = b0;
            a0_3 = b0;
            a1_0 = b1;
            a1_1 = b1;
            a1_2 = b1;
            a1_3 = b1;
            a2_0 = b2;
            a2_1 = b2;
            a2_2 = b2;
            a2_3 = b2;
            a3_0 = b3;
            a3_1 = b3;
            a3_2 = b3;
            a3_3 = b3;

            for (c = 0; c < C; ++c) {
                x0 = xt[xi];
                x1 = xt[xi + 1];
                x2 = xt[xi + 2];
                x3 = xt[xi + 3];
                xi += 4;
                w0 = w[wo0 + c];
                a0_0 += x0 * w0;
                a0_1 += x1 * w0;
                a0_2 += x2 * w0;
                a0_3 += x3 * w0;
                w1 = w[wo1 + c];
                a1_0 += x0 * w1;
                a1_1 += x1 * w1;
                a1_2 += x2 * w1;
                a1_3 += x3 * w1;
                w2 = w[wo2 + c];
                a2_0 += x0 * w2;
                a2_1 += x1 * w2;
                a2_2 += x2 * w2;
                a2_3 += x3 * w2;
                w3 = w[wo3 + c];
                a3_0 += x0 * w3;
                a3_1 += x1 * w3;
                a3_2 += x2 * w3;
                a3_3 += x3 * w3;
            }

            o = oOff + g * 4 * oSP + (m0 + 0) * oSM;
            out[o] = a0_0;
            out[o + oSP] = a0_1;
            out[o + 2 * oSP] = a0_2;
            out[o + 3 * oSP] = a0_3;
            o = oOff + g * 4 * oSP + (m0 + 1) * oSM;
            out[o] = a1_0;
            out[o + oSP] = a1_1;
            out[o + 2 * oSP] = a1_2;
            out[o + 3 * oSP] = a1_3;
            o = oOff + g * 4 * oSP + (m0 + 2) * oSM;
            out[o] = a2_0;
            out[o + oSP] = a2_1;
            out[o + 2 * oSP] = a2_2;
            out[o + 3 * oSP] = a2_3;
            o = oOff + g * 4 * oSP + (m0 + 3) * oSM;
            out[o] = a3_0;
            out[o + oSP] = a3_1;
            out[o + 2 * oSP] = a3_2;
            out[o + 3 * oSP] = a3_3;
        }
    }

    return m0;
}

function gemm4x1(xt, G, C, w, wOff, M, bias, out, oOff, oSP, oSM) {
    var m0, g, c, xi, o;
    var wo0, b0, w0, wo1, b1, w1, wo2, b2;
    var w2, wo3, b3, w3, x0;
    var a0_0, a1_0, a2_0, a3_0;

    for (m0 = 0; m0 + 3 < M; m0 += 4) {
        wo0 = wOff + (m0 + 0) * C;
        b0 = bias[m0 + 0];
        wo1 = wOff + (m0 + 1) * C;
        b1 = bias[m0 + 1];
        wo2 = wOff + (m0 + 2) * C;
        b2 = bias[m0 + 2];
        wo3 = wOff + (m0 + 3) * C;
        b3 = bias[m0 + 3];
        xi = 0;

        for (g = 0; g < G; ++g) {
            a0_0 = b0;
            a1_0 = b1;
            a2_0 = b2;
            a3_0 = b3;

            for (c = 0; c < C; ++c) {
                x0 = xt[xi];
                ++xi;
                w0 = w[wo0 + c];
                a0_0 += x0 * w0;
                w1 = w[wo1 + c];
                a1_0 += x0 * w1;
                w2 = w[wo2 + c];
                a2_0 += x0 * w2;
                w3 = w[wo3 + c];
                a3_0 += x0 * w3;
            }

            o = oOff + g * 1 * oSP + (m0 + 0) * oSM;
            out[o] = a0_0;
            o = oOff + g * 1 * oSP + (m0 + 1) * oSM;
            out[o] = a1_0;
            o = oOff + g * 1 * oSP + (m0 + 2) * oSM;
            out[o] = a2_0;
            o = oOff + g * 1 * oSP + (m0 + 3) * oSM;
            out[o] = a3_0;
        }
    }

    return m0;
}


/*
 * Pack G groups of 4 consecutive pixels, starting at pixel p0, from a
 * channel-major [C][P] block into xt[g][c][0..3].
 */
function packChannelMajor4(x, xOff, C, P, p0, G, xt) {
    var step = C * 4;
    var c, g, src, dst;

    for (c = 0; c < C; ++c) {
        src = xOff + c * P + p0;
        dst = c * 4;

        for (g = 0; g < G; ++g) {
            xt[dst] = x[src];
            xt[dst + 1] = x[src + 1];
            xt[dst + 2] = x[src + 2];
            xt[dst + 3] = x[src + 3];
            src += 4;
            dst += step;
        }
    }
}

function packChannelMajor1(x, xOff, C, P, p0, G, xt) {
    var c, g, src, dst;

    for (c = 0; c < C; ++c) {
        src = xOff + c * P + p0;
        dst = c;

        for (g = 0; g < G; ++g) {
            xt[dst] = x[src + g];
            dst += C;
        }
    }
}

/*
 * out[oOff + m * P + p] =
 *     bias[m] + sum_c x[xOff + c * P + p] * w[wOff + m * C + c]
 *
 * x is channel-major ([C][P]), w is row-major ([M][C]) and M must be a
 * multiple of 4. This is a pointwise Conv as written, and also
 * MatMul(A, B) with x = B, w = A.
 */
function gemmChannelMajor(x, xOff, C, P, w, wOff, M, bias, out, oOff) {
    var maxG = Math.floor(PACK_CHUNK_BYTES / (16 * C));
    var P4 = P - (P % 4);
    var xt, p0, G;

    if (maxG < 1) {
        maxG = 1;
    }

    xt = scratch("pack", maxG * 4 * C);

    for (p0 = 0; p0 < P4; p0 += G * 4) {
        G = (P4 - p0) >> 2;
        if (G > maxG) {
            G = maxG;
        }

        packChannelMajor4(x, xOff, C, P, p0, G, xt);
        gemm4x4(xt, G, C, w, wOff, M, bias, out, oOff + p0, 1, P);
    }

    if (P4 < P) {
        packChannelMajor1(x, xOff, C, P, P4, P - P4, xt);
        gemm4x1(xt, P - P4, C, w, wOff, M, bias, out, oOff + P4, 1, P);
    }
}

function convPointwiseNCHW(x, w, bias, xShape, wShape) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var P = H * W;
    var out;

    if (N !== 1 || (M % 4) !== 0) {
        return Base.convPointwiseNCHW(x, w, bias, xShape, wShape);
    }

    out = new Float32Array(M * P);
    gemmChannelMajor(x, 0, C, P, w, 0, M, bias, out, 0);

    return {
        data: out,
        shape: [1, M, H, W]
    };
}

/*
 * im2col for G groups of 4 consecutive output pixels starting at p0,
 * written straight into the packed layout xt[g][q][0..3] with
 * q = (ic * KH + ky) * KW + kx, the order of one weight row.
 */
function packIm2col4(
    x, C, H, W, KH, KW, sh, sw, pt, pl, OW, p0, G, xt
) {
    var image = H * W;
    var Q4 = C * KH * KW * 4;
    var g, t, p, oy, ox, ic, ky, kx, iy, ix, dst, plane;

    for (g = 0; g < G; ++g) {
        for (t = 0; t < 4; ++t) {
            p = p0 + g * 4 + t;
            oy = Math.floor(p / OW);
            ox = p - oy * OW;
            dst = g * Q4 + t;

            for (ic = 0; ic < C; ++ic) {
                plane = ic * image;

                for (ky = 0; ky < KH; ++ky) {
                    iy = oy * sh - pt + ky;

                    for (kx = 0; kx < KW; ++kx) {
                        ix = ox * sw - pl + kx;

                        if (iy < 0 || iy >= H || ix < 0 || ix >= W) {
                            xt[dst] = 0.0;
                        } else {
                            xt[dst] = x[plane + iy * W + ix];
                        }

                        dst += 4;
                    }
                }
            }
        }
    }
}

/*
 * Dense (group = 1) Conv as im2col + GEMM. Only the first layer of the
 * model takes this path.
 */
function convDenseNCHW(x, w, bias, xShape, wShape, attrs) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var KH = wShape[2];
    var KW = wShape[3];
    var pads = attrs.pads || [0, 0, 0, 0];
    var strides = attrs.strides || [1, 1];
    var dilations = attrs.dilations || [1, 1];
    var group = attrs.group || 1;
    var OH = Math.floor((H + pads[0] + pads[2] - KH) / strides[0]) + 1;
    var OW = Math.floor((W + pads[1] + pads[3] - KW) / strides[1]) + 1;
    var P = OH * OW;
    var Q = C * KH * KW;
    var maxG = Math.floor(PACK_CHUNK_BYTES / (16 * Q));
    var out, xt, p0, G;

    if (
        N !== 1 ||
        group !== 1 ||
        wShape[1] !== C ||
        dilations[0] !== 1 ||
        dilations[1] !== 1 ||
        (M % 4) !== 0 ||
        (P % 4) !== 0
    ) {
        return Base.convGroupedNCHW(x, w, bias, xShape, wShape, attrs);
    }

    if (maxG < 1) {
        maxG = 1;
    }

    out = new Float32Array(M * P);
    xt = scratch("pack", maxG * 4 * Q);

    for (p0 = 0; p0 < P; p0 += G * 4) {
        G = (P - p0) >> 2;
        if (G > maxG) {
            G = maxG;
        }

        packIm2col4(
            x, C, H, W, KH, KW,
            strides[0], strides[1], pads[0], pads[1],
            OW, p0, G, xt
        );
        gemm4x4(xt, G, Q, w, 0, M, bias, out, p0, 1, P);
    }

    return {
        data: out,
        shape: [1, M, OH, OW]
    };
}

/*
 * MatMul with equal (non-broadcast) batch dimensions.
 * B is [K][N], i.e. channel-major with K channels and N "pixels", and
 * the rows of A are the weight rows, so no transposition is needed.
 */
function matMul(a, aShape, b, bShape) {
    var ar = aShape.length;
    var br = bShape.length;
    var M, K, N, batch, zero, out, i;

    if (ar < 2 || br < 2) {
        return Base.matMul(a, aShape, b, bShape);
    }

    M = aShape[ar - 2];
    K = aShape[ar - 1];
    N = bShape[br - 1];
    batch = Base.product(aShape, 0, ar - 2);

    if (
        K !== bShape[br - 2] ||
        (M % 4) !== 0 ||
        batch !== Base.product(bShape, 0, br - 2) ||
        (br > 2 && br !== ar)
    ) {
        return Base.matMul(a, aShape, b, bShape);
    }

    for (i = 0; i < br - 2; ++i) {
        if (aShape[i] !== bShape[i]) {
            return Base.matMul(a, aShape, b, bShape);
        }
    }

    zero = scratch("zeroBias", M);
    out = new Float32Array(batch * M * N);

    for (i = 0; i < batch; ++i) {
        gemmChannelMajor(
            b, i * K * N, K, N,
            a, i * M * K, M,
            zero,
            out, i * M * N
        );
    }

    return {
        data: out,
        shape: aShape.slice(0, ar - 1).concat([N])
    };
}


/*
 * Depthwise row kernels. pad is one zero-padded input plane, Wp its row
 * length. Four neighbouring outputs share the loaded input values.
 */
function depthwisePlaneK7(
    pad, Wp, w, wb, KH, b, out, yb, OH, OW
) {
    var OW4 = OW - (OW % 4);
    var oy, ox, ky, kx, r, wi, o, acc;
    var a0, a1, a2, a3;
    var w0, w1, w2, w3, w4, w5, w6;
    var x0, x1, x2, x3, x4, x5, x6, x7, x8, x9;

    for (oy = 0; oy < OH; ++oy) {
        o = yb + oy * OW;

        for (ox = 0; ox < OW4; ox += 4) {
            a0 = b;
            a1 = b;
            a2 = b;
            a3 = b;
            r = oy * Wp + ox;
            wi = wb;

            for (ky = 0; ky < KH; ++ky) {
                x0 = pad[r];
                x1 = pad[r + 1];
                x2 = pad[r + 2];
                x3 = pad[r + 3];
                x4 = pad[r + 4];
                x5 = pad[r + 5];
                x6 = pad[r + 6];
                x7 = pad[r + 7];
                x8 = pad[r + 8];
                x9 = pad[r + 9];
                w0 = w[wi];
                w1 = w[wi + 1];
                w2 = w[wi + 2];
                w3 = w[wi + 3];
                w4 = w[wi + 4];
                w5 = w[wi + 5];
                w6 = w[wi + 6];
                a0 = a0 + x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3 +
                    x4 * w4 + x5 * w5 + x6 * w6;
                a1 = a1 + x1 * w0 + x2 * w1 + x3 * w2 + x4 * w3 +
                    x5 * w4 + x6 * w5 + x7 * w6;
                a2 = a2 + x2 * w0 + x3 * w1 + x4 * w2 + x5 * w3 +
                    x6 * w4 + x7 * w5 + x8 * w6;
                a3 = a3 + x3 * w0 + x4 * w1 + x5 * w2 + x6 * w3 +
                    x7 * w4 + x8 * w5 + x9 * w6;
                r += Wp;
                wi += 7;
            }

            out[o + ox] = a0;
            out[o + ox + 1] = a1;
            out[o + ox + 2] = a2;
            out[o + ox + 3] = a3;
        }

        for (; ox < OW; ++ox) {
            acc = b;
            wi = wb;

            for (ky = 0; ky < KH; ++ky) {
                r = (oy + ky) * Wp + ox;
                for (kx = 0; kx < 7; ++kx) {
                    acc += pad[r + kx] * w[wi++];
                }
            }

            out[o + ox] = acc;
        }
    }
}

function depthwisePlaneK3(
    pad, Wp, w, wb, KH, b, out, yb, OH, OW
) {
    var OW4 = OW - (OW % 4);
    var oy, ox, ky, kx, r, wi, o, acc;
    var a0, a1, a2, a3;
    var w0, w1, w2;
    var x0, x1, x2, x3, x4, x5;

    for (oy = 0; oy < OH; ++oy) {
        o = yb + oy * OW;

        for (ox = 0; ox < OW4; ox += 4) {
            a0 = b;
            a1 = b;
            a2 = b;
            a3 = b;
            r = oy * Wp + ox;
            wi = wb;

            for (ky = 0; ky < KH; ++ky) {
                x0 = pad[r];
                x1 = pad[r + 1];
                x2 = pad[r + 2];
                x3 = pad[r + 3];
                x4 = pad[r + 4];
                x5 = pad[r + 5];
                w0 = w[wi];
                w1 = w[wi + 1];
                w2 = w[wi + 2];
                a0 = a0 + x0 * w0 + x1 * w1 + x2 * w2;
                a1 = a1 + x1 * w0 + x2 * w1 + x3 * w2;
                a2 = a2 + x2 * w0 + x3 * w1 + x4 * w2;
                a3 = a3 + x3 * w0 + x4 * w1 + x5 * w2;
                r += Wp;
                wi += 3;
            }

            out[o + ox] = a0;
            out[o + ox + 1] = a1;
            out[o + ox + 2] = a2;
            out[o + ox + 3] = a3;
        }

        for (; ox < OW; ++ox) {
            acc = b;
            wi = wb;

            for (ky = 0; ky < KH; ++ky) {
                r = (oy + ky) * Wp + ox;
                for (kx = 0; kx < 3; ++kx) {
                    acc += pad[r + kx] * w[wi++];
                }
            }

            out[o + ox] = acc;
        }
    }
}

/* Any kernel size and stride. */
function depthwisePlaneGeneric(
    pad, Wp, w, wb, KH, KW, sh, sw, b, out, yb, OH, OW
) {
    var OW4 = OW - (OW % 4);
    var sw2 = sw * 2;
    var sw3 = sw * 3;
    var oy, ox, ky, kx, r, wi, o, acc, wv;
    var a0, a1, a2, a3;

    for (oy = 0; oy < OH; ++oy) {
        o = yb + oy * OW;

        for (ox = 0; ox < OW4; ox += 4) {
            a0 = b;
            a1 = b;
            a2 = b;
            a3 = b;
            wi = wb;

            for (ky = 0; ky < KH; ++ky) {
                r = (oy * sh + ky) * Wp + ox * sw;

                for (kx = 0; kx < KW; ++kx) {
                    wv = w[wi++];
                    a0 += pad[r] * wv;
                    a1 += pad[r + sw] * wv;
                    a2 += pad[r + sw2] * wv;
                    a3 += pad[r + sw3] * wv;
                    ++r;
                }
            }

            out[o + ox] = a0;
            out[o + ox + 1] = a1;
            out[o + ox + 2] = a2;
            out[o + ox + 3] = a3;
        }

        for (; ox < OW; ++ox) {
            acc = b;
            wi = wb;

            for (ky = 0; ky < KH; ++ky) {
                r = (oy * sh + ky) * Wp + ox * sw;
                for (kx = 0; kx < KW; ++kx) {
                    acc += pad[r + kx] * w[wi++];
                }
            }

            out[o + ox] = acc;
        }
    }
}

/*
 * Conv with one input channel per group: true depthwise (M = C) and the
 * channel-multiplier layers (M = 2 * C).
 */
function convDepthwiseNCHW(x, w, bias, xShape, wShape, attrs) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var KH = wShape[2];
    var KW = wShape[3];
    var group = attrs.group || 1;
    var pads = attrs.pads || [0, 0, 0, 0];
    var strides = attrs.strides || [1, 1];
    var dilations = attrs.dilations || [1, 1];
    var pt = pads[0];
    var pl = pads[1];
    var sh = strides[0];
    var sw = strides[1];
    var Hp = H + pt + pads[2];
    var Wp = W + pl + pads[3];
    var OH = Math.floor((Hp - KH) / sh) + 1;
    var OW = Math.floor((Wp - KW) / sw) + 1;
    var image = H * W;
    var outImage = OH * OW;
    var mult, out, pad, c, j, oc, y, i, src, dst;

    if (
        N !== 1 ||
        wShape[1] !== 1 ||
        group !== C ||
        (M % C) !== 0 ||
        dilations[0] !== 1 ||
        dilations[1] !== 1
    ) {
        return Base.convGroupedNCHW(x, w, bias, xShape, wShape, attrs);
    }

    mult = M / C;
    out = new Float32Array(M * outImage);
    pad = scratch("pad", Hp * Wp);

    /* The border is never written below, so clear it once per layer. */
    for (i = 0; i < Hp * Wp; ++i) {
        pad[i] = 0.0;
    }

    for (c = 0; c < C; ++c) {
        src = c * image;

        for (y = 0; y < H; ++y) {
            dst = (y + pt) * Wp + pl;
            for (i = 0; i < W; ++i) {
                pad[dst + i] = x[src + i];
            }
            src += W;
        }

        for (j = 0; j < mult; ++j) {
            oc = c * mult + j;

            if (sh === 1 && sw === 1 && KW === 7) {
                depthwisePlaneK7(
                    pad, Wp, w, oc * KH * 7, KH, bias[oc],
                    out, oc * outImage, OH, OW
                );
            } else if (sh === 1 && sw === 1 && KW === 3) {
                depthwisePlaneK3(
                    pad, Wp, w, oc * KH * 3, KH, bias[oc],
                    out, oc * outImage, OH, OW
                );
            } else {
                depthwisePlaneGeneric(
                    pad, Wp, w, oc * KH * KW, KH, KW, sh, sw, bias[oc],
                    out, oc * outImage, OH, OW
                );
            }
        }
    }

    return {
        data: out,
        shape: [1, M, OH, OW]
    };
}


/*
 * erf, Cephes double-precision rational approximation. Only used to
 * build the lookup table below.
 */
function polevl(x, coef) {
    var ans = coef[0];
    var i;

    for (i = 1; i < coef.length; ++i) {
        ans = ans * x + coef[i];
    }

    return ans;
}

function p1evl(x, coef) {
    var ans = x + coef[0];
    var i;

    for (i = 1; i < coef.length; ++i) {
        ans = ans * x + coef[i];
    }

    return ans;
}

var ERF_T = [
    9.60497373987051638749E0,
    9.00260197203842689217E1,
    2.23200534594684319226E3,
    7.00332514112805075473E3,
    5.55923013010394962768E4
];

var ERF_U = [
    3.35617141647503099647E1,
    5.21357949780142679795E2,
    4.59432382970980127987E3,
    2.26290000613890934246E4,
    4.92673942608635921086E4
];

var ERFC_P = [
    2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2
];

var ERFC_Q = [
    1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2
];

/* x >= 0 */
function erfPositive(x) {
    var z;

    if (x <= 1.0) {
        z = x * x;
        return x * polevl(z, ERF_T) / p1evl(z, ERF_U);
    }

    return 1.0 - Math.exp(-x * x) *
        polevl(x, ERFC_P) / p1evl(x, ERFC_Q);
}

/*
 * erf(t) for t in [0, ERF_LUT_RANGE], ERF_LUT_SCALE samples per unit,
 * linearly interpolated. The interpolation error is below 3e-8, under
 * half a float32 ulp of the result.
 */
var ERF_LUT_RANGE = 4.0;
var ERF_LUT_SCALE = 2048.0;
var erfLut = null;

function getErfLut() {
    var n;
    var i;

    if (!erfLut) {
        n = ERF_LUT_RANGE * ERF_LUT_SCALE + 2;
        erfLut = new Float32Array(n);

        for (i = 0; i < n; ++i) {
            erfLut[i] = erfPositive(i / ERF_LUT_SCALE);
        }
    }

    return erfLut;
}

/*
 * out[i] = x[i] * (erf(x[i] / divScalar) + addScalar) * mulScalar
 *
 * out may be x itself.
 */
function geluErfLut(x, out, n, divScalar, addScalar, mulScalar) {
    var lut = getErfLut();
    var inv = 1.0 / divScalar;
    var i, xv, d, u, k, e;

    for (i = 0; i < n; ++i) {
        xv = x[i];
        d = xv * inv;

        if (d < 0.0) {
            d = -d;

            if (d >= ERF_LUT_RANGE) {
                e = -1.0;
            } else {
                u = d * ERF_LUT_SCALE;
                k = u | 0;
                e = lut[k];
                e = -(e + (u - k) * (lut[k + 1] - e));
            }
        } else if (d >= ERF_LUT_RANGE) {
            e = 1.0;
        } else {
            u = d * ERF_LUT_SCALE;
            k = u | 0;
            e = lut[k];
            e = e + (u - k) * (lut[k + 1] - e);
        }

        out[i] = xv * (e + addScalar) * mulScalar;
    }
}

function geluSlice(x, shape, scalars) {
    var out = new Float32Array(x.length);

    geluErfLut(
        x, out, out.length,
        scalars.div, scalars.add, scalars.mul
    );

    return {
        data: out,
        shape: shape
    };
}


/*
 * Elementwise helpers. "Mid" broadcasting: the big operand is laid out
 * as [outer][C][inner] and the small one has C values. That covers
 * scalars, per-channel scales and trailing-dimension bias vectors.
 * out may be the big operand itself.
 */
function mulMid(big, small, outer, C, inner, out) {
    var i = 0;
    var o, c, k, s;

    for (o = 0; o < outer; ++o) {
        for (c = 0; c < C; ++c) {
            s = small[c];
            for (k = 0; k < inner; ++k) {
                out[i] = big[i] * s;
                ++i;
            }
        }
    }
}

function addMid(big, small, outer, C, inner, out) {
    var i = 0;
    var o, c, k, s;

    for (o = 0; o < outer; ++o) {
        for (c = 0; c < C; ++c) {
            s = small[c];
            for (k = 0; k < inner; ++k) {
                out[i] = big[i] + s;
                ++i;
            }
        }
    }
}

function mulSame(a, b, out, n) {
    var i;

    for (i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

function addSame(a, b, out, n) {
    var i;

    for (i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

/*
 * Transpose without a division per element per dimension.
 * Returns the output shape.
 */
function transpose(x, shape, perm, out) {
    var rank = shape.length;
    var inS = Base.computeStrides(shape);
    var outShape = new Array(rank);
    var step = new Array(rank);
    var coord = new Array(rank);
    var n = x.length;
    var last = rank - 1;
    var i = 0;
    var src = 0;
    var d, k, s, nLast, stLast;

    for (d = 0; d < rank; ++d) {
        outShape[d] = shape[perm[d]];
        step[d] = inS[perm[d]];
        coord[d] = 0;
    }

    nLast = outShape[last];
    stLast = step[last];

    while (i < n) {
        s = src;
        for (k = 0; k < nLast; ++k) {
            out[i++] = x[s];
            s += stLast;
        }

        for (d = last - 1; d >= 0; --d) {
            coord[d] += 1;
            src += step[d];
            if (coord[d] < outShape[d]) {
                break;
            }
            src -= step[d] * outShape[d];
            coord[d] = 0;
        }
    }

    return outShape;
}

/*
 * Mean over the trailing `inner` elements of each of `outer` blocks,
 * accumulated in float32 like the reference executor.
 */
function reduceMeanTrailing(x, outer, inner, out) {
    var i = 0;
    var o, k, acc;

    for (o = 0; o < outer; ++o) {
        acc = 0.0;
        for (k = 0; k < inner; ++k) {
            acc = Math.fround(acc + x[i]);
            ++i;
        }
        out[o] = acc / inner;
    }
}


/*
 * One kernel call described as plain data, so that the same slice of
 * work can run on this thread or be posted to a helper worker.
 * For "matmul", x is B and w holds the rows of A; "gelu" only uses x.
 */
function runSlice(t) {
    if (t.kind === "pointwise") {
        return convPointwiseNCHW(t.x, t.w, t.b, t.xShape, t.wShape);
    }
    if (t.kind === "depthwise") {
        return convDepthwiseNCHW(
            t.x, t.w, t.b, t.xShape, t.wShape, t.attrs
        );
    }
    if (t.kind === "dense") {
        return convDenseNCHW(
            t.x, t.w, t.b, t.xShape, t.wShape, t.attrs
        );
    }
    if (t.kind === "matmul") {
        return matMul(t.w, t.wShape, t.x, t.xShape);
    }
    if (t.kind === "gelu") {
        return geluSlice(t.x, t.xShape, t.attrs);
    }
    throw new Error("unknown slice kind: " + t.kind);
}


module.exports = {
    runSlice: runSlice,
    convPointwiseNCHW: convPointwiseNCHW,
    convDepthwiseNCHW: convDepthwiseNCHW,
    convDenseNCHW: convDenseNCHW,
    matMul: matMul,
    geluErfLut: geluErfLut,
    mulMid: mulMid,
    addMid: addMid,
    mulSame: mulSame,
    addSame: addSame,
    transpose: transpose,
    reduceMeanTrailing: reduceMeanTrailing
};

};
deps[3] = {'./mobileclip-fp32-kernels-v1.js':2};
modules[4] = function (module, exports, require) {
"use strict";

/*
 * Bookkeeping for the helper workers used by the v3 executor.
 *
 * The executor cuts a heavy kernel call into slices of output rows
 * (see runSlice in the v3 kernels). Each slice is self-contained: its
 * inputs are copied into fresh ArrayBuffers and transferred, the helper
 * computes it and transfers the result back. Helpers hold no model
 * state, so a helper that dies only costs the slice it was working on,
 * which is then recomputed on the calling thread.
 *
 * send(helper, message, transferList) is supplied by the caller. Old
 * browsers cannot start a worker from inside a worker, so in the
 * browser the page owns the helpers and relays these messages.
 */
function createPool(send) {
    var alive = [];
    var pending = {};
    var nextId = 1;

    function setAlive(helper, isAlive) {
        alive[helper] = isAlive === true;

        if (!alive[helper]) {
            failHelper(helper);
        }
    }

    /* Indices of the helpers that can take work. */
    function helpers() {
        var list = [];
        var k;

        for (k = 0; k < alive.length; ++k) {
            if (alive[k]) {
                list.push(k);
            }
        }

        return list;
    }

    /*
     * Posts one slice. onData(Float32Array) receives the result;
     * onFail() is called instead if the helper is lost. Returns false,
     * calling neither, if the slice could not be posted.
     */
    function submit(helper, slice, onData, onFail) {
        var id = nextId++;
        var x = new Float32Array(slice.x);
        var w = new Float32Array(slice.w);
        var b = new Float32Array(slice.b);

        try {
            send(
                helper,
                {
                    type: "slice",
                    id: id,
                    kind: slice.kind,
                    x: x.buffer,
                    w: w.buffer,
                    b: b.buffer,
                    xShape: slice.xShape,
                    wShape: slice.wShape,
                    attrs: slice.attrs
                },
                [x.buffer, w.buffer, b.buffer]
            );
        } catch (e) {
            alive[helper] = false;
            return false;
        }

        pending[id] = {
            helper: helper,
            onData: onData,
            onFail: onFail
        };

        return true;
    }

    function onResult(id, buffer) {
        var p = pending[id];

        /* Unknown id: a late answer from a helper already given up on. */
        if (p) {
            delete pending[id];
            p.onData(new Float32Array(buffer));
        }
    }

    function failHelper(helper) {
        var lost = [];
        var id;
        var i;

        alive[helper] = false;

        for (id in pending) {
            if (
                pending.hasOwnProperty(id) &&
                pending[id].helper === helper
            ) {
                lost.push(pending[id]);
                delete pending[id];
            }
        }

        for (i = 0; i < lost.length; ++i) {
            lost[i].onFail();
        }
    }

    /* Gives up on every helper that still owes a result. */
    function failPending() {
        var owing = [];
        var id;
        var i;

        for (id in pending) {
            if (pending.hasOwnProperty(id)) {
                owing.push(pending[id].helper);
            }
        }

        for (i = 0; i < owing.length; ++i) {
            failHelper(owing[i]);
        }
    }

    return {
        setAlive: setAlive,
        helpers: helpers,
        submit: submit,
        onResult: onResult,
        failHelper: failHelper,
        failPending: failPending
    };
}

module.exports = {
    createPool: createPool
};

};
deps[4] = {};

var cache = {};
function load(id) {
    var module;
    var localRequire;
    if (cache[id]) {
        return cache[id].exports;
    }
    module = {exports: {}};
    cache[id] = module;
    localRequire = function (name) {
        var target = deps[id][name];
        if (target === undefined) {
            throw new Error("bundle require not found: module=" + id + " name=" + name);
        }
        return load(target);
    };
    modules[id](module, module.exports, localRequire);
    return module.exports;
}
load(0);
}());
