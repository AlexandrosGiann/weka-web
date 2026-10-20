/* Weka Web — Web Worker for the Needle 2 assistant model (keeps the page responsive). */
'use strict';
importScripts('cact-full.js', 'tokenizer-full.js', 'engine-full.js', 'needle-runner.js');
var runner = null;
function post(m) { self.postMessage(m); }
async function download(url) {
  var res = await fetch(url);
  if (!res.ok) throw new Error('HTTP ' + res.status + ' for ' + url);
  var total = +res.headers.get('content-length') || 0, got = 0, chunks = [];
  var reader = res.body.getReader();
  for (;;) {
    var x = await reader.read();
    if (x.done) break;
    chunks.push(x.value); got += x.value.length;
    post({ type: 'progress', stage: 'download', fraction: total ? got / total : 0 });
  }
  var buf = new Uint8Array(got), o = 0;
  chunks.forEach(function (c) { buf.set(c, o); o += c.length; });
  return buf.buffer;
}
self.onmessage = async function (ev) {
  var m = ev.data;
  try {
    if (m.type === 'load') {
      var buf = m.buffer || await download('needle2.cact');
      runner = new NeedleRunner(buf);
      post({ type: 'ready' });
    } else if (m.type === 'run') {
      if (!runner) throw new Error('model is not loaded');
      var r = await runner.run(m.req, function (f, stage) { post({ type: 'progress', id: m.id, stage: stage, fraction: f }); }, null);
      post({ type: 'result', id: m.id, result: r });
    }
  } catch (e) {
    post({ type: 'error', id: m.id, message: String(e && e.message || e) });
  }
};
