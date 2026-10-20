/* Weka Web — "Web data": responsible import of public tables and open data (HTML tables, CSV, JSON, ARFF, Wikipedia).
 * Safeguards: explicit consent, robots.txt check, per-site and per-minute rate limits, size and time caps,
 * no cookies/credentials, no private-network addresses, no CORS proxies, source attribution in the report. */
(function () {
  'use strict';
  const W = window.WEKA, App = window.App, h = App.h;

  const LIMITS = { maxBytes: 10 * 1024 * 1024, timeoutMs: 20000, perHostGapMs: 5000, perMinute: 10, maxRows: 200000 };
  const UA_TOKEN = 'wekaweb';
  const ws = { tab: 'url', url: '', wikiLang: I18N.lang === 'el' ? 'el' : 'en', wikiTitle: '', consent: false, busy: false, result: null, log: [], robots: {}, lastByHost: {}, recent: [] };

  const EXAMPLES = [
    { kind: 'url', title: 'Palmer penguins', url: 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv', licence: 'CC0 1.0', desc: L`344 penguins, 3 species, bill and flipper measurements (CSV on GitHub).` },
    { kind: 'wiki', title: 'List of countries by life expectancy', lang: 'en', licence: 'CC BY-SA 4.0', desc: L`Wikipedia table via the official MediaWiki API.` },
    { kind: 'wiki', title: 'List of largest cities', lang: 'en', licence: 'CC BY-SA 4.0', desc: L`Population and area of the world's largest cities (Wikipedia).` },
  ];

  // ------------------------------------------------------------------ safety checks
  /** Accepts only public http(s) URLs without embedded credentials. */
  function checkUrl(raw) {
    let u;
    try { u = new URL(String(raw).trim()); } catch (e) { throw new Error(L`This is not a valid URL. It should start with https://`); }
    if (u.protocol !== 'https:' && u.protocol !== 'http:') throw new Error(L`Only http(s) addresses are allowed.`);
    if (u.username || u.password) throw new Error(L`URLs with a username or password are not allowed. Weka Web never logs in to websites.`);
    const host = u.hostname.toLowerCase().replace(/^\[|\]$/g, '');
    const priv = host === 'localhost' || /\.(local|localhost|internal|lan|home|intranet)$/.test(host) || host === '::1' || /^f[cd][0-9a-f]{2}:/.test(host) || /^fe80:/.test(host) ||
      /^(127\.|10\.|0\.|169\.254\.|192\.168\.)/.test(host) || /^172\.(1[6-9]|2\d|3[01])\./.test(host) || /^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\./.test(host);
    if (priv) throw new Error(L`Private or local network addresses are blocked. Only public websites can be used.`);
    return u;
  }

  /** robots.txt rules for our user agent (or *): longest match wins, Allow wins ties. */
  function robotsVerdict(text, path) {
    const groups = []; let cur = null, lastWasAgent = false;
    for (let line of text.split(/\r?\n/)) {
      line = line.replace(/#.*/, '').trim();
      const m = line.match(/^([A-Za-z-]+)\s*:\s*(.*)$/);
      if (!m) continue;
      const key = m[1].toLowerCase(), val = m[2].trim();
      if (key === 'user-agent') {
        if (!lastWasAgent || !cur) { cur = { agents: [], rules: [] }; groups.push(cur); }
        cur.agents.push(val.toLowerCase()); lastWasAgent = true;
      } else {
        lastWasAgent = false;
        if (cur && (key === 'allow' || key === 'disallow')) cur.rules.push({ allow: key === 'allow', path: val });
      }
    }
    const mine = groups.filter((g) => g.agents.some((a) => a !== '*' && UA_TOKEN.includes(a)));
    const use = mine.length ? mine : groups.filter((g) => g.agents.includes('*'));
    let best = null;
    for (const g of use) for (const r of g.rules) {
      if (!r.path) continue;
      let pat = r.path.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
      if (pat.endsWith('\\$')) pat = pat.slice(0, -2) + '$';
      const re = new RegExp('^' + pat);
      if (re.test(path) && (!best || r.path.length > best.path.length || (r.path.length === best.path.length && r.allow))) best = r;
    }
    return best ? { allowed: best.allow, rule: (best.allow ? 'Allow: ' : 'Disallow: ') + best.path } : { allowed: true, rule: '' };
  }

  async function checkRobots(u) {
    const origin = u.origin;
    if (!(origin in ws.robots)) {
      try {
        const res = await rawFetch(origin + '/robots.txt', 512 * 1024, true);
        ws.robots[origin] = res.status === 200 ? res.text : res.status >= 400 && res.status < 500 ? '' : null;
      } catch (e) { ws.robots[origin] = null; }
    }
    const txt = ws.robots[origin];
    if (txt === null) return { status: 'unknown' };
    const v = robotsVerdict(txt, u.pathname + u.search);
    return { status: v.allowed ? 'allowed' : 'disallowed', rule: v.rule };
  }

  async function throttle(host) {
    const now = Date.now();
    ws.recent = ws.recent.filter((t) => now - t < 60000);
    if (ws.recent.length >= LIMITS.perMinute) throw new Error(L`Rate limit: at most ${LIMITS.perMinute} requests per minute. Please wait a little.`);
    const wait = (ws.lastByHost[host] || 0) + LIMITS.perHostGapMs - now;
    if (wait > 0) { App.toast(L`Waiting ${Math.ceil(wait / 1000)} s to be polite to ${host}…`, 'info', wait); await new Promise((r) => setTimeout(r, wait)); }
    ws.lastByHost[host] = Date.now();
    ws.recent.push(Date.now());
  }

  /** GET without cookies or referrer, with a timeout and a byte cap. */
  async function rawFetch(url, maxBytes, quiet) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), LIMITS.timeoutMs);
    let res;
    try {
      res = await fetch(url, { method: 'GET', mode: 'cors', credentials: 'omit', referrerPolicy: 'no-referrer', cache: 'no-store', redirect: 'follow', signal: ctl.signal });
    } catch (e) {
      clearTimeout(timer);
      if (e.name === 'AbortError') throw new Error(L`The site did not answer within ${LIMITS.timeoutMs / 1000} seconds.`);
      const err = new Error(L`The browser could not read this address. Most often the site does not allow other web pages to read it (CORS), or you are offline. Weka Web does not bypass this with proxies: download the file yourself and open it with “Open”, or use the site's official API.`);
      err.cors = true; throw err;
    }
    if (res.url) { try { checkUrl(res.url); } catch (e) { clearTimeout(timer); ctl.abort(); throw e; } }
    const len = +res.headers.get('content-length');
    if (len && len > maxBytes) { clearTimeout(timer); ctl.abort(); throw new Error(L`The file is larger than the ${Math.round(maxBytes / 1048576)} MB limit.`); }
    const reader = res.body && res.body.getReader ? res.body.getReader() : null;
    let text;
    if (reader) {
      const chunks = []; let size = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.length;
        if (size > maxBytes) { ctl.abort(); clearTimeout(timer); throw new Error(L`The file is larger than the ${Math.round(maxBytes / 1048576)} MB limit.`); }
        chunks.push(value);
      }
      const buf = new Uint8Array(size); let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.length; }
      text = new TextDecoder('utf-8').decode(buf);
    } else text = await res.text();
    clearTimeout(timer);
    if (!quiet && !res.ok) throw new Error(L`The site answered with HTTP ${res.status}.`);
    return { status: res.status, text, type: (res.headers.get('content-type') || '').toLowerCase(), url: res.url || url, bytes: new Blob([text]).size };
  }

  /** The guarded fetch used by the page and the assistant. official = documented public API (robots.txt targets crawlers, not APIs). */
  async function guardedGet(rawUrl, opts = {}) {
    if (!ws.consent) throw new Error(L`Please confirm the responsible-use statement first.`);
    const u = checkUrl(rawUrl);
    const entry = { time: new Date(), host: u.host, path: u.pathname, status: '…', robots: opts.official ? 'API' : '…', bytes: 0 };
    ws.log.unshift(entry); ws.log = ws.log.slice(0, 30);
    try {
      if (!opts.official) {
        const rb = await checkRobots(u);
        entry.robots = rb.status;
        if (rb.status === 'disallowed') throw new Error(L`robots.txt of ${u.host} does not allow automated access to this page (${rb.rule}). Weka Web respects that.`);
        if (rb.status === 'unknown' && !opts.robotsUnknownOk) {
          const err = new Error(L`Could not read robots.txt of ${u.host}, so it is not clear whether automated access is welcome.`); err.robotsUnknown = true; throw err;
        }
      }
      await throttle(u.host);
      const res = await rawFetch(u.href, LIMITS.maxBytes);
      entry.status = res.status; entry.bytes = res.bytes;
      return res;
    } catch (e) { if (entry.status === '…') entry.status = e.cors ? 'CORS' : L`blocked`; throw e; }
  }

  // ------------------------------------------------------------------ parsing
  const MISSING = new Set(['', '?', '—', '–', '-', 'n/a', 'N/A', 'NA', 'null', 'None', '…']);
  function cleanCell(s) { return String(s).replace(/\[[^\]]{1,12}\]/g, '').replace(/ /g, ' ').replace(/\s+/g, ' ').trim(); }
  function asNumber(s) {
    let t = s.replace(/[−–]/g, '-').replace(/[\s ,]/g, '').replace(/^[$€£¥~≈]+/, '').replace(/%$/, '');
    if (/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(t)) return t;
    return null;
  }
  /** Turns a header + string rows into a Dataset (numeric columns detected after cleaning). */
  function tableToDataset(t) {
    const header = t.header.map((x, i) => cleanCell(x) || 'col' + (i + 1));
    const seen = {};
    const names = header.map((n) => { seen[n] = (seen[n] || 0) + 1; return seen[n] > 1 ? n + '_' + seen[n] : n; });
    const rows = t.rows.slice(0, LIMITS.maxRows).map((r) => names.map((_, j) => cleanCell(r[j] ?? '')));
    const numeric = names.map((_, j) => {
      let ok = 0, tot = 0;
      for (const r of rows) { const v = r[j]; if (MISSING.has(v)) continue; tot++; if (asNumber(v) !== null) ok++; }
      return tot > 0 && ok / tot >= 0.8;
    });
    const q = (s) => (/[",\n;\t]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s);
    const lines = [names.map(q).join(',')];
    for (const r of rows) lines.push(r.map((v, j) => (MISSING.has(v) ? '?' : numeric[j] ? asNumber(v) ?? '?' : q(v))).join(','));
    const d = W.parseCSV(lines.join('\n'), (t.name || 'web-table').replace(/[^\w.-]+/g, '_').slice(0, 60));
    return d;
  }

  function htmlTables(html, base) {
    const doc = new DOMParser().parseFromString(html, 'text/html'); // inert document: scripts never run
    doc.querySelectorAll('script, style, sup.reference, .reference, .sortkey, .mw-ref, .noprint, [style*="display:none"]').forEach((e) => e.remove());
    doc.querySelectorAll('br').forEach((b) => b.replaceWith(' '));
    const title = cleanCell((doc.querySelector('title') || {}).textContent || base || 'page');
    const out = [];
    doc.querySelectorAll('table').forEach((tb, ti) => {
      if (tb.querySelector('table')) return; // layout tables that contain tables
      const grid = [], isHead = [];
      [...tb.rows].forEach((tr, ri) => {
        grid[ri] = grid[ri] || [];
        let c = 0;
        const cells = [...tr.cells];
        isHead[ri] = cells.length > 0 && cells.every((x) => x.tagName === 'TH') || (tr.parentElement && tr.parentElement.tagName === 'THEAD');
        for (const cell of cells) {
          while (grid[ri][c] !== undefined) c++;
          const txt = cleanCell(cell.textContent);
          const cs = Math.min(+cell.getAttribute('colspan') || 1, 50), rs = Math.min(+cell.getAttribute('rowspan') || 1, 500);
          for (let a = 0; a < rs; a++) for (let b = 0; b < cs; b++) { grid[ri + a] = grid[ri + a] || []; grid[ri + a][c + b] = txt; }
          c += cs;
        }
      });
      if (grid.length < 3) return;
      let nh = 0; while (nh < grid.length - 1 && isHead[nh]) nh++;
      if (nh === 0) nh = 1;
      const width = Math.max(...grid.map((r) => r.length));
      if (width < 2) return;
      const header = [];
      for (let j = 0; j < width; j++) {
        const parts = [];
        for (let i = 0; i < nh; i++) { const v = (grid[i] || [])[j]; if (v && !parts.includes(v)) parts.push(v); }
        header.push(parts.join(' ') || 'col' + (j + 1));
      }
      const rows = grid.slice(nh).filter((r, i) => !isHead[i + nh] && r.filter((x) => x !== undefined && x !== '').length >= Math.max(2, width / 2));
      if (rows.length < 2) return;
      let name = cleanCell((tb.querySelector('caption') || {}).textContent || '');
      if (!name) {
        let p = tb.previousElementSibling, hops = 0;
        while (p && hops < 8 && !/^H[2-4]$/.test(p.tagName) && !p.querySelector('h2,h3,h4')) { p = p.previousElementSibling; hops++; }
        if (p) name = cleanCell((/^H[2-4]$/.test(p.tagName) ? p : p.querySelector('h2,h3,h4')).textContent).replace(/\[edit\]$/i, '');
      }
      name = name.length > 90 ? name.slice(0, 87) + '…' : name;
      out.push({ name: name || title + ' — table ' + (ti + 1), header, rows });
    });
    return { title, tables: out };
  }

  function jsonTable(text, name) {
    const root = JSON.parse(text);
    let best = null;
    const walk = (o, depth) => {
      if (depth > 6 || !o || typeof o !== 'object') return;
      if (Array.isArray(o)) {
        const objs = o.filter((x) => x && typeof x === 'object' && !Array.isArray(x));
        if (objs.length >= 2 && (!best || objs.length > best.length)) best = objs;
        else if (o.length >= 2 && o.every((x) => Array.isArray(x)) && (!best || o.length > best.length)) best = o;
        o.slice(0, 50).forEach((x) => walk(x, depth + 1));
      } else Object.values(o).forEach((x) => walk(x, depth + 1));
    };
    walk(root, 0);
    if (!best) throw new Error(L`No table-like array (a list of records) was found in this JSON.`);
    if (Array.isArray(best[0])) return { name, header: best[0].map(String), rows: best.slice(1).map((r) => r.map((v) => (v === null ? '' : String(v)))) };
    const flat = (o, p = '', acc = {}) => {
      for (const [k, v] of Object.entries(o)) {
        if (v && typeof v === 'object' && !Array.isArray(v) && p.split('.').length < 3) flat(v, p + k + '.', acc);
        else acc[p + k] = Array.isArray(v) ? v.filter((x) => typeof x !== 'object').join(' ') : v === null || v === undefined ? '' : String(v);
      }
      return acc;
    };
    const recs = best.map((o) => flat(o));
    const keys = [];
    for (const r of recs.slice(0, 2000)) for (const k of Object.keys(r)) if (!keys.includes(k)) keys.push(k);
    return { name, header: keys, rows: recs.map((r) => keys.map((k) => r[k] ?? '')) };
  }

  /** Detects the format of a response and returns {title, tables[]} or {dataset}. */
  function parseResponse(res, nameHint) {
    const t = res.text, type = res.type, path = new URL(res.url).pathname.toLowerCase();
    const name = nameHint || decodeURIComponent(path.split('/').pop() || new URL(res.url).hostname).replace(/\.[^.]+$/, '') || 'web';
    if (/@relation/i.test(t.slice(0, 4000)) && !/<html/i.test(t.slice(0, 500))) return { title: name, dataset: W.parseARFF(t) };
    if (type.includes('json') || path.endsWith('.json') || /^\s*[[{]/.test(t)) {
      try { return { title: name, tables: [jsonTable(t, name)] }; } catch (e) { if (type.includes('json')) throw e; }
    }
    if (type.includes('html') || /^\s*<(!doctype|html)/i.test(t)) return htmlTables(t, name);
    if (type.includes('csv') || /\.(csv|tsv|txt)$/.test(path) || type.includes('text/plain')) return { title: name, dataset: W.parseCSV(t, name) };
    throw new Error(L`Unsupported content (${type || L`unknown type`}). Supported: HTML tables, CSV/TSV, JSON and ARFF.`);
  }

  // ------------------------------------------------------------------ actions
  async function fetchUrl(url, opts = {}) {
    ws.busy = true; ws.result = null; App.go('web');
    try {
      const res = await guardedGet(url, opts);
      const parsed = parseResponse(res, opts.name);
      ws.result = Object.assign({ url: res.url, bytes: res.bytes, type: res.type, licence: opts.licence || '' }, parsed);
      if (!parsed.dataset && !(parsed.tables || []).length) throw new Error(L`No usable tables were found on this page (a table needs at least 2 columns and 2 data rows).`);
    } catch (e) {
      ws.result = { error: e.message, robotsUnknown: !!e.robotsUnknown, url, opts };
    } finally { ws.busy = false; App.go('web'); }
    return ws.result;
  }
  function wikiUrl(lang, title) {
    return `https://${lang}.wikipedia.org/w/api.php?action=parse&format=json&formatversion=2&origin=*&redirects=1&prop=text%7Cdisplaytitle&page=${encodeURIComponent(title.trim())}`;
  }
  async function fetchWiki(lang, title) {
    if (!/^[a-z]{2,3}$/.test(lang)) throw new Error('lang');
    ws.busy = true; ws.result = null; App.go('web');
    try {
      const res = await guardedGet(wikiUrl(lang, title), { official: true });
      const j = JSON.parse(res.text);
      if (j.error) throw new Error(L`Wikipedia: ${j.error.info || j.error.code}`);
      const page = j.parse.title;
      const parsed = htmlTables(j.parse.text, page);
      parsed.title = page;
      const human = `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(page.replace(/ /g, '_'))}`;
      ws.result = Object.assign({ url: human, bytes: res.bytes, type: 'Wikipedia (MediaWiki API)', licence: 'CC BY-SA 4.0 — ' + L`Wikipedia contributors` }, parsed);
      if (!parsed.tables.length) throw new Error(L`This article has no data tables.`);
    } catch (e) { ws.result = { error: e.message }; } finally { ws.busy = false; App.go('web'); }
    return ws.result;
  }
  function loadDataset(d, label) {
    const r = ws.result;
    App.loadNew(d, label);
    App.state.source = { url: r.url, time: new Date(), licence: r.licence || '' };
    App.state.ops[0].label = L`Imported from the web: ${new URL(r.url).host}`;
    App.toast(L`Remember to cite the source: ${r.url}`, 'info', 6000);
    App.go('preprocess');
  }

  // ------------------------------------------------------------------ page
  function preview(t) {
    const cols = t.header.slice(0, 8);
    return h('div.table-wrap', { style: { maxHeight: '220px', border: '1px solid var(--border)', borderRadius: '10px' } },
      h('table.tbl', h('thead', h('tr', cols.map((c) => h('th', c)), t.header.length > 8 ? h('th', '…') : null)),
        h('tbody', t.rows.slice(0, 6).map((r) => h('tr', cols.map((_, j) => h('td', cleanCell(r[j] ?? '').slice(0, 40))), t.header.length > 8 ? h('td', '…') : null)))));
  }
  function resultCard() {
    const r = ws.result;
    if (ws.busy) return h('div.card', h('div.card-body', h('div.status-line', App.icon('globe'), L`Fetching… (checking robots.txt and rate limits first)`)));
    if (!r) return null;
    if (r.error) {
      const body = h('div.card-body', h('div.callout.warn', App.icon('info'), h('div', r.error)));
      if (r.robotsUnknown) body.appendChild(h('div', { style: { marginTop: '12px' } },
        h('p.small.muted', L`If you have checked the site's terms of use and they allow it, you can continue once.`),
        h('button.btn', { on: { click: () => fetchUrl(r.url, Object.assign({}, r.opts, { robotsUnknownOk: true })) } }, L`I checked the terms — fetch once`)));
      return h('div.card', App.cardHead('info', L`Could not import`), body);
    }
    const head = h('div.small.muted', { style: { marginBottom: '12px', wordBreak: 'break-all' } }, L`Source: `, h('a', { href: r.url, target: '_blank', rel: 'noopener noreferrer' }, r.url), ` · ${(r.bytes / 1024).toFixed(1)} KB` + (r.licence ? ' · ' + r.licence : ''));
    if (r.dataset) {
      const d = r.dataset;
      return h('div.card', App.cardHead('data', r.title), h('div.card-body', head,
        h('p', L`${d.numInstances} instances · ${d.numAttributes} attributes`),
        h('button.btn.primary', { on: { click: () => loadDataset(d, r.title) } }, App.icon('upload'), L`Load as dataset`)));
    }
    return h('div.card', App.cardHead('table', L`${r.tables.length} table(s) found in “${r.title}”`), h('div.card-body', head,
      ...r.tables.map((t, i) => h('div', { style: { marginBottom: '18px' } },
        h('div.row', { style: { marginBottom: '6px' } }, h('strong', t.name), h('span.badge', L`${t.rows.length} rows × ${t.header.length} columns`), h('div.spacer'),
          h('button.btn.sm.primary', { on: { click: () => { try { const d = tableToDataset(t); loadDataset(d, t.name); } catch (e) { App.error(e); } } } }, App.icon('upload'), L`Load as dataset`)),
        preview(t)))));
  }

  App.pages.web = {
    render() {
      const page = h('div.page', App.pageHead(L`Web data`, L`Import tables from public web pages and open-data sources — responsibly.`));
      page.appendChild(h('div.card', App.cardHead('shield', L`Responsible use`), h('div.card-body',
        h('ul', { style: { margin: '0 0 12px', paddingLeft: '20px', lineHeight: 1.65 } },
          h('li', L`Only public pages: no logins, cookies or passwords are ever sent, and local/private network addresses are blocked.`),
          h('li', L`robots.txt is checked before reading any page; paths it disallows are refused.`),
          h('li', L`At most one request every ${LIMITS.perHostGapMs / 1000} s per site and ${LIMITS.perMinute} per minute; downloads are limited to ${LIMITS.maxBytes / 1048576} MB and ${LIMITS.timeoutMs / 1000} s.`),
          h('li', L`If a site does not allow browser access (CORS), it is not bypassed with proxies — download the file yourself or use the site's official API.`),
          h('li', L`Respect the site's terms of use and licence, avoid personal data (GDPR), and cite the source — it is added to your report automatically.`)),
        h('label.check', h('input', { type: 'checkbox', checked: ws.consent, on: { change: (e) => { ws.consent = e.target.checked; App.go('web'); } } }),
          L`I will only import public data that I am allowed to use, and I will respect the site's terms and licence.`))));

      const tabs = h('div.seg', [['url', L`Page or file URL`], ['wiki', 'Wikipedia'], ['examples', L`Open-data examples`]].map(([k, l]) =>
        h('button' + (ws.tab === k ? '.active' : ''), { type: 'button', on: { click: () => { ws.tab = k; App.go('web'); } } }, l)));
      const body = h('div.card-body', h('div', { style: { marginBottom: '14px' } }, tabs));
      const dis = !ws.consent || ws.busy;
      if (ws.tab === 'url') {
        const inp = h('input.input', { type: 'url', placeholder: 'https://…', value: ws.url, on: { input: (e) => { ws.url = e.target.value; } }, style: { flex: 1, minWidth: '220px' } });
        inp.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !dis) fetchUrl(ws.url); });
        body.append(h('div.field', h('label', L`Address of a web page with tables, or of a CSV / JSON / ARFF file`),
          h('div.row', { style: { flexWrap: 'wrap' } }, inp, h('button.btn.primary', { disabled: dis, on: { click: () => fetchUrl(ws.url) } }, App.icon('globe'), L`Fetch`)),
          h('div.hint', L`Tip: many open-data portals (data.gov, data.europe.eu, GitHub raw files) allow browser access.`)));
      } else if (ws.tab === 'wiki') {
        const sel = h('select.select', { style: { width: '90px' }, on: { change: (e) => { ws.wikiLang = e.target.value; } } }, ['en', 'el', 'de', 'fr', 'es', 'it'].map((l) => h('option', { value: l, selected: l === ws.wikiLang }, l)));
        const inp = h('input.input', { placeholder: L`Article title, e.g. List of countries by GDP (nominal)`, value: ws.wikiTitle, on: { input: (e) => { ws.wikiTitle = e.target.value; } }, style: { flex: 1, minWidth: '220px' } });
        inp.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !dis && ws.wikiTitle.trim()) fetchWiki(ws.wikiLang, ws.wikiTitle); });
        body.append(h('div.field', h('label', L`Wikipedia article`), h('div.row', { style: { flexWrap: 'wrap' } }, sel, inp,
          h('button.btn.primary', { disabled: dis, on: { click: () => ws.wikiTitle.trim() && fetchWiki(ws.wikiLang, ws.wikiTitle) } }, App.icon('globe'), L`Fetch tables`)),
          h('div.hint', L`Uses the official MediaWiki API. Wikipedia text is licensed CC BY-SA 4.0 — credit “Wikipedia contributors”.`)));
      } else {
        body.append(h('div.samples', EXAMPLES.map((x) => h('button.sample', { disabled: dis, on: { click: () => (x.kind === 'wiki' ? fetchWiki(x.lang, x.title) : fetchUrl(x.url, { licence: x.licence, name: x.title })) } },
          h('div.n', x.title), h('div.d', x.desc), h('div.tags', h('span.badge', x.kind === 'wiki' ? 'Wikipedia' : 'CSV'), h('span.badge.primary', x.licence))))));
      }
      if (!ws.consent) body.append(h('div.callout', { style: { marginTop: '10px' } }, App.icon('info'), h('div', L`Tick the responsible-use statement above to enable fetching.`)));
      page.appendChild(h('div.card', App.cardHead('globe', L`Import`), body));
      const rc = resultCard(); if (rc) page.appendChild(rc);
      if (ws.log.length) page.appendChild(h('div.card', App.cardHead('list', L`Request log (this session)`), h('div.card-body', h('div.table-wrap', h('table.tbl',
        h('thead', h('tr', h('th', L`Time`), h('th', 'Host'), h('th', L`Path`), h('th', 'robots.txt'), h('th', 'HTTP'), h('th.num', 'KB'))),
        h('tbody', ws.log.map((e) => h('tr', h('td', App.timeStr(e.time)), h('td', e.host), h('td', e.path.slice(0, 50)), h('td', e.robots), h('td', String(e.status)), h('td.num', (e.bytes / 1024).toFixed(1))))))))));
      return page;
    },
  };

  /** Used by the assistant: opens the page with an address filled in (the user still confirms and clicks Fetch). */
  App.web = {
    prefill(url) { ws.tab = 'url'; ws.url = url; App.go('web'); },
    prefillWiki(title, lang) { ws.tab = 'wiki'; ws.wikiTitle = title; if (lang) ws.wikiLang = lang; App.go('web'); },
    get consent() { return ws.consent; },
    fetchUrl, fetchWiki, LIMITS,
    _test: { checkUrl, robotsVerdict, htmlTables, jsonTable, tableToDataset, asNumber },
  };
})();
