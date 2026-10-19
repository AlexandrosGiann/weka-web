/* Weka Web — Video tutorials page. Videos and their metadata are listed in videos/index.js (works from file:// too). */
(function () {
  'use strict';
  const W = window.WEKA, App = window.App, h = App.h;
  const ts = { active: null };
  const pick = (o) => (o && typeof o === 'object' ? o[I18N.lang] || o.en : o || '');
  const fmtT = (s) => Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');

  App.pages.tutorials = {
    render() {
      const vids = window.WEKA_VIDEOS || [];
      const page = h('div.page', App.pageHead(L`Video tutorials`, L`Short screen recordings that show, step by step, how to use every part of the app.`));
      if (!vids.length) {
        page.appendChild(h('div.card', h('div.empty', App.icon('video'), h('h3', L`No videos found`), h('p', L`The videos folder is missing. Keep the “videos” folder next to index.html.`))));
        return page;
      }
      if (!ts.active || !vids.find((v) => v.id === ts.active)) ts.active = vids[0].id;
      const cur = vids.find((v) => v.id === ts.active);
      const file = typeof cur.file === 'object' ? cur.file[I18N.lang] || cur.file.en : cur.file;
      const video = h('video', { controls: true, preload: 'metadata', src: 'videos/' + file, style: { width: '100%', display: 'block', borderRadius: '12px', background: '#000', aspectRatio: '16 / 9' } });
      const chs = Array.isArray(cur.chapters) ? cur.chapters : (cur.chapters && (cur.chapters[I18N.lang] || cur.chapters.en)) || [];
      const chapters = h('div.history', chs.map((c) => h('button.result-item', { style: { padding: '7px 10px' }, on: { click: () => { video.currentTime = c.t; video.play(); } } },
        h('span.badge.primary', fmtT(c.t)), h('div.meta', h('div.t', { style: { fontWeight: 520 } }, pick(c.label))))));
      page.appendChild(h('div.grid', { style: { gridTemplateColumns: 'minmax(0,2fr) minmax(0,1fr)', gap: '18px', alignItems: 'start' }, class: 'tut-grid' },
        h('div.card', h('div.card-body', video, h('h3', { style: { marginTop: '14px', fontSize: '18px' } }, pick(cur.title)), h('p.muted', pick(cur.desc)),
          cur.dataset ? h('button.btn.sm', { on: { click: () => App.openSample(cur.dataset) } }, App.icon('data'), L`Open the dataset used in this video (${cur.dataset})`) : null)),
        h('div.card', App.cardHead('list', L`Chapters`), h('div.card-body', chapters.childNodes.length ? chapters : h('div.muted.small', '—')))));
      page.appendChild(h('div.samples', vids.map((v) => h('button.sample' + (v.id === ts.active ? '.active-sample' : ''), { on: { click: () => { ts.active = v.id; App.go('tutorials'); } } },
        h('div.row', { style: { flexWrap: 'nowrap' } }, h('span.badge.primary', fmtT((v.duration && typeof v.duration === 'object' ? v.duration[I18N.lang] || v.duration.en : v.duration) || 0)), h('div.n', pick(v.title))), h('div.d', pick(v.desc))))));
      return page;
    },
  };
})();
