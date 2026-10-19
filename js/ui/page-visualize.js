/* Weka Web — σελίδα Οπτικοποίησης: scatter plot matrix και αναλυτικό διάγραμμα. */
(function () {
  'use strict';
  const W = window.WEKA, U = W.Utils, App = window.App, h = App.h, C = App.charts;
  const vs = { selected: null, color: null, x: null, y: null, jitter: 0, size: 3, hidden: new Set(), cell: 120, relation: null };

  App.pages.visualize = {
    render() {
      const d = App.state.data;
      if (!d) return h('div.page', App.pageHead(L`Οπτικοποίηση`, L`Διαγράμματα διασποράς για κάθε ζεύγος χαρακτηριστικών.`), App.needData());
      const usable = d.attributes.map((a, j) => j).filter((j) => d.attributes[j].type !== 'string');
      if (vs.relation !== d || !vs.selected) {
        vs.selected = usable.slice(0, Math.min(6, usable.length));
        vs.relation = d;
        vs.color = d.classIndex;
        vs.x = usable[0]; vs.y = usable[Math.min(1, usable.length - 1)];
        vs.hidden.clear();
      }
      vs.selected = vs.selected.filter((j) => j < d.numAttributes);
      if (vs.color >= d.numAttributes) vs.color = -1;
      const page = h('div.page', App.pageHead(L`Οπτικοποίηση`, L`Διαγράμματα διασποράς για κάθε ζεύγος χαρακτηριστικών. Κλικ σε κελί για μεγέθυνση.`));
      const colorAtt = vs.color >= 0 ? d.attributes[vs.color] : null;
      const colorSel = h('select.select', { style: { width: 'auto' }, on: { change: (e) => { vs.color = Number(e.target.value); vs.hidden.clear(); App.go('visualize'); } } },
        h('option', { value: -1 }, L`Ενιαίο χρώμα`), usable.map((j) => h('option', { value: j, selected: j === vs.color }, L`Χρώμα: ` + d.attributes[j].name)));
      if (vs.color < 0) colorSel.value = -1;
      // χρώματα: nominal → παλέτα, numeric → κλίμακα
      let colors, legend = null;
      if (colorAtt && colorAtt.type === 'nominal') {
        colors = d.rows.map((r) => (r[vs.color] === r[vs.color] ? App.color(r[vs.color]) : '#9ca3af'));
        legend = C.legend(colorAtt.values, App.color, (i) => { if (vs.hidden.has(i)) vs.hidden.delete(i); else vs.hidden.add(i); big.redraw(); }, vs.hidden);
      } else if (colorAtt) {
        const st = d.attributeStats(vs.color);
        colors = d.rows.map((r) => { const v = r[vs.color]; if (v !== v) return '#9ca3af'; const t = (v - st.min) / (st.max - st.min || 1); return `hsl(${240 - t * 200} 75% 52%)`; });
        legend = h('div.legend', h('span', h('i', { style: { background: 'hsl(240 75% 52%)' } }), U.d2s(st.min, 3)), h('span', '→'), h('span', h('i', { style: { background: 'hsl(40 75% 52%)' } }), U.d2s(st.max, 3)));
      } else colors = d.rows.map(() => App.cssVar('--primary') || '#4f46e5');
      const hiddenFn = colorAtt && colorAtt.type === 'nominal' ? (i) => vs.hidden.has(d.rows[i][vs.color]) : null;

      // ------------------------------------------------- μεγάλο διάγραμμα
      const ax = d.attributes[vs.x], ay = d.attributes[vs.y];
      const axisSel = (k) => h('select.select', { style: { width: 'auto' }, on: { change: (e) => { vs[k] = Number(e.target.value); App.go('visualize'); } } }, usable.map((j) => h('option', { value: j, selected: j === vs[k] }, d.attributes[j].name)));
      const big = C.scatter({
        xs: d.rows.map((r) => r[vs.x]), ys: d.rows.map((r) => r[vs.y]), colors, hidden: hiddenFn,
        xLabel: ax.name, yLabel: ay.name, xCats: ax.type === 'nominal' ? ax.values : null, yCats: ay.type === 'nominal' ? ay.values : null,
        jitter: vs.jitter / 10, size: vs.size, height: 480,
        info: (i) => L`<b>Εγγραφή ${i + 1}</b><table>` + d.attributes.slice(0, 16).map((a, j) => `<tr><td style="color:var(--text-3)">${App.esc(a.name)}</td><td><b>${App.esc(d.stringValue(d.rows[i], j))}</b></td></tr>`).join('') + (d.numAttributes > 16 ? '<tr><td colspan=2>…</td></tr>' : '') + '</table>',
      });
      const jitter = h('input', { type: 'range', min: 0, max: 10, value: vs.jitter, style: { width: '120px', accentColor: 'var(--primary)' }, on: { input: (e) => { vs.jitter = Number(e.target.value); big.redraw({ jitter: vs.jitter / 10 }); } } });
      const size = h('input', { type: 'range', min: 1, max: 8, value: vs.size, style: { width: '100px', accentColor: 'var(--primary)' }, on: { input: (e) => { vs.size = Number(e.target.value); big.redraw({ size: vs.size }); } } });
      page.appendChild(h('div.card', App.cardHead('scatter', L`Διάγραμμα διασποράς`),
        h('div.card-body',
          h('div.row', { style: { marginBottom: '12px' } }, h('span.small.muted', 'X'), axisSel('x'), h('span.small.muted', 'Y'), axisSel('y'), colorSel,
            h('div', { style: { flex: 1 } }), h('span.small.muted', 'Jitter'), jitter, h('span.small.muted', L`Μέγεθος`), size),
          big, h('div.row', { style: { marginTop: '10px', justifyContent: 'space-between' } }, legend || h('span'), h('span.muted.small', L`Σύρε για zoom · κλικ στο υπόμνημα για απόκρυψη κλάσης`)))));

      // ------------------------------------------------- scatter matrix
      const pickList = h('div.pill-list', usable.map((j) => {
        const on = vs.selected.includes(j);
        return h('button.pill', { style: { border: '1px solid ' + (on ? 'var(--primary)' : 'transparent'), background: on ? 'var(--primary-soft)' : null, color: on ? 'var(--primary)' : null, cursor: 'pointer' },
          on: { click: () => { if (on) vs.selected = vs.selected.filter((x) => x !== j); else if (vs.selected.length < 12) vs.selected.push(j); else App.toast(L`Έως 12 χαρακτηριστικά στο scatter matrix.`); vs.selected.sort((a, b) => a - b); App.go('visualize'); } } }, d.attributes[j].name);
      }));
      const k = vs.selected.length;
      const cellSize = Math.max(70, Math.min(150, Math.floor(1000 / Math.max(1, k))));
      const grid = h('div.matrix-grid', { style: { gridTemplateColumns: `90px repeat(${k}, ${cellSize}px)`, gridAutoRows: `${cellSize}px` } });
      const cells = [];
      for (let yi = 0; yi < k; yi++) {
        const ja = vs.selected[k - 1 - yi];
        grid.appendChild(h('div.matrix-label', d.attributes[ja].name));
        for (let xi = 0; xi < k; xi++) {
          const jb = vs.selected[xi];
          const cv = h('canvas');
          const cell = h('div.matrix-cell', { title: `${d.attributes[jb].name} × ${d.attributes[ja].name}`, on: { click: () => { vs.x = jb; vs.y = ja; App.go('visualize'); } } }, cv);
          grid.appendChild(cell);
          cells.push([cv, jb, ja]);
        }
      }
      grid.appendChild(h('div'));
      for (let xi = 0; xi < k; xi++) grid.appendChild(h('div.matrix-label', { style: { height: '32px', alignSelf: 'start' } }, d.attributes[vs.selected[xi]].name));
      grid.style.gridAutoRows = '';
      grid.style.gridTemplateRows = `repeat(${k}, ${cellSize}px) 34px`;
      page.appendChild(h('div.card', App.cardHead('grid', 'Scatter plot matrix', h('span.muted.small', k + L` χαρακτηριστικά`)),
        h('div.card-body', h('div.field-label', { style: { marginBottom: '8px' } }, L`Χαρακτηριστικά στη μήτρα`), pickList, h('div', { style: { overflow: 'auto', marginTop: '14px' } }, grid))));
      requestAnimationFrame(() => {
        for (const [cv, jb, ja] of cells) {
          const xs = d.rows.map((r) => r[jb]), ys = d.rows.map((r) => r[ja]);
          C.miniScatter(cv, xs, ys, colors, d.attributes[jb].type === 'nominal' ? d.attributes[jb].values.length : 0, d.attributes[ja].type === 'nominal' ? d.attributes[ja].values.length : 0);
        }
      });
      return page;
    },
  };
})();
