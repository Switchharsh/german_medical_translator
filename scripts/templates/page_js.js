/* ================= 2026-10 update: new metrics, control, agreement, perturbation, report types ================= */
const FAMC = {surface:'var(--surface-hue)', learned:'var(--semantic-hue)', cef:'var(--cef-hue)', detectors:'var(--clinical-hue)'};
const FAMN = {surface:'Surface overlap', learned:'Learned semantic', cef:'Cross-examination (CEF)', detectors:'Clinical detectors'};
const MET = {
  crit:['crit%',0,'detectors',1], words:['words%',0,'detectors',1], numbers:['numbers%',0,'detectors',1], terms:['term%',0,'detectors',1],
  bleu:['BLEU',1,'surface',1], chrf:['chrF++',1,'surface',1], ter:['TER',0,'surface',1],
  comet:['COMET',1,'learned',3], xcomet:['xCOMET',1,'learned',3], xcomet_qe:['xCOMET-QE',1,'learned',3], xcomet_mqm:['xCOMET-MQM',1,'learned',3],
  metricx_ref:['MetricX',0,'learned',2], metricx_qe:['MetricX-QE',0,'learned',2],
  cef_conformity:['CEF conformity',1,'cef',1], cef_coverage:['CEF coverage',1,'cef',1], cef_consistency:['CEF consistency',1,'cef',1],
};
const SYS = Object.keys(D.singlepass).filter(m => m !== 'identity');
const $ = id => document.getElementById(id);
const hasV = v => v !== null && v !== undefined && !Number.isNaN(v);
const val = (s, k) => D.singlepass[s][k];
const fm = (k, v) => hasV(v) ? Number(v).toFixed(MET[k][3]) : '&mdash;';
const arrow = k => MET[k][1] ? '&uarr;' : '&darr;';
const better = (k, a, b) => MET[k][1] ? a > b : a < b;
const ordered = (k, list = SYS) => list.filter(s => hasV(val(s, k))).sort((a, b) => MET[k][1] ? val(b, k) - val(a, k) : val(a, k) - val(b, k));
const escH = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const blab = s => escH(String(s).replace('<=', '\u2264 ').replace(/^>/, '> ').replace('-', '\u2013'));   // length buckets, e.g. '<=500' -> '\u2264 500'
const opts = (arr, sel) => arr.map(([v, l]) => `<option value="${v}"${v === sel ? ' selected' : ''}>${l}</option>`).join('');

/* ---- 1. who wins, by metric ---- */
const RK = [['surface','BLEU','bleu'],['learned','xCOMET','xcomet'],['learned','MetricX','metricx_ref'],
            ['cef','CEF conformity','cef_conformity'],['detectors','words%','words'],['detectors','crit%','crit']];
$('rankings').innerHTML = RK.map(([fam, label, k]) =>
  `<div class="rankcol fam-${fam}"><h3>${label}</h3><ol>${ordered(k).slice(0, 5).map(s => `<li>${s}</li>`).join('')}</ol></div>`).join('');
{
  const best = D.ranks.best, ms = D.ranks.metrics, winners = {};
  ms.forEach(m => { (winners[best[m.key]] = winners[best[m.key]] || []).push(m.label); });
  const n = Object.keys(winners).length;
  $('dis-callout').innerHTML = `<b>${ms.length} metrics name ${n} different best systems among ${D.ranks.systems.length}.</b> ` +
    Object.entries(winners).map(([s, l]) => `<span class="mono">${s}</span>: ${l.join(', ')}`).join(' &middot; ') +
    `. The surface metrics agree with each other far more than they agree with anything else (see the agreement matrix), and the clinical detectors and the cross-examination metric each rank the systems differently again.`;
}

/* ---- 2. every metric, every system, sortable ---- */
const SPCOLS = ['crit','words','numbers','terms','bleu','chrf','ter','comet','xcomet','xcomet_qe','metricx_ref','metricx_qe','cef_conformity','cef_coverage','cef_consistency'];
let spSort = 'words';
function renderSP() {
  const best = {};
  SPCOLS.forEach(k => { const o = ordered(k); best[k] = o.length ? val(o[0], k) : null; });
  const order = ordered(spSort).concat(SYS.filter(s => !hasV(val(s, spSort))));
  const head = SPCOLS.map(k => `<th class="sortable f-${MET[k][2]}${k === spSort ? ' sorted' : ''}" data-k="${k}" title="${FAMN[MET[k][2]]} &mdash; click to sort best first">${MET[k][0]} ${arrow(k)}</th>`).join('');
  const row = (s, label, cls) => `<tr class="${cls || ''}"><td class="name">${label}</td>` +
    SPCOLS.map(k => `<td class="${!cls && hasV(val(s, k)) && val(s, k) === best[k] ? 'best' : ''}">${fm(k, val(s, k))}</td>`).join('') + `<td>${D.singlepass[s].n}</td></tr>`;
  $('sp-table').innerHTML = `<thead><tr><th>System</th>${head}<th>n</th></tr></thead><tbody>` +
    order.map(s => row(s, nm(s))).join('') + row('identity', 'identity (control)', 'control') + '</tbody>';
}
$('sp-table').addEventListener('click', e => { const th = e.target.closest && e.target.closest('th.sortable'); if (th) { spSort = th.dataset.k; renderSP(); } });
renderSP();

/* ---- 3. the control: does each metric put the untranslated source last? ---- */
{
  const fams = ['surface','learned','cef','detectors'];
  const ks = SPCOLS.filter(k => k !== 'numbers' && ordered(k).length >= 10 && hasV(val('identity', k)))   // numbers% is excluded: a copy keeps every number, so it scores 0 by construction.sort((a, b) => fams.indexOf(MET[a][2]) - fams.indexOf(MET[b][2]));
  let pass = 0;
  const body = ks.map(k => {
    const o = ordered(k), idv = val('identity', k), beats = o.filter(s => better(k, idv, val(s, k))).length, ok = beats === 0;
    pass += ok;
    return `<tr><td style="color:${FAMC[MET[k][2]]}">${MET[k][0]}</td><td>${fm(k, idv)}</td><td>${fm(k, val(o[o.length - 1], k))}</td><td>${fm(k, val(o[0], k))}</td>` +
      `<td>${beats} of ${o.length}</td><td style="text-align:left;white-space:normal"><span class="chip ${ok ? 'pass' : 'fail'}">${ok ? 'passes' : 'fails'}</span></td></tr>`;
  }).join('');
  $('ctl-table').innerHTML = `<thead><tr><th>Metric</th><th>identity</th><th>weakest real</th><th>best real</th><th>real systems identity beats</th><th style="text-align:left">Control</th></tr></thead><tbody>${body}</tbody>`;
  const failed = ks.filter(k => ordered(k).filter(s => better(k, val('identity', k), val(s, k))).length > 0).map(k => MET[k][0]);
  $('ctl-note').innerHTML = `<b>${pass} of ${ks.length} metrics put the untranslated source last; ${failed.join(', ')} do not.</b> ` +
    `The control returns the German input unchanged for a German&rarr;English task, so any working metric must rank it below every real translation. ` +
    `The cross-examination scores compare what two texts say and never check the language, so an untranslated copy trivially &ldquo;conforms&rdquo;; xCOMET, run without a reference, also prefers it. ` +
    `Both need a language check beside them before their scores can be trusted at the top of a ranking. numbers% is left out of this test: an untranslated copy keeps every number, so it scores zero errors by construction.`;
}

/* ---- 4. do the metrics agree? ---- */
{
  const order = ['bleu','chrf','ter','comet','xcomet','xcomet_qe','metricx_ref','metricx_qe','cef_conformity','cef_coverage','cef_consistency','crit','words'].filter(k => D.ranks.spearman[k]);
  const n = order.length, cell = 42, left = 20, top = 96, W = left + n * cell + 8, H = top + n * cell + 20;
  let s = `<svg viewBox="0 0 ${W + 110} ${H}" role="img" aria-label="Spearman rank correlation between metrics across systems">`;
  order.forEach((k, j) => { s += `<text class="lab" transform="translate(${left + j * cell + cell / 2 + 4},${top - 8}) rotate(-50)" style="fill:${FAMC[MET[k][2]]}">${MET[k][0]}</text>`; });
  order.forEach((a, i) => {
    s += `<text class="lab" x="${left + n * cell + 8}" y="${top + i * cell + cell / 2 + 4}" style="fill:${FAMC[MET[a][2]]}">${MET[a][0]}</text>`;
    order.forEach((b, j) => {
      const r = D.ranks.spearman[a][b];
      if (r === null || r === undefined) return;
      const col = r >= 0 ? 'var(--semantic-hue)' : 'var(--clinical-hue)';
      s += `<rect x="${left + j * cell}" y="${top + i * cell}" width="${cell - 2}" height="${cell - 2}" rx="4" fill="${col}" fill-opacity="${(Math.min(1, Math.abs(r)) * 0.82).toFixed(2)}"><title>${MET[a][0]} vs ${MET[b][0]}: rho ${r.toFixed(2)}</title></rect>` +
           `<text class="val" x="${left + j * cell + (cell - 2) / 2}" y="${top + i * cell + cell / 2 + 3}" text-anchor="middle">${r.toFixed(2)}</text>`;
    });
  });
  $('heat').innerHTML = s + '</svg>';
  const fam = Object.entries(D.ranks.family_mean_rho).map(([k, v]) => { const [a, b] = k.split('|'); return {a, b, v, n: D.ranks.family_pairs[k]}; }).sort((x, y) => y.v - x.v);
  $('fam-rho').innerHTML = `<thead><tr><th>Family pair</th><th>mean rho</th><th>metric pairs</th></tr></thead><tbody>` +
    fam.map(x => `<tr><td>${FAMN[x.a]}${x.a === x.b ? ' (within)' : ' &harr; ' + FAMN[x.b]}</td><td>${x.v.toFixed(2)}</td><td>${x.n}</td></tr>`).join('') + '</tbody>';
}

/* ---- 5. round trip: learned metrics along the cycles ---- */
const RTM = [
  ['bleu', 'BLEU', s => D.roundtrip[s] && D.roundtrip[s].bleu, 'surface', 1],
  ['xcomet', 'xCOMET', s => D.rt_learned[s] && D.rt_learned[s].xcomet, 'learned', 3],
  ['metricx_ref', 'MetricX (lower is better)', s => D.rt_learned[s] && D.rt_learned[s].metricx_ref, 'learned', 2],
  ['cef_conformity', 'CEF conformity', s => D.rt_learned[s] && D.rt_learned[s].cef_conformity, 'cef', 1],
  ['cef_coverage', 'CEF coverage', s => D.rt_learned[s] && D.rt_learned[s].cef_coverage, 'cef', 1],
];
function lineFig(title, get, fam, dec, sel) {
  const W = 320, H = 190, l = 44, r = 64, t = 8, b = 28, sys = SYS.filter(s => get(s) && get(s).length >= 2);
  const all = sys.flatMap(s => get(s)), lo0 = Math.min(...all), hi0 = Math.max(...all), pad = (hi0 - lo0) * 0.08 || 1, lo = lo0 - pad, hi = hi0 + pad;
  const X = i => l + i * (W - l - r) / 9, Y = v => t + (hi - v) * (H - t - b) / (hi - lo);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title} over ten round-trip cycles">`;
  for (let g = 0; g <= 3; g++) { const v = lo + (hi - lo) * g / 3; s += `<line class="gl" x1="${l}" x2="${W - r}" y1="${Y(v)}" y2="${Y(v)}"/><text class="mut" x="${l - 6}" y="${Y(v) + 3}" text-anchor="end">${v.toFixed(dec)}</text>`; }
  [1, 5, 10].forEach(c => { s += `<text class="mut" x="${X(c - 1)}" y="${H - 8}" text-anchor="middle">${c}</text>`; });
  s += `<text class="mut" x="${l + (W - l - r) / 2}" y="${H - 0}" text-anchor="middle">cycle</text>`;
  sys.filter(x => x !== sel).forEach(x => { s += `<polyline fill="none" stroke="var(--line-strong)" stroke-width="1.2" points="${get(x).map((v, i) => `${X(i)},${Y(v)}`).join(' ')}"><title>${x}</title></polyline>`; });
  if (get(sel)) {
    const v = get(sel);
    s += `<polyline fill="none" stroke="${FAMC[fam]}" stroke-width="2.6" stroke-linejoin="round" points="${v.map((q, i) => `${X(i)},${Y(q)}`).join(' ')}"/>` +
         v.map((q, i) => `<circle cx="${X(i)}" cy="${Y(q)}" r="2.6" fill="${FAMC[fam]}" stroke="var(--surface)" stroke-width="1"/>`).join('') +
         `<text class="val" x="${X(9) + 6}" y="${Y(v[9]) + 3}">${v[9].toFixed(dec)}</text>`;
  }
  return `<div class="fig"><h3>${title}</h3>${s}</svg></div>`;
}
function renderRT() {
  const sel = $('rt-sys').value;
  $('rt-figs').innerHTML = RTM.map(([k, label, get, fam, dec]) => lineFig(label, get, fam, dec, sel)).join('');
}
$('rt-sys').innerHTML = opts(SYS.map(s => [s, s]), 'deepl');
$('rt-sys').addEventListener('change', renderRT);
renderRT();

/* ---- 6. perturbation sensitivity ---- */
const PNAME = {'DeepSeek-V4-Flash@de->en': 'DeepSeek output, DE→EN', 'ref@de->en': 'Human reference, DE→EN', 'ref@en->de': 'Human reference, EN→DE'};
const PORD = [['bleu','BLEU','surface'],['chrf','chrF++','surface'],['ter','TER','surface'],['xcomet','xCOMET','learned'],['xcomet_qe','xCOMET-QE','learned'],
  ['xcomet_mqm','xCOMET-MQM','learned'],['metricx_ref','MetricX','learned'],['metricx_qe','MetricX-QE','learned'],['cef_conformity','CEF conformity','cef'],
  ['cef_coverage','CEF coverage','cef'],['cef_consistency','CEF consistency','cef']];
const PBASES = Object.keys(D.perturb);
function renderPerturb() {
  const key = $('pb-base').value, b = D.perturb[key];
  if (!b) { $('pb-auc').innerHTML = '<p class="note">No perturbation data.</p>'; return; }
  // AUC bars
  const rows = PORD.map(([k, label, fam]) => ({k, label, fam, a: (b.summary.find(x => x.metric === k) || {}).auc})).filter(x => hasV(x.a));
  const W = 440, lw = 118, bw = W - lw - 40, rh = 24;
  let s = `<svg viewBox="0 0 ${W} ${rows.length * rh + 34}" role="img" aria-label="Probability that a metric penalises a dangerous edit more than a harmless one">`;
  [0, 0.25, 0.5, 0.75, 1].forEach(g => { s += `<line class="${g === 0.5 ? 'ref' : 'gl'}" x1="${lw + g * bw}" x2="${lw + g * bw}" y1="0" y2="${rows.length * rh}"/><text class="mut" x="${lw + g * bw}" y="${rows.length * rh + 14}" text-anchor="middle">${g}</text>`; });
  s += `<text class="mut" x="${lw + bw / 2}" y="${rows.length * rh + 30}" text-anchor="middle">P(dangerous edit penalised more than a harmless one)</text>`;
  rows.forEach((x, i) => { const y = i * rh;
    s += `<text class="lab" x="${lw - 8}" y="${y + 15}" text-anchor="end">${x.label}</text>` +
         `<rect x="${lw}" y="${y + 4}" width="${Math.max(0, x.a) * bw}" height="${rh - 9}" rx="3" fill="${FAMC[x.fam]}" fill-opacity="0.85"><title>${x.label}: ${x.a.toFixed(3)}</title></rect>` +
         `<text class="val" x="${lw + x.a * bw + 5}" y="${y + 15}">${x.a.toFixed(2)}</text>`; });
  $('pb-auc').innerHTML = s + '</svg>';
  // per-kind table
  const KN = [['negation_drop','dropped negation'],['laterality_flip','flipped side'],['number_change','changed number'],['harmless','harmless rewording']];
  // Shading = how much MORE often the metric scores this kind of edit worse than it scores a harmless rewording
  // (percentage points over its own baseline, full strength at 50). Plain "share scored worse" is not enough: a metric can
  // mark 53% of harmless edits worse by a thousandth of a point. Effect size z is printed beside it.
  const cell = (k, kind) => { const x = b.kinds.find(y => y.metric === k && y.kind === kind); if (!x) return '<td>&mdash;</td>';
    const h = b.kinds.find(y => y.metric === k && y.kind === 'harmless'), excess = kind === 'harmless' || !h ? 0 : Math.max(0, x.caught - h.caught);
    const shade = Math.round(Math.min(1, excess / 50) * 45), st = shade ? ` style="background:color-mix(in srgb, var(--semantic-hue) ${shade}%, transparent)"` : '';
    return `<td${st}>${x.caught}% <span class="minor">z ${x.z === null ? '&mdash;' : x.z.toFixed(2)}</span></td>`; };
  $('pb-kinds').innerHTML = `<thead><tr><th>Metric</th>${KN.map(([, l]) => `<th>${l}</th>`).join('')}</tr></thead><tbody>` +
    PORD.map(([k, label, fam]) => `<tr><td style="color:${FAMC[fam]}">${label}</td>${KN.map(([kind]) => cell(k, kind)).join('')}</tr>`).join('') +
    `<tr class="control"><td>rule-based detectors</td>${KN.map(([kind]) => { const d = b.detectors[kind]; return d ? `<td>${Math.round(100 * d.hit / d.n)}% <span class="minor">${kind === 'harmless' ? 'false alarms' : 'recall'}</span></td>` : '<td>&mdash;</td>'; }).join('')}</tr></tbody>`;
}
$('pb-base').innerHTML = opts(PBASES.map(k => [k, PNAME[k] || k]), PBASES.includes('DeepSeek-V4-Flash@de->en') ? 'DeepSeek-V4-Flash@de->en' : PBASES[0]);
$('pb-base').addEventListener('change', renderPerturb);
renderPerturb();
{ // detector recall by how many facts the report contains
  const R = D.recall.kinds, groups = [
    ['negation_drop', 'Dropped negation', ['1 negation', '2', '3', '4 or more'], ['1','2','3','4']],
    ['laterality_flip', 'Flipped side', ['one side mentioned', 'both sides'], ['1','2']],
    ['number_change', 'Changed measurement', ['1', '2', '3', '4 or more'], ['1','2','3','4']]];
  const W = 320, H = 190, l = 36, t = 10, b = 44;
  $('pb-recall').innerHTML = groups.map(([kind, title, labels, keys]) => {
    const bars = keys.map(k => R[kind][k]).map(x => x ? Math.round(100 * x.caught / x.n) : null), ns = keys.map(k => R[kind][k] ? R[kind][k].n : 0);
    const bw = (W - l - 8) / keys.length;
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Detector recall for ${title}">`;
    [0, 50, 100].forEach(g => { const y = t + (100 - g) * (H - t - b) / 100; s += `<line class="gl" x1="${l}" x2="${W}" y1="${y}" y2="${y}"/><text class="mut" x="${l - 5}" y="${y + 3}" text-anchor="end">${g}%</text>`; });
    bars.forEach((v, i) => { if (v === null) return; const h = v * (H - t - b) / 100, x = l + i * bw + bw * 0.18, y = t + (H - t - b) - h;
      s += `<rect x="${x}" y="${y}" width="${bw * 0.64}" height="${Math.max(h, 1)}" rx="3" fill="var(--clinical-hue)" fill-opacity="0.85"/>` +
           `<text class="val" x="${x + bw * 0.32}" y="${y - 4}" text-anchor="middle">${v}%</text>` +
           `<text class="mut" x="${x + bw * 0.32}" y="${H - b + 14}" text-anchor="middle">${labels[i]}</text><text class="mut" x="${x + bw * 0.32}" y="${H - b + 27}" text-anchor="middle">n=${ns[i]}</text>`; });
    return `<div class="fig"><h3>${title}</h3>${s}</svg></div>`; }).join('');
}

/* ---- 7. report types ---- */
function hbars(items, max, fmtv, color) {
  const lw = 128, W = 330, bw = W - lw - 56, rh = 22;
  let s = `<svg viewBox="0 0 ${W} ${items.length * rh + 4}" role="img">`;
  items.forEach((x, i) => { const y = i * rh, w = max ? x.v / max * bw : 0;
    s += `<text class="lab" x="${lw - 8}" y="${y + 15}" text-anchor="end">${x.l}</text><rect x="${lw}" y="${y + 4}" width="${Math.max(w, 1)}" height="${rh - 9}" rx="3" fill="${color}" fill-opacity="0.85"/>` +
         `<text class="val" x="${lw + w + 5}" y="${y + 15}">${fmtv(x)}</text>`; });
  return s + '</svg>';
}
{
  const T = D.types, tot = T.n, top = (o, n) => Object.entries(o).sort((a, b) => b[1] - a[1]).slice(0, n);
  const fig = (title, items, color) => `<div class="fig"><h3>${title}</h3>${hbars(items.map(([l, v]) => ({l, v})), Math.max(...items.map(x => x[1])), x => `${x.v} (${Math.round(100 * x.v / tot)}%)`, color)}</div>`;
  const lenItems = D.groups.groupings.length.map(b => [blab(b) + ' chars', D.groups.groups.length[b].n]);
  $('ty-comp').innerHTML = fig('Modality', top(T.modality, 6), 'var(--surface-hue)') + fig('Body region', top(T.region, 8), 'var(--semantic-hue)') +
    fig('Primary subspecialty', top(T.subspecialty, 7), 'var(--cef-hue)') + fig('Source length', lenItems, 'var(--clinical-hue)');
  const P = T.profile, rowsP = T.order.concat(['ALL']);
  $('ty-profile').innerHTML = `<thead><tr><th>Modality</th><th>n</th><th>median chars</th><th>sentences</th><th>numbers / report</th><th>negations / report</th><th>&ge;2 negations</th><th>both sides mentioned</th><th>has preamble</th></tr></thead><tbody>` +
    rowsP.map(m => { const p = P[m]; return `<tr${m === 'ALL' ? ' class="control"' : ''}><td class="name">${m === 'ALL' ? 'all reports' : m}</td><td>${p.n}</td><td>${p.chars.toFixed(0)}</td><td>${p.units.toFixed(0)}</td><td>${p.nums.toFixed(1)}</td><td>${p.neg.toFixed(1)}</td><td>${(100 * p.neg2).toFixed(0)}%</td><td>${(100 * p.both_sides).toFixed(0)}%</td><td>${(100 * p.preamble).toFixed(0)}%</td></tr>`; }).join('') + '</tbody>';
  const C = D.groups.cross, bs = D.groups.groupings.length;
  $('ty-cross').innerHTML = `<thead><tr><th>Modality</th>${bs.map(b => `<th>${blab(b)}</th>`).join('')}<th>all</th></tr></thead><tbody>` +
    Object.entries(C).map(([m, r]) => `<tr><td class="name">${m}</td>${bs.map(b => `<td>${r[b]}</td>`).join('')}<td>${bs.reduce((a, b) => a + r[b], 0)}</td></tr>`).join('') + '</tbody>';
}
const EXM = ['words','crit','numbers','terms','bleu','chrf','ter','xcomet','xcomet_qe','metricx_ref','metricx_qe','cef_conformity','cef_coverage','cef_consistency'];
function groupVal(g, name, m, sys) {
  const e = D.groups.groups[g][name];
  if (sys === 'ALL') { const p = e.pooled[m]; return p ? {v: p.mean, lo: p.lo, hi: p.hi} : null; }
  const v = (e.system[sys] || {})[m]; return hasV(v) ? {v} : null;
}
function ciBars(g, m, sys) {
  const names = D.groups.groupings[g], items = names.map(n => ({n, e: D.groups.groups[g][n], x: groupVal(g, n, m, sys)}));
  const mx = Math.max(...items.map(i => i.x ? (i.x.hi !== undefined ? i.x.hi : i.x.v) : 0)) * 1.08 || 1;
  const lw = 112, W = 400, bw = W - lw - 78, rh = 28, dec = MET[m][3];
  let s = `<svg viewBox="0 0 ${W} ${items.length * rh + 6}" role="img" aria-label="${MET[m][0]} by ${g}">`;
  items.forEach((it, i) => { const y = i * rh; if (!it.x) return;
    const w = it.x.v / mx * bw;
    s += `<text class="lab" x="${lw - 8}" y="${y + 17}" text-anchor="end">${blab(it.n)}</text><text class="mut" x="${lw - 8}" y="${y + 27}" text-anchor="end" style="font-size:9px">n=${it.e.n}</text>` +
         `<rect x="${lw}" y="${y + 5}" width="${Math.max(w, 1)}" height="${rh - 12}" rx="3" fill="${FAMC[MET[m][2]]}" fill-opacity="0.8"/>`;
    if (it.x.lo !== undefined && it.e.n >= 2) s += `<line x1="${lw + it.x.lo / mx * bw}" x2="${lw + it.x.hi / mx * bw}" y1="${y + 15}" y2="${y + 15}" stroke="var(--ink)" stroke-width="1.6"/><line x1="${lw + it.x.lo / mx * bw}" x2="${lw + it.x.lo / mx * bw}" y1="${y + 11}" y2="${y + 19}" stroke="var(--ink)" stroke-width="1.6"/><line x1="${lw + it.x.hi / mx * bw}" x2="${lw + it.x.hi / mx * bw}" y1="${y + 11}" y2="${y + 19}" stroke="var(--ink)" stroke-width="1.6"/>`;
    s += `<text class="val" x="${lw + (it.x.hi !== undefined ? it.x.hi : it.x.v) / mx * bw + 6}" y="${y + 18}">${it.x.v.toFixed(dec)}</text>`; });
  return s + '</svg>';
}
function groupTable(g, sys) {
  const names = D.groups.groupings[g];
  return `<thead><tr><th>${g === 'modality' ? 'Modality' : 'Source length (chars)'}</th><th>n</th>${EXM.map(m => `<th class="f-${MET[m][2]}">${MET[m][0]} ${arrow(m)}</th>`).join('')}</tr></thead><tbody>` +
    names.map(n => { const e = D.groups.groups[g][n];
      return `<tr><td class="name">${blab(n)}</td><td>${e.n}</td>${EXM.map(m => { const x = groupVal(g, n, m, sys); return `<td>${x ? x.v.toFixed(MET[m][3]) : '&mdash;'}</td>`; }).join('')}</tr>`; }).join('') + '</tbody>';
}
function renderExplorer() {
  const sys = $('ex-sys').value, m = $('ex-metric').value;
  $('ex-fig-mod').innerHTML = ciBars('modality', m, sys); $('ex-fig-len').innerHTML = ciBars('length', m, sys);
  $('ex-tab-mod').innerHTML = groupTable('modality', sys); $('ex-tab-len').innerHTML = groupTable('length', sys);
  $('ex-cap').textContent = sys === 'ALL' ? 'All 13 systems pooled: each report is averaged over the systems first. Whiskers are 95% bootstrap intervals over reports.' : `${sys} alone, no interval (one system, so the spread is over reports only).`;
}
$('ex-sys').innerHTML = opts([['ALL', 'All systems pooled']].concat(SYS.map(s => [s, s])), 'ALL');
$('ex-metric').innerHTML = opts(EXM.map(k => [k, `${MET[k][0]} (${MET[k][1] ? 'higher' : 'lower'} is better)`]), 'words');
$('ex-sys').addEventListener('change', renderExplorer); $('ex-metric').addEventListener('change', renderExplorer);
renderExplorer();

/* ---- 8. EN -> DE (standard metrics only, where the translations are complete) ---- */
{
  const E = D.ende, sysn = Object.keys(E.systems).filter(s => s !== 'identity').sort((a, b) => E.systems[a].words - E.systems[b].words);
  const cols = ['crit','words','numbers','terms','bleu','chrf','ter'];
  const row = (s, label, cls) => `<tr class="${cls || ''}"><td class="name">${label}</td>${cols.map(k => `<td>${E.systems[s][k].toFixed(MET[k][3])}</td>`).join('')}<td>${E.systems[s].n}</td></tr>`;
  $('en-table').innerHTML = `<thead><tr><th>System</th>${cols.map(k => `<th class="f-${MET[k][2]}">${MET[k][0]} ${arrow(k)}</th>`).join('')}<th>n</th></tr></thead><tbody>` +
    sysn.map(s => row(s, nm(s))).join('') + (E.systems.identity ? row('identity', 'identity (control)', 'control') : '') + '</tbody>';
  const ex = Object.entries(E.excluded).map(([s, w]) => `<span class="mono">${s}</span>: ${w}`), pe = Object.entries(E.pending).map(([s, w]) => `<span class="mono">${s}</span>: ${w}`);
  $('en-banner').innerHTML = `<b>EN&rarr;DE is partial.</b> ${sysn.length} systems are complete and scored with the surface metrics and the detectors. ` +
    (ex.length ? `Not shown: ${ex.join('; ')}. ` : '') + (pe.length ? `Still running: ${pe.join('; ')}. ` : '') +
    `Two rows are not like-for-like with the DE\u2192EN runs: <span class="mono">MiniMax-M3</span> was served by the SCADS API here and by a different gateway there, and the GLM row, when it lands, is the newer <span class="mono">glm-5.3</span>, not <span class="mono">glm-5.2</span>. The learned metrics and CEF for this direction have not been run yet. The English input is itself a human translation of the German original, which is the reference, so this measures translating a translation.`;
}
