#!/usr/bin/env python
"""Render the metrics page: inject fresh data and the 2026-10 sections into the previous page.

    python scripts/render_metrics_page.py --data results/page_data_X/page_data.json \\
        --base metrics_reference.html --out metrics_reference_X.html

The previous page supplies the design and the sections that did not change (worked example,
interventions, ceiling). Everything numeric comes from --data. Never overwrites.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "scripts/templates"


def section(html: str, sid: str, new: str) -> str:
    a = html.index(f'<section id="{sid}">')
    b = html.index("</section>", a) + len("</section>")
    return html[:a] + new + html[b:]


def cut(js: str, start: str, end: str) -> str:
    a = js.index(start); b = js.index(end)
    assert a < b, (start, end)
    return js[:a] + js[b:]


HEADER = '''<header class="mast">
  <div class="eyebrow">German&#8596;English radiology reports &middot; PARROT</div>
  <h1>Clinical MT metrics, defined and measured</h1>
  <p class="sub">Every metric this project computes &mdash; what it counts, how it is calculated, how to read it &mdash;
  and the numbers it produced across fourteen systems. Surface, learned, cross-examination and rule-based metrics rank the same
  systems in different orders, two of them fail a basic sanity test, and one corrupted clinical fact barely moves most of them.</p>
  <div class="meta">
    <span>14 systems</span><span>296 reports</span><span>15 metrics</span><span>DE&rarr;EN single pass &middot; round trip &middot; EN&rarr;DE (partial)</span><span>sacreBLEU 2.6.0</span>
  </div>
</header>'''

NAV = '''<nav class="toc"><ul>
  <li><a href="#disagreement">Who wins</a></li>
  <li><a href="#worked">Worked example</a></li>
  <li><a href="#reference">Metric reference</a></li>
  <li><a href="#singlepass">Single-pass scores</a></li>
  <li><a href="#control">Sanity test</a></li>
  <li><a href="#agreement">Agreement</a></li>
  <li><a href="#roundtrip">Round trip</a></li>
  <li><a href="#perturbation">Perturbation test</a></li>
  <li><a href="#types">Report types</a></li>
  <li><a href="#ende">EN&rarr;DE</a></li>
  <li><a href="#interventions">Interventions</a></li>
  <li><a href="#ceiling">The ceiling</a></li>
  <li><a href="#reading">Reading notes</a></li>
</ul></nav>'''

DISAGREEMENT = '''<section id="disagreement">
  <h2>Who wins depends on what you measure</h2>
  <p class="lede">The five best systems under one metric from each family, on the 296 German&rarr;English reports. Colour marks the
  family: <span style="color:var(--surface-hue)">surface overlap</span>, <span style="color:var(--semantic-hue)">learned semantic</span>,
  <span style="color:var(--cef-hue)">cross-examination</span>, <span style="color:var(--clinical-hue)">clinical detectors</span>.</p>
  <div class="rank" id="rankings"></div>
  <div class="callout" id="dis-callout"></div>
</section>'''

REFERENCE_LEDE = '''<p class="lede">Colour marks the family: <span style="color:var(--surface-hue)">surface overlap</span>,
  <span style="color:var(--semantic-hue)">learned semantic</span>, <span style="color:var(--cef-hue)">cross-examination (CEF)</span>,
  <span style="color:var(--clinical-hue)">clinical fidelity</span>. The last three cards describe how the newer tests work.</p>'''

SINGLEPASS = '''<section id="singlepass">
  <h2>Single-pass scores</h2>
  <p class="lede">PARROT German radiology reports, 296 documents, German&rarr;English. Every metric for every system; click a column
  heading to sort by it (best first), and the best value in each column is shaded. COMET is <code>Unbabel/wmt22-comet-da</code>, run on
  twelve systems before the newer metrics existed.</p>
  <div class="tablebox"><table id="sp-table"></table></div>
  <p class="note" style="margin-top:14px"><b>crit%</b> counts reports with any critical finding from the rule-based detectors; <b>words%</b>
  keeps only negation and laterality findings and leaves out numbers and measurements, which are about two thirds of the critical findings
  here; <b>numbers%</b> is the number-and-measurement share on its own. All four come from one detector version, rescored on 2026-09-29.
  <b>term%</b> is the share of reports with a terminology finding (major severity, not counted in crit%).</p>
  <p class="note" style="margin-top:8px"><b>How the newer columns were computed.</b> xCOMET and MetricX score <em>aligned segments</em>
  (a report cut into short parallel pieces, median 41 tokens) and a report's score is the length-weighted mean of its segments. CEF scores the
  whole report, with N = 10 questions per text, judged by Qwen3.8-27B. All three are the mean over reports; a one-point difference is far inside
  the noise for the rule-based columns (one report is 0.34 points) and should be read against the intervals in the report-types section.</p>
  <p class="note" style="margin-top:8px"><b>DeepL</b> ran on the free API plan (EN-US target). It and <code>translategemma-27b</code> have no COMET score.
  DeepL writes <code>5-mm</code> for a compound modifier in 5 of 296 reports, and the number detector reads that as a mismatch against <code>5 mm</code>;
  normalising the hyphen would clear 2 of its 76 critical reports. words% is not affected.</p>
</section>

<section id="control">
  <h2>A sanity test: does the metric put the untranslated source last?</h2>
  <p class="lede">The control system returns its German input unchanged for a German&rarr;English task. Whatever else a metric does, it must rank this below
  every real translation; otherwise its scale has no floor.</p>
  <div class="tablebox"><table id="ctl-table"></table></div>
  <div class="callout" id="ctl-note"></div>
</section>

<section id="agreement">
  <h2>Do the metrics agree on the ranking?</h2>
  <p class="lede">Spearman rank correlation between metrics, over the systems both scored (the control is excluded: every metric agrees it is worst, which would dominate
  the numbers). Green means two metrics order the systems alike; orange means they order them in opposite ways. Signs are aligned so that a higher number always means
  &ldquo;same verdict&rdquo;, including for lower-is-better metrics.</p>
  <div class="figs" style="grid-template-columns:minmax(0,1.7fr) minmax(0,1fr)">
    <div class="fig"><h3>Rank correlation between metrics</h3><div id="heat"></div>
      <div class="legend"><span><i style="background:var(--semantic-hue)"></i>agree</span><span><i style="background:var(--clinical-hue)"></i>disagree</span><span>depth of colour = strength</span></div></div>
    <div class="fig"><h3>Agreement within and between families</h3><div class="tablebox"><table id="fam-rho"></table></div>
      <p class="note" style="margin-top:10px">Thirteen systems is a small sample: treat a difference of a few hundredths as nothing.</p></div>
  </div>
</section>'''

ROUNDTRIP_EXTRA = '''
  <h3 style="margin:30px 0 0">Learned metrics along the cycles</h3>
  <p class="note" style="margin-top:8px;max-width:68ch">English steps only. Grey lines are the other systems; pick one to highlight it. xCOMET, MetricX and CEF
  are computed on aligned segments for the 20-report sample, so differences between neighbouring cycles of a few hundredths are within the noise.</p>
  <div class="ctrls"><label>Highlight <select id="rt-sys"></select></label></div>
  <div class="figs" id="rt-figs"></div>
</section>'''

NEW_SECTIONS = '''
<section id="perturbation">
  <h2>Perturbation sensitivity: what does one corrupted fact do to each metric?</h2>
  <p class="lede">The worked example above shows a blind spot on one sentence. This runs the same experiment across the corpus. For each report, one edit is made to a
  translation: <b>drop a negation</b>, <b>flip left and right</b>, <b>change a measurement</b> (all of which change what a clinician would conclude), or <b>swap a phrase for a
  clinically equivalent one</b> (which does not). Each metric then scores the original and the edited text.</p>
  <div class="ctrls"><label>Text that was edited <select id="pb-base"></select></label></div>
  <div class="figs" style="grid-template-columns:minmax(0,1fr) minmax(0,1.25fr)">
    <div class="fig"><h3>Can the metric tell a real error from a harmless rewording?</h3><div id="pb-auc"></div>
      <p class="note" style="margin-top:10px">0.5 is a coin flip; 1 means a dangerous edit is always penalised harder than a harmless one. All three dangerous kinds are pooled.</p></div>
    <div class="fig"><h3>Edits scored worse, by kind</h3><div class="tablebox"><table id="pb-kinds"></table></div>
      <p class="note" style="margin-top:10px">Each cell is the share of edits the metric scored strictly worse, with the effect size z (the mean penalty in standard deviations of that metric over the real systems&rsquo; per-report scores). <b>Green shading shows how much more often the metric penalises this kind of edit than it penalises a harmless rewording</b> (percentage points, full strength at 50), because a metric can mark half of all harmless edits &ldquo;worse&rdquo; by a thousandth of a point. A pale cell means the metric cannot tell that error from a rewording, which is a finding, not a missing value.</p></div>
  </div>
  <div class="callout"><b>Read across the kinds, not the pooled number.</b> Every learned regression metric reacts to a dropped negation and a changed measurement, but a flipped
  left/right barely moves xCOMET or MetricX, hardly more than a harmless rewording. The cross-examination conformity score reacts to all three kinds, because its questions
  (&ldquo;is the nodule in the left upper lobe?&rdquo;) put the changed fact to the text directly. The n-gram metrics cannot separate the two: a harmless rewording costs them about as much as a real error.
  The effect sizes are small for every metric except CEF conformity, whose penalty per edit is large relative to how little it varies between real systems; for the rest a single
  wrong fact is invisible in a system-level score.</div>

  <h3 style="margin:30px 0 0">Why the rule-based detectors miss most single errors</h3>
  <p class="note" style="margin-top:8px;max-width:72ch">The negation detector compares whether a report has <em>any</em> negation cue on each side, and the laterality detector compares the
  <em>set</em> of sides mentioned. Neither compares statements. So the same edit is caught when the report holds one such fact and missed when it holds several. Recall on the human
  reference, DE&rarr;EN:</p>
  <div class="figs" id="pb-recall"></div>
  <div class="callout"><b>The published critical-error rates are lower bounds.</b> Most reports contain several negations, which is exactly where the negation detector is blind, and the
  report-types section shows how the share varies by modality. Measurements are caught more often because their comparison is a multiset rather than a presence test.</div>
</section>

<section id="types">
  <h2>Report types: what is in the corpus, and how it translates</h2>
  <p class="lede">PARROT carries its own metadata for every report. Modality, region and subspecialty are as recorded (with <code>German</code>/<code>Germany</code> merged and the 48 free-text area
  labels grouped into eight regions by keyword); length is the character count of the German source.</p>
  <div class="figs" id="ty-comp"></div>
  <h3 style="margin:30px 0 0">Content profile by modality</h3>
  <p class="note" style="margin-top:8px;max-width:72ch">What each kind of report is exposed to. Numbers per report counts every numeric token, so angiography&rsquo;s doses and acquisition parameters are included.
  The last columns are where the rule-based detectors are blind: two or more negations, and both sides mentioned.</p>
  <div class="tablebox"><table id="ty-profile"></table></div>
  <h3 style="margin:30px 0 0">Modality and length overlap</h3>
  <p class="note" style="margin-top:8px;max-width:72ch">The two are tangled: X-ray and ultrasound reports are almost all short, CT and MRI are long. Read a modality effect and a length effect together, not separately.</p>
  <div class="tablebox"><table id="ty-cross"></table></div>

  <h3 style="margin:30px 0 0">Scores by modality and by report length</h3>
  <div class="ctrls">
    <label>System <select id="ex-sys"></select></label>
    <label>Chart metric <select id="ex-metric"></select></label>
  </div>
  <p class="note" id="ex-cap"></p>
  <div class="figs" style="grid-template-columns:repeat(auto-fit,minmax(340px,1fr))">
    <div class="fig"><h3>By modality</h3><div id="ex-fig-mod"></div></div>
    <div class="fig"><h3>By source length (characters)</h3><div id="ex-fig-len"></div></div>
  </div>
  <div class="tablebox" style="margin-top:18px"><table id="ex-tab-mod"></table></div>
  <div class="tablebox" style="margin-top:14px"><table id="ex-tab-len"></table></div>
  <div class="callout"><b>Length matters more than modality, and BLEU cannot see it.</b> Pooled over the systems, the share of reports with a critical finding climbs steadily from the shortest
  to the longest bucket, while BLEU is nearly flat across the first three buckets. Angiography combines the best BLEU with the worst clinical error rate: its reports are
  preamble-heavy and number-dense. Mammography (2 reports) and angiography (15) are too small to support a conclusion; read them as descriptive. One exception to the trend: pooled CEF conformity <em>rises</em> with length, the opposite of every other metric. That is most likely a resolution effect (ten questions sample a smaller share of a long report&rsquo;s facts, so fewer contradictions are found), not better translation, though that explanation has not been tested.</div>
</section>

<section id="ende">
  <h2>English&rarr;German</h2>
  <p class="lede">The same benchmark in the other direction: the human English translation is the input and the original German is the reference. One pass, no round trip.</p>
  <div class="banner" id="en-banner"></div>
  <div class="tablebox"><table id="en-table"></table></div>
</section>
'''

READING = '''<section id="reading">
  <h2>Reading notes</h2>
  <div class="cards">
    <div class="card"><h3>Metric signatures</h3>
      <div class="formula" id="sigs"></div>
      <p class="note">A BLEU number without a signature is not reproducible; different tokenisers move it by
      several points. Corpus scores use sacreBLEU&rsquo;s corpus scorers, never the mean of sentence scores.</p></div>
    <div class="card"><h3>Polarity and resolution</h3>
      <p class="note"><b>Lower is better</b> for TER, MetricX, crit%, words%, numbers% and term%; higher is better for the rest. The rate columns are shares of
      <b>reports</b>, so one report is 0.34 points at n = 296 (5 points at n = 20). Treat BLEU differences under about a point as noise and never compare BLEU across test sets.
      Group breakdowns hold far fewer reports; the whiskers show how much that matters.</p></div>
    <div class="card"><h3>Detector recall</h3>
      <p class="note">The rule-based detectors compare whole-document presence rather than individual statements, so a single wrong negation in a report with several is invisible to
      them, and a flipped side is missed when both sides appear. The perturbation section measures this. The published critical-error rates are <b>lower bounds</b>, and negation errors
      are the most under-counted.</p></div>
    <div class="card"><h3>The control, and what failing it means</h3>
      <p class="note">xCOMET and CEF rank an untranslated copy above real translations. CEF compares what two texts say but never checks the language; xCOMET without a reference prefers a copy.
      Use them next to a language check, and do not read their position at the top of a ranking as quality.</p></div>
    <div class="card"><h3>CEF depends on its judge</h3>
      <p class="note">The method&rsquo;s authors chose DeepSeek-V3 after a stability analysis of five judges, in which Qwen3-235B was the least stable. This run uses Qwen3.8-27B, which that
      analysis does not cover, and CEF conformity ranks the sibling model <code>qwen35-27b</code> first although the detectors rank it near the bottom. That may be the judge favouring its own family;
      a second judge on a subset would test it. N = 10 questions also cannot cover every fact in a long report.</p></div>
    <div class="card"><h3>Learned metrics score segments, not reports</h3>
      <p class="note">xCOMET reads source, translation and reference as one 512-token sequence, and about half of these reports exceed that. Reports are therefore cut into aligned segments
      (Gale&ndash;Church length alignment, checked against the reports whose sentence counts already match). A segmentation error would show up as noise in these columns.</p></div>
    <div class="card"><h3>Two known biases in the clinical layer</h3>
      <p class="note"><b>Under-translation flatters it.</b> A detector cannot flag a number that was never written, so a system that omits content can look safer; <code>opus</code> looks middling on crit% but is
      worst on words%. <b>Numbers dominate.</b> About two thirds of critical findings are measurements, which is why words% is shown beside crit%.</p></div>
  </div>
</section>'''

FOOTER = '''<footer>
  <p>Every figure on this page is read from result files at build time, not typed: the single-pass rows from <code>results/parrot_de/rescored_20260929/</code>, the learned metrics from
  <code>results/xcomet_*</code>, <code>results/metricx_*</code> and <code>results/cef_*</code>, the rank and group analyses from <code>results/page_data_*/</code>, the perturbation test from
  <code>results/perturbation_*/</code>, and EN&rarr;DE from <code>results/parrot_en_de_*/</code>.</p>
  <p>Full write-up: <code>THESIS.md</code>. Method sources: xCOMET (Guerreiro et al. 2024), MetricX-24 (Juraska et al. 2024), Cross-Examination Framework (Raha et al. 2026, arXiv:2601.19350).</p>
</footer>'''

CARDS_NEW = r'''  {fam:'clinical', name:'words% (negation and laterality only)', range:'0&ndash;100% &middot; <b>lower better</b> &middot; no reference',
   what:'The share of reports with a critical <b>negation</b> or <b>laterality</b> finding. It drops the number-and-measurement findings, which are about two thirds of all critical findings on this corpus, so that systems are compared on the words rather than on how they typeset a measurement.',
   formula:`words% = reports with ≥ 1 critical negation / laterality finding
         ─────────────────────────────────────────────────── × 100
                          all reports`,
   read:'It is a <b>lower bound</b>: the detectors compare whole-document presence, not individual statements (see the perturbation test), so the true rate is higher, most of all for negation.',
   cant:'It cannot see an omitted fact or a wrong term, only a changed negation or side that the presence test happens to register.'},
  {fam:'semantic', name:'xCOMET-XL', range:'0&ndash;1 &middot; higher better &middot; with or without a reference',
   what:'Unbabel&rsquo;s xCOMET-XL (3.5 B parameters), trained on direct-assessment scores <em>and</em> on error-span annotations. It returns a score and the words it believes are wrong, each tagged minor, major or critical. Run here per aligned segment, with the reference (xCOMET) and without (xCOMET-QE).',
   formula:`score = 0.12·src-head + 0.33·ref-head + 0.33·unified-head + 0.22·span-score
span-score = ( 25 − min(25, Σ points) ) / 25         minor 1 · major 5 · critical 10`,
   read:'The span-score alone is shown in the perturbation test as xCOMET-MQM. The reference-free score (QE) never sees the reference, so it cannot be anchored to one translator&rsquo;s word choice.',
   cant:'It fails the sanity test: an untranslated copy scores above real translations when no reference is given. It is also blind to a flipped left/right, and trained on news-like text.'},
  {fam:'semantic', name:'MetricX-24 (hybrid)', range:'0&ndash;25 &middot; <b>lower better</b> &middot; with or without a reference',
   what:'Google&rsquo;s MetricX-24 hybrid, large, a regression model trained on MQM and direct-assessment ratings that predicts an <em>error score</em>. The same model scores with a reference (<code>source / candidate / reference</code>) and without (<code>source / candidate</code>).',
   formula:`input  : "source: S candidate: H reference: R"   (reference part dropped for QE)
output : predicted error points, clipped to [0, 25]   (0 = perfect)`,
   read:'The opposite polarity to BLEU and COMET: a falling number is an improvement. It takes 1,536 tokens, so segments here are large.',
   cant:'Like xCOMET it reacts strongly to a dropped negation and barely to a flipped side. Its reference-free score is the second signal that does not depend on the single gold translation.'},
  {fam:'cef', name:'Cross-Examination Framework (CEF)', range:'0&ndash;100 &middot; higher better &middot; <b>reference-free</b>',
   what:'Raha et al. 2026. The source and the translation are treated as two independent knowledge bases. A judge model writes 10 closed-ended questions from each, always answerable YES, then asks each text the <em>other</em> text&rsquo;s questions and answers YES, NO or IDK (cannot tell). Here the judge is Qwen3.8-27B at temperature 0, with the paper&rsquo;s own prompts.',
   formula:`Coverage    = 100 − %( source questions the translation answers IDK )   omissions
Conformity  = 100 − %( source questions the translation answers NO  )   contradictions
Consistency = 100 − %( translation questions the source answers IDK )   hallucinations`,
   read:'<b>Conformity</b> is the one that reacts to a changed negation, side or number: the changed fact turns a YES into a NO. Coverage and consistency measure omission and addition, so they barely move on these edits.',
   cant:'It never checks the language, so an untranslated copy scores perfectly. It depends on its judge, and 10 questions cannot sample every fact in a long report.'},
  {fam:'clinical', name:'Perturbation sensitivity test', range:'AUC 0&ndash;1 &middot; effect size in SD',
   what:'One controlled edit per report: drop a negation, flip left/right, change a measurement (dangerous), or swap an equivalent phrase (harmless). The metric scores the original and the edit; the question is whether it penalises the dangerous edits more.',
   formula:`penalty   = score(original) − score(edit)      sign-flipped for lower-is-better metrics
z         = mean penalty / SD of the metric over real systems' per-report scores
AUC       = P( penalty of a dangerous edit  >  penalty of a harmless edit )`,
   read:'AUC near 0.5 means the metric cannot separate real errors from rewordings. z puts BLEU, xCOMET and MetricX on one scale.',
   cant:'Edits are single and synthetic. The cue lists overlap the detectors&rsquo; lexicons, so the detectors&rsquo; own hit rate is partly circular; learned and n-gram metrics have no such lexicon.'},
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=ROOT / "metrics_reference.html")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    html = a.base.read_text(encoding="utf-8")
    data = a.data.read_text(encoding="utf-8")

    # ---- CSS ----
    html = html.replace("</style>", (T / "page_css.css").read_text(encoding="utf-8") + "</style>", 1)
    # ---- header / nav ----
    s = html.index('<header class="mast">'); e = html.index("</header>", s) + len("</header>")
    html = html[:s] + HEADER + html[e:]
    s = html.index('<nav class="toc">'); e = html.index("</nav>", s) + len("</nav>")
    html = html[:s] + NAV + html[e:]
    # ---- sections ----
    html = section(html, "disagreement", DISAGREEMENT)
    s = html.index('<section id="reference">'); a0 = html.index('<p class="lede">', s); b0 = html.index("</p>", a0) + 4
    html = html[:a0] + REFERENCE_LEDE + html[b0:]
    html = section(html, "singlepass", SINGLEPASS)
    # round trip: keep its table and note, append the figure block before its closing tag
    s = html.index('<section id="roundtrip">'); e = html.index("</section>", s)
    html = html[:e] + ROUNDTRIP_EXTRA.rstrip().removesuffix("</section>") + html[e:]
    # new sections go between the round trip and the interventions
    html = html.replace('<section id="interventions">', NEW_SECTIONS + '\n<section id="interventions">', 1)
    html = section(html, "reading", READING)
    s = html.index("<footer>"); e = html.index("</footer>", s) + len("</footer>")
    html = html[:s] + FOOTER + html[e:]

    # ---- data block ----
    tag = '<script id="data" type="application/json">'
    s = html.index(tag) + len(tag); e = html.index("</script>", s)
    html = html[:s] + data.replace("</", "<\\/") + html[e:]

    # ---- script: cards, drop superseded blocks, append the new renderers ----
    anchor = "];\ndocument.getElementById('cards').innerHTML"
    assert html.count(anchor) == 1, "cards anchor"
    html = html.replace(anchor, "  " + CARDS_NEW.strip() + "\n" + anchor, 1)
    # the last card object ends with a trailing comma-less brace in the old list; make sure a comma separates them
    html = html.replace("}\n  {fam:'clinical', name:'words% (negation", "},\n  {fam:'clinical', name:'words% (negation", 1)
    sc = html.index("<script>\nconst D = ")
    js_head, js_body = html[:sc], html[sc:]
    js_body = cut(js_body, "/* ---- rankings ---- */", "/* ---- worked example ---- */")
    js_body = cut(js_body, "/* ---- single pass ---- */", "/* ---- round trip ---- */")
    js_body = js_body.replace("const API = new Set(['glm-5.2','MiniMax-M3','DeepSeek-V4-Flash']);",
                              "const API = new Set(['glm-5.2','glm-5.3','MiniMax-M3','DeepSeek-V4-Flash','deepl']);", 1)
    end = js_body.rindex("</script>")
    js_body = js_body[:end] + "\n" + (T / "page_js.js").read_text(encoding="utf-8") + "\n" + js_body[end:]
    html = js_head + js_body
    a.out.write_text(html, encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size/1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
