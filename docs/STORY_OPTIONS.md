# Story options for the slowdown work (2026-09-15)

The current `main.tex` tries to be an audit narrative, a methods paper, a sea-ice predictability paper and an
observational application at once. This note lays out what actually survives, then three ways to package it,
each with one or two concrete organisations (Intro / Data / Methods / Results / Discussion / Conclusion) and a
figure list, and ends with a recommendation.

## 0. What we have, sorted by how much we trust it

**Trust fully (arithmetic or replicated many ways).**
- The construction: Eq. coupling; onset-inclusive R² 0.28 → 0.075 → 0.04 (SIE), 0.26 → 0.03 → 0.00 (GMT).
- The CNN's onset-inclusive skill was the ice state read through summer SST: occlusion, aux tracking r = −0.77,
  four XAI methods agreeing on the same (misleading) map, extra-Arctic network (pending) as the clean test.
- LB22's predictor inside CESM2-LE: ¾ construction, ¼ real, and the real part is the IPO (GMT ledger).

**Trust as a model result (one model, offset labels, clean design).**
- The sea-ice ledger: ~11% of the coming decade's September SIE trend variance is foreseeable from the onset
  state (extent + volume ≈ 9%, IPO ≈ 2.5%); the ocean surface and upper-ocean heat content add nothing beyond
  the ice; the CNN equals the ice-state logistic (0.62 vs 0.61); Pacific–Arctic bridge present at ~2% of
  interannual variance with the model's sign. This *is* trustworthy — it is exactly what the lessons paper says
  a clean design should produce. What it is not is a positive result: it is a quantified limit.

**Trust as an application only (26 verifiable onsets; conclusion depends on σ).**
- Observations: CESM2-trained baselines sat below the base rate through the observed 2005–2010 onsets; by
  the observations' own σ that decade was a slowdown, by CESM2's σ nothing since 1990 is. Useful as a
  demonstration, weak as a finding. In the lessons story its one job is: *an observational hindcast did not
  catch the construction either* (in the GRL v1 it looked fine).

So: do we still trust the sea-ice predictions? Yes, the offset-label results are the trustworthy ones; the
original (onset-inclusive) predictions are not, and nothing in v1 survives except the ledger. The question is
not trust but publishability of a "little is foreseeable, and not from the ocean surface" result.

---

## Case 1 — one manuscript for everything (AIES regular article, or JAMES)

Why it is currently all over the place: the audit is told in the order we discovered things, so the reader
meets a result, then its refutation, then a new result, five times. Two organisations that fix that.

### 1A. Two-part results: "the construction" then "what survives"
1. Introduction — decadal slowdowns (GMT, SIE); ML+XAI precursor claims; the question "how much of the
   coming decade is foreseeable from the state at onset, and does XAI locate it"; the two case studies.
2. Data — CESM2-LE (SST, SIE, siv, GMT, OHC100/300 from the AWS store); observations (NSIDC, OISST/ERSST).
3. Methods — slowdown labels, onset-inclusive vs offset; Eq. coupling; CNN + XAI + occlusion; scalar baselines;
   continuous-target ledger; the LB22-predictor test; observational application.
4. Results A: the construction — Fig. 1 mechanism (GMT & SIE bars); Fig. 2 CNN on onset-inclusive labels:
   XAI vs occlusion; Fig. 3 skill vs baselines on both label sets incl. extra-Arctic CNN; Fig. 4 LB22 predictor
   with the missing control.
5. Results B: what survives — Fig. 5 ledger GMT vs SIE (where the memory lives); Fig. 6 observations
   (baselines on the current pause; σ dependence).
6. Discussion — two case studies, portability; familiar problem; where the other 89% is; limits.
7. Conclusions — lessons + the sea-ice ledger in one paragraph each.

Pros: everything in one place, one citation. Cons: ~9,000 words, two audiences (ML-methods readers stop
after Part A, sea-ice readers skip Part A), and the sea-ice "result" reads as an afterthought to the audit.
AIES's regular-article format is fine with the length; a "Lessons Learned" piece is not.

### 1B. Question-led: "how much is foreseeable, and can XAI find it?"
Same sections, but the construction is demoted to a methods subsection ("why the window starts at t+1") and
the paper is a decadal-predictability ledger for SIE with GMT as the comparison system. The CNN/XAI audit
becomes one results subsection ("a CNN adds nothing and XAI cannot tell"). Figures: mechanism → SI; ledger
first. This is the "positive" framing of the same material.

Pros: single clear question, sea-ice readers served. Cons: hides the lesson that is the most transferable
finding; the LB22 comparison sits awkwardly in a sea-ice paper; the honest answer to the title question
("about 11%, from the ice") is modest for a long paper.

---

## Case 2 — two manuscripts

### 2A. AIES "Lessons Learned" (the Mamalakis shape: ~8 pages, one lesson, 3–4 figures)
Title (imperative): *Keep the onset year out of the window: lessons learned from predicting decadal trend
slowdowns with neural networks and XAI.*
1. Introduction — moving-trend slowdown labels; ML+XAI precursor studies; the one fact overlooked; the two
   case studies (ours and LB22, one author each); Mamalakis23 as the template.
2. The construction — Eq. coupling, Oldham/Kenney/Kaufman/Kapoor lineage; Fig. 1 mechanism (white-noise line,
   GMT & SIE bars).
3. Case study 1: Arctic SIE from summer SST — Fig. 2 (a) LRP composite, (b) occlusion, (c) skill vs baselines
   on onset-inclusive and offset labels incl. extra-Arctic CNN. One page. Message: every check passed and every
   check was consistent with the artefact.
4. Case study 2: GMT from OHC100 — Fig. 3 (a) LB22-predictor table as bars, (b,c) GMT ledger + residual map.
   Message: ¾ construction, ¼ real, and the real part is the IPO; physical reading survives, numbers do not.
   One paragraph on SIE ledger for contrast (Fig. 3d or text): memory in the ice, not the ocean.
5. Implications — (i) offset the window or use a forward-mean/state-free target (as Gordon22, Rader23 do);
   (ii) state baseline first; (iii) occlusion beside relevance; (iv) verify scalar inputs; (v) portability: ask
   what the new target's own state explains before transplanting a design. Observations: one paragraph — the
   hindcast looked consistent too; no figure, or SI.
6. Summary — the one-line lesson.

Figures: 3 main (+ optional Fig. 4 = the extra-Arctic CNN if it deserves its own panel); SI: XAI-method
comparison, learning curves/frozen scalar, LB22 CV table, obs figure, decadal-ML design table.

### 2B. Sea-ice paper (GRL, or JGR-Oceans / J. Climate if GRL wants more novelty)
Title: *How much of the coming decade's Arctic sea-ice trend is foreseeable from its onset state?* (or
*Decadal slowdowns in Arctic sea-ice decline are not foreseeable from the ocean surface in CESM2-LE*).
1. Introduction — observed slowdown since ~2005 (England25, Wang25); proposed drivers (PARC, NAO, Atlantic);
   decadal initial-value predictability of SIE from volume/OHC (Blanchard-W, Day, Bushuk, Yeager); the
   question stated as a ledger; one sentence citing 2A for the label design.
2. Data — CESM2-LE 100 members, SIE/siv/SST/indices, OHC100/300 (50 members); NSIDC, OISST/ERSST.
3. Methods — offset trend anomaly as a continuous target and as a 1σ label; OLS/logistic ledger on member-block
   splits; regional OHC and PCs; interannual bridge; observational application with calibration and
   observation-only σ.
4. Results — Fig. 1 definition + example member + observed record (offset window); Fig. 2 ledger: ice state
   → +siv → +IPO → +OHC → +SST map (CNN as the last bar: adds nothing); Fig. 3 interannual bridge (sign,
   stability, 2%) and residual maps; Fig. 4 observations: calibrated baseline probabilities across the record,
   the 2005–2010 pause, σ dependence, 2016–2025.
5. Discussion — memory is in the ice; the ocean surface pictures it; OHC adds nothing for SIE (unlike GMT);
   what the other ~89% is (decade-integrated circulation); model dependence (teleconnection sign); the
   observed pause is a slowdown only by the observations' own variability.
6. Conclusions.

Pros: each paper has one audience and one message; the GRL is short and timely (the pause is in the news); the
AIES cites it for the physics and it cites the AIES for the design. Cons: two review cycles; the GRL's message
is a quantified negative — GRL takes those when the question is topical and the ledger is clean, but the
introduction must position it against the decadal-predictability literature, not against v1. Order: AIES
first (or simultaneous with a preprint), because the GRL's design choice depends on it.

---

## Case 3 — one manuscript, lessons only (the sea-ice results are the case study, not a result)

Same as 2A, with the sea-ice ledger reduced to what the lesson needs: "a clean design leaves ~11% foreseeable,
from the ice, and the ocean surface adds nothing", one panel in Fig. 3. No observations section; the current
pause appears in one sentence of the Implications ("an observational hindcast was consistent with the artefact
and would have been consistent with its absence"). Repository and SI carry the full ledger and obs figures so
nothing is lost.

Pros: the cleanest paper we can write; fastest; the most transferable contribution; strongest position for a
coauthored self-audit. Cons: the sea-ice science (ledger, bridge sign, pause vs σ) goes unpublished as a
finding; a year of sea-ice work is represented by one panel.

---

## Recommendation

Write 2A now, exactly as Case 3 would (three figures, one lesson), and decide about 2B after the extra-Arctic
CNN and the observational votes are in. The reason to keep 2B alive: the ledger and the σ-dependence of the
observed pause are real, clean, topical findings that do not fit in a Lessons Learned piece without diluting
it; the reason not to commit yet: if the ledger stays at ~11% and the CNN adds nothing, GRL is a stretch and
JGR/J. Climate is the honest venue, which is a different writing job. Case 1 is the option to avoid: it is the
structure we have, and it is the reason the draft reads as it does.

Practical consequence for `main.tex`: cut, don't rearrange. Sections Data/Methods shrink to a page; Results
become the two case studies; the observations section and the "what is foreseeable" section move to the SI
(and become the seed of 2B).

---

## Decision (2026-09-15, evening): two papers

**A. AIES Lessons Learned** — as 2A / Case 3. Three figures, one lesson, observations reduced to a sentence.

**B. Sea-ice paper (GRL target)** — redesigned target so the paper does not rest on the decadal linear trend alone.

Targets (all windows start at t+1; onset predictors at t):
1. September SIE anomaly at lead L = 1 … 10 yr (standard initial-value framework; comparable to
   Blanchard-Wrigglesworth 2011, Day 2014, Bushuk 2019, Guemas 2016).
2. Forward means: t+1…t+5 and t+1…t+10 (Gordon22 / Rader23 convention).
3. Offset 10-yr trend as a continuous target — both tails: slowdowns (+) and RILEs (−) in one regression;
   1σ labels only as a secondary view. NOTE: never use "change" SIE(t+L) − SIE(t) as a target — that is
   Oldham's problem again. Predict the future state with the onset state as covariate.

Predictor ledger at onset, added test R² beyond the ice state (SIE, siv; thickness pattern optional):
- ocean surface: SST indices, SST map PCs, the CNN (one bar);
- ocean interior: OHC100 / OHC300 / **OHC700 (to extract, ~7.5 h)**, regional + PCs;
- upper bound: decade-mean (concurrent) SST and OHC — if even the realised ocean over the decade adds nothing,
  the residual is not ocean-driven;
- attribution of the residual: concurrent decade-mean Z500 / SLP (the Ding 2017 circulation pathway) — the
  "other 89%" as attribution, not prediction.

Figures: (1) targets + observed record; (2) predictability vs lead, ice state alone vs + ocean, with the
re-emergence literature bands; (3) trend ledger by depth and concurrent, residual maps (onset and concurrent);
(4) observations — model-trained ice(+IPO) predictions across the record, the 2005–2010 pause under observed
and model σ, 2016–2025.

Framing cautions: the observed-slowdown literature (England25, Wang25) is about 20-yr trends / the recent
pause; PARC (Ding, Baxter) is about interannual minima (2007, 2012). Do not present them as proposed decadal
slowdown mechanisms. The CESM2 Pacific–Arctic bridge (~2%, model sign) is a caveat on model-trained
predictors, not a result; observations (36 yr, nonstationary sign — Bonan20) cannot establish it either way.
