# Revision plan — Hoffman & Massonnet, "XAI reveals SST patterns associated with slowdowns in Arctic September SIE decline in CESM2-LE"

This plan maps each reviewer concern to concrete changes in the `slowdown_arctic_seaice` repo and the manuscript. It is ordered by *decision risk*: the first phase decides whether the paper's central claim survives, so nothing downstream should be rewritten until Phase 1 is done. Every new experiment writes a new output under `results/` with a distinct suffix rather than overwriting the current cached predictions, so the current figures remain reproducible for comparison.

## How the Labe & Barnes (2022) lineage changes the picture

The framework is a near-direct transplant of Labe & Barnes (2022, GRL; hereafter LB22): 10-yr moving trends from 1990, a mean ± 1σ threshold defined on observations and rescaled to the model as a fraction of the forced response, input maps contemporaneous with the first year of the trend window ("a map for the year 2000 predicts whether 2000–2009 is a slowdown"), F1 compared against random chance, LRP-z composites over correct predictions, linearly detrended observations, and out-of-sample "future" onset predictions for 2016–2026. Because LB22 is published and cited, many of the reviewer's design objections can be answered by *disclosure plus sensitivity* rather than redesign. But three things get worse, not better, once the lineage is made explicit.

**What the precedent buys you (handle by transparency, not redesign).** The window length, 1990 start, 1σ threshold, fraction-of-forced-response scaling, TP-only LRP composites, and the observational future-onset panel all have a published precedent. For each, state plainly in Methods that it follows LB22 and add the cheap sensitivity (Phase 4.2). Reviewer comments 4(i–iii), part of 5, and the *form* of 9–10 are downgraded from "flawed" to "under-explained".

**What gets sharper.**

1. *The endpoint artifact (comment 1) is specific to the sea-ice transplant.* In LB22 the predictor (detrended upper-OHC anomaly maps) is not a near-identity of the target's initial condition (GMST anomaly at year t), and the skill they found was *remote* (off-equatorial Pacific). In your adaptation the predictor field over the Arctic is essentially a map of where the ice edge sits in the onset year, i.e. SIE(t) itself, and your headline finding is that this *local* region dominates. A skeptical reader will conclude that the one place the framework differs from LB22 in its result is exactly where the framework becomes degenerate. Phase 1 is therefore not optional; it is the paper.

2. *The "beats random chance" comparison is much weaker here than in LB22 because of the base rate.* LB22: prevalence ≈ 0.10, so always-positive F1 ≈ 0.19 vs ANN F1 = 0.40 — a clear margin. Yours: prevalence = 0.28, always-positive F1 = 0.44 vs CNN median 0.48 — no margin. Same evaluation practice, very different evidential weight. Explain *why* the base rate is 2.8× higher than LB22's and than the ~16% implied by the observed 1σ definition (floor effect near ice-free conditions is the leading suspect, Phase 4.1), and report the always-positive baseline explicitly.

3. *LB22 ran the index-only logistic-regression baseline; you did not.* LB22 §3.2 fits a logistic regression on the IPO index alone (F1 = 0.28 vs ANN 0.40) precisely to show the network learns more than the index. Following LB22 faithfully requires the same test here with the Arctic SST index and with SIE(t) — and there is a real risk the logistic model matches or beats the CNN, given that every quantitative result in §3.2 is already stated in terms of the scalar index. Phase 1.1 becomes mandatory on precedent grounds alone.

**Novelty.** With the framework inherited, the contribution has to be the sea-ice-specific physics: local (Arctic) versus remote (Pacific) sources, and the incomplete correspondence between GMT and SIE slowdowns (currently buried in Fig. S12). If Phase 1 confirms the local signal is real at lead, that comparison *is* the paper; promote Fig. S12 and frame the abstract around "same framework, different target, opposite answer about where predictability lives". If Phase 1 shows the local signal is the initial-condition effect, the contribution becomes the methodological caution (Branch B in Phase 6.6), which is still a meaningful addendum to the LB22 line of work. The title should stop echoing LB22's key point ("Explainable AI reveals…"); it invites the direct comparison on the one axis where the paper is weakest.

## What the Supporting Information changes

Reading the SI resolves a few reviewer points and sharpens others. Net effect: the plan's priorities stand, three items get added, and several "missing" analyses turn out to exist and just need to be moved into the main text.

**Already answered in the SI (move to main text or cite explicitly):**
- VE is defined (Text S5, law of total variance) — reviewer comment 6 becomes "define it in the main text and fix the `(ref)` placeholder".
- P(slowdown | phase) for *all* slowdowns exists (Fig. S11, grey bars) — reviewer comment 5 is half-answered. But the main text quotes the TP numbers (Arctic VE = 11.4%) while the all-slowdown VE is 5.2% (IPO 2.4%, Niño 3.4 0.5%). The physical claim in the abstract must use the all-slowdown values, and "strongly" is even less defensible at 5%.
- Confusion matrices, precision/recall, and per-model metric spread exist (Figs. S3–S5). Test set: TP 67, FP 69, FN 69, TN 305 → prevalence 0.267, precision = recall = F1 = 0.49. Always-positive F1 = 0.42. **Accuracy ≈ 71% is below the majority-class baseline of 73%.** These numbers should be in the main text with the baselines beside them.
- Full-sample SST composites for slowdown vs non-slowdown exist (Fig. S7a,c) — good; add the corresponding relevance-free evidence to the main text.

**New problems the SI exposes (add to the plan):**

1. **The forced response itself contains a slowdown (Fig. S1c–d).** The CESM2-LE *ensemble-mean* 10-yr trend flattens to ~0 and briefly turns positive around 2005–2015. This is almost certainly the CESM2-LE biomass-burning forcing artifact (DeRepentigny et al. 2022, cited in the intro). Two consequences. (a) The model threshold is `f × ensemble-mean trend(t)`; when the ensemble-mean trend → 0 the threshold → 0 and the definition degenerates: any member with a flat decade is a "slowdown". Fig. S12 shows ~65 of 100 members labelled slowdown around 2010 versus ~10 in 1995. (b) The labels therefore carry a strong *forced, time-dependent* signal while the SST inputs have the forced signal removed — the CNN is asked to classify a partly forced event from internal-variability maps. → **Phase 1.1 gains a "year-climatology" baseline**: predict P(slowdown | onset year) from the training members alone. If that beats the CNN, the network may simply be inferring the epoch (which is recoverable from the Arctic SST anomaly *variance* pattern, since open-water area grows with time). → **Phase 4 gains a biomass-burning test**: retrain on members 51–100 (smoothed BB forcing) only, or use `--member-groups last50`, and compare labels, base rate and skill against members 1–50. Zach raised exactly this.

2. **Fig. S2h is direct evidence for the endpoint artifact.** Slowdown onsets sit at SIE anomaly ≈ −1 M km² relative to the ensemble mean; non-slowdowns at ≈ +0.5. The separation between the two distributions is large and already in your SI. A logistic regression on that single number (Phase 1.1, item 3) can be computed in an afternoon from the existing slowdown files, before any retraining, and will tell you where the paper stands.

3. **Test-set slowdowns cluster in time across members (Fig. S6):** 2000–2012, 2018–2026, 2030–2040. That is the forced signature again, and it means the "independent" test members share the same label epochs; block-bootstrap by member (Phase 3.2) does not remove this dependence — a leave-years-out or leave-epoch-out check should be added.

**Housekeeping in the SI itself:** Text numbering is inconsistent between the contents list and the body (S1 is titled "Model Architecture" but listed as "CESM2"; LRP is S3 in the body, S4 in the list; the main text cites S2 for metrics and S4 for indices). Fig. S13 caption promises IPO panels (e,f) that do not exist, and the main text cites Fig. S13e; panels (c,d) labelled Niño 3.4 phases do not look like El Niño/La Niña composites (Zach flagged this too). `(ref)` placeholder in Text S5; "CESMS2-LE" typo. `configs/model.py` and `configs/training.py` in the repo describe a *different* network (3 conv layers, dense 128/64, softmax, categorical CE, lr 1e‑3, batch 32) from the one actually trained and described in Table S1 (`src/cnn/model.py`, `scripts/04_cesm2le_cnn_train.py`: 2 conv, sigmoid, focal loss, lr 1e‑4, batch 120). The training code is right; the config files are stale and should be deleted or synced before a reviewer opens the repo.

## Coauthor comments (Z. Labe, annotated v1)

Zach's 44 annotations fall into five groups. Where they overlap with the reviewer plan the step number is given; new items are marked ★. A separate response document (`ZL_RESPONSE.md`) addresses each in plain language.

*Science / analysis*
- Forced slowdown in the CESM2-LE ensemble mean and the biomass-burning issue; suggests sub-setting by the 50 smooth-forcing members → new Phase 4.4 ★ (and SI point 1 above).
- "Would a simple regression model with a sea ice index predict a slowdown just as well?" → Phase 1.1 (he independently arrived at the reviewer's central concern).
- Sensitivity to the 10-yr window (England et al. use 20 yr) → Phase 4.2.
- Other SST months/seasons (winter/spring preconditioning, May melt ponds) → Phase 4.2 season sweep; cheap version via logistic baselines.
- SST under sea ice: what value does each product assign in ice-covered cells, and is it driving the Arctic relevance? → new Phase 1.6 ★ (mask cells with aice > 15% at onset, or replace with climatology, and retrain/occlude).
- ERSSTv5 quality in the Arctic; recommends OISSTv2.1 (used for the Arctic Report Card SST chapter), notes ERSSTv6 exists → Phase 6.2, now OISSTv2.1 specifically (see Phase 6.2 for the download step).
- What do LRP maps and SST composites look like for observations? → new Phase 6.7 ★.
- Additional XAI robustness/trustworthiness approaches from the sea ice motion paper → Phase 5.3.
- Concurrent/lagged atmospheric composites (Z200, U200) for correct slowdown predictions → optional Phase 5.5 ★, SI only.
- Possible Atlantic/Bering ocean heat transport link to the Arctic SST anomaly; PDO signature in the composite → discussion text only, flagged as speculative.
- Fig. S13 confusion (missing e,f; Niño 3.4 phase composites don't look right) → Phase 7.

*Framing / presentation*
- Key Points 2 and 3 read as contradictory → Phase 5.4.
- Abstract's "end of the recent pause" should be hedged ("according to this prediction framework") → Phase 6.6.
- Split Conclusions into Discussion + short Conclusions; explain up front why *warm* Arctic SST goes with a *slower* decline → Phase 6.6/7.
- State the GMT-vs-SIE slowdown mismatch (Fig. S12) more forcefully → Novelty section above.
- Define the Arctic SST index briefly in the main text; mention LRP-z rule in the main text; "training data, not testing?" for LRP (yes — training; state why); explain why Fig. 4a shows one particular CNN → Phase 6.4 removes it.
- Expand intro on proposed drivers of the observed slowdown (three DOIs supplied) → Phase 7.
- Colour-blind-safe palette (replace green with blue in Figs. 1, 4, S1) → Phase 7.

*Text fixes*
- "cliamte"; "El Niño" spelling; "in CESM2-LE" qualifier on line 234; "again" at line 371; "or accelerations, rapid ice loss events"; "version 4?" for the NSIDC Sea Ice Index (check: Fetterer et al. 2017 is v3; current is G02135 v4).

*Administrative*
- Affiliation: Climate Central Inc., Princeton, NJ; ORCID 0000-0002-6394-7651. He will add a sentence to Acknowledgments after Climate Central's internal review — leave a placeholder.

Repo facts the plan relies on (verified in the code):

- `src/cnn/splits.py`: "JJA of year *t* predicts September of year *t*" — the SST input is contemporaneous with the *first* point of the 10-yr trend window. This is the root of reviewer comment 1.
- `src/data/cesm2le/slowdowns.py`: model threshold is `fraction_nsidc × ensemble_mean_trend(t)`; no ice-free screening.
- `src/cnn/model.py::METRIC_NAMES` already computes AUPRC, AUROC, Brier, Precision, Recall, F1, Prevalence — the metrics exist, they just aren't in the main text and have no baselines beside them.
- `src/data/observations/ersst/test_cnn.py` supports `forced_method in ('ensmean', 'linear')`, with `ensmean` (mean/trend-corrected CESM2 forced response) as the default in `03_ersst_test.py`, while the manuscript (§2.2) says observations are *linearly* detrended. Whichever was actually used for Fig. 4 must be stated, and both must be shown.

---

## Phase 1 — Is the Arctic SST signal an initial-condition artifact? (reviewer comments 1, 11)

> **Status (2026-09-12) — Step 1.1 run on the server, original labels.** CNN median test F1 ≈ 0.48, AUPRC ≈ 0.50, AUROC ≈ 0.71. Year-climatology alone (no SST, no ice): 0.48 / 0.50 / 0.71. Logistic on SIE anomaly at onset: 0.53 / 0.54 / 0.74. SIE + year: 0.61 / 0.67 / 0.81. Arctic SST index alone ≈ always-positive (F1 0.44). Adding Arctic SST to SIE(t) changes nothing. **Decision gate: not passed.** The CNN is indistinguishable from the forced label epoch and is beaten by the initial ice state. Next: (i) epoch-free labels (`02_cesm2le_slowdowns_relative.py`, per-forcing-group demeaning, window sweep 3–15 yr), (ii) rerun 07 with `--labels-file`, (iii) `cnn_attribution.json` to quantify epoch vs ice-state content of the CNN output, (iv) only then retrain (lag-1 SST, SIE(t) as auxiliary input, group-wise SST demeaning in `splits.py`). Branch B framing (§6.6) is now the default unless (iv) clears `logit_sie_anom` across splits.

This is the make-or-break phase. Goal: show that skill and the Arctic relevance survive when the trivial "low ice at t ⇒ flatter trend from t" pathway is removed.

**Step 1.1 — Build the scalar baselines (no CNN).** New script `scripts/07_baselines.py` and module `src/analysis/baselines.py`.
For every split, fit on the training members and evaluate on the test members, using the *same* labels as the CNN:

1. Constant classifier "always slowdown" → F1 = 2p/(1+p) at prevalence p (analytic, but compute it).
2. Random classifier at rate p (permutation, 1000 draws) → distribution of F1/AUPRC.
3. Logistic regression on **September SIE anomaly at year t** (ensemble-demeaned). This is the endpoint-leverage baseline.
4. Logistic regression on the **Arctic JJA SST index** alone.
5. Logistic regression on Arctic SST index + Niño 3.4 + IPO.
6. Logistic regression on SIE(t) + Arctic SST index (tests whether SST adds anything beyond ice state).
7. **Year-climatology baseline**: P(slowdown | onset year) estimated from the training members, applied to test members (no SST at all). Tests whether the CNN is exploiting the non-stationary label base rate (Fig. S12) rather than variability.
8. Logistic regression on SIE(t) + year (items 3 and 7 combined) — the "nothing about ocean variability" ceiling the CNN must clear.

Inputs for items 3, 7, 8 are already on disk (`results/` slowdown label files + SIE metrics from `01_cesm2le_preprocessing.py`); Fig. S2h shows the SIE(t) separation is large. Do these first.

Save to `results/baselines/baselines_split{k}.nc` with the same metric names as `METRIC_NAMES`. Output a single table (metric × model × split) that becomes new **main-text Table 1 / Figure**.

**Step 1.2 — Lagged-input experiment.** Add a `sst_lag` argument to `load_jja_sst()` and `build_split()` in `src/cnn/splits.py` (default 0 = current behaviour). Regenerate splits with `sst_lag=1` (JJA of year t−1 predicts trend starting September t) to `results/tvt_splits/..._lag1_split{k}.nc`, retrain (`04_cesm2le_cnn_train.py --tag lag1`), predict, LRP. Compare F1/AUPRC/AUROC and the Arctic-vs-tropics relevance fraction to the lag-0 run.

**Step 1.3 — Trend-window-without-the-first-year experiment.** In `compute_decadal_trends_ensemble()` add `skip_first: int = 0`; relabel with the trend fitted over years t+1 … t+9 (keeping the same onset-year indexing so the SST input is unchanged). Retrain with tag `skip1`. If skill collapses to the baseline in both 1.2 and 1.3, the result is the endpoint artifact and the paper must be reframed (see Phase 6, branch B).

**Step 1.4 — Partial-out the initial ice state.** Regress the binary label (or the continuous trend) on SIE(t) across the training set, and train the CNN on the residualised target (continuous version, see 1.5) or with SIE(t) supplied as an auxiliary scalar input so the CNN can only earn skill *beyond* it. Report the change in relevance maps.

**Step 1.5 (recommended, addresses the "future work" paragraph too) — Continuous target.** Add a regression head option to `build_cnn()` (`n_classes=1`, linear output, MSE/Huber loss) predicting the decadal trend anomaly relative to the ensemble mean. Compare against a linear regression on SIE(t). This side-steps the arbitrary 1σ threshold entirely and makes skill quantification cleaner (r², MSE skill score vs climatology and vs SIE(t) persistence).

**Step 1.6 — Ice-covered cells (Zach's under-ice SST question).** In CESM2 the SST in ice-covered cells is pinned near freezing, so the Arctic SST anomaly map is partly an ice-concentration map. Build a variant of the splits in which cells with `aice > 0.15` in JJA of the onset year are set to the land sentinel (or to the ensemble-mean value, i.e. zero anomaly), retrain with tag `openwater`, and repeat the Arctic occlusion test (Phase 5.3). If skill and Arctic relevance survive using open-water SST only, the "ocean preconditioning" language is earned; if not, the signal is ice cover. Also document what ERSSTv5/OISST assign under ice, because the observational input inherits whatever convention the product uses.

**Decision gate.** Proceed to the "physically meaningful preconditioning" framing only if, with lag-1 SST *or* skip-first labels, the CNN (or logistic Arctic-SST model) still beats the SIE(t) baseline with non-overlapping block-bootstrap CIs (Phase 3). Otherwise, reframe (Phase 6B).

---

## Phase 2 — Skill reporting and baselines in the main text (comments 2, 12)

**Step 2.1 — Replace "exceeds random" with a proper skill figure.** New main-text figure (replacing current Fig. S5 role): per-split box/strip plots of F1, AUPRC and AUROC on the test set for the CNN, with horizontal lines for always-positive, random (permutation 95th percentile), logistic-SIE(t), and logistic-Arctic-SST. Use `results/metrics/cnn_jja_metrics_split{k}.nc` + `results/baselines/`.

**Step 2.2 — Report precision and recall separately** at the PR-optimal threshold (already computed in `06_cnn_predict_cesm2le.py`), and state the threshold selection rule in Methods (it is currently chosen on the training set — say so).

**Step 2.3 — Permutation significance test.** Shuffle labels *by ensemble member × contiguous year block* (not per sample; see Phase 3) 1000 times, retrain a cheap surrogate (logistic on CNN penultimate features, or simply recompute F1 of the fixed CNN against shuffled labels) to get the null F1 distribution. Report the p-value.

**Step 2.4 — Fix units/language.** F1 = 0.48, not 48%. Replace "predict" with "classify/diagnose" wherever the input is contemporaneous with the window start (unless Phase 1.2 becomes the headline configuration, in which case "predict at one-year lead" is defensible).

---

## Phase 3 — Autocorrelation and effective sample size (comment 3)

**Step 3.1 — Count independent events.** New function `src/analysis/events.py::label_events()` that merges consecutive positive labels within a member into one event. Report: number of events per member, mean event duration, total independent events in train/val/test. Add to Methods and Fig. S2.

**Step 3.2 — Block bootstrap everywhere.** Replace i.i.d. bootstrap in the Fig. 3 notebook (`F2-3_FS8-S9_composite_pdf.ipynb`) and in the `ci_low/ci_high` in `collect_metrics_dataset()` with a block bootstrap that resamples *whole ensemble members* (the only truly independent unit). Rerun all CI numbers (VE brackets, F1 spread).

**Step 3.3 — Event-level metrics.** Add an event-based hit rate: an observed/simulated slowdown event counts as "hit" if ≥ X% of its onset years are flagged. Report alongside sample-level F1. This is also what makes the observational evaluation honest (two independent observed events).

---

## Phase 4 — Slowdown definition robustness and label contamination (comment 4)

**Step 4.1 — Ice-free floor screening.** In `02_cesm2le_slowdowns.py`, add a flag to mark any window in which the member's September SIE falls below 1 M km² (or the ensemble-mean reaches a chosen fraction of its 1990 value). Plot slowdown frequency vs onset year (new Fig. S2 panel). If frequency rises toward 2040, either (a) end the analysis at an onset year where all members are still well above the floor (e.g., 2030), or (b) drop screened windows. Retrain with the cleaned labels; report the new base rate (currently 28% vs ~16% by construction in observations — explain the difference explicitly in the text).

**Step 4.2 — Sensitivity sweep.** Parameterise and rerun labels for: threshold at 0.5σ, 1σ, 1.5σ; window 8, 10, 12, 15 yr; SST season JJA vs MJJ vs annual. Cheap version: rerun only the logistic-regression baselines (Phase 1.1) across this grid; expensive version: retrain the CNN at 1σ/10yr plus one alternative. Summarise as a heat-map Figure S.

**Step 4.3 — State the threshold construction explicitly.** Write out in Methods that trends are negative, so f = (μ+σ)/μ < 1 and the model threshold is `f × ensemble-mean trend(t)`, which is time-varying, exactly as in LB22 (their f = 0.44 for GMST). Add the observed values of μ, σ, f, and the resulting model base rate next to LB22's, with the explanation for the difference. Justify (or drop) the 1990 "slowdown" in observations — consider defining slowdowns only relative to the *preceding* decades' rate, or adding an absolute criterion (e.g., trend not significantly different from zero, following England et al. 2025) as a sensitivity.

**Step 4.4 — Biomass-burning forcing subset (Zach).** CESM2-LE members 1–50 use the CMIP6 biomass-burning emissions with spurious interannual variability over 1997–2014; members 51–100 use the smoothed version. `03_cesm2le_tvt_splits.py` already loads by `MEMBER_GROUPS = ['first50', 'last50']`. Recompute the ensemble-mean trend and slowdown labels separately for each 50-member group (Fig. S1c–d equivalent), report the base rate and the 2005–2015 label peak for each, and train a `last50`-only CNN ensemble (5 blocks → 3 train / 1 val / 1 test). If the forced slowdown and the label peak largely vanish with smoothed forcing, the current labels are contaminated by the forcing artifact and the `last50` results should become the headline, with the full-ensemble results as a sensitivity.

**Step 4.5 — Threshold degeneracy guard.** Because the threshold is `f × ensemble-mean trend(t)`, it collapses to ~0 where the forced trend flattens (Fig. S1d). Add a floor: define the model threshold as `max(f × ens-mean trend(t), μ_m + σ_m)` where μ_m, σ_m come from the pooled member trends over 1990–2040, or simply report how many labels change when the 2005–2015 windows are treated with the pooled threshold. State the choice in Methods.

---

## Phase 5 — Conditioning and XAI analyses (comments 5, 6, 7)

**Step 5.1 — Condition on truth, not on TPs.** In the Fig. 3 notebook, produce P(slowdown | phase) using *all* labelled samples (train+test, both model and, separately, observations), and only then P(TP | phase) as a diagnostic of *model* behaviour. The physical claim in the abstract must rest on the former. Same for the SST composite: composite over all slowdown onsets, then over TP/FP/FN separately.

**Step 5.2 — Define VE.** Write the formula used (presumably η² / R² of the binary indicator on the categorical phase). Report it with block-bootstrap CIs and replace "strongly" with language proportional to ~11%.

**Step 5.3 — XAI robustness.** Add to `src/xai/`:
- a second attribution method (Integrated Gradients or SmoothGrad via `innvestigate`/TF), compare spatial correlation with LRP-z;
- an occlusion test: zero (set to climatology) the Arctic (>60°N), tropical Pacific, and North Pacific boxes already defined in `XAI_CONFIG['regions']` and record the F1 drop for each — this is the cleanest "what matters" evidence and does not depend on LRP;
- a sanity check: LRP from randomly-initialised and from label-shuffled-trained networks (Adebayo-style), to show the Arctic pattern is not architecture-driven.

**Step 5.4 — Drop or support the CP-El Niño narrative.** Unless the occlusion test shows a measurable skill contribution from the central Pacific, remove Key Point 3 and the CP/EP discussion, or move it to a hedged sentence. Reconcile Key Points with results (Niño 3.4 VE = 1.3% cannot coexist with "highly relevant").

---

## Phase 6 — Observational application (comments 8, 9, 10)

**Step 6.1 — Forced-signal removal.** Run `03_ersst_test.py` and `06_cnn_predict_ersst.py` with both `linear` and `ensmean`, plus a quadratic detrend option (add to `test_cnn.py`). Plot the resulting Arctic SST index for 2010–2025 under each method. If the 2016–2020 positive Arctic anomaly and the ~85% 2016 slowdown vote disappear under `ensmean`/quadratic, the current Fig. 4 result is a detrending artifact and must be presented as such (or removed).

**Step 6.2 — Alternative SST product: OISST v2.1 (Zach's recommendation).** OISSTv2.1 (NOAA, daily, 0.25°, Sept 1981–present, satellite AVHRR + in situ, with an explicit sea-ice-concentration-based proxy SST under ice) is what the Arctic Report Card SST chapter uses. Steps: (i) add `src/data/observations/oisst/download.py` pulling monthly means from NOAA PSL (`sst.mon.mean.nc`, ~1 GB) and its ice-concentration companion if available; (ii) reuse `regrid_to_cesm2le.py` (bilinear, ocean-only) — check the land/ice mask handling because OISST has values under ice while the CESM2 grid treats those cells as ocean with SST ≈ −1.8 °C; (iii) run `03_ersst_test.py`-equivalent for OISST with both forced-removal methods; (iv) show the Arctic SST index and the ensemble vote fraction for ERSSTv5 and OISST side by side (new Fig. S). Also note ERSSTv6 exists (Zach); a one-line justification for whichever product is primary is needed in §2.2. This step is independent of everything in Phases 1–5 and can be done by a student/RA in parallel.

**Step 6.3 — Fix text/figure mismatches.** Recompute the fraction of CNNs predicting a slowdown for each onset year 2016–2025 and state the numbers; the current text ("fewer than 20% … 2017–2025") disagrees with Fig. 4b (2017 ≈ 0.35–0.4; 2016 ≈ 0.85 is not mentioned). Extend observed labels to onset year 2016 (2016–2025 needs September 2025, which is available).

**Step 6.4 — Replace Fig. 4a.** Remove the single-CNN panel; show the ensemble fraction with block-bootstrap uncertainty, and mark event-level hits/misses for the two independent observed events.

**Step 6.5 — Sign discrepancy caveat.** Add a paragraph making explicit that CESM2-LE's positive-IPO/ENSO association is opposite to the observed 2007–2013 case, and that this limits confidence in the observational forecast. Cite Topál & Ding 2023 / Bonan & Blanchard-Wrigglesworth 2020 here, not as an aside.

**Step 6.7 — LRP and SST composites for observations (Zach).** Run `05_cesm2le_lrp.py`-equivalent on the observational inputs (all 45 CNNs), composite relevance and SST over the years flagged by >15% of CNNs (1990, 2007, 2016–2020) and over the observed slowdown onsets. Compare with the model TP composite (Fig. 2). This is also the cleanest way to see whether the 2016–2020 votes rest on the Arctic (detrending artifact suspicion, Step 6.1) or on the Pacific. SI figure.

**Step 6.6 — Rewrite the conclusion.** Two branches depending on Phase 1:

- **Branch A (signal survives lagged/skip-first tests):** "Arctic SST preconditioning at one-year lead carries skill beyond the initial ice state; tropical Pacific modulation is weak and model-dependent." Keep a *heavily* hedged observational application with no "end of the pause" claim in the abstract.
- **Branch B (signal does not survive):** reframe the paper as a cautionary methodological result: a CNN+XAI pipeline trained on a large ensemble recovers a known statistical property (initial-condition dependence of decadal trends) and presents it as ocean preconditioning; the paper's contribution becomes the demonstration of the necessary controls (baselines, lag tests, occlusion) for XAI-based "precursor" claims in sea ice. This is publishable and consistent with the first author's XAI-trustworthiness work.

---

## Phase 7 — Manuscript clean-up (minor comments)

- Fig. 1 caption: two "(b)" labels. Line 120 "cliamte"; line 162 "an analogous". Niño spelling consistent everywhere including Fig. 3.
- Units: "M km²" vs "Mkm²/decade" — pick one; note trends in code are M km² yr⁻¹ (per-year) while text quotes per-decade.
- Methods: state 45 = 9 splits × 5 seeds; define "test data unique in each split"; state that the SIE domain is poleward of 60°N while NSIDC is hemispheric.
- Methods §2.2: state the forced-removal method actually used for observations (code default is `ensmean`, text says linear).
- References: add authors to Notz et al. (2020); fix CO₂ subscript in Notz & Stroeve (2016); de-duplicate DOI/URL fields; check Danabasoglu capitalisation; check the Swart et al. (2015) citation for slowdown identification.
- Move at least one skill figure (Phase 2.1) into the main text; move Fig. 2 or 3 to SI if the GRL figure limit forces it.
- Structure (Zach): split §4 into a Discussion (physical interpretation of warm-Arctic-SST → slower decline, model–obs differences, GMT vs SIE) and a short Conclusions.
- Main text should briefly define the Arctic SST index, name the LRP-z rule, and say LRP is computed on training data and why.
- Intro: one or two sentences on proposed drivers of the observed slowdown, using the three references Zach supplied (10.1126/science.adh5158; 10.1088/1748-9326/abc047; 10.1038/s43247-025-02882-1).
- Colour-blind-safe palette: replace green/red pairs in Figs. 1, 4, S1, S6 (blue/orange).
- NSIDC Sea Ice Index version: confirm v3 vs v4 and cite accordingly.
- SI: fix Text S-numbering and cross-references; Fig. S13 (add IPO panels e,f or drop the reference; verify the Niño 3.4 phase composites); `(ref)` in Text S5; "CESMS2-LE".
- Repo: delete or sync `configs/model.py` / `configs/training.py` (they describe a network that was never trained); add Zach's affiliation and ORCID; leave an Acknowledgments placeholder for Climate Central's internal-review sentence.

---

## Suggested execution order and rough effort

| Order | Steps | Retraining? | Effort |
|---|---|---|---|
| 1 | 1.1 baselines (incl. SIE(t) and year-climatology), 3.1 event counts, 4.1 frequency-vs-year, 4.4 first50/last50 label comparison | no | 2 days |
| 2 | 1.2 lag-1, 1.3 skip-first labels, 1.6 open-water-only input | yes (3 × 45 CNNs) | 3–4 days compute |
| 3 | 6.1 detrending comparison, 6.3 recount, 6.7 obs LRP composites | LRP only | 1–2 days |
| 4 | 5.3 occlusion + second XAI method, 5.1 recondition (all-slowdown numbers to main text) | LRP only | 2 days |
| 5 | 3.2 block bootstrap, 5.2 VE definition, 4.5 threshold guard | no | 1 day |
| 6 | 4.2 sensitivity sweep (window, σ, season; logistic only) | no | 1 day |
| 7 | 4.4 last50-only CNN retrain; 1.5 regression head (optional) | yes | 3 days |
| 8 | 6.2 OISST v2.1 download, regrid, rerun obs pipeline | no | 1–2 days (parallelisable) |
| 9 | Rewrite: Discussion/Conclusions split, Key Points, abstract hedge, SI/repo housekeeping (Phases 6.6, 7) | — | 1 week |

Everything in rows 1, 3 and 5 can be done from cached outputs and settles most of the factual disputes (and most of Zach's analysis questions) before any GPU time is spent. Row 1 is the decision gate: the SIE(t) and year-climatology baselines will show within a day whether the CNN is learning ocean variability or the initial ice state plus the forced epoch.
