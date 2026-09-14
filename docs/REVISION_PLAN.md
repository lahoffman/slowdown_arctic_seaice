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

## Phase 1 — Does the SST pattern carry information beyond the ice state? (reviewer comments 1, 11; Zach's regression and under-ice comments)

> **Status (2026-09-13) — Steps 1.1–1.3 complete. Decision: retrain on the relative labels, pooled σ, onsets 1990–2030, SST demeaned per forcing group.**
>
> *Original labels.* CNN median test F1 ≈ 0.48, AUPRC ≈ 0.50, AUROC ≈ 0.71 — indistinguishable from the onset-year climatology (0.48 / 0.50 / 0.71), a predictor that never sees SST or ice. Logistic regression on SIE anomaly at onset: 0.53 / 0.54 / 0.74. SIE + year: 0.61 / 0.67 / 0.81. Arctic SST index alone ≈ always-positive. Conclusion: with the LB22-style labels the CNN's skill is fully explained by the forced label epoch; the labels are unusable for an internal-variability attribution.
>
> *Relative labels (`02_cesm2le_slowdowns_relative.py`, w10, 1σ, group demeaning).* Base rate 0.15; onset-year climatology falls below always-positive (F1 0.17 vs 0.26) — the epoch is gone. SIE anomaly alone: F1 0.43, AUPRC 0.41, AUROC 0.76. Arctic SST alone 0.31 / 0.28 / 0.68 (some skill, but adding it to SIE(t) changes nothing). Pacific indices on top of SIE(t) raise AUROC to ~0.79 — the honest upper bound on remote-index information given the ice state. The existing CNN scored against the new labels (a transfer test): 0.35 / 0.30 / 0.70 — above climatology, below the one-variable SIE regression.
>
> *σ mode and onset range (Step 1.2, Fig. S2).* The member spread of the decadal-trend anomaly is flat at 0.12–0.15 M km² yr⁻¹ until ≈2025 and then collapses with the ice (0.02 by 2060). Pooled σ (0.130, onsets 1990–2040) keeps the base rate flat at ≈0.15–0.20 until ≈2028 and then rolls it off to zero by 2040; a year-dependent σ keeps 15% flagged forever but only by calling 0.02 M km² yr⁻¹ anomalies at 0.4 M km² of ice "slowdowns" (191 of 5100 training-window labels differ, essentially all after 2030). Every window length (3–15 yr) rolls off from the same year, so this is a property of the ice, not of the 10-yr window. **Decision: pooled σ, onsets capped at 2030** (`03 --end-year 2030`): 41 onset years × 100 members = 4100 samples (3280 / 410 / 410 per split), 20% fewer than before, base rate flat throughout. The yearly-σ label file is kept on disk but not used.
>
> *Forcing groups (Step 1.3, Fig. S3).* The SMBB members are 0.1–0.3 °C cooler than the CMIP6-BB members across the northern extratropics in 2000–2020 (North Pacific and North Atlantic/Labrador Sea strongest); the Arctic-mean forced difference peaks at −0.14 °C around 2009. Removing the 100-member mean therefore leaves a systematic ±0.05–0.1 °C anomaly of opposite sign in the two groups' "internal variability" maps for exactly the decades where most of the original labels lived — small against the ±0.5 °C member spread, but a group-identity shortcut a CNN can learn. All retrained configurations remove each group's own 50-member mean instead. Consequence for the observations: the ERSST forced reference must be chosen (either group or the 100-member mean; ≤0.1 °C difference in the Arctic) — Phase 6.1.
>
> *Decisions taken.* (1) The manuscript adopts the relative definition (§2.3 rewritten, tracked). (2) Baselines are in the SI (now Fig. S5) and §3.1 is reframed around them. (3) The CNN is retrained on the relative labels with the settings above; Steps 1.4–1.8 are the retrain. (4) Two SI figures document the two free choices: Fig. S2 (pooled σ, onset cap) and Fig. S3 (forcing-group forced response).

> **Status (2026-09-13, evening) — first retrained configuration and the cheap analyses.**
>
> *`rel_base` (SST map only, 45 CNNs, 4 h 20 min).* Test F1 0.37 / AUPRC 0.39 / AUROC 0.72 at prevalence 0.18; onset-year climatology at chance (AUROC 0.51) — no epoch left. Above the old CNN transferred to these labels (0.35 / 0.30 / 0.70) but below the one-variable SIE-anomaly regression (0.43 / 0.48 / 0.77) and SIE + Pacific (0.45 / 0.52 / 0.80); it matches the three-index regression `logit_indices` (0.37 / 0.35 / 0.72) almost exactly. Reading: from the full map the network recovers what three area indices contain, and less than the ice state. `rel_aux`, `rel_lag1`, `rel_openwater` training; LRP, occlusion and observational predictions queued (`run_postprocess.sh`).
>
> *Label sensitivity (4.2).* Ice state > indices by 0.05–0.09 AUROC and Pacific increment +0.01–0.03 in every window × threshold cell; skill rises with threshold, not window. Our 10 yr / 1σ cell has the largest Pacific increment (+0.033; grid median +0.014).
>
> *Forcing groups (4.4).* SMBB test members ≈0.05 AUROC more predictable than CMIP6-BB for the scalar regression and the CNN alike — physical, not a shortcut. Closed without the last50-only retrain.
>
> *Events (3.1/3.3).* 702 positive member-years = 239 events of mean 2.9 yr, onsets flat in time. `rel_base` event hit rate 0.62, false-alarm ratio 0.61.
>
> *Observations (6.1/6.2).* OISST v2.1 downloaded, monthly-averaged and block-averaged to the CESM2 grid. OISST is ≈0.5 °C warmer than ERSST in the central Arctic in the 1990s, converging by 2015, so ERSST's 1990–2025 Arctic JJA trend is ≈2× OISST's — ERSST leaves the ice-covered interior empty (74 % vs 99 % coverage) and pins under-ice cells near freezing, so cells switching to open water inflate its trend. After forced-signal removal the 2016–2025 Arctic anomaly is ≈0 or negative for all five references × both products, and 2010–2015 positive for all: the observational "no warm Arctic since 2016" statement is robust. Decision: OISST primary product (coverage), `group_cmip6` primary forced reference (observation-based emissions; CNN trained group-demeaned), others as range.

> **Status (2026-09-14) — retrain complete for all four configurations; postprocess (occlusion, LRP, observational predictions) complete. Decision gate 1.8: Branch B unless the reformulation tests (Phase 8) change the picture.**
>
> *Skill on the Fig. S5 axes (test members, median over 9 splits × 5 seeds; `results/baselines/<tag>/baselines_summary.md`).* Prevalence 0.176; always-positive F1 0.30.
>
> | configuration | input | F1 | AUPRC | AUROC |
> |---|---|---|---|---|
> | `rel_base` | JJA SST map, onset year | 0.37 | 0.39 | 0.720 |
> | `rel_aux` | + September SIE anomaly as scalar input | 0.39 | 0.39 | 0.723 |
> | `rel_openwater` | `rel_aux` with ice-covered cells (aice > 0.15) set to zero anomaly | 0.34 | 0.35 | 0.692 |
> | `rel_lag1` | `rel_aux` with SST of the year *before* onset | — | — | 0.624 |
> | `logit_sie_anom` | SIE anomaly alone (linear) | 0.43 | 0.48 | 0.765 |
> | `logit_indices` | Arctic SST + Niño 3.4 + IPO (linear) | 0.37 | 0.35 | 0.719 |
> | `logit_sie_pacific` | SIE + Niño 3.4 + IPO (linear) | 0.45 | 0.52 | 0.798 |
> | `logit_sie_ipo` | SIE + IPO (linear) | 0.45 | 0.53 | 0.800 |
>
> Reading: (i) no CNN configuration reaches the one-variable SIE regression; the bar for Branch A (`logit_sie_pacific`, 0.80) is not approached. (ii) `rel_aux` scores the same as `rel_base` although it is *given* the SIE anomaly: the network does not exploit the scalar it was handed (a single standardised scalar concatenated to the dense layer, dropout 0.2, early stopping after ~5–10 epochs) and still reads the ice state off the Arctic part of the map. The CNN is therefore a probe, not a tuned predictor, and must be described as such. (iii) The full SST map is worth what three area indices are worth (`rel_base` ≈ `logit_indices` on all three metrics). (iv) `rel_lag1`: at one-year lead the skill collapses to 0.62 — there is no *prediction*, only concurrent classification (Phase 2.4: "predict" → "classify" throughout). (v) `rel_openwater` loses 0.03 AUROC; the mask pattern itself (where the zeros are) still encodes the ice edge, so this is a lower bound on how much of the Arctic signal is ice cover, not a clean open-water test.
>
> *Region occlusion (5.3; `results/occlusion/<tag>/occlusion_summary.md`; ΔAUROC = occluded − full, median over 45 models; regions: Arctic >65°N, North Pacific, tropical Pacific, North Atlantic, everything except the Arctic).*
>
> | configuration | full | −Arctic | −N. Pacific | −trop. Pacific | −N. Atlantic | −all but Arctic |
> |---|---|---|---|---|---|---|
> | `rel_base` | 0.720 | **−0.096** | −0.001 | +0.003 | −0.001 | **+0.013** |
> | `rel_aux` | 0.723 | **−0.077** | +0.001 | +0.002 | −0.000 | **+0.010** |
> | `rel_openwater` | 0.679 | **−0.049** | −0.007 | +0.003 | −0.002 | +0.009 |
> | `rel_lag1` | 0.624 | −0.027 | +0.002 | −0.001 | −0.004 | +0.014 |
>
> Reading: every configuration's skill is the Arctic; zeroing the whole extra-Arctic field costs nothing and *improves* the median by 0.01 (the network fits extra-Arctic noise). The Pacific boxes contribute ≤ 0.007 in every configuration, of either sign. Per-split (rel_base) the Arctic drop is 0.07–0.14 in all nine splits and the extra-Arctic drop is ≤ 0.016 and negative in six of nine. This settles 5.4: Key Point 3 (central-Pacific relevance) is not supported; the LRP hotspots in the Pacific (v1 Fig. 2, and the same hotspots in LB22's Fig. 2) are where the network *places* relevance, not where its skill comes from — an LRP-on-standardised-anomalies artefact (relevance follows variance) that the occlusion test exposes. Figure: `make_figure.py occlusion` (all configurations, one panel) — candidate main-text figure.
>
> *Training diagnostics (`results/logs/<tag>/history_split{k}_run{r}.json`; `make_figure.py learning_curves --tag <tag>`).* Validation focal loss is flat from epoch 1 (≈0.18 → 0.15 by epoch 30, noise ±0.03 on 410 validation samples / ~70 positives) while training loss falls 0.25 → 0.10: the generalisable signal is learnt in the first few epochs, the rest is memorisation. Train > val is expected (class-weighted, dropout, L2 in the training loss only). Early stopping (`val_loss`, patience 10, restore best) picks epochs 1–31; for `rel_base`, models whose best epoch was ≥ 3 show no relation between chosen epoch and test AUROC (r = 0.08; within-split anomaly +0.01 for epochs 3–12, −0.003 for 13–31), but the five models that stopped at epoch 1–2 are 0.04 AUROC below their split mean — essentially untrained networks that won a coin flip on validation noise. Between-split spread of AUROC (sd 0.044; split 7 ≈ 0.80, split 5 ≈ 0.65) is twice the seed spread (0.021): *which members are in the test set matters more than anything about training* (→ 3.2 report the split range, not a seed ±). Fix for any further training (not applied retroactively): `EarlyStopping(start_from_epoch=5)`, monitor `val_auprc` (add `keras.metrics.AUC(curve="PR")`). Learning-curve figure → SI as a training diagnostic.
>
> *Observational predictions (6.3).* `results/predictions/{ersst,oisst}/forced_{ensmean,group_cmip6,group_smbb,linear}/<tag>/cnn_prediction_<product>_M{k}_{r}.nc` exist for all four tags (ERSST × `rel_openwater` skipped by design: no ice field). Vote-fraction recount and Fig. 4 not yet done.
>
> *Events (3.3) for the other tags, LRP composites (Fig. 2) for the retrained models: not yet looked at.*
>
> *Why LB22 "worked" and this did not (for the Discussion).* (a) Physics: tropical-Pacific OHC drives GMST directly (ocean heat uptake; Kosaka & Xie 2013, England et al. 2014); its link to September SIE is an atmospheric bridge (Ding et al. 2014/2017, Baxter et al. 2019) that acts mainly in summer, competes with local circulation noise and then with the ice's own persistence/thickness memory — a second-order pathway with far lower signal-to-noise. (b) Predictor: OHC100 annual mean is an integrated, low-pass quantity; JJA SST is one noisy season of a surface field; the GMST target has no ice-floor problem. (c) Baseline: LB22's only reference model is a logistic regression on the IPO index (F1 0.28 vs ANN 0.40 at prevalence 0.10; always-positive F1 0.19); there is no regression on the GMST state at onset and no occlusion, i.e. the test this revision applies is one the parent paper did not have to pass. The paper should state (a) and (b) explicitly and make (c) as a general methodological point (persistence baselines and occlusion are needed before XAI hotspots are read as precursors), not as a criticism of LB22.
>
> *Decision gate 1.8.* Branch A requires `rel_aux` > `logit_sie_pacific` on AUROC/AUPRC in most splits: it is 0.08 AUROC *below* and below the one-variable regression. **Branch B** on the predictive question. Before the rewrite, Phase 8 asks the *concurrent/diagnostic* version of the SST–ice question (what accompanies a slow decade, rather than what precedes it), which is the question Baxter et al. (2019) answered with composites and which the atmospheric-bridge mechanism makes more plausible; its outcome decides between "clean negative" and "no predictive skill, but a concurrent Pacific modulation". The lead-0 negative result stays in the paper under either outcome.

> **SI figure numbering (v2).** S1 definition (unchanged) · **S2 pooled σ and onset cap (new)** · **S3 forcing-group forced response (new)** · S4 label distributions · S5 baselines · S6 PR curve · S7 confusion matrices · S8 metric strip · S9 test-member timeline · S10 composites all vs CNN · S11–S13 FP/TN/FN composites · S14 P(event | phase) train · S15 SIE vs GMT · S16 obs phase composites. The review sections above this block quote v1 numbers (v1 S3–S14 = v2 S5–S16).

The phase has two halves: 1.1 established the baselines; 1.2–1.8 retrain the CNN so that its skill is measured *beyond* the scalar predictors (SIE anomaly at onset, Arctic SST index, Niño 3.4, IPO). Everything is designed so the answer is unambiguous either way.

**Step 1.1 — DONE. Scalar baselines** (`scripts/07_baselines.py`, `src/analysis/baselines.py`). Always-positive, random-at-prevalence, onset-year climatology, and logistic regressions on SIE anomaly, Arctic SST, Niño 3.4, IPO and their combinations, fit on training members and scored on test members for each of the 9 splits, with member-block bootstrap CIs. Run on both label sets; results in the status block above and in Fig. S3. The CNN-output attribution diagnostic (`cnn_attribution.json`: R² of CNN probability on year-climatology and SIE anomaly) is part of this step.

**Step 1.2 — DONE. Labels for the retrain: pooled σ, onsets 1990–2030 (Fig. S2).** Relative definition (`02_cesm2le_slowdowns_relative.py`), 10-yr window, +1σ, demeaned per forcing group. `--sigma-mode both` now writes the pooled and the year-dependent σ label files side by side (`…_group_1990-2100.nc` and `…_group_yearly_1990-2100.nc`) and draws the decision figure `…_sigma_modes.png`: (a) member spread of the trend anomaly by onset year with the ensemble-mean SIE on the right axis, (b) base rate by onset year under each σ, (c) how many of the 100 labels per year differ. Outcome in the status block: pooled σ, onsets restricted with `03_cesm2le_tvt_splits.py --end-year 2030`. Fig. S2 (`make_figure.py S2`) shows the spread collapse and the base rate under the chosen definition; §2.3 gets one sentence on the cap (tracked change, pending). Split-aligned climate indices are regenerated by `03` for every tag (or `03 --climate-indices-only --tag <tag>`); `make_figure.py` takes `--sigma-mode yearly` for Figs S1/S2.

```
python scripts/02_cesm2le_slowdowns_relative.py --sigma-mode both        # labels + sigma_modes figure
```

**Step 1.3 — DONE. SST preprocessing to match the labels (Fig. S3).**
- *Group-wise forced response.* `src/data/cesm2le/forced.py::forced_response(sst, demean='all'|'group')` is the single definition, used on the fly by `splits.py::load_jja_sst_demeaned(demean=…)` and persisted by `02_cesm2le_forced.py` (now also writes `cesm2le_groupmean_jja_sst.nc` and two diagnostics in `results/figures/diagnostics/`: `forced_group_difference.png` — SMBB − CMIP6 forced JJA SST map and the Arctic-mean series per group; `forced_demeaned_arctic.png` — member Arctic SST anomalies under both demeanings). Land sentinel and global standardisation unchanged.
- *Lag.* `03 --sst-lag L` uses JJA of year *t − L* for target year *t* (needs `--start-year 1990+L`); the split file records `sst_years` and `target_years`.
- *Auxiliary input.* `03 --aux sie_anom` stores the onset-year September SIE anomaly (demeaned the same way as the SST, standardised with training statistics) as `aux_tr/va/te`; `model.py::build_cnn(n_aux=…)` concatenates it with the flattened map features before the output layer (n_aux = 0 reproduces the original network; old `.h5` files still load); `04`/`05`/`06` detect the auxiliary columns from the split file (`--no-aux` to ignore). LRP (`compute_lrp_z`) accepts `[maps, aux]` and returns map relevance.
- *Tags.* `configs/paths.py` path helpers take `tag`; `03`/`04`/`05`/`06` take `--tag` and write to `tvt_splits/<tag>`, `models/<tag>`, `metrics/<tag>`, `attributions/<tag>`, `predictions/cesm2le/<tag>`. `07_baselines.py --cnn-tag <tag> --demean group` scores a tagged configuration on the Fig. S3 axes; `make_figure.py --tag <tag>` draws every CNN-based figure for it. Untagged = original outputs, untouched.

```
python scripts/02_cesm2le_forced.py                                       # forced fields + diagnostics
LBL=$SLOWDOWN_DATA_ROOT/cesm2le/slowdowns/cesm2le_sie_slowdown_relative_SEP_w10_s1_group_1990-2100.nc
python scripts/03_cesm2le_tvt_splits.py --labels-file $LBL --demean group --end-year 2030 --tag rel_base
python scripts/03_cesm2le_tvt_splits.py --labels-file $LBL --demean group --end-year 2030 --aux sie_anom --tag rel_aux
python scripts/03_cesm2le_tvt_splits.py --labels-file $LBL --demean group --end-year 2030 --aux sie_anom --sst-lag 1 --start-year 1991 --tag rel_lag1
```
Split files record `target_years = 1990-2030` (`1991-2030` for lag1), `demean = group`, `aux`, `labels_file`.

*Manuscript changes from 1.2/1.3 (tracked changes pending).* New Fig. S2 (σ / cap) and Fig. S3 (forcing groups); old S2–S14 renumbered S4–S16 (`make_figure.py` and `paper.py` already use the new ids). §2.1: forced response removed per forcing group, with the obs reference deferred to Phase 6.1. §2.3: pooled σ, onsets 1990–2030, one clause that all windows roll off from ≈2028. Fig. S1 and S4 unchanged in content. After 1.4: every CNN-based figure — Figs 2, 3, S6–S14 — is regenerated with `make_figure.py --tag rel_aux` (or whichever configuration 1.8 selects), Fig. S5 gains the retrained CNN bars, Fig. 4 waits for Phase 6.

**Step 1.4 — DONE (2026-09-13/14; `rel_base` 4 h 20 min, four tags ≈ 17 h on profx). Train three configurations, same architecture, 9 splits × 5 seeds each** (`scripts/run_retrain.sh`, which chains 04 → 06 → 07 per tag, is resumable via `04 --skip-existing`, and has a `--smoke` mode: 1 split, 1 seed, 2 epochs).
- `base`: JJA SST of the onset year (as now), on the new labels. Reference run.
- `aux`: same SST plus the September SIE anomaly at onset as a scalar input concatenated before the dense layer. The CNN can then only earn skill *beyond* the ice state. Headline configuration.
- `lag1`: JJA SST of the year *before* onset (+ the SIE scalar). Tests whether SST-pattern skill exists at one year lead, i.e. whether "predict" is defensible.
(Absorbs the earlier lag-1, skip-first-year and residualised-target ideas.)

**Step 1.5 — DONE (`results/baselines/<tag>/baselines_summary.md`; table in the 2026-09-14 status block).** Evaluate every configuration on the Fig. S5 axes. `06_cnn_predict_cesm2le.py` then `07_baselines.py --labels-file <relative> --tag <config>` for each, so all three land next to the same baselines. Member-block bootstrap (Phase 3.2). The bar to clear is `logit_sie_pacific` (SIE + Niño 3.4 + IPO, AUROC ≈ 0.79), not always-positive.

**Step 1.6 — DONE (`rel_openwater`: AUROC 0.69, Arctic occlusion −0.049 vs −0.096 for `rel_base`; caveat: the zero pattern still encodes the ice edge).** `scripts/02_cesm2le_icemask.py` nearest-neighbour regrids the JJA-mean CICE concentration of every member-year to the SST grid and saves the `aice > 0.15` mask (`cesm2le/aice/icemask/`, diagnostic `figures/diagnostics/icemask_summary.png`); `03 --openwater` zeroes the demeaned SST anomaly under that mask before standardisation (lagged inputs use the mask of the SST year); the split records `openwater` / `ice_threshold`. In CESM2 the SST in ice-covered cells is pinned near freezing, so the Arctic SST anomaly map is partly an ice-concentration map. Train an `openwater` variant of `aux` in which cells with `aice > 0.15` in JJA of the onset year are set to zero anomaly, and repeat the Arctic occlusion test (Phase 5.3). If skill and Arctic relevance survive on open-water SST alone, the "ocean preconditioning" language is earned; if not, the Arctic signal is ice cover. Also document what ERSSTv5 / OISST assign under ice, since the observational input inherits the product's convention.

**Step 1.7 — LRP DONE for all four tags (`results/attributions/<tag>/`, 45 files each); occlusion DONE (table above); TP composites (Fig. 2 with `--tag`) not yet drawn.** `05_cesm2le_lrp.py` on `aux`, `lag1` and `openwater`; TP composites (Fig. 2) and the occlusion test (Phase 5.3) on the same models. With SIE(t) supplied as a scalar, Arctic relevance in the SST map can no longer be "the ice edge"; whatever remains there or in the Pacific is the pattern information.

**Step 1.8 — DECIDED: Branch B on the predictive question (see 2026-09-14 status block); Phase 8 tests the concurrent question before the rewrite.** Branch A: `aux` beats `logit_sie_pacific` on AUROC/AUPRC across most splits → the paper claims SST-pattern predictability of Arctic SIE slowdowns beyond the ice state, with `lag1` deciding whether it is prediction or diagnosis. Branch B: it does not → the contribution is methodological (Fig. S5 as the central result, Phase 6.6). Either branch is publishable; §3.1 was written to hold under both. Phase 6 and 7 rewriting waits for this gate, except the branch-independent items (6.1, 6.2, 6.7, clean-up).

---

## Phase 2 — Skill reporting and baselines in the main text (comments 2, 12)

**Step 2.1 — DONE (Fig. S5, `make_figure.py S5`).** Replace "exceeds random" with a proper skill figure. New main-text figure (replacing current Fig. S5 role): per-split box/strip plots of F1, AUPRC and AUROC on the test set for the CNN, with horizontal lines for always-positive, random (permutation 95th percentile), logistic-SIE(t), and logistic-Arctic-SST. Use `results/metrics/cnn_jja_metrics_split{k}.nc` + `results/baselines/`.

**Step 2.2 — Report precision and recall separately** at the PR-optimal threshold (already computed in `06_cnn_predict_cesm2le.py`), and state the threshold selection rule in Methods (it is currently chosen on the training set — say so).

**Step 2.3 — Permutation significance test.** Shuffle labels *by ensemble member × contiguous year block* (not per sample; see Phase 3) 1000 times, retrain a cheap surrogate (logistic on CNN penultimate features, or simply recompute F1 of the fixed CNN against shuffled labels) to get the null F1 distribution. Report the p-value.

**Step 2.4 — Fix units/language.** F1 = 0.48, not 48%. Replace "predict" with "classify/diagnose" wherever the input is contemporaneous with the window start (unless the `lag1` configuration of Step 1.4 shows skill, in which case "predict at one-year lead" is defensible).

---

## Phase 3 — Autocorrelation and effective sample size (comment 3)

**Step 3.1 — DONE (`09_event_stats.py`).** 702 positive member-years in the 1990–2030 relative labels form 239 events (runs of consecutive flagged onset years) of mean 2.9 yr; onsets flat in time. v1 labels: 1417 → 364 events, 3.9 yr. This is the effective positive sample size for Methods and for the observational evaluation (2–3 events in 1990–2025). *Original plan:* New function `src/analysis/events.py::label_events()` that merges consecutive positive labels within a member into one event. Report: number of events per member, mean event duration, total independent events in train/val/test. Add to Methods and Fig. S2.

**Step 3.2 — Block bootstrap everywhere.** Replace i.i.d. bootstrap in the Fig. 3 notebook (`F2-3_FS8-S9_composite_pdf.ipynb`) and in the `ci_low/ci_high` in `collect_metrics_dataset()` with a block bootstrap that resamples *whole ensemble members* (the only truly independent unit). Rerun all CI numbers (VE brackets, F1 spread).

**Step 3.3 — DONE for `rel_base`, rerun per tag (`09_event_stats.py --tag`).** Event hit rate (≥1 flagged year in an event) 0.62, false-alarm ratio (predicted runs touching no event) 0.61, vs sample F1 0.38; hit rate saturates at 0.6–0.7 once sample F1 > 0.3. *Original plan:* Add an event-based hit rate: an observed/simulated slowdown event counts as "hit" if ≥ X% of its onset years are flagged. Report alongside sample-level F1. This is also what makes the observational evaluation honest (two independent observed events).

---

## Phase 4 — Slowdown definition robustness and label contamination (comment 4)

**Step 4.1 — SUPERSEDED by the relative definition (see Step 1.2).** Ice-free floor screening. In `02_cesm2le_slowdowns.py`, add a flag to mark any window in which the member's September SIE falls below 1 M km² (or the ensemble-mean reaches a chosen fraction of its 1990 value). Plot slowdown frequency vs onset year (new Fig. S2 panel). If frequency rises toward 2040, either (a) end the analysis at an onset year where all members are still well above the floor (e.g., 2030), or (b) drop screened windows. Retrain with the cleaned labels; report the new base rate (currently 28% vs ~16% by construction in observations — explain the difference explicitly in the text).

**Step 4.2 — DONE for window × threshold (`09_sensitivity_sweep.py`; SST season → optional).** Windows 8/10/12/15 yr × 0.5/1/1.5σ, onsets 1990–2030, baselines only: ice state > indices by 0.05–0.09 AUROC and SIE + Pacific − SIE = +0.007…+0.033 in every cell; skill rises with threshold, not window. The paper's 10 yr/1σ cell has the largest Pacific increment (+0.033; median +0.014) — state this. Heat-map `diagnostics/sensitivity_sweep.png` → SI. *Original plan:* Parameterise and rerun labels for: threshold at 0.5σ, 1σ, 1.5σ; window 8, 10, 12, 15 yr; SST season JJA vs MJJ vs annual. Cheap version: rerun only the logistic-regression baselines (Phase 1.1) across this grid; expensive version: retrain the CNN at 1σ/10yr plus one alternative. Summarise as a heat-map Figure S.

**Step 4.3 — DONE in §2.3 (tracked change).** State the threshold construction explicitly. Write out in Methods that trends are negative, so f = (μ+σ)/μ < 1 and the model threshold is `f × ensemble-mean trend(t)`, which is time-varying, exactly as in LB22 (their f = 0.44 for GMST). Add the observed values of μ, σ, f, and the resulting model base rate next to LB22's, with the explanation for the difference. Justify (or drop) the 1990 "slowdown" in observations — consider defining slowdowns only relative to the *preceding* decades' rate, or adding an absolute criterion (e.g., trend not significantly different from zero, following England et al. 2025) as a sensitivity.

**Step 4.4 — DONE (`09_baselines_by_group.py`; last50-only CNN dropped to optional).** Same fits, test members scored per group: SMBB more predictable than CMIP6-BB by ≈0.05 AUROC for the SIE-anomaly regression (0.82 vs 0.77) *and* for the CNN (0.77 vs 0.72). A regression with one scalar input cannot exploit group identity, so the gap is a property of the members, not an input artefact; the CNN adds no extra group dependence. Group-wise demeaning (1.3) stays as the fix. *Original plan:* Biomass-burning forcing subset (Zach). CESM2-LE members 1–50 use the CMIP6 biomass-burning emissions with spurious interannual variability over 1997–2014; members 51–100 use the smoothed version. `03_cesm2le_tvt_splits.py` already loads by `MEMBER_GROUPS = ['first50', 'last50']`. Recompute the ensemble-mean trend and slowdown labels separately for each 50-member group (Fig. S1c–d equivalent), report the base rate and the 2005–2015 label peak for each, and train a `last50`-only CNN ensemble (5 blocks → 3 train / 1 val / 1 test). If the forced slowdown and the label peak largely vanish with smoothed forcing, the current labels are contaminated by the forcing artifact and the `last50` results should become the headline, with the full-ensemble results as a sensitivity.

**Step 4.5 — SUPERSEDED by the relative definition.** Threshold degeneracy guard. Because the threshold is `f × ensemble-mean trend(t)`, it collapses to ~0 where the forced trend flattens (Fig. S1d). Add a floor: define the model threshold as `max(f × ens-mean trend(t), μ_m + σ_m)` where μ_m, σ_m come from the pooled member trends over 1990–2040, or simply report how many labels change when the 2005–2015 windows are treated with the pooled threshold. State the choice in Methods.

---

## Phase 5 — Conditioning and XAI analyses (comments 5, 6, 7)

**Step 5.1 — Condition on truth, not on TPs.** In the Fig. 3 notebook, produce P(slowdown | phase) using *all* labelled samples (train+test, both model and, separately, observations), and only then P(TP | phase) as a diagnostic of *model* behaviour. The physical claim in the abstract must rest on the former. Same for the SST composite: composite over all slowdown onsets, then over TP/FP/FN separately.

**Step 5.2 — Define VE.** Write the formula used (presumably η² / R² of the binary indicator on the categorical phase). Report it with block-bootstrap CIs and replace "strongly" with language proportional to ~11%.

**Step 5.3 — Occlusion DONE (`08_occlusion.py`, `src/analysis/occlusion.py`; results in the 2026-09-14 status block; figure `make_figure.py occlusion`). Second attribution method and shuffled-labels check still to do.** Add to `src/xai/`:
- a second attribution method (Integrated Gradients or SmoothGrad via `innvestigate`/TF), compare spatial correlation with LRP-z;
- an occlusion test: zero (set to climatology) the Arctic (>60°N), tropical Pacific, and North Pacific boxes already defined in `XAI_CONFIG['regions']` and record the F1 drop for each — this is the cleanest "what matters" evidence and does not depend on LRP;
- a sanity check: LRP from randomly-initialised and from label-shuffled-trained networks (Adebayo-style), to show the Arctic pattern is not architecture-driven.

**Step 5.4 — DECIDED by occlusion: tropical/central Pacific contributes ≤ 0.007 AUROC in every configuration → Key Point 3 and the CP/EP discussion go; LRP Pacific hotspots are reported as an attribution artefact, with the occlusion figure as the evidence.** Unless the occlusion test shows a measurable skill contribution from the central Pacific, remove Key Point 3 and the CP/EP discussion, or move it to a hedged sentence. Reconcile Key Points with results (Niño 3.4 VE = 1.3% cannot coexist with "highly relevant").

---

## Phase 6 — Observational application (comments 8, 9, 10)

**Step 6.1 — DONE (`09_obs_forced_removal.py`).** Observed Arctic (>65°N) JJA SST anomaly under linear / quadratic / ensmean / group_cmip6 / group_smbb × ERSST / OISST: 2016–2025 mean ≈ 0 or negative in all ten combinations (−0.08…+0.06 °C; 3–5 of 10 years positive), 2010–2015 positive in all. Method and product each move the index by ≈0.1 °C; the large raw ERSST/OISST trend difference disappears after trend-corrected removal. Decision: `group_cmip6` primary (observation-based emissions; CNN trained group-demeaned), others as range. Figure `diagnostics/obs_forced_removal.png` → SI. *Original plan:* Run `03_ersst_test.py` and `06_cnn_predict_ersst.py` with both `linear` and `ensmean`, plus a quadratic detrend option (add to `test_cnn.py`). Plot the resulting Arctic SST index for 2010–2025 under each method. If the 2016–2020 positive Arctic anomaly and the ~85% 2016 slowdown vote disappear under `ensmean`/quadratic, the current Fig. 4 result is a detrending artifact and must be presented as such (or removed).

**Step 6.2 — PIPELINE DONE, evaluation pending retrained CNNs.** OISST v2.1 downloaded (1990–2025), monthly `sst` + `ice`, block-averaged to the CESM2 grid (`01_oisst_preprocessing.py`, check `02_oisst_regrid_check.py`). `02_obs_compare_products.py`: OISST ≈0.5 °C warmer than ERSST in the central Arctic in the 1990s, converging by 2015 → ERSST Arctic trend ≈2× OISST; coverage 99 % vs 74 % of Arctic ocean cells (ERSST leaves the ice-covered interior empty). Decision: OISST primary product. Observational predictions for every tag × product × reference run in `run_postprocess.sh`. Figure `diagnostics/obs_products_compare.png` → SI. References Huang20 / Banzon16 / Reynolds07 in references.bib. *Original plan:* OISSTv2.1 (NOAA, daily, 0.25°, Sept 1981–present, satellite AVHRR + in situ, with an explicit sea-ice-concentration-based proxy SST under ice) is what the Arctic Report Card SST chapter uses. Steps: (i) add `src/data/observations/oisst/download.py` pulling monthly means from NOAA PSL (`sst.mon.mean.nc`, ~1 GB) and its ice-concentration companion if available; (ii) reuse `regrid_to_cesm2le.py` (bilinear, ocean-only) — check the land/ice mask handling because OISST has values under ice while the CESM2 grid treats those cells as ocean with SST ≈ −1.8 °C; (iii) run `03_ersst_test.py`-equivalent for OISST with both forced-removal methods; (iv) show the Arctic SST index and the ensemble vote fraction for ERSSTv5 and OISST side by side (new Fig. S). Also note ERSSTv6 exists (Zach); a one-line justification for whichever product is primary is needed in §2.2. This step is independent of everything in Phases 1–5 and can be done by a student/RA in parallel. *Code (2026-09-13):* `src/data/observations/oisst/` downloads the daily AVHRR-only files month by month, averages them to monthly `sst` and `ice` on the 0.25° grid (dailies deleted), and block-averages onto the CESM2 grid with the same `sst_obs` layout as ERSST plus `ice_obs`; `scripts/02_obs_compare_products.py` compares the two products on the CESM2 grid (Arctic JJA series, climatology and trend difference maps, coverage/ice) so the choice of primary product in §2.2 is documented rather than asserted.

**Step 6.3 — Predictions DONE for all tags × products × references (`run_postprocess.sh`); recount and text fix pending.** Recompute the fraction of CNNs predicting a slowdown for each onset year 2016–2025 and state the numbers; the current text ("fewer than 20% … 2017–2025") disagrees with Fig. 4b (2017 ≈ 0.35–0.4; 2016 ≈ 0.85 is not mentioned). Extend observed labels to onset year 2016 (2016–2025 needs September 2025, which is available).

**Step 6.4 — Replace Fig. 4a.** Remove the single-CNN panel; show the ensemble fraction with block-bootstrap uncertainty, and mark event-level hits/misses for the two independent observed events.

**Step 6.5 — Sign discrepancy caveat.** Add a paragraph making explicit that CESM2-LE's positive-IPO/ENSO association is opposite to the observed 2007–2013 case, and that this limits confidence in the observational forecast. Cite Topál & Ding 2023 / Bonan & Blanchard-Wrigglesworth 2020 here, not as an aside.

**Step 6.7 — LRP and SST composites for observations (Zach).** Run `05_cesm2le_lrp.py`-equivalent on the observational inputs (all 45 CNNs), composite relevance and SST over the years flagged by >15% of CNNs (1990, 2007, 2016–2020) and over the observed slowdown onsets. Compare with the model TP composite (Fig. 2). This is also the cleanest way to see whether the 2016–2020 votes rest on the Arctic (detrending artifact suspicion, Step 6.1) or on the Pacific. SI figure.

**Step 6.6 — Rewrite the conclusion.** Two branches depending on Phase 1:

- **Branch A (signal survives lagged/skip-first tests):** "Arctic SST preconditioning at one-year lead carries skill beyond the initial ice state; tropical Pacific modulation is weak and model-dependent." Keep a *heavily* hedged observational application with no "end of the pause" claim in the abstract.
- **Branch B (signal does not survive):** reframe the paper as a cautionary methodological result: a CNN+XAI pipeline trained on a large ensemble recovers a known statistical property (initial-condition dependence of decadal trends) and presents it as ocean preconditioning; the paper's contribution becomes the demonstration of the necessary controls (baselines, lag tests, occlusion) for XAI-based "precursor" claims in sea ice. This is publishable and consistent with the first author's XAI-trustworthiness work.

---

## Phase 8 — Reformulating the SST–ice question (added 2026-09-14)

Phase 1 answered the *predictive* question (does the SST state at onset carry information about the coming decade beyond the ice state?) with a no. Baxter et al. (2019) and the Ding et al. teleconnection literature ask a *concurrent* question — which SST pattern accompanies fast versus slow decline — and the atmospheric-bridge mechanism (summer tropical SST → Arctic circulation → same-summer melt) makes that the version where a Pacific signal should show if it exists at all. These are different scientific questions, decided on before seeing the numbers; both answers go in the paper. Order: cheap → one retrain → second retrain only if warranted.

**Step 8.1 — Residual on baselines (hours, no training; `scripts/09_residual_analysis.py`, `src/analysis/residual.py`, `src/plotting/residual.py`).** Continuous target: the 10-yr trend anomaly (`trend_anom` in the relative label file), no 1σ threshold. Stage 1: OLS trend ~ SIE anomaly at onset, fit on training members per split, test R². Stage 2: OLS of the *residual* on each index set (Arctic SST, Niño 3.4, IPO, Pacific, all indices), with the indices taken (a) at onset and (b) averaged over the 10 JJA seasons of the trend window (concurrent); test R² and ΔR² over the SIE-only model. Descriptive maps: pointwise correlation of the pooled residual with JJA SST at onset and with the decade-mean JJA SST (Baxter-style). Outputs `results/residual/residual_skill.nc`, `residual_summary.md`, `residual_corr_maps.nc`, `figures/diagnostics/residual_analysis.png`. *Read-out:* if the concurrent Pacific indices explain a non-trivial share of the residual (ΔR² ≳ 0.05) while the onset indices do not, 8.2 is worth a retrain and the paper gets a "concurrent modulation, no predictability" result; if both are ≈ 0, the paper is the clean negative and 8.2 is optional.

```
python scripts/09_residual_analysis.py            # ~10 min incl. the two SST loads
```

**Step 8.2 — Concurrent-decade CNN (one overnight retrain).** `03 --sst-window 10`: the input map for onset *t* is the mean of the demeaned JJA SST over years *t … t+9* (running mean in `load_jja_sst_demeaned(sst_window=…)`; split attr `sst_window`), with the SIE scalar (`--aux sie_anom`) so that the map can only earn credit beyond the ice state; tag `rel_concurrent`. Same 04 → 06 → 07 → 08 → 05 chain. The observational version is possible only for onsets ≤ 2016 (needs SST through 2025); `03_obs_test.py` caps `--end-year` accordingly. *Read-out:* occlusion on `rel_concurrent` — if the tropical/North-Pacific drop becomes O(0.03–0.05) while it stays ≈ 0 at lead 0, that is the paper's positive result; if the Arctic is still all there is, the negative is complete.

```
python scripts/03_cesm2le_tvt_splits.py --labels-file $LBL --demean group --end-year 2030 --aux sie_anom --sst-window 10 --tag rel_concurrent --no-fig
scripts/run_retrain.sh --smoke rel_concurrent && rm $SLOWDOWN_DATA_ROOT/results/models/rel_concurrent/cnn_jja_split0_run0.h5
nohup scripts/run_retrain.sh rel_concurrent > $SLOWDOWN_DATA_ROOT/results/logs/retrain_concurrent_nohup.out 2>&1 &
# afterwards
nohup scripts/run_postprocess.sh rel_concurrent > $SLOWDOWN_DATA_ROOT/results/logs/postprocess_concurrent_nohup.out 2>&1 &
```

**Step 8.3 — Regional target (second retrain, only if 8.1/8.2 suggest a Pacific signal).** Pan-Arctic SIE dilutes the Pacific pathway; Ding's and Baxter's signals are strongest in the Pacific sector (Chukchi / Beaufort / East Siberian, ≈ 120°E–120°W, > 65°N). Compute September sector SIE from `aice` × `tarea` on the CICE grid (`02_cesm2le_slowdowns_relative.py --sector pacific`, to be written), relative labels as before, `rel_concurrent`-style inputs, tag `rel_pacsector`. Not started.

**Step 8.1 result (2026-09-14).** SIE anomaly at onset explains R² 0.28 (0.20–0.34 across splits) of the decadal trend anomaly (r = −0.53: low ice → shallower decline, i.e. mean reversion). Residual on the indices: onset IPO/Pacific ΔR² +0.026, all indices +0.029; *concurrent* decade-mean indices explain less (Pacific +0.005, IPO −0.002, all +0.010; Arctic +0.016). Correlation maps: at onset a faint (|r| ≲ 0.1) warm-tropical-Pacific / positive-IPO pattern with slower decline (CESM2's sign, opposite to the observed 2007–2013 case); concurrently the Arctic saturates at r < −0.3 (warm decade ↔ faster decline — the response of SST to ice loss, and dominated by under-ice cells whose tiny pinned anomaly tracks concentration, which is why the area-weighted Arctic index adds only 0.016), North Pacific cooling and subpolar-Atlantic warming at |r| ≈ 0.1. Read-out: the Baxter-style concurrent hypothesis fails at the index level; 8.2 is run only as the door-closer (GPU idle), 8.3 dropped. Figure `diagnostics/residual_analysis.png` → SI.

**Step 8.5 — Where the other 72 % is: sea-ice volume (and, later, subsurface Atlantic heat).** Much of the remaining variance is decade-integrated summer circulation with no memory at onset (Ding et al. 2017/2019) — which is also why Z200/U200 at onset made the CNN *worse* in earlier tests. The reducible part is initial-condition memory the SST does not carry: sea-ice thickness/volume (Blanchard-Wrigglesworth et al. 2011; Bushuk et al. 2017; Bonan et al. 2019) and Barents/Kara–Nordic-seas ocean heat content / Atlantic inflow (Årthun et al. 2012; Onarheim et al. 2015; Yeager et al. 2015). Code: `download.py` has `HI` (`ice/proc/tseries/month_1/hi`, grid-cell mean thickness); `01_cesm2le_preprocessing.py --variable hi` downloads, combines, splits by month and writes NH volume `sivoln_{group}members_{MON}.nc` (10³ km³, Σ hi·tarea north of 50°N) next to the SIE metrics; `load_sie_monthly_files(variable='siv')`; `07_baselines.py` adds `logit_siv`, `logit_sie_siv`, `logit_sie_siv_pacific` and `09_residual_analysis.py` a "SIE + volume" stage-1 block, both automatically once the files exist. OHC needs ocean `TEMP` (heavy) — deferred to the follow-up. *Read-out:* SIE + volume R² well above 0.28 → the follow-up paper ("initial-condition sources of decadal Arctic SIE trend predictability in CESM2-LE"); for this paper one Discussion sentence naming thickness and subsurface Atlantic heat as the next test.

```
python scripts/01_cesm2le_preprocessing.py --variable hi --dry-run | head     # check URLs (hi, else try sivol)
nohup python scripts/01_cesm2le_preprocessing.py --variable hi > $SLOWDOWN_DATA_ROOT/results/logs/hi_download.out 2>&1 &
python scripts/09_residual_analysis.py --no-maps                               # picks up siv_anom
```

**Training changes for 8.2/8.3 (from the learning-curve diagnosis):** `EarlyStopping(start_from_epoch=5)` and monitor `val_auprc` (`keras.metrics.AUC(curve="PR", name="auprc")` in `compile`), patience 15. Applied in `src/cnn/train.py` for new tags only; the four existing tags are not retrained for this.

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

## Optional / future work (not needed for the revision)

- **Continuous target.** Regression head on `build_cnn()` predicting the decadal trend anomaly relative to the group mean (MSE/Huber), compared against a linear regression on SIE(t). Removes the 1σ threshold entirely and gives cleaner skill scores (r², MSE skill vs climatology and vs SIE(t) persistence). Fits the "future work" paragraph in the Conclusions.
- **last50-only CNN** (smoothed biomass-burning members) as a sensitivity on the forcing artifact, once the `aux` results are in.
- **Shorter windows** (3–7 yr) as a check on the re-emergence-timescale discussion; the label statistics are already available from the window sweep.
- **Atmospheric composites** (Z200, U200) concurrent with correct predictions (Zach's suggestion); SI only.

## Suggested execution order and rough effort

| Order | Steps | Retraining? | Effort | Status |
|---|---|---|---|---|
| 1 | 1.1 baselines on both label sets; relative labels + window sweep; Fig. S3; §2.3 and §3.1 rewritten | no | — | **done** |
| 2 | 1.2 labels (pooled σ, onsets ≤ 2030; Fig. S2); 1.3 group-wise SST demeaning (Fig. S3), `--sst-lag`, auxiliary input, `--tag`; splits `rel_base` / `rel_aux` / `rel_lag1` built | no | — | **done** (tracked changes for §2.1/§2.3 + SI renumbering pending) |
| 3 | 1.4 train `base`, `aux`, `lag1` (3 × 45 CNNs); 1.5 predict + baselines per config; 1.6 `openwater` variant | yes | ≈17 h on profx | **done** |
| 4 | 1.7 LRP on retrained models; 5.3 occlusion (second XAI method pending); 5.1 all-slowdown conditioning (pending) | LRP only | 2 days | **LRP + occlusion done** |
| 5 | 1.8 decision gate → **Branch B on prediction**; 8.1 residual baselines → 8.2 concurrent CNN (→ 8.3 regional) before the 6.6 rewrite | 8.2: one retrain | 1 day + 1 night | next |
| 6 | 6.1 detrending comparison, 6.3 recount, 6.7 obs LRP composites (on the retrained `aux` models) | LRP only | 1–2 days | |
| 7 | 3.2 block bootstrap, 5.2 VE in main text, 4.2 sensitivity sweep (logistic only) | no | 1–2 days | |
| 8 | 6.2 OISST v2.1 (parallel); optional items as time allows | partly | 2–3 days | parallel |
| 9 | Phase 7 clean-up: Discussion/Conclusions split, Key Points, SI numbering, repo configs | — | 1 week | |

Row 1 answered the original decision gate: the CNN was learning the forced epoch plus the initial ice state. Rows 2–5 are the retrain that decides between Branch A and B; nothing in Phase 6 or 7 should be written until 1.8 is resolved, except the parts that are branch-independent (6.1, 6.2, 6.7, and the clean-up items).
