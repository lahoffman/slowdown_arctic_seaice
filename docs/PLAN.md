# Revision checklist

One line per step; status in the first column. Reasoning, results and history live in
`REVISION_PLAN.md`; this file only says what to do next and what is done. Keep the two in
sync: when a step changes there, change the line here.

Status: `[x]` done · `[~]` in progress / partly done · `[ ]` to do · `[-]` dropped or superseded

## Phase 1 — Does the SST pattern carry information beyond the ice state?

- [x] 1.1 Scalar baselines (`07_baselines.py`) on original and relative labels; Fig. S5; §3.1 reframed. *Result: CNN ≈ onset-year climatology on old labels; SIE anomaly alone beats it on both.*
- [x] 1.2 Labels for the retrain: pooled σ (0.130), onsets 1990–2030 (`03 --end-year 2030`); Fig. S2. *Result: trend spread collapses with the ice after ≈2028 for every window; yearly σ would manufacture events near the ice floor.* §2.3 sentence pending (7.7).
- [x] 1.3 Preprocessing: SST demeaned per forcing group (`forced.py::forced_response`, Fig. S3); `--sst-lag`; SIE-anomaly auxiliary input; `--tag` everywhere. Splits `rel_base` / `rel_aux` / `rel_lag1` built (4100 samples each). *Result: SMBB 0.1–0.3 °C cooler than CMIP6-BB in 2000–2020; Arctic forced difference −0.14 °C peak.*
- [x] 1.4 Train `rel_base`, `rel_aux`, `rel_lag1`, `rel_openwater` — 9 splits × 5 seeds each (`scripts/run_retrain.sh`; ≈17 h on profx). *Result (test AUROC, median): base 0.720, aux 0.723, openwater 0.692, lag1 0.624 — vs 0.765 for the SIE-anomaly regression and 0.80 for SIE + IPO. No CNN reaches the one-variable linear model; `rel_aux` does not use the scalar it is given.*
- [x] 1.5 Predict + baselines per configuration (`results/baselines/<tag>/baselines_summary.md`). Bar (`logit_sie_pacific`, 0.80) not approached by any configuration. Fig. S5 per tag: `make_figure.py S5 --tag <tag>` (not yet drawn).
- [x] 1.6 Under-ice SST (Zach): `rel_openwater` AUROC 0.69, Arctic occlusion drop −0.049 (vs −0.096 with ice-covered cells present). *Caveat for the text: the zero pattern still encodes the ice edge — a lower bound on the ice-cover share of the Arctic signal.* ERSST/OISST under-ice note → §2.2 (7.7).
- [~] 1.7 LRP done for all four tags (`results/attributions/<tag>/`, 45 each); occlusion done (5.3). *To do:* Fig. 2 TP composites with `make_figure.py 2 --tag rel_aux`.
- [x] 1.8 Decision gate → **Branch B on the predictive question** (no configuration beats the ice-state regression; occlusion: skill is the Arctic, extra-Arctic SST contributes ≤ 0.01 and removing it helps). Phase 8 asks the concurrent question before the 6.6 rewrite.

## Phase 2 — Skill reporting

- [x] 2.1 Baselines figure in the SI (Fig. S5) and §3.1 framed against baselines.
- [ ] 2.2 Report precision and recall separately; state the threshold rule (chosen on training data) in Methods.
- [ ] 2.3 Permutation significance test (member × year-block shuffles) → p-value.
- [~] 2.4 Units/language: F1 as a fraction (done in §3.1 rewrite); "predict" → "classify" everywhere — `rel_lag1` (one-year lead) AUROC 0.62, Arctic occlusion −0.03: no lead-time skill. *Text change pending (7.7).*

## Phase 3 — Autocorrelation and effective sample size

- [x] 3.1 Independent events (`09_event_stats.py`): 702 positive member-years = 239 events, mean 2.9 yr, onsets flat 1990–2030 (v1 labels: 1417 → 364, 3.9 yr). *To write into Methods / Fig. S4 caption (7.7).*
- [ ] 3.2 Member-block bootstrap for every CI (Fig. 3 / S14 VE brackets, metric spread); baselines already do this.
- [~] 3.3 Event-level skill (`09_event_stats.py --tag`): `rel_base` hit rate 0.62 (0.33–0.76), false-alarm ratio 0.61, vs sample F1 0.38. Rerun for the other tags; use for the observational evaluation.

## Phase 4 — Slowdown definition

- [-] 4.1 Ice-free floor screening — superseded by the relative definition (1.2).
- [x] 4.2 Sensitivity sweep (windows 8/10/12/15 yr × 0.5/1/1.5σ, baselines only; `09_sensitivity_sweep.py`). *Result: ice state > indices by 0.05–0.09 AUROC and Pacific adds +0.01–0.03 in every cell; skill rises with threshold, not window; our 10 yr/1σ cell has the largest Pacific increment (+0.033 vs grid median +0.014) — say so.* SI heat-map = `diagnostics/sensitivity_sweep.png`. SST-season variant → optional.
- [x] 4.3 Explain in §2.3 why the LB22 scaling is not used (tracked change in manuscript).
- [x] 4.4 Biomass-burning forcing: labels and SST demeaned per forcing group (1.2/1.3, Fig. S3). Per-group check (`09_baselines_by_group.py`): SMBB members more predictable than CMIP6-BB by ≈0.05 AUROC for the scalar regression *and* the CNN alike — a physical difference, not a group shortcut. last50-only CNN → optional.
- [-] 4.5 Threshold degeneracy guard — superseded by the relative definition.

## Phase 5 — Conditioning and XAI

- [ ] 5.1 Report P(slowdown | phase) for *all* slowdowns in the main text (abstract claims must use these numbers, not TP-only).
- [ ] 5.2 Define VE in the main text; fix `(ref)` in Text S5; tone down "strongly".
- [~] 5.3 XAI robustness: occlusion done for all four tags (`results/occlusion/<tag>/`). *Result: −Arctic costs 0.10 / 0.08 / 0.05 / 0.03 AUROC (base / aux / openwater / lag1); −everything-else *gains* +0.01 in all four; Pacific boxes ≤ 0.007 either sign.* Figure `make_figure.py occlusion` (candidate main-text figure). Second attribution method and shuffled-labels check still to do.
- [~] 5.4 Decided by occlusion: Key Point 3 and the CP/EP-El Niño narrative go; LRP Pacific hotspots (ours and LB22's) reported as relevance-follows-variance, not skill. *Text change pending.*

## Phase 6 — Observations

- [x] 6.1 Forced-signal removal on the observed Arctic index (`09_obs_forced_removal.py`, 5 references × 2 products). *Result: 2016–2025 Arctic anomaly ≈ 0 or negative in every combination (ERSST −0.04…+0.06 °C, OISST −0.08…+0.02), 2010–2015 positive in all; product and method spreads both ≈0.1 °C.* Decision: `group_cmip6` as primary reference for observations (CMIP6 BB emissions are the observation-based ones; CNN trained on group-demeaned SST), ensmean / group_smbb reported as range. Write into §2.2 and the Fig. 4 caption (7.7).
- [x] 6.2 OISST v2.1 as second SST product: downloaded, regridded, compared (Figs S18/S19); obs pipeline generalised (`03_obs_test.py --product --forced-method --tag`, `06_cnn_predict_obs.py`).
- [~] 6.3 Observational predictions done for all tags × products × references (`results/predictions/{ersst,oisst}/forced_<method>/<tag>/`). *To do:* recount vote fractions 2016–2025, Fig. 4, fix the text; extend observed labels to onset 2016.
- [~] 6.4 Fig. 4: single-CNN panel removed by default (`--single-model` to add back); add bootstrap uncertainty and event-level hits.
- [ ] 6.5 Model–observation sign discrepancy (IPO/ENSO phase) as an explicit caveat paragraph.
- [ ] 6.6 Rewrite Conclusions per Branch A or B; split into Discussion + short Conclusions.
- [ ] 6.7 LRP and SST composites on the observational inputs (Zach). *Branch-independent.*

## Phase 7 — Clean-up

- [~] 7.1 Figures: colour-blind palette (done), no grid lines (done), font size via `paper_rc` (done), baselines figure added (done); Figs S2/S3 added and `make_figure.py` renumbered S4–S15 (done); Fig. 1 caption fixed (done). Remaining: `.tex` renumbering to v2 ids (7.7); Fig. S16 (obs phase composites) generator; verify cartopy maps on the server.
- [ ] 7.2 Text fixes from Zach: "cliamte", Niño spelling, "in CESM2-LE" (l. 234), "again" (l. 371), rapid-ice-loss wording, NSIDC index version, 5-month running mean note, define Arctic SST index and LRP-z in main text.
- [ ] 7.3 Intro: 1–2 sentences on proposed drivers of the observed slowdown (Zach's three DOIs).
- [ ] 7.4 SI: Text S-numbering and cross-references; Fig. S16 caption/panels (e,f) and Niño 3.4 phase composites; "CESMS2-LE".
- [ ] 7.5 Repo: delete or sync stale `configs/model.py` / `configs/training.py` (and fix `configs/__init__.py`); remove `figures/legacy/`, `_to_delete/`, `PROJECT_STRUCTURE.md` when ready.
- [~] 7.7 Tracked changes: done — §2.1 (group demeaning), §2.3 (pooled σ, onsets ≤ 2030, 239 events), §2.2 (OISST primary, group_cmip6 reference, ERSST check; Huang20/Banzon16/Reynolds07), §3.1 sensitivity sentence, SI captions S2/S3/S4 and new S17–S19 (provisional numbers via `\figsweep` `\figproducts` `\figforced` macros), S4–S16 renumbered. *Remaining:* §3.1 `0.XX` from `rel_aux`; final SI renumbering pass at the end of Phase 1.
- [ ] 7.6 Add Zach's affiliation/ORCID; Acknowledgments placeholder for Climate Central; title no longer echoing LB22's key point.

SI numbering (v2): S1 definition · S2 pooled σ/cap · S3 forcing groups · S4 label stats · S5 baselines · S6 PR · S7 confusion · S8 metrics · S9 timeline · S10 composites · S11–S13 FP/TN/FN · S14 P(event|phase) train · S15 SIE vs GMT · S16 obs phase composites · S17 label sensitivity · S18 ERSST vs OISST · S19 obs Arctic index vs forced reference (S17–S19 provisional, set by macros in the .tex; final order at end of Phase 1).

## Phase 8 — Reformulating the SST–ice question (concurrent, not predictive)

- [~] 8.1 Residual on baselines (`09_residual_analysis.py`, no training): trend anomaly ~ SIE(t), then residual ~ indices at onset vs averaged over the trend decade; residual–SST correlation maps. *Code done; run on profx.* Read-out: concurrent ΔR² ≳ 0.05 → 8.2 is worth it.
- [~] 8.2 Concurrent-decade CNN: `03 --sst-window 10 --aux sie_anom --tag rel_concurrent` → `run_retrain.sh rel_concurrent` → `run_postprocess.sh rel_concurrent`. *Code done (`sst_window` in `load_jja_sst_demeaned`, `03`, obs input); one overnight retrain.* Read-out: Pacific occlusion drop O(0.03–0.05) → "concurrent modulation, no predictability"; ≈ 0 → clean negative.
- [ ] 8.3 Pacific-sector September SIE as target (Chukchi/Beaufort/E. Siberian) — only if 8.1/8.2 show a Pacific signal. Needs sector SIE from `aice` × `tarea`.
- [ ] 8.4 Training tweak for new tags: `start_from_epoch=5`, monitor `val_auprc`, patience 15 (`src/cnn/train.py`). Existing tags not retrained.

## Optional / future work

- [~] Continuous (regression) target instead of the 1σ threshold — linear version is 8.1; CNN regression head still optional.
- [ ] last50-only CNN as a forcing-artifact sensitivity.
- [ ] Shorter trend windows (3–7 yr) vs re-emergence timescales.
- [ ] Concurrent atmospheric composites (Z200, U200) for correct predictions.
