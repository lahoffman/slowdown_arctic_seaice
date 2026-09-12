# Revision checklist

One line per step; status in the first column. Reasoning, results and history live in
`REVISION_PLAN.md`; this file only says what to do next and what is done. Keep the two in
sync: when a step changes there, change the line here.

Status: `[x]` done · `[~]` in progress / partly done · `[ ]` to do · `[-]` dropped or superseded

## Phase 1 — Does the SST pattern carry information beyond the ice state?

- [x] 1.1 Scalar baselines (`07_baselines.py`) on original and relative labels; Fig. S5; §3.1 reframed. *Result: CNN ≈ onset-year climatology on old labels; SIE anomaly alone beats it on both.*
- [x] 1.2 Labels for the retrain: pooled σ (0.130), onsets 1990–2030 (`03 --end-year 2030`); Fig. S2. *Result: trend spread collapses with the ice after ≈2028 for every window; yearly σ would manufacture events near the ice floor.* §2.3 sentence pending (7.7).
- [x] 1.3 Preprocessing: SST demeaned per forcing group (`forced.py::forced_response`, Fig. S3); `--sst-lag`; SIE-anomaly auxiliary input; `--tag` everywhere. Splits `rel_base` / `rel_aux` / `rel_lag1` built (4100 samples each). *Result: SMBB 0.1–0.3 °C cooler than CMIP6-BB in 2000–2020; Arctic forced difference −0.14 °C peak.*
- [~] 1.4 Train `base`, `aux` (headline), `lag1` — 9 splits × 5 seeds each. Driver: `scripts/run_retrain.sh [--smoke] [tags]` (04 → 06 → 07 per tag, resumable with `--skip-existing`, logs in `results/logs/retrain_<tag>_<stamp>.log`). *Smoke test passes on synthetic data; full run queued on profx.*
- [~] 1.5 Predict + `07_baselines.py --cnn-tag <config> --demean group` for each (done by `run_retrain.sh`); then `make_figure.py S5 --baselines-tag <tag>` and compare on the Fig. S5 axes. Bar to clear: `logit_sie_pacific` (AUROC ≈ 0.79).
- [~] 1.6 Under-ice SST (Zach): `openwater` variant of `aux` — code done (`02_cesm2le_icemask.py` → `03 --openwater --tag rel_openwater` → `run_retrain.sh rel_openwater`); Arctic occlusion test (5.3) and the ERSST/OISST under-ice note still to do.
- [ ] 1.7 LRP on retrained models; TP composites (Fig. 2) and occlusion on the same models.
- [ ] 1.8 Decision gate → Branch A (pattern skill beyond ice state) or Branch B (methodological paper). Phase 6/7 rewrite waits for this.

## Phase 2 — Skill reporting

- [x] 2.1 Baselines figure in the SI (Fig. S5) and §3.1 framed against baselines.
- [ ] 2.2 Report precision and recall separately; state the threshold rule (chosen on training data) in Methods.
- [ ] 2.3 Permutation significance test (member × year-block shuffles) → p-value.
- [~] 2.4 Units/language: F1 as a fraction (done in §3.1 rewrite); "predict" → "classify" unless `lag1` shows skill.

## Phase 3 — Autocorrelation and effective sample size

- [ ] 3.1 Count independent events (merge consecutive positive years per member); report in Methods / Fig. S2.
- [ ] 3.2 Member-block bootstrap for every CI (Fig. 3 / S14 VE brackets, metric spread); baselines already do this.
- [ ] 3.3 Event-level hit rate alongside sample-level F1; use for the observational evaluation.

## Phase 4 — Slowdown definition

- [-] 4.1 Ice-free floor screening — superseded by the relative definition (1.2).
- [ ] 4.2 Sensitivity sweep (σ threshold, window 8–15 yr, SST season) with the logistic baselines only; SI heat-map.
- [x] 4.3 Explain in §2.3 why the LB22 scaling is not used (tracked change in manuscript).
- [x] 4.4 Biomass-burning forcing: labels and SST demeaned per forcing group (1.2/1.3, Fig. S3); last50-only CNN → optional.
- [-] 4.5 Threshold degeneracy guard — superseded by the relative definition.

## Phase 5 — Conditioning and XAI

- [ ] 5.1 Report P(slowdown | phase) for *all* slowdowns in the main text (abstract claims must use these numbers, not TP-only).
- [ ] 5.2 Define VE in the main text; fix `(ref)` in Text S5; tone down "strongly".
- [ ] 5.3 XAI robustness: second attribution method, region occlusion test, randomised-weights / shuffled-labels sanity check.
- [ ] 5.4 Reconcile Key Points 2 and 3; drop or hedge the CP-El Niño narrative unless occlusion supports it.

## Phase 6 — Observations

- [ ] 6.1 Forced-signal removal comparison (linear vs ensmean vs quadratic) and choice of forced reference for ERSST now that the CNN is trained on group-demeaned SST (either group or 100-member mean; ≤0.1 °C in the Arctic); Arctic SST index 2010–2025 under each. *Branch-independent.*
- [ ] 6.2 OISST v2.1 as second SST product (download, regrid, rerun obs pipeline). *Branch-independent; can run in parallel.*
- [ ] 6.3 Recount vote fractions 2016–2025 and fix the text ("fewer than 20% … 2017–2025" ≠ Fig. 4b); extend observed labels to onset 2016.
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
- [ ] 7.7 Tracked changes for 1.2/1.3: §2.1 (group demeaning), §2.3 (pooled σ, onsets ≤ 2030, all windows roll off), SI captions for Figs S2/S3, renumber S4–S16 in both `.tex` files and `sync_figures.sh`.
- [ ] 7.6 Add Zach's affiliation/ORCID; Acknowledgments placeholder for Climate Central; title no longer echoing LB22's key point.

SI numbering (v2): S1 definition · S2 pooled σ/cap · S3 forcing groups · S4 label stats · S5 baselines · S6 PR · S7 confusion · S8 metrics · S9 timeline · S10 composites · S11–S13 FP/TN/FN · S14 P(event|phase) train · S15 SIE vs GMT · S16 obs phase composites.

## Optional / future work

- [ ] Continuous (regression) target instead of the 1σ threshold.
- [ ] last50-only CNN as a forcing-artifact sensitivity.
- [ ] Shorter trend windows (3–7 yr) vs re-emergence timescales.
- [ ] Concurrent atmospheric composites (Z200, U200) for correct predictions.
