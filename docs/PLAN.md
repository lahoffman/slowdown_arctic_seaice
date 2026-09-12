# Revision checklist

One line per step; status in the first column. Reasoning, results and history live in
`REVISION_PLAN.md`; this file only says what to do next and what is done. Keep the two in
sync: when a step changes there, change the line here.

Status: `[x]` done · `[~]` in progress / partly done · `[ ]` to do · `[-]` dropped or superseded

## Phase 1 — Does the SST pattern carry information beyond the ice state?

- [x] 1.1 Scalar baselines (`07_baselines.py`) on original and relative labels; Fig. S3; §3.1 reframed. *Result: CNN ≈ onset-year climatology on old labels; SIE anomaly alone beats it on both.*
- [~] 1.2 Labels for the retrain: code done (`02_cesm2le_slowdowns_relative.py --sigma-mode both` → `…_sigma_modes.png`). *To do on profx:* run it, pick pooled vs yearly σ (or `03 --end-year 2030`), document in §2.3.
- [~] 1.3 Preprocessing: code done — `forced.py::forced_response` + group-mean file and diagnostics (`02_cesm2le_forced.py`); `03 --labels-file/--demean group/--sst-lag/--aux sie_anom/--tag`; `build_cnn(n_aux)`; `--tag` in `04`/`05`/`06`, `07 --cnn-tag`, `make_figure.py --tag`. *To do on profx:* `02_cesm2le_forced.py`, then `03` for `rel_base`, `rel_aux`, `rel_lag1`.
- [ ] 1.4 Train `base`, `aux` (headline), `lag1` — 9 splits × 5 seeds each (`04 --tag rel_base` etc.).
- [ ] 1.5 Predict + `07_baselines.py --tag <config>` for each; all on the Fig. S3 axes. Bar to clear: `logit_sie_pacific` (AUROC ≈ 0.79).
- [ ] 1.6 Under-ice SST (Zach): `openwater` variant of `aux` (aice > 15% → zero anomaly) + Arctic occlusion test; document under-ice SST in ERSST/OISST.
- [ ] 1.7 LRP on retrained models; TP composites (Fig. 2) and occlusion on the same models.
- [ ] 1.8 Decision gate → Branch A (pattern skill beyond ice state) or Branch B (methodological paper). Phase 6/7 rewrite waits for this.

## Phase 2 — Skill reporting

- [x] 2.1 Baselines figure in the SI (Fig. S3) and §3.1 framed against baselines.
- [ ] 2.2 Report precision and recall separately; state the threshold rule (chosen on training data) in Methods.
- [ ] 2.3 Permutation significance test (member × year-block shuffles) → p-value.
- [~] 2.4 Units/language: F1 as a fraction (done in §3.1 rewrite); "predict" → "classify" unless `lag1` shows skill.

## Phase 3 — Autocorrelation and effective sample size

- [ ] 3.1 Count independent events (merge consecutive positive years per member); report in Methods / Fig. S2.
- [ ] 3.2 Member-block bootstrap for every CI (Fig. 3 / S12 VE brackets, metric spread); baselines already do this.
- [ ] 3.3 Event-level hit rate alongside sample-level F1; use for the observational evaluation.

## Phase 4 — Slowdown definition

- [-] 4.1 Ice-free floor screening — superseded by the relative definition (1.2).
- [ ] 4.2 Sensitivity sweep (σ threshold, window 8–15 yr, SST season) with the logistic baselines only; SI heat-map.
- [x] 4.3 Explain in §2.3 why the LB22 scaling is not used (tracked change in manuscript).
- [~] 4.4 Biomass-burning forcing: labels demeaned per group (done); last50-only CNN → optional.
- [-] 4.5 Threshold degeneracy guard — superseded by the relative definition.

## Phase 5 — Conditioning and XAI

- [ ] 5.1 Report P(slowdown | phase) for *all* slowdowns in the main text (abstract claims must use these numbers, not TP-only).
- [ ] 5.2 Define VE in the main text; fix `(ref)` in Text S5; tone down "strongly".
- [ ] 5.3 XAI robustness: second attribution method, region occlusion test, randomised-weights / shuffled-labels sanity check.
- [ ] 5.4 Reconcile Key Points 2 and 3; drop or hedge the CP-El Niño narrative unless occlusion supports it.

## Phase 6 — Observations

- [ ] 6.1 Forced-signal removal comparison (linear vs ensmean vs quadratic); Arctic SST index 2010–2025 under each. *Branch-independent.*
- [ ] 6.2 OISST v2.1 as second SST product (download, regrid, rerun obs pipeline). *Branch-independent; can run in parallel.*
- [ ] 6.3 Recount vote fractions 2016–2025 and fix the text ("fewer than 20% … 2017–2025" ≠ Fig. 4b); extend observed labels to onset 2016.
- [~] 6.4 Fig. 4: single-CNN panel removed by default (`--single-model` to add back); add bootstrap uncertainty and event-level hits.
- [ ] 6.5 Model–observation sign discrepancy (IPO/ENSO phase) as an explicit caveat paragraph.
- [ ] 6.6 Rewrite Conclusions per Branch A or B; split into Discussion + short Conclusions.
- [ ] 6.7 LRP and SST composites on the observational inputs (Zach). *Branch-independent.*

## Phase 7 — Clean-up

- [~] 7.1 Figures: colour-blind palette (done), no grid lines (done), font size via `paper_rc` (done), Fig. S3 added and SI renumbered (done); Fig. 1 caption fixed (done). Remaining: Fig. S14 (obs phase composites) generator; verify cartopy maps on the server.
- [ ] 7.2 Text fixes from Zach: "cliamte", Niño spelling, "in CESM2-LE" (l. 234), "again" (l. 371), rapid-ice-loss wording, NSIDC index version, 5-month running mean note, define Arctic SST index and LRP-z in main text.
- [ ] 7.3 Intro: 1–2 sentences on proposed drivers of the observed slowdown (Zach's three DOIs).
- [ ] 7.4 SI: Text S-numbering and cross-references; Fig. S14 caption/panels (e,f) and Niño 3.4 phase composites; "CESMS2-LE".
- [ ] 7.5 Repo: delete or sync stale `configs/model.py` / `configs/training.py` (and fix `configs/__init__.py`); remove `figures/legacy/`, `_to_delete/`, `PROJECT_STRUCTURE.md` when ready.
- [ ] 7.6 Add Zach's affiliation/ORCID; Acknowledgments placeholder for Climate Central; title no longer echoing LB22's key point.

## Optional / future work

- [ ] Continuous (regression) target instead of the 1σ threshold.
- [ ] last50-only CNN as a forcing-artifact sensitivity.
- [ ] Shorter trend windows (3–7 yr) vs re-emergence timescales.
- [ ] Concurrent atmospheric composites (Z200, U200) for correct predictions.
