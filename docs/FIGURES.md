# Figure inventory — two papers (updated 2026-09-16)

Two manuscripts share one figure pipeline:

| paper | manuscript | manifest | build | sync into |
|---|---|---|---|---|
| A — AIES Lessons Learned | `manuscript_aies/main.tex` | `configs/figures_aies.py` | `python scripts/make_figures.py` | `manuscript_aies/sync_figures.sh` → `manuscript_aies/figures/` |
| B — GRL sea-ice ledger | `manuscript/manuscript_v2.tex` | `configs/figures_grl.py` | `python scripts/make_figures.py --paper grl` | `manuscript/sync_figures.sh` → `manuscript/figures/` |

Each build writes `results/figures/<paper>/` plus `STATUS.md` (what is current, what still waits on a run).
`make_figure` entries are per-id builders (`scripts/make_figure.py <id>`); `diagnostic` entries are PNGs written by the
`09_*`/`10_*` scripts into `results/figures/diagnostics/` and copied under the manuscript's name. Never edit files in
`manuscript*/figures/` by hand — they are overwritten on every sync.

---

## Paper A — AIES Lessons Learned (three main figures)

| # | file | generator | content | status |
|---|---|---|---|---|
| 1 | `gmt_coupling_check.png` | `09_gmt_coupling_check.py` | R² of the decadal trend from the onset state, SIE and GMT, window at t / t+1 / t+2, white-noise line | done |
| 2 | `fig_2.pdf` + `fig_occlusion.pdf` + `fig_S5.pdf` (stacked in LaTeX) | `make_figure 2 --tag off1_aux`, `occlusion --occlusion-tags …`, `S5 --baselines-tag off1_pacific --tag off1_pacific` | (a) SST + LRP composite over correct slowdowns; (b) occlusion ΔAUROC by region per configuration; (c) skill vs baselines, onset-inclusive and offset labels, incl. the extra-Arctic CNN row | (a,b) done; (c) needs `off1_pacific` baselines (queue) |
| 3 | `ohc_lb22_test.png` + `fig_ledger.pdf` (stacked) | `09_ohc_lb22_test.py --cv --n-pcs 20 50 --ann`; `make_figure ledger` | (a) LB22 predictor with the GMT-state control, window at t vs t+1; (b–e) OHC added R² beyond the state and residual maps, GMT over SIE | done (five-block after `lb22_cv`) |

To do for A: a single `make_figure` id per composite (`sie_case`, `gmt_case`) so panel letters run (a)–(e) instead of
restarting in each stacked file; until then the LaTeX stack is the draft.

### Paper A supplement

| file | generator | content |
|---|---|---|
| `xai_compare_off1_aux.png` (`rel_aux` twin) | `05b_xai_compare.py --tag …` | LRP-z, α2β1, DeepTaylor, input×gradient composites + Arctic share |
| `fig_learning_curves.pdf` | `make_figure learning_curves --tag off1_aux` | loss vs epoch, 45 networks; stopping epochs |
| frozen-scalar synthetic panel | Appendix B numbers; figure to make (`09_synthetic_frozen_scalar.py`, not yet written) | oracle vs frozen vs warm-started |
| LB22 CV table | `results/ohc/lb22_test/summary.md` | five-block medians, 20/50 PCs, MLP row |
| `regression_off1_aux.png`, `regression_off1_pacific.png` | `04_cesm2le_cnn_regress.py --tag …` | CNN on the continuous target vs the state regression |
| `xai_compare_off1_pacific.png` | `05b --tag off1_pacific` | what the extra-Arctic network looks at |
| decadal-ML design table | `docs/LITERATURE_DECADAL_ML.md` | Gordon21/22, Rader23, Fernandez25, LB22: target window vs input |
| Table S1 | hand-written | architecture, training, and **all configurations** (see below) |
| `fig_S15.pdf` | `make_figure S15` | SIE vs GMT slowdown coincidence (27 %) |

Observations do not appear in paper A (removed 2026-09-16).

### Table S1 — configurations to list (paper A)

| tag | labels | input | scalar | notes |
|---|---|---|---|---|
| `rel_base` | onset-inclusive | JJA SST map, year t | — | the original result |
| `rel_aux` | onset-inclusive | map | SIE anomaly, Glorot init (frozen) | the control that could not work |
| `rel_lag1` | onset-inclusive | map, year t−1 | — | one-year lead |
| `rel_openwater` | onset-inclusive | map, ice-covered cells zeroed | — | |
| `rel_concurrent` | onset-inclusive | decade-mean map over the target window | — | diagnostic, not a forecast |
| `off1` | offset (t+1…t+10) | map, year t | — | |
| `off1_aux` | offset | map | SIE anomaly, warm-started at the logistic fit | the paper's CNN |
| `off1_pacific` | offset | map with SST poleward of 50°N zeroed | SIE anomaly, warm-started | the extra-Arctic test |
| regression twins | offset | as `off1_aux` / `off1_pacific` | OLS warm start | Huber loss, continuous target |

Each row: 9 member-block splits × 5 seeds = 45 networks; Adam 1e-4, focal loss with class weights, dropout 0.2,
early stopping (patience 10). Baselines (`07_baselines.py`) are fitted per split on the same training members.

---

## Paper B — GRL sea-ice ledger (four main figures; 12 publication units)

| # | file | generator | content | status |
|---|---|---|---|---|
| 1 | `fig_1.pdf` | `make_figure 1 --trend-offset 1` | schematic (drop the CNN box or keep small), NSIDC record with the offset-window classification, one member with slowdown and rapid-loss decades | needs `--trend-offset` regeneration; tail histogram panel (c) still to add from `09_tail_asymmetry` |
| 2 | `fig_ledger_grl.pdf` | `make_figure ledger_grl --trend-offset 1 --tag off1_aux` | (a) ice state R², then ΔR² from indices, SST PCs, CNN — onset (grey) vs decade mean (green); (b) OHC ΔR² heat-map, regions × (100/300/700 m × onset/decade mean); (c) GMT ledger | code written 2026-09-16; needs `run_ohc2.sh` outputs |
| 3 | `fig_maps_grl.pdf` | `make_figure maps_grl --trend-offset 1` | residual correlation with SST (onset, decade mean) and OHC100 (onset, decade mean) | code written; needs `--concurrent` ledger |
| 4 | `fig_obs.png` | `10_obs_baseline_predict.py --labels-file $LBL1 --forced linear --tag off1_aux` | calibrated ice/IPO probabilities on the record, CNN votes, observed slowdowns by observed and model σ | done except CNN votes (queue) |

### Paper B supplement

| file | generator | content |
|---|---|---|
| `fig_S1`, `fig_S2`, `fig_S4` (offset) | `make_figure S1/S2/S4 --trend-offset 1` | definition, pooled σ and onset cap, label distributions |
| `fig_S3` | `S3` | forced response SMBB − CMIP6 |
| `fig_S5.pdf` | `S5 --tag off1_aux --trend-offset 1` | classification view: baselines vs CNN AUROC/F1/AUPRC |
| `tail_asymmetry.png` | `09_tail_asymmetry.py` | slowdowns vs rapid-loss decades as two tails of one regression |
| `interannual_check.png` | `09_interannual_check.py` | Pacific–Arctic bridge: sign, stability, 2 % |
| `residual_analysis.png` | `09_residual_analysis.py --sst-pcs 10 20` | onset vs concurrent index ΔR², residual maps (SST) |
| `regression_off1_aux.png` | `04 regress` | CNN vs OLS on the continuous target |
| `ohc_sie_ledger_{100,300,700}.png`, `ohc_sie_ledger_100_conc.png` | `09_ohc_ledger.py` | per-depth ledgers, onset and concurrent |
| `ohc_gmt_ledger_100.png` | `09_ohc_ledger.py --target gmt` | the GMT contrast in full |
| `fig_S17.pdf` | `09_sensitivity_sweep.py` (rerun on offset labels) | window 8–15 yr × threshold sweep |
| `fig_S15.pdf` | `S15` | SIE vs GMT slowdown coincidence |
| `fig_S18`, `fig_S19`, `obs_oisst.png`, `obs_sigmodel.png` | products / forced reference / OISST twin / model-σ classification | observational robustness |
| `thickness_SEP.png` | `09_thickness_maps.py` | thickness pattern at onset vs the trend (ice-state robustness) |

---

## v1 GRL figures no longer used anywhere

`fig_3.pdf` (P(TP | phase)), `fig_4.pdf` (v1 CNN votes), `fig_S6–S14`, `fig_S16` (static). Kept in `manuscript/figures_old/`.
