# Figure inventory — v1/v2 GRL manuscript → AIES manuscript

File names are what `make_figure.py <id>` writes to `results/figures/paper/` (synced to `manuscript/figures/`), or the
`results/figures/diagnostics/*.png` a script writes directly. "Reuse" is the proposal for the AIES paper; nothing is
deleted from the GRL manuscript, which stays as the backup.

## Main text (current GRL v2)

| file | id / generator | content | AIES proposal |
|---|---|---|---|
| `fig_1.pdf` | `make_figure.py 1` | schematic (CNN), NSIDC record, one member's slowdowns | **reuse** as Fig. 1 — redraw the definition panel with the window starting at t+1 (offset labels) |
| `fig_2.pdf` | `make_figure.py 2 --tag <cnn>` | TP composite: SST + LRP (signed) | **reuse** as half of the "relevance vs skill" figure, paired with the occlusion bars; regenerate on the final CNN |
| `fig_3.pdf` | `make_figure.py 3` | P(TP \| phase), test members | drop from main text → SI (superseded by ΔR² figure); or delete |
| `fig_4.pdf` | `make_figure.py 4` | observations: CNN votes + indices | SI; becomes "obs predictions, both products, all forced references" once the final CNN exists (6.3/6.4) |

## Supplementary (current GRL v2)

| file | id / generator | content | AIES proposal |
|---|---|---|---|
| `fig_S1.pdf` | `S1` | slowdown definition, 6 panels | **reuse**, add offset window |
| `fig_S2.pdf` | `S2` | pooled σ and onset cap | SI |
| `fig_S3.pdf` | `S3` | forced response SMBB − CMIP6 | SI |
| `fig_S4.pdf` | `S4` | label distributions, 8 panels | SI |
| `fig_S5.pdf` | `S5 --tag <cnn>` | baselines vs CNN skill | **main text** (offset labels, `--tag off1`) |
| `fig_S6.pdf` | `S6` | PR curve, one CNN | SI (final CNN) |
| `fig_S7.pdf` | `S7` | confusion matrices | SI (final CNN) |
| `fig_S8.pdf` | `S8` | metric strip, all CNNs | SI (final CNN) |
| `fig_S9.pdf` | `S9` | test-member slowdown timeline | SI |
| `fig_S10.pdf` | `S10` | SST composites all vs CNN-filtered | SI or drop |
| `fig_S11–S13.pdf` | `S11 S12 S13` | FP / TN / FN composites | drop, or condense to one SI figure |
| `fig_S14.pdf` | `S14` | P(event \| phase), train, all vs TP | drop (ΔR² figure carries this) |
| `fig_S15.pdf` | `S15` | SIE vs GMT slowdown counts | **keep**, SI or main (27 % coincidence; decoupling from LB22's target) |
| `fig_S16.pdf` | static | obs phase composites | drop |
| `fig_S17.pdf` | `S17` | label sensitivity heat-maps (`09_sensitivity_sweep.py`) | SI — rerun on offset labels |
| `fig_S18.pdf` | `S18` | ERSST vs OISST on the CESM2 grid | SI |
| `fig_S19.pdf` | `S19` | observed Arctic index vs forced reference | SI (or main, observational anchor) |

## New since v2 (diagnostics/*.png; to be given `make_figure.py` ids before the AIES draft)

| file | generator | content | AIES proposal |
|---|---|---|---|
| `fig_occlusion.pdf` | `make_figure.py occlusion` | ΔAUROC per region, all configurations | **main** — the "what the CNN uses" half of the relevance-vs-skill figure |
| `fig_learning_curves.pdf` | `make_figure.py learning_curves --tag` | loss vs epoch, all 45 CNNs; stopping epoch | **main or SI** — training diagnostic (flat validation curve, epoch-1 models) |
| `interannual_check.png` | `09_interannual_check.py` | (a–b) Pacific–SIE interannual link & nonstationarity, (c) year-ahead R², (d) **onset-year-in-window R² 0.28→0.075→0.04**, (e) SIE–SST correlation map (model's Ding/Baxter pattern) | **main** — panel (d) is the paper's central finding; (a–c, e) SI or a second main figure |
| `residual_analysis.png` | `09_residual_analysis.py --labels-file <off1>` | trend vs ice state; ΔR² of index sets (onset vs concurrent); residual–SST maps | **main** — the 7 / 3 / 0 ledger |
| `regression_<tag>.png` | `04_cesm2le_cnn_regress.py` | CNN R² vs OLS references per split; predicted vs true | main or SI (pending run) |
| `event_stats[_<tag>].png` | `09_event_stats.py` | event durations, onsets, event vs sample skill | SI |
| `sensitivity_sweep.png` | `09_sensitivity_sweep.py` | = S17 | SI |
| `obs_products_compare.png`, `obs_forced_removal.png` | `02_obs_compare_products.py`, `09_obs_forced_removal.py` | = S18, S19 | SI |
| `sigma_modes.png`, `forced_group_difference.png`, `forced_demeaned_arctic.png` | `02_*` | sources of S2, S3 | SI / not shown |
| `openwater_check_<tag>.png`, `icemask_summary.png`, `oisst_regrid_check.png` | `03 --openwater`, `02_cesm2le_icemask.py`, `02_oisst_regrid_check.py` | pipeline checks | not shown (repo only) |
| *(to make)* `aux_frozen_synthetic.png` | small script | synthetic test: scalar-only label, aux-CNN 0.62 vs oracle 0.84; warm start 0.84 | **main or SI** — the design lesson (8.8) |
| *(to make)* `arithmetic_synthetic.png` | small script | white-noise slope-vs-first-value R² 0.25 at offset 0, 0 at offset 1 | **main** — pairs with interannual (d) |

## Proposed AIES main-text figures (≈ 8)

1. Definition and data: schematic + slowdown definition with the offset window (fig_1 + S1 material).
2. The construction artefact: white-noise synthetic + CESM2 offset test (interannual (d) + `arithmetic_synthetic`).
3. Skill against baselines on the honest labels (S5, `--tag off1`).
4. Relevance vs skill: LRP TP composite (fig_2) beside occlusion bars (`fig_occlusion`).
5. What is foreseeable: ΔR² ledger + residual–SST maps (`residual_analysis`, offset labels, incl. volume).
6. The bridge that exists: Pacific–SIE interannual link, model sign, nonstationarity (interannual (a, b, e)).
7. Training and design diagnostics: learning curves + frozen-scalar synthetic (`learning_curves`, `aux_frozen_synthetic`).
8. Observations: Arctic index under all forced references × products (S19), and what the retrained CNN says (fig_4-style).
