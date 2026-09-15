# Revision checklist

One line per step; status in the first column. Reasoning, results and history live in
`REVISION_PLAN.md`; this file only says what to do next and what is done. Keep the two in
sync: when a step changes there, change the line here.

Status: `[x]` done · `[~]` in progress / partly done · `[ ]` to do · `[-]` dropped or superseded

## Phase 1 — Does the SST pattern carry information beyond the ice state?

- [x] 1.1 Scalar baselines (`07_baselines.py`) on original and relative labels; Fig. S5; §3.1 reframed. *Result: CNN ≈ onset-year climatology on old labels; SIE anomaly alone beats it on both.*
- [x] 1.2 Labels for the retrain: pooled σ (0.130), onsets 1990–2030 (`03 --end-year 2030`); Fig. S2. *Result: trend spread collapses with the ice after ≈2028 for every window; yearly σ would manufacture events near the ice floor.* §2.3 sentence pending (7.7).
- [x] 1.3 Preprocessing: SST demeaned per forcing group (`forced.py::forced_response`, Fig. S3); `--sst-lag`; SIE-anomaly auxiliary input; `--tag` everywhere. Splits `rel_base` / `rel_aux` / `rel_lag1` built (4100 samples each). *Result: SMBB 0.1–0.3 °C cooler than CMIP6-BB in 2000–2020; Arctic forced difference −0.14 °C peak.*
- [~] 1.4 Train `rel_base`, `rel_aux`, `rel_lag1`, `rel_openwater` (aux configurations to be redone with the 8.8 warm start) — 9 splits × 5 seeds each (`scripts/run_retrain.sh`; ≈17 h on profx). *Result (test AUROC, median): base 0.720, aux 0.723, openwater 0.692, lag1 0.624 — vs 0.765 for the SIE-anomaly regression and 0.80 for SIE + IPO. No CNN reaches the one-variable linear model; `rel_aux` does not use the scalar it is given.*
- [x] 1.5 Predict + baselines per configuration (`results/baselines/<tag>/baselines_summary.md`). Bar (`logit_sie_pacific`, 0.80) not approached by any configuration. Fig. S5 per tag: `make_figure.py S5 --tag <tag>` (not yet drawn).
- [x] 1.6 Under-ice SST (Zach): `rel_openwater` AUROC 0.69, Arctic occlusion drop −0.049 (vs −0.096 with ice-covered cells present). *Caveat for the text: the zero pattern still encodes the ice edge — a lower bound on the ice-cover share of the Arctic signal.* ERSST/OISST under-ice note → §2.2 (7.7).
- [~] 1.7 LRP done for all four tags (`results/attributions/<tag>/`, 45 each); occlusion done (5.3). *To do:* Fig. 2 TP composites with `make_figure.py 2 --tag rel_aux`.
- [~] 1.8 Decision gate → Branch B on the predictive question, **re-opened by 8.8** (aux input was frozen; rerun as `rel_aux_ws`) (no configuration beats the ice-state regression; occlusion: skill is the Arctic, extra-Arctic SST contributes ≤ 0.01 and removing it helps). Phase 8 asks the concurrent question before the 6.6 rewrite.

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
- [~] 5.3 XAI robustness: occlusion done for all four tags; **multi-method attribution** coded (`05b_xai_compare.py`: LRP-z/α2β1, DeepTaylor, IG, SmoothGrad, input×grad, SHAP-Deep if `pip install shap`) → composites, inter-method correlation, region share vs occlusion ΔAUROC. Run on `rel_aux` (v1 reference) and the final `off1_aux`. Occlusion done for all four tags (`results/occlusion/<tag>/`). *Result: −Arctic costs 0.10 / 0.08 / 0.05 / 0.03 AUROC (base / aux / openwater / lag1); −everything-else *gains* +0.01 in all four; Pacific boxes ≤ 0.007 either sign.* Figure `make_figure.py occlusion` (candidate main-text figure). Second attribution method and shuffled-labels check still to do.
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
- [x] 8.5 Sea-ice volume: *SEP or MAR volume adds ΔR² ≈ 0.01 to extent (0.075 → 0.087) — initial ice conditions of any kind foresee ~9 %.* Was: Sea-ice thickness → NH volume as a second ice-state scalar (`01_cesm2le_preprocessing.py --variable hi`; `sivoln_*.nc`; `siv_anom` picked up automatically by `07_baselines.py` and `09_residual_analysis.py`). *Code done; download pending (~same size as aice).* Read-out: SIE + volume R² ≫ 0.28 → the follow-up paper's opening result; this paper gets one Discussion sentence. Barents–Kara OHC deferred (needs ocean TEMP).
- [~] 8.7 CNN regression head on the continuous trend anomaly (`04_cesm2le_cnn_regress.py --tag rel_aux --splits 2 5 7 --n-runs 2`): *code done.* Scored as test R² vs OLS on SIE / SIE + Pacific on the same samples.
- [~] 8.8 **Bug: aux scalar frozen at lr 1e-4 → `rel_aux` was not a fair test.** Fix: warm-start output layer at the scalar logistic fit (`build_cnn(aux_init=…)`, default in `04`). *To do:* `rel_aux_ws` subset (splits 2 5 7) → full retrain + postprocess if it moves; then `rel_openwater`/`rel_lag1`/`rel_concurrent` with the fix. Decision gate 1.8 re-opened for this one question.
- [x] 8.6 Interannual + onset-in-window check run. *Result: Pacific bridge present with the model sign (r ≈ −0.1…−0.15, 2 % of year-ahead variance); **trend-on-SIE(t) R² 0.28 → 0.075 → 0.04 for windows starting t, t+1, t+2** — the ice-state result is ¾ arithmetic.*
- [~] 8.9 **Offset labels** — residual analysis run: *ice 0.075, + volume 0.087, + IPO at onset +0.028, concurrent ≈ 0 → ≈ 11 % of the following decade's trend variance is foreseeable.* Remaining: (`02 --trend-offset 1`, onsets 1990–2029): rebuild baselines (`07 --tag off1`), residual analysis, then decide CNN retrains (`base`, warm-started `aux`) on this target. Supersedes `rel_aux_ws` / `rel_concurrent` runs.
- [-] 8.10 `rel_concurrent` subset uninterpretable (ran to the epoch cap under the AUPRC rule); stopping rule reverted to val_loss + min 5 epochs.
- [~] 8.12 Thickness *pattern* as predictor (`09_thickness_maps.py --labels-file $LBL1 --month SEP|MAR`): hi on the CICE grid north of 60°N at onset → corr maps with the offset trend (raw and after SIE), EOFs, sector volumes; out-of-sample R²/AUROC of SIE + pan volume / + sectors / + 5–20 PCs. *Code done.* Read-out: if SIE + PCs or + sectors clears ~0.15, thickness *maps* carry memory that scalars miss → CNN on hi maps (`aice`-style nearest regrid exists) is the follow-up; if ≈ 0.09, the ice interior adds nothing at pan-Arctic decadal scale and paper two closes.
- [~] 8.11 RILE vs slowdown figure: `09_rile_compare.py` after the six `07 --label-var` runs (definition verified: riles = z < −1σ).
- [~] 5.3b Multi-method XAI (`05b_xai_compare.py`), see 5.3.
- [x] 8.4 Training tweak for new tags: `start_from_epoch=5`, monitor `val_auprc`, patience 15 (`src/cnn/train.py`). Existing tags not retrained.

## Phase 9 — AIES manuscript (decided 2026-09-14; GRL manuscript kept as backup in `manuscript/`)

- [x] 9.1 `manuscript_aies/` built (AMS `ametsocV6.2`, private repo `lahoffman/slowdown_arctic_seaice-manuscript-aies`, purged from the public history): `main.tex` (10 sections + App. A derivation + App. B frozen scalar), `supplement.tex` (Texts S1–S4, Tables S1–S3, 16 SI figs), `references.bib`, `Makefile`, `sync_figures.sh`. Story/abstract/section map in `docs/AIES_DRAFT.md`; figure inventory in `docs/FIGURES.md`.
- [~] 9.2 `make_figure.py` ids: `occlusion`, `learning_curves` exist; still to add `residual`, `interannual`, `regression`, `xai_compare`, `obs_predict`, `arithmetic_synthetic`, `aux_frozen_synthetic`.
- [ ] 9.3 Zach: send offset figure + panel (d) + the ledger; AIES vs GRL.
- [x] 9.4 Literature due diligence on the coupling (`docs/LITERATURE_COUPLING.md`, 2026-09-15): the phenomenon is Oldham's problem / "mathematical coupling" (Oldham 1962; Tu & Gilthorpe 2007; Kenney 1982 in geoscience); the OLS-trend form corr = −√(3(w−1)/(w(w+1))) is not in the climate or ML literature. **LB22 confirmed onset-inclusive** ("OHC100 for the year 2000 → whether 2000–2009 is a slowdown"; only baseline an IPO logistic). Hoffman 2025 / PROCAST are state targets with a persistence null → no arithmetic coupling. Framing for §1/§9/App. A fixed accordingly.
- [~] 9.5 **Reframe (2026-09-15):** headline = (i) the coupling, derived and quantified; (ii) seven XAI methods agree with each other and still point at the proxy (occlusion prices it zero); (iii) the 11 % ledger with IPO ≥ ice extent on honest labels. "Skill before XAI" checklist → one paragraph in Conclusions, not the contribution. Abstract/§1 ¶3/§6 order to be rewritten on this basis.
- [~] 9.6 Observations restored (`10_obs_baseline_predict.py`, `src/analysis/obs_predict.py`, `src/plotting/obs_predict.py`): logistic SIE / IPO / SIE+IPO / SIE+Pacific fitted on CESM2-LE (9 split-fits → band), applied to NSIDC SIE anomaly (forced removed as in the CNN pipeline) and ERSSTv5 JJA indices; observed offset-window z for verification where the decade is complete; off1 CNN vote fraction overlaid when present. *Code done; run on profx.* This is Fig. 8b: CNN and logistic side by side on the observed record.

### Status of runs (2026-09-15, evening)
- [x] `07_baselines.py --tag off1` — AUROC: SIE 0.612, IPO 0.658, Pacific 0.662, SIE+IPO 0.681, volume 0.651, SIE+vol+Pacific 0.686. F1 ≈ always-positive at this skill level → report AUROC + AUPRC.
- [x] 8.12 thickness maps (SEP, MAR): SIE + 20 PCs ≤ 0.10 → thickness *pattern* adds nothing; closed.
- [~] `04_cesm2le_cnn_regress.py --tag off1_aux --splits 2 5 7 --n-runs 2` — **first run used the onset-inclusive labels by default (bug fixed: default is now the split's own labels file); result 0.286 vs 0.280 was the old target.** Rerun with the offset target in progress (screen `regress`). Read: `cat results/regression/off1_aux/regress_summary.md` (first line must name `…off1…`).
- [~] **off1 CNN training running** (screen `off1`, `LABELS=$LBL1 scripts/run_retrain.sh off1 off1_aux`, log `results/logs/retrain_off1_*.log`; 90 models, overnight). Then `scripts/run_postprocess.sh off1 off1_aux` (occlusion, obs predictions × products × forced, LRP), then `make_figure.py S5 --baselines-tag off1 --tag off1`, `05b --tag off1_aux --split 2 --run 0`, `10_obs_baseline_predict.py … --tag off1 --product ersst|oisst`.
- [~] 5.3b `05b_xai_compare.py --tag rel_aux --split 2 --run 0` rerunning (screen `xai`) with the new defaults: LRP-z, α2β1, DeepTaylor, input×gradient, SHAP; IG/SmoothGrad opt-in (ran >1 h at 9 cores and were killed); 4-thread cap; composite normalisation at the 99th pct. Check `grep -i "FAILED\|skip" results/logs/xai_compare_rel_aux.log`.
- [x] **9.6 Observations (`10_obs_baseline_predict.py`) run for ERSST and OISST** — see REVISION_PLAN status block for the result. Defaults now: predictor anomaly per `--forced`; observed classification against an observation-only linear fit with the observed trend σ (`--label-ref linear --label-sigma obs`); calibrated probabilities. Outputs per product: `results/obs_predict/w10_off1/<product>/`, `diagnostics/obs_predict_w10_off1_<product>_<forced>[_sigmodel].png`.
- [ ] `09_rile_compare.py --windows 3 5 10` after the six `07 --label-var` runs (8.11).
- [ ] `09_sensitivity_sweep.py` offset option → S17 on honest labels.
- [ ] Synthetic figures (arithmetic; frozen scalar) — `scripts/10_synthetic_checks.py`; AR(1) check: does the fitted SIE autocorrelation alone reproduce the offset R² 0.075?
- [ ] GMST version of the coupling test for the LB22 sentence — optional, one script.
- Housekeeping learned today: `device_commit` drops the exec bit → `chmod +x scripts/*.sh` after pulling shell scripts; every new screen needs `SLOWDOWN_DATA_ROOT`/`LBL1` (add both `export`s to `~/.bashrc`); name screens (`screen -S off1|regress|xai`); `ps -u $USER -o pid,etime,pcpu,args | grep python` is the truth about what is running.

### 8.11 RILE / short-window variant
```
python scripts/02_cesm2le_slowdowns_relative.py --window 3 5 --trend-offset 1 --no-fig
for w in 3 5; do
  L=$SLOWDOWN_DATA_ROOT/cesm2le/slowdowns/cesm2le_sie_slowdown_relative_SEP_w${w}_s1_group_off1_1990-2100.nc
  python scripts/07_baselines.py --labels-file $L --label-var riles    --demean group --start-year 1990 --end-year 2029 --tag rile_w${w}_off1 --no-fig
  python scripts/07_baselines.py --labels-file $L --label-var slowdown --demean group --start-year 1990 --end-year 2029 --tag slow_w${w}_off1 --no-fig
  python scripts/09_residual_analysis.py --labels-file $L --end-year 2029 --no-maps      # continuous, sign-agnostic
done
```
Read-out: short windows are dominated by interannual noise; expect the ice-state R² to *rise* (persistence matters more over 3 yr than 10) and the Pacific increment to stay ≈ 0.02–0.03. Both tails scored; the continuous analysis covers RILEs and slowdowns at once.

## Optional / future work

- [~] Continuous (regression) target instead of the 1σ threshold — linear version is 8.1; CNN regression head still optional.
- [ ] last50-only CNN as a forcing-artifact sensitivity.
- [ ] Shorter trend windows (3–7 yr) vs re-emergence timescales.
- [ ] Concurrent atmospheric composites (Z200, U200) for correct predictions.
