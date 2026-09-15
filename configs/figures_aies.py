"""
figures_aies.py — the AIES manuscript's figure manifest.

One entry per file that main.tex / supplement.tex includes (the key is the filename used in the
\\fig{} macro). Two kinds:
  make_figure : run scripts/make_figure.py with ``ids`` and ``args``; the output fig_<id><suffix>.pdf is
                renamed to the key. Entries with identical ``args`` are built in one call (shared cache).
  diagnostic  : copy results/figures/diagnostics/<src> (written by a 09_*/10_*/05b script) to the key.
``needs`` names the run that must exist first (for the status table only).

Edit here, never in sync_figures.sh. scripts/make_figures_aies.py reads this.
"""

OFF1 = ["--trend-offset", "1"]                 # offset labels (window t+1 … t+10, onsets ≤ 2029)
CNN = ["--tag", "off1_aux"]                    # the paper's final CNN configuration
ALL_TAGS = ["--occlusion-tags", "off1", "off1_aux", "off1_pacific", "rel_base", "rel_aux", "rel_openwater", "rel_lag1"]

MAIN = {
    "fig_1.pdf":               dict(kind="make_figure", ids=["1"],  args=OFF1, needs="offset labels"),
    "fig_2.pdf":               dict(kind="make_figure", ids=["2"],  args=CNN + OFF1, needs="off1_aux LRP (run_postprocess)"),
    "fig_S5.pdf":              dict(kind="make_figure", ids=["S5"], args=["--baselines-tag", "off1_pacific", "--tag", "off1_pacific"] + OFF1,
                                    needs="07 --tag off1_pacific (run_retrain) — the CNN row is the extra-Arctic network; off1/off1_aux rows via the baselines table"),
    "fig_occlusion.pdf":       dict(kind="make_figure", ids=["occlusion"], args=ALL_TAGS, needs="08_occlusion for off1 tags"),
    "fig_learning_curves.pdf": dict(kind="make_figure", ids=["learning_curves"], args=CNN, needs="off1_aux training histories"),
    "fig_S19.pdf":             dict(kind="make_figure", ids=["S19"], args=[], needs="09_obs_forced_removal (done)"),
    "interannual_check.png":         dict(kind="diagnostic", src="interannual_check.png", needs="09_interannual_check (done)"),
    "residual_analysis_w10_off1.png": dict(kind="diagnostic", src="residual_analysis_w10_off1.png", needs="09_residual_analysis (done)"),
    "gmt_coupling_check.png":        dict(kind="diagnostic", src="gmt_coupling_check.png", needs="09_gmt_coupling_check"),
    "ohc_lb22_test.png":             dict(kind="diagnostic", src="ohc_lb22_test.png", needs="01_cesm2le_ohc + 09_ohc_lb22_test (run_ohc.sh)"),
    "ohc_sie_ledger_100.png":        dict(kind="diagnostic", src="ohc_sie_ledger_100_JJA.png", needs="09_ohc_ledger --depth 100 (run_ohc.sh)"),
    "ohc_gmt_ledger_100.png":        dict(kind="diagnostic", src="ohc_gmt_ledger_100_annual.png", needs="09_ohc_ledger --target gmt --depth 100 (run_ohc.sh)"),
    "regression_off1_aux.png":       dict(kind="diagnostic", src="regression_off1_aux.png", needs="04_cesm2le_cnn_regress --tag off1_aux (offset target)"),
    "xai_compare_off1_aux.png":      dict(kind="diagnostic", src="xai_compare_off1_aux.png", needs="05b --tag off1_aux (after training)"),
    "obs_predict_ersst.png":         dict(kind="diagnostic", src="obs_predict_w10_off1_ersst_linear.png", needs="10_obs_baseline_predict (done; CNN votes after postprocess)"),
    "xai_compare_off1_pacific.png":  dict(kind="diagnostic", src="xai_compare_off1_pacific.png", needs="05b --tag off1_pacific (queue)"),
    "regression_off1_pacific.png":   dict(kind="diagnostic", src="regression_off1_pacific.png", needs="04 regress --tag off1_pacific (queue)"),
}

SUPPLEMENT = {
    "fig_S1.pdf":  dict(kind="make_figure", ids=["S1"], args=OFF1, needs="offset labels"),
    "fig_S2.pdf":  dict(kind="make_figure", ids=["S2"], args=OFF1, needs="offset labels"),
    "fig_S3.pdf":  dict(kind="make_figure", ids=["S3"], args=[], needs="02_cesm2le_forced (done)"),
    "fig_S4.pdf":  dict(kind="make_figure", ids=["S4"], args=OFF1, needs="offset labels"),
    "fig_S5_rel_aux.pdf": dict(kind="make_figure", ids=["S5"], args=["--tag", "rel_aux", "--suffix", "_rel_aux"], needs="rel_aux (done)"),
    "fig_S6.pdf":  dict(kind="make_figure", ids=["S6"], args=CNN + OFF1, needs="off1_aux predictions"),
    "fig_S7.pdf":  dict(kind="make_figure", ids=["S7"], args=CNN + OFF1, needs="off1_aux predictions"),
    "fig_S8.pdf":  dict(kind="make_figure", ids=["S8"], args=CNN + OFF1, needs="off1_aux metrics"),
    "fig_S9.pdf":  dict(kind="make_figure", ids=["S9"], args=CNN + OFF1, needs="off1_aux predictions"),
    "fig_S10.pdf": dict(kind="make_figure", ids=["S10"], args=CNN + OFF1, needs="off1_aux predictions (slow: pass over all splits)"),
    "fig_S15.pdf": dict(kind="make_figure", ids=["S15"], args=[], needs="02_cesm2le_slowdowns_gmt (done)"),
    "fig_S17.pdf": dict(kind="make_figure", ids=["S17"], args=[], needs="09_sensitivity_sweep (offset version pending)"),
    "fig_S18.pdf": dict(kind="make_figure", ids=["S18"], args=[], needs="02_obs_compare_products (done)"),
    "fig_learning_curves.pdf": MAIN["fig_learning_curves.pdf"],
    "event_stats_off1_aux.png":       dict(kind="diagnostic", src="event_stats_off1_aux.png", needs="09_event_stats --tag off1_aux"),
    "residual_analysis_w3_off1.png":  dict(kind="diagnostic", src="residual_analysis_w3_off1.png", needs="09_residual_analysis w3 (done)"),
    "residual_analysis_w5_off1.png":  dict(kind="diagnostic", src="residual_analysis_w5_off1.png", needs="09_residual_analysis w5 (done)"),
    "rile_vs_slowdown.png":           dict(kind="diagnostic", src="rile_vs_slowdown.png", needs="09_rile_compare (pending)"),
    "thickness_SEP_w10_off1.png":     dict(kind="diagnostic", src="thickness_SEP_w10_off1.png", needs="09_thickness_maps (done)"),
    "xai_compare_rel_aux.png":        dict(kind="diagnostic", src="xai_compare_rel_aux.png", needs="05b --tag rel_aux (rerunning)"),
    "obs_predict_oisst.png":          dict(kind="diagnostic", src="obs_predict_w10_off1_oisst_linear.png", needs="10_obs_baseline_predict --product oisst (done)"),
    "obs_predict_ersst_sigmodel.png": dict(kind="diagnostic", src="obs_predict_w10_off1_ersst_linear_sigmodel.png", needs="10_obs_baseline_predict --label-sigma model"),
}

MANIFEST = {**SUPPLEMENT, **MAIN}          # MAIN last so shared keys keep the main-text definition
