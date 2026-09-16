"""
figures_grl.py — manifest of the GRL paper's figures (manuscript/manuscript_v2.tex, supplement).

Same schema as figures_aies.py: manuscript filename → dict(kind, ids/args | src, needs).
Built by `python scripts/make_figures.py --paper grl` into results/figures/grl/, synced by
manuscript/sync_figures.sh into manuscript/figures/.
"""

OFF1 = ["--trend-offset", "1"]
CNN = ["--tag", "off1_aux"]

MAIN = {
    "fig_1.pdf":          dict(kind="make_figure", ids=["1"], args=OFF1, needs="offset labels (definition panel with window t+1 … t+10)"),
    "fig_ledger_grl.pdf": dict(kind="make_figure", ids=["ledger_grl"], args=OFF1 + CNN,
                               needs="09_residual_analysis --sst-pcs, 09_ohc_ledger sie 100/300/700 (+ --concurrent), gmt 100, 04 regress off1_aux"),
    "fig_maps_grl.pdf":   dict(kind="make_figure", ids=["maps_grl"], args=OFF1,
                               needs="09_residual_analysis (maps), 09_ohc_ledger --depth 100 and --concurrent"),
    "fig_obs.png":        dict(kind="diagnostic", src="obs_predict_w10_off1_ersst_linear.png", needs="10_obs_baseline_predict (CNN votes via --tag off1_aux)"),
}

SUPPLEMENT = {
    "fig_S1.pdf":  dict(kind="make_figure", ids=["S1"], args=OFF1, needs="slowdown definition, offset window"),
    "fig_S2.pdf":  dict(kind="make_figure", ids=["S2"], args=OFF1, needs="pooled σ and onset cap"),
    "fig_S3.pdf":  dict(kind="make_figure", ids=["S3"], args=[], needs="forced response SMBB − CMIP6"),
    "fig_S4.pdf":  dict(kind="make_figure", ids=["S4"], args=OFF1, needs="label distributions"),
    "fig_S5.pdf":  dict(kind="make_figure", ids=["S5"], args=CNN + OFF1, needs="baselines vs CNN, offset labels (the classification view)"),
    "fig_S15.pdf": dict(kind="make_figure", ids=["S15"], args=[], needs="SIE vs GMT slowdown counts"),
    "fig_S17.pdf": dict(kind="make_figure", ids=["S17"], args=[], needs="window / threshold sweep (rerun on offset labels)"),
    "fig_S18.pdf": dict(kind="make_figure", ids=["S18"], args=[], needs="ERSST vs OISST"),
    "fig_S19.pdf": dict(kind="make_figure", ids=["S19"], args=[], needs="observed Arctic index vs forced reference"),
    "tail_asymmetry.png":         dict(kind="diagnostic", src="tail_asymmetry_w10_off1.png", needs="09_tail_asymmetry"),
    "interannual_check.png":      dict(kind="diagnostic", src="interannual_check.png", needs="09_interannual_check (Pacific–Arctic bridge)"),
    "residual_analysis.png":      dict(kind="diagnostic", src="residual_analysis_w10_off1.png", needs="09_residual_analysis"),
    "regression_off1_aux.png":    dict(kind="diagnostic", src="regression_off1_aux.png", needs="04 regress --tag off1_aux (CNN on the continuous target)"),
    "ohc_sie_ledger_100.png":     dict(kind="diagnostic", src="ohc_sie_ledger_100_JJA.png", needs="09_ohc_ledger --depth 100"),
    "ohc_sie_ledger_300.png":     dict(kind="diagnostic", src="ohc_sie_ledger_300_JJA.png", needs="09_ohc_ledger --depth 300"),
    "ohc_sie_ledger_700.png":     dict(kind="diagnostic", src="ohc_sie_ledger_700_JJA.png", needs="09_ohc_ledger --depth 700 (run_ohc2)"),
    "ohc_sie_ledger_100_conc.png": dict(kind="diagnostic", src="ohc_sie_ledger_100_JJA_conc.png", needs="09_ohc_ledger --concurrent (run_ohc2)"),
    "ohc_gmt_ledger_100.png":     dict(kind="diagnostic", src="ohc_gmt_ledger_100_annual.png", needs="09_ohc_ledger --target gmt"),
    "obs_oisst.png":              dict(kind="diagnostic", src="obs_predict_w10_off1_oisst_linear.png", needs="10_obs --product oisst"),
    "obs_sigmodel.png":           dict(kind="diagnostic", src="obs_predict_w10_off1_ersst_linear_sigmodel.png", needs="10_obs --label-sigma model --forced linear"),
    "thickness_SEP.png":          dict(kind="diagnostic", src="thickness_SEP_w10_off1.png", needs="09_thickness_maps"),
}

MANIFEST = {**SUPPLEMENT, **MAIN}
