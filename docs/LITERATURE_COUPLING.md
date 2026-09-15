# The onset-year-in-window coupling: what the literature says (2026-09-14)

Purpose: due diligence before the AIES paper claims the trend-window coupling as a result. Read for §1, §3, §9 and Appendix A.

## The result, stated once

Label = OLS slope of y over a window of w years starting at onset t. Predictor = any function of the state at t.
For white noise, corr(y_t, slope) = (t−t̄)/√S_tt = −√(3(w−1)/(w(w+1))): −0.71 (w=2, Oldham), −0.71 (w=3), −0.63 (w=5), −0.50 (w=10, R² 0.25).
Once t leaves the window the white-noise correlation is exactly 0; any remaining correlation is physical relaxation (AR(1) memory).
CESM2 SIE: R² 0.276 (window t…t+9) → 0.075 (t+1…t+10) → 0.040 (t+2…t+11). Onset-inclusive ≈ 0.25 arithmetic + ~0.03 physics.

## Where the coupling is already written down

| paper | field | what it says | how we use it |
|---|---|---|---|
| Oldham 1962, *J. Chron. Dis.* | biostatistics | corr(x₀, x₁−x₀) = −1/√2 for iid; regress change on the mean of the two values instead | origin; Appendix A cites as the w=2 case |
| Blomqvist 1977; Svärdsudd & Blomqvist 1978 | biostat | generalisation with measurement error and true dependence of change on level | shows the community separated arithmetic from physics 50 years ago |
| Hayes 1988, *Stat. Med.* | biostat | review of methods to test whether change depends on initial value | cite |
| Tu & Gilthorpe 2007, *Stat. Med.*; Blance, Tu & Gilthorpe 2005, *IJE*; Tu et al. 2004 *IJE* "mathematical coupling: a multilevel approach" | biostat | "mathematical coupling": when one variable contains the other, correlation is partly algebraic | the name we adopt |
| Pearson 1897 (spurious correlation of indices); Kenney 1982 *WRR* "Beware of spurious self-correlations!"; Brett 2004 *Oikos* | geoscience/ecology | shared-term spurious correlation is known in the geosciences under "self-correlation" | shows the phenomenon is not foreign to our field, only the trend-window form is |
| Barnett, van der Pols & Dobson 2005, *IJE* | epidemiology | regression to the mean: what it is and how to deal with it | general citation |

None of these treat an OLS trend over w points; none appear in the climate-ML literature. Not found in any climate or ML paper as a predictor–target coupling statement (searched: "mathematical coupling" + trend/climate; leakage surveys; hiatus-prediction papers). Standard geoscience texts (Bendat & Piersol; Emery & Thomson; von Storch & Zwiers; Wilks) — not verified from the PDFs; Wilks has a short regression-to-the-mean forecasting note; a clean miss there supports writing it up. (Gille SIOC 221A notes: not reachable from the session; the least-squares/trend lecture is the one to check.)

## Papers in hand (uploaded 2026-09-14) and how each bears on the claim

**Labe & Barnes 2022 GRL (+SI).** Confirmed onset-inclusive: "if we input a map of OHC100 anomalies for the year 2000, the ANN will output whether the decade from 2000 to 2009 will be a slowdown event or not" (§2.4). Label: 10-yr moving trend of GMST below 0.44 × ensemble-mean trend; onset defined as the first year of the window. Only baseline: logistic regression on the IPO index (F1 0.28 vs ANN 0.40); no GMST-anomaly (state) baseline. LRP-z, positive relevance only. The coupling enters through OHC100(t) ↔ GMST(t): annual-mean upper-100 m heat content and GMST are strongly correlated, so any OHC map that encodes the global-mean state predicts the onset-inclusive trend by construction. The exact size for GMST is unknown (we have not run it) — state this as "applies in principle; magnitude to be tested", not as a demonstrated flaw. A cheap test if we want it: GMST anomaly at t vs 10-yr trend from t in CESM2-LE, onset-inclusive vs offset. Their F1-vs-random framing and single-index baseline are the same structure we started with.

**Mayer & Barnes 2021 GRL (+SI).** Subseasonal forecasts of opportunity (tropical OLR → N. Atlantic Z500 sign, 3-class ANN, confidence-based selection). Target is a future state, not a trend containing the input → no coupling. Baseline: comparison to multinomial logistic regression in the SI; skill measured against random chance at 90% confidence. Relevant to us as the "confident-prediction subset" idea (our TP composites), and as another LRP-only interpretation; not relevant to the coupling.

**Bommer et al. 2024 AIES.** Five XAI properties (robustness, faithfulness, randomisation, complexity, localisation) evaluated on the Labe & Barnes "predict the decade" task, with a uniform-random-baseline skill score. Key for us: they show methods *disagree* and rank them; they do not ask whether an agreeing set of methods can still describe a proxy. Our multi-method figure is the complement: agreement across LRP-z, α2β1, DeepTaylor, input×gradient (and IG/SHAP when they run) on the Arctic, with occlusion showing the Arctic is a scalar proxy. Cite for method choice and for "faithfulness ≠ physical meaning". Note their remark that neglecting negative neurons in LRP-αβ worsens robustness — matches our α2β1/DeepTaylor maps being smooth positive bands.

**Hoffman et al. 2025 JGR-MLC (own paper).** A PROCAST-style transfer operator (and an NN) predicting the September SIE *state* at lags τ = 1–10 yr and averaging times T = 1–10 yr, with anomaly persistence as the benchmark throughout ("skillful" = R² > persistence with CI above zero). State targets do not contain a slope through the initial point, so there is no arithmetic coupling. For τ < T the input average (July over t, t−1, t−2) and the target average (September over the same years) share years — inherited from PROCAST's lag × averaging grid; persistence shares the same years, so relative skill is fair, and the only caveat is that the effective lead in those cells is months rather than τ. One sentence in §9 as the contrast case: state target + persistence null = clean; trend target containing the initial year + no state baseline = coupled.

**Sévellec & Drijfhout 2018 Nat. Comm. (PROCAST).** Predicts GMT/SST anomalies at lead (not trends) from the current anomaly via a transfer operator; persistence is the null hypothesis for all lags and averaging times. Clean by construction; the natural contrast to LB22 in §9 ("state targets with a persistence null vs trend targets containing the initial year").

**DelSole & Tippett 2018 Clim. Dyn.** Generalised predictability: distinguishes initial-value from forced predictability and proposes filtering so that long-time-scale processes do not count as predictability when the climatological distribution is broad. Use in §9: our group-demeaning removes forced predictability; the offset window removes the arithmetic part; what remains is the initial-value predictability of the residual — 7.5 % ice, +1 % volume, +2.5 % IPO.

**Weisheimer & Palmer 2014 J. R. Soc. Interface.** Reliability categories for seasonal forecasts; reliability as distinct from skill. Use for the observations section (our vote fractions are not calibrated probabilities) and the probabilistic-metrics recommendation in the checklist.

**Hawkins et al. 2016 QJRMS.** Design of Arctic sea-ice prediction systems: damped persistence as a benchmark level of skill (Blanchard-Wrigglesworth 2011a; Merryfield 2013), potential vs actual predictability, detrended climatological σ. Use for the "persistence baseline first" point — in sea-ice prediction this has been standard for a decade; our v1 omitted it.

**Li et al. 2026 npj (Atlantic DNN).** Reconstruction of daily SIE from SST at 20–60-day leads; baseline is a ridge regression on the same SST; integrated gradients + occlusion sensitivity both shown (Fig. 3). No ice-persistence baseline. Not a trend target → no arithmetic coupling, but the "SST predicts SIE" claim without an ice-state baseline is the same missing control at a shorter timescale. They *do* show occlusion beside attribution, which we should acknowledge.

## What this means for the claims in the AIES draft

1. Say "Oldham's problem / mathematical coupling in trend-window labels", cite Oldham, Tu & Gilthorpe, Kenney; derive the OLS form in Appendix A. Do not say "nobody noticed".
2. LB22: state precisely what is verified (onset-inclusive window, no state baseline) and what is not (magnitude for GMST/OHC). Offer the one-line test if we run it.
3. Use PROCAST / Hoffman 2025 as the clean contrast (state target, persistence null); note the τ < T effective-lead caveat in one clause.
4. Bommer 2024 + Mamalakis 2022: method agreement is the recommended fix; our figure shows agreement on a proxy → occlusion is not optional.
5. DelSole & Tippett 2018 supplies the language for what survives: initial-value predictability of the demeaned residual after the arithmetic part is removed.


## Added 2026-09-15 (evening): machine-learning leakage and initial-value predictability

| paper | what it gives us | where used |
|---|---|---|
| Kaufman, Rosset, Perlich & Stitelman 2012, *ACM TKDD* 6(4):15, doi 10.1145/2382577.2382579 | canonical formulation of *leakage*: information about the target reaching the model by a route other than the one under test; detection and avoidance | §9 "A familiar problem in an unfamiliar place"; checklist |
| Kapoor & Narayanan 2023, *Patterns* 4:100804, doi 10.1016/j.patter.2023.100804 | survey of 294 papers in 17 fields: leakage is the leading cause of over-optimistic ML-in-science results; taxonomy begins with "no clean separation between predictor and target"; model-info sheets as remedy | §9; motivates the checklist as an instance of theirs |
| Archie 1981, *Ann. Surg.* 193:296, doi 10.1097/00000658-198103000-00008 | "mathematical coupling of data" in clinical physiology — shared measured quantity on both sides of a correlation | §3.2 / §9 naming |
| Blanchard-Wrigglesworth et al. 2011a (J. Clim.), Day et al. 2014, Bushuk et al. 2020 (2024), Tietsche et al. 2014, Hoffman et al. 2025 | sea-ice initial-value predictability: state (thickness) memory of 1–3 yr; damped persistence as benchmark | §9: the 0.075 that survives the offset is this memory |
| Meehl et al. 2011/2018, Yeager et al. 2015 (and Yeager 2018, Smith 2019 to add) | decadal GMT predictability from the ocean interior (heat uptake, AMOC), not the surface | §9: why OHC100 is both a GMT proxy and a memory carrier |
| Tietsche et al. 2011 | sea-ice recovery = relaxation; the case where the coupling *is* the hypothesis | §9 exemption |

Key distinction for the writing (Lauren, 2026-09-15): the trend-window coupling defeats the usual temporal-leakage check — no target year lies in the predictor's past — because the leak is through the algebra of the label, not through time. Say this once, plainly.

Positioning on LB22 (agreed 2026-09-15): the GMT-state test bounds the coupling's contribution to *their* predictor at 0.25 r² with r unknown; it does **not** show their result is coupling. Saying more requires their predictor (OHC100 maps) → see `docs/OHC_PLAN.md`. Until then §9 says "the control was not applied and would be cheap to apply", nothing stronger.
