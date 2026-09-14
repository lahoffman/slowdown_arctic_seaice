# Branch B — draft front matter (for discussion with Zach; not yet in the .tex)

Drafted 2026-09-14 from the Phase-1 / Phase-8 results. Numbers are the current medians
(`results/baselines/rel_aux/baselines_summary.md`, `results/occlusion/*/occlusion_summary.md`,
`results/residual/residual_summary.md`) and will be re-checked before they go into the manuscript.
The v1 text is quoted at the bottom for side-by-side reading.

---

## Title

**Sea surface temperature patterns add little to the ice state for classifying slowdowns in the decline of Arctic September sea ice extent in CESM2-LE**

Alternatives:
- *Arctic sea ice state, not remote sea surface temperature, governs decadal slowdowns in Arctic sea ice decline in CESM2-LE*
- *Explainable AI as a test, not a map: global sea surface temperature carries little information on decadal Arctic sea ice slowdowns beyond the ice state*

## Key Points (≤ 140 characters each, no special characters)

1. In CESM2-LE a logistic regression on the September sea ice anomaly at onset classifies decadal slowdowns better than a CNN on global SST maps
2. Occlusion shows all CNN skill comes from the Arctic part of the map and none from the Pacific, and skill vanishes with a one year lead
3. Pacific SST at onset or during the decade adds less than three percent of trend variance beyond the ice state across twelve slowdown definitions

## Abstract (≈ 240 words)

Since 2007 the decline of September Arctic sea ice extent (SIE) has slowed relative to the preceding decade, and tropical Pacific variability has been proposed as a driver of such decadal fluctuations. Here we test whether global sea surface temperature (SST) patterns carry information about decadal SIE slowdowns beyond what the ice state itself provides, using the 100-member Community Earth System Model Version 2 Large Ensemble (CESM2-LE). Slowdowns are defined relative to each forcing group's mean decline, and an ensemble of convolutional neural networks (CNNs) is trained to classify them from summer SST maps, with and without the September SIE anomaly at onset as an additional input. A logistic regression on the SIE anomaly alone outperforms every CNN configuration (test AUROC 0.77 versus 0.72); adding Pacific indices to the regression raises it to 0.80, an increment the CNN never recovers. Region occlusion shows that the CNN's skill resides entirely in the Arctic part of the map, where SST anomalies track ice concentration, and that removing all extra-Arctic SST does not reduce skill; with SST from the year before onset, skill collapses. The result holds across twelve slowdown definitions, both forcing groups, and for SST averaged over the trend decade rather than at onset. Layer-wise relevance propagation nevertheless places relevance in the tropical Pacific, as in earlier applications, illustrating that attribution maps must be paired with occlusion or baseline tests before being read as precursors. In CESM2-LE, decadal Arctic SIE slowdowns are an ice-state phenomenon; the remaining variance lies in ice thickness and subsurface ocean heat, not in the SST field.

## Plain Language Summary (sketch)

Arctic September sea ice has declined for decades, but not steadily: some decades lose ice more slowly than others. Because sea surface temperature patterns in the Pacific have been linked to Arctic ice loss, we asked whether a machine-learning model could tell, from a global map of summer ocean temperatures, which decades in a large climate-model ensemble would be slow ones — and whether it could do so better than simply knowing how much ice there was at the start. It could not. A one-number statistical model using the ice extent at the start of the decade did better than the neural network given the whole map, and when we blanked out parts of the map the network only lost skill when the Arctic itself was removed. Explainability maps still highlighted the Pacific, which shows that such maps must be checked against simpler tests before they are read as physical causes. In this model, slow decades are set by the state of the ice, not by the oceans around it.

## What changes in the body (for the rewrite plan)

- §1 Introduction: keep the mechanism review; replace the final paragraph ("we assess the predictability … origins of predictive skill") with the test framing: does SST add information beyond the ice state, and does XAI locate it correctly.
- §3.1: baselines figure (now S5) moves to the main text; occlusion figure new in the main text; report AUROC/AUPRC/F1 with the 9-split range.
- §3.2 (LRP): keep Fig. 2 but caption it as "where the network looks", paired with occlusion "what it uses"; drop the CP-El Niño narrative.
- §3.3 (phase conditioning): P(slowdown | phase) on all slowdowns; state the 2–3 % Pacific increment as the size of the effect.
- §3.4 (observations): recast as the check that the observed 2016–2025 Arctic SST anomaly is neutral-to-cool under all forced references and both products (Figs S18/S19); drop the "end of the pause" forecast.
- §4 Discussion: why GMST (LB22) and SIE differ (direct ocean heat uptake vs atmospheric bridge; integrated OHC vs one summer of SST; baseline asymmetry stated as a general point); where the other 70 % is (summer circulation with no memory at onset; thickness and Barents–Kara heat as the reducible part); LRP-follows-variance caution.
- §5 Conclusions: short.

---

## v1 for comparison

**Title.** eXplainable AI reveals sea surface temperature patterns associated with slowdowns in the decline of Arctic September sea ice extent in CESM2-LE

**Key Points.**
1. Slowdowns in the decadal decline of Arctic sea ice extent depend strongly on the state of Arctic sea surface temperature at the onset.
2. Tropical Pacific modes of variability exert a weak modulation on the likelihood of a slowdown for a given onset year.
3. Sea surface temperature anomalies in the central Pacific El Niño region are highly relevant for correctly predicting slowdowns in the Arctic sea ice decline.

**Abstract.** Since 2007, September Arctic sea ice extent (SIE) has experienced a slowdown in decline that contrasts with the accelerated decline during the early years of the 21st century. Understanding the origins of interannual to decadal fluctuations of summer Arctic sea ice extent around the long-term trend remains challenging. Here, we train an ensemble of convolutional neural networks (CNN) to predict the onset of a slowdown in the decadal trend of September Arctic SIE from global maps of boreal summertime sea surface temperature (SST) using output from the Community Earth System Model Version 2 Large Ensemble (CESM2-LE). Explainable artificial intelligence (XAI) applied to the trained CNNs identifies SST anomalies in both the Arctic and tropical Pacific as important predictors of slowdown onset. Statistical analyses reveal that a slowdown is more likely to occur in the years following anomalously warm Arctic SST or following the positive phases of the El Niño-Southern Oscillation (ENSO) and Interdecadal Pacific Oscillation (IPO). However, Arctic SST explains substantially more variance in slowdown occurrence than either ENSO or IPO, indicating that the local sources of the Arctic predictability play a greater role than remote ones on driving decadal variability. Fewer than 20% of trained CNN ensemble members predict a slowdown onset beginning during 2017-2025 when applied to observations and reanalyses, indicating that in the coming years we are likely to see the end of the recent pause in Arctic sea ice decline.
