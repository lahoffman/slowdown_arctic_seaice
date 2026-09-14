# Response to Zach's comments on v1

Plain-language summary of each comment, what we'll do about it, and where it sits in the revision plan (`docs/REVISION_PLAN.md`). Grouped by theme rather than page order. "Plan §" refers to that document. Figure numbers are the v2 SI numbering (S2 and S3 are new; v1 S3–S14 are now S5–S16) unless marked v1.

---

## The big ones (change the analysis)

**1. The CESM2-LE ensemble mean itself has a slowdown around 2005–2020, so part of what we call a "slowdown" is forced, not internal variability. Could the biomass-burning forcing issue be behind this? Try sub-setting to the 50 smooth-forcing members.**
*Addressed (labels, inputs, retrain done).* Agreed, and it mattered more than it first looked. The old threshold was a fraction of the ensemble-mean trend, so when that trend flattened to zero around 2010 the threshold collapsed and ~65% of members were labelled "slowdown" that decade (vs ~10% in the mid-1990s); a "year-only" baseline matched the CNN's skill exactly, i.e. the network was recognising the epoch. Three changes: (a) slowdowns are now defined relative to each *forcing group's* mean trend — a member's decadal trend anomaly exceeding one pooled standard deviation (Fig. S1; base rate flat at ≈15% across 1990–2030, Fig. S2); (b) the SST inputs have each group's own 50-member mean removed rather than the 100-member mean, because the two groups' forced responses differ by 0.1–0.3 °C across the northern extratropics in 2000–2020 (new Fig. S3) and the residual is a group-identity shortcut a CNN could learn; (c) onsets are restricted to 1990–2030 because the spread of decadal trends collapses as the ice approaches zero (Fig. S2a). The CNN has been retrained on these labels and inputs (four configurations, 45 networks each). A per-group check shows SMBB members are ≈0.05 AUROC more predictable than CMIP6-BB for the scalar regression *and* the CNN alike — a property of the members, not a group shortcut — so the smooth-forcing-only CNN was not needed. *Plan §1.1–1.4, §4.4.*

**2. Would a simple regression on a sea ice index predict slowdowns just as well?**
*Addressed — and it decides the paper.* Yes, and it was the most important comment. New Figure S5 compares the CNN, split by split, with baselines fit on the same training members: always-positive, the onset-year slowdown climatology, and logistic regressions on the September SIE anomaly at onset, the Arctic SST index, Niño 3.4 and IPO, and their combinations (`scripts/07_baselines.py`). A logistic regression on the SIE anomaly at onset alone is a skilful predictor, adding the Arctic SST index to it changes nothing (the Arctic SST anomaly at onset is largely a proxy for the ice state), and the Pacific indices add a small increment. §3.1 now frames CNN skill against these baselines instead of "beats random chance" (which is a weak test at a 28% base rate), and the Conclusions sentence on Goosse & Zunz (2014) points to Fig. S5 as the CESM2-LE analogue. After the retrain on the epoch-free labels the picture is unambiguous: the best CNN configuration reaches test AUROC 0.72 (F1 0.39), the SIE-anomaly regression 0.77 (0.43), SIE + IPO 0.80 (0.45). Even when the SIE anomaly is handed to the CNN as a scalar input (`rel_aux`) it does not improve, and a region-occlusion test shows all of the CNN's skill sits in the Arctic part of the map while zeroing everything outside the Arctic *raises* skill by 0.01. With SST from the year before onset the skill collapses (AUROC 0.62). We read this as: in CESM2-LE, decadal slowdowns are classifiable from the ice state alone, and global SST patterns add nothing measurable — a negative result we now propose to make the paper's main finding (see "Where this leaves the paper" below). Labe & Barnes (2022) ran an IPO-only regression as their reference; the ice-state baseline here is the analogue of a GMST-state baseline, which the parent paper did not include.

**3. What is the SST under sea ice in each product, and is it driving the Arctic relevance?**
*Done.* It is most of it. In CESM2, SST under ice is pinned near freezing, so the Arctic anomaly map is partly an ice-concentration map. Retrained with ice-covered cells (aice > 15 % in JJA of the onset year) set to zero anomaly, the CNN loses 0.03 AUROC (0.72 → 0.69) and the Arctic occlusion drop halves (−0.10 → −0.05); since the pattern of zeroed cells itself still marks the ice edge, this is a lower bound on the ice-cover share. ERSSTv5 leaves the ice-covered interior empty (74 % Arctic coverage) and pins under-ice cells near freezing; OISST v2.1 fills it with a proxy SST (99 %) — documented in §2.2 and new Fig. S18. *Plan §1.6, §5.3.*

**4. Is ERSSTv5 good enough in the Arctic? Consider OISSTv2.1 (used for the Arctic Report Card); ERSSTv6 also exists.**
*Done.* OISST v2.1 is now the primary product (§2.2, Fig. S18): it is ≈0.5 °C warmer than ERSSTv5 in the central Arctic in the 1990s and converges by 2015, so ERSST's 1990–2025 Arctic JJA trend is about twice OISST's — a coverage/under-ice artefact. After forced-signal removal the two products agree (2010–2015 warm, 2016–2025 neutral-to-cool under every reference; Fig. S19); ERSST is kept as a check. Observational predictions exist for every configuration × both products × four forced references. v6 mentioned. *Plan §6.1–6.3.*

**5. Sensitivity to the 10-yr window (England et al. use 20 yr)?**
*Done for window × threshold* (new Fig. S17): windows 8/10/12/15 yr × 0.5/1/1.5σ with the scalar baselines. The ice state beats the indices by 0.05–0.09 AUROC in all twelve cells and the Pacific indices add +0.01–0.03; skill rises with threshold, not window. Our 10-yr / 1σ cell is, if anything, the one most favourable to a Pacific signal. SST-season variants were not run given the occlusion result. *Plan §4.2.*

**6. Other SST months/seasons — winter/spring preconditioning, May melt ponds?**
*Lead-time answered; seasons deferred.* The configuration using JJA SST from the year before onset scores AUROC 0.62 — no skill at one-year lead, so the paper will say "classify", not "predict", throughout. Given that and the occlusion result, other seasons were not pursued for this revision. *Plan §1.4, §2.4.*

**7. What do the LRP maps and SST composites look like for observations?**
We'll compute LRP for all 45 CNNs on the observational inputs and composite over the years with >15% votes (1990, 2007, 2016–2020) and over the observed slowdown onsets, compared with the model composite in Fig. 2. This also tests whether the 2016–2020 predictions rest on the Arctic or the Pacific. *Plan §6.7.*

**8. Any extra XAI robustness/trustworthiness checks from your sea-ice-motion paper?**
*Occlusion done; it is the paper's key figure.* Zeroing the Arctic (>65°N) costs 0.08–0.10 AUROC in every configuration; zeroing the North Pacific, tropical Pacific or North Atlantic costs ≤ 0.007 (either sign); zeroing *everything except the Arctic* improves the median by 0.01. The Pacific LRP hotspots — which look like those in Labe & Barnes (2022) — are therefore where the network places relevance, not where its skill comes from; we will say this explicitly as a caution about reading LRP composites as precursors. Second attribution method and shuffled-labels check still planned as SI robustness. *Plan §5.3, §5.4.*

**9. Concurrent or lagged atmospheric composites (Z200, U200) for correct slowdown predictions?**
Not yet looked at. Cheap to do from existing fields; we'll add it as an SI figure if it shows anything. *Plan §5.5 (optional).*

**10. Possible ocean heat transport link (Atlantic warming-hole relevance, Bering Strait); big PDO signature in the composite.**
We'll add a sentence or two in the new Discussion section as hypotheses, clearly flagged as speculative, and note the PDO-like structure explicitly. No new analysis planned for this round.

---

## Where this leaves the paper (added 2026-09-14, for discussion with Zach)

Taken together, items 1–3, 5, 6 and 8 give one result: in CESM2-LE, decadal September-SIE slowdowns are classifiable from the ice state at onset (a one-variable logistic regression reaches AUROC 0.77, SIE + IPO 0.80), and neither the CNN on the global SST map (0.72), the CNN given the ice state as an extra input (0.72), the CNN on open-water SST (0.69), nor the CNN with one-year lead (0.62) adds to it; occlusion places all CNN skill in the Arctic and none outside it, across 12 label definitions and both forcing groups. We propose to make this the paper: a rigorously tested negative on the "Pacific SST patterns precede Arctic slowdowns" narrative, with XAI (LRP + occlusion) as the tool that diagnosed why the v1 result looked positive. Before rewriting we are running the *concurrent* version of the question (decade-mean SST over the trend window, Baxter et al. 2019-style; plan Phase 8), which the atmospheric-bridge mechanism makes more plausible than prediction from the onset state; if a Pacific signal appears there the paper becomes "concurrent modulation without predictability", otherwise the clean negative. Either way Key Points, abstract, title and Conclusions will be rewritten around the baselines and occlusion figures rather than around LRP hotspots. Zach's view on venue and framing is wanted before that rewrite.

---

## Framing and structure

**Key Points 2 and 3 sound contradictory.** They are — and occlusion settled it: the central/tropical Pacific contributes nothing measurable to skill, so KP3 goes and the CP-El Niño discussion with it. *Plan §5.4.*

**Hedge the abstract's "end of the recent pause".** Agreed; we'll rephrase along the lines of "according to this framework…" and, depending on the detrending test (below), may remove the forecast claim from the abstract entirely. We also found the current text ("fewer than 20% … 2017–2025") doesn't match Fig. 4b (2017 ≈ 0.35–0.4; 2016 ≈ 0.85 is not mentioned) — will recount and fix. *Plan §6.3, §6.6.*

**Split Conclusions into a Discussion plus a shorter Conclusions; explain up front why warm Arctic SST goes with a slower decline.** Yes. The Discussion will lead with the physical interpretation (low ice → warm open water at onset → regression toward the mean trend, plus whatever survives the tests above), then model–obs differences, then GMT vs SIE. *Plan §6.6, §7.*

**State the GMT-vs-SIE slowdown mismatch more forcefully.** Agreed — Fig. S15 (v1 S12) (only 27.5% of SIE slowdowns coincide with a GMT slowdown) is one of the more novel results and will be promoted, possibly to the main text.

**Expand the intro on proposed drivers of the observed slowdown (three DOIs supplied).** Will add one or two sentences with those references. *Plan §7.*

**Define the Arctic SST index in the main text; mention the LRP-z rule in the main text; LRP on training data, not testing?** Will do all three. LRP is on training data because that's where the TP sample is large enough for stable composites; we'll say so, and note LB22 used testing. *Plan §7.*

**Why show this particular CNN in Fig. 4a?** No good reason — we'll remove the single-model panel and show only the ensemble vote fraction with uncertainty. *Plan §6.4.*

**Colour-blind-safe colours (green → blue) in Fig. 1 and wherever red/green overlap.** Will switch to a blue/orange scheme in Figs. 1, 4, S1, S6. *Plan §7.*

**Fig. S16 (v1 S13) is confusing: panels (e,f) are referenced but missing, and the Niño 3.4 phase composites don't look like El Niño / La Niña.** You're right on both. The caption and main text reference IPO panels that were dropped, and the Niño 3.4 phase composites need re-checking (they look more like a North Pacific pattern). Will fix. *Plan §7.*

---

## Small text fixes

"cliamte" → "climate"; consistent "El Niño"; add "in CESM2-LE" to the skill statement (line 234); remove the repeated "again" (line 371); add "or accelerations (rapid ice loss events)" where slowdowns-only is mentioned; confirm the NSIDC Sea Ice Index version (v3 cited via Fetterer et al. 2017; check whether we used v4); note the 5-month running mean for Niño 3.4 in the main text.

---

## Administrative

Zach's affiliation (Climate Central Inc., Princeton, NJ) and ORCID (0000-0002-6394-7651) to be added. A placeholder will be left in Acknowledgments for the sentence he needs to add after Climate Central's internal review.

---

## One thing we found while going through this that Zach didn't flag

Our test-set accuracy (~71%, v1 Fig. S5) is below the majority-class baseline (73%), and the F1 of 0.48–0.49 is only marginally above the always-predict-slowdown baseline of 0.42–0.44 because the base rate is 27–28%. Labe & Barnes had a ~10% base rate, where "beats random" was a meaningful test; here it isn't. This is why items 1–3 above come first.
