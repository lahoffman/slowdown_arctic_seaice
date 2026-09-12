# Response to Zach's comments on v1

Plain-language summary of each comment, what we'll do about it, and where it sits in the revision plan (`docs/REVISION_PLAN.md`). Grouped by theme rather than page order. "Plan §" refers to that document. Figure numbers are the v2 SI numbering (S2 and S3 are new; v1 S3–S14 are now S5–S16) unless marked v1.

---

## The big ones (change the analysis)

**1. The CESM2-LE ensemble mean itself has a slowdown around 2005–2020, so part of what we call a "slowdown" is forced, not internal variability. Could the biomass-burning forcing issue be behind this? Try sub-setting to the 50 smooth-forcing members.**
*Addressed (labels and inputs); CNN retrain in progress.* Agreed, and it mattered more than it first looked. The old threshold was a fraction of the ensemble-mean trend, so when that trend flattened to zero around 2010 the threshold collapsed and ~65% of members were labelled "slowdown" that decade (vs ~10% in the mid-1990s); a "year-only" baseline matched the CNN's skill exactly, i.e. the network was recognising the epoch. Three changes: (a) slowdowns are now defined relative to each *forcing group's* mean trend — a member's decadal trend anomaly exceeding one pooled standard deviation (Fig. S1; base rate flat at ≈15% across 1990–2030, Fig. S2); (b) the SST inputs have each group's own 50-member mean removed rather than the 100-member mean, because the two groups' forced responses differ by 0.1–0.3 °C across the northern extratropics in 2000–2020 (new Fig. S3) and the residual is a group-identity shortcut a CNN could learn; (c) onsets are restricted to 1990–2030 because the spread of decadal trends collapses as the ice approaches zero (Fig. S2a). The CNN is being retrained on these labels and inputs. A smooth-forcing-only CNN is kept as an optional sensitivity test rather than the main line, since the group-relative definition removes the artifact directly. *Plan §1.1–1.3, §4.4.*

**2. Would a simple regression on a sea ice index predict slowdowns just as well?**
*Addressed.* Yes, and it was the most important comment. New Figure S5 compares the CNN, split by split, with baselines fit on the same training members: always-positive, the onset-year slowdown climatology, and logistic regressions on the September SIE anomaly at onset, the Arctic SST index, Niño 3.4 and IPO, and their combinations (`scripts/07_baselines.py`). A logistic regression on the SIE anomaly at onset alone is a skilful predictor, adding the Arctic SST index to it changes nothing (the Arctic SST anomaly at onset is largely a proxy for the ice state), and the Pacific indices add a small increment. §3.1 now frames CNN skill against these baselines instead of "beats random chance" (which is a weak test at a 28% base rate), and the Conclusions sentence on Goosse & Zunz (2014) points to Fig. S5 as the CESM2-LE analogue. The numbers in §3.1 are placeholders until the CNN is retrained on the epoch-free labels (see item 1); with the current model they are F1 0.35 / AUROC 0.70 for the CNN vs 0.43 / 0.76 for the SIE-anomaly regression. Labe & Barnes (2022) ran the equivalent index-only regression, so this also brings the paper in line with the parent framework.

**3. What is the SST under sea ice in each product, and is it driving the Arctic relevance?**
Very likely part of it. In CESM2, SST under ice is pinned near freezing, so the Arctic anomaly map is partly an ice-concentration map. We'll retrain with ice-covered cells masked (aice > 15% in JJA of the onset year) and rerun the Arctic occlusion test. We'll also document what ERSSTv5 and OISST assign under ice. *Plan §1.6, §5.3.*

**4. Is ERSSTv5 good enough in the Arctic? Consider OISSTv2.1 (used for the Arctic Report Card); ERSSTv6 also exists.**
Yes — we'll add OISSTv2.1 as a second observational input, regrid it to the CESM2 grid, rerun the observational predictions and the Arctic SST index, and show both products side by side. We'll justify the primary product in §2.2 and mention v6. This is independent of the model work and can run in parallel. *Plan §6.2.*

**5. Sensitivity to the 10-yr window (England et al. use 20 yr)?**
Not tested yet. We'll run the cheap version (logistic baselines) over windows of 8/10/12/15 yr, thresholds of 0.5/1/1.5σ, and SST seasons, and retrain the CNN for one alternative. Reported as an SI heat-map. *Plan §4.2.*

**6. Other SST months/seasons — winter/spring preconditioning, May melt ponds?**
Good idea; same sensitivity sweep covers MAM and MJJ inputs. Separately, one of the retrained configurations uses JJA SST from the *year before* the window (`lag1`), so we will get a direct answer on whether there is skill at one year lead. *Plan §1.4, §4.2.*

**7. What do the LRP maps and SST composites look like for observations?**
We'll compute LRP for all 45 CNNs on the observational inputs and composite over the years with >15% votes (1990, 2007, 2016–2020) and over the observed slowdown onsets, compared with the model composite in Fig. 2. This also tests whether the 2016–2020 predictions rest on the Arctic or the Pacific. *Plan §6.7.*

**8. Any extra XAI robustness/trustworthiness checks from your sea-ice-motion paper?**
Yes: a second attribution method (Integrated Gradients or SmoothGrad), a region-occlusion test (Arctic / tropical Pacific / North Pacific boxes), and a randomised-weights / shuffled-labels sanity check. Occlusion is the most direct "does this region matter for skill" test and doesn't depend on LRP at all. *Plan §5.3.*

**9. Concurrent or lagged atmospheric composites (Z200, U200) for correct slowdown predictions?**
Not yet looked at. Cheap to do from existing fields; we'll add it as an SI figure if it shows anything. *Plan §5.5 (optional).*

**10. Possible ocean heat transport link (Atlantic warming-hole relevance, Bering Strait); big PDO signature in the composite.**
We'll add a sentence or two in the new Discussion section as hypotheses, clearly flagged as speculative, and note the PDO-like structure explicitly. No new analysis planned for this round.

---

## Framing and structure

**Key Points 2 and 3 sound contradictory.** They are — KP2 says tropical Pacific modulation is weak, KP3 says the central Pacific is highly relevant. We'll drop or heavily hedge KP3 unless the occlusion test shows the central Pacific actually contributes skill. *Plan §5.4.*

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
