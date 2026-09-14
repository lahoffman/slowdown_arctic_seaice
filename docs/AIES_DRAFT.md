# AIES manuscript — story, front matter, and what carries over from the GRL draft

Drafted 2026-09-14. Numbers are the current medians (`results/residual/w10_off1/`, `results/occlusion/*/`,
`results/interannual/`, `results/baselines/<tag>/`); re-check before they go in. Supersedes `BRANCH_B_DRAFT.md`
(kept for the GRL variant). Target: *Artificial Intelligence for the Earth Systems* (AMS), Article.

---

## The story in one paragraph

A CNN + XAI framework inherited from the GMST-hiatus literature (Labe & Barnes 2022) was applied to decadal slowdowns
in Arctic September sea-ice decline and appeared to show Arctic and tropical-Pacific SST precursors. Audited against
the controls that framework never had — persistence baselines, a label whose window does not contain the onset year,
region occlusion, a scalar input the network can actually use, and training diagnostics — the result dissolves in a
specific, instructive order: (1) three quarters of the "ice-state" skill is regression to the mean built into the label
(R² 0.28 → 0.075 when the window starts one year after onset); (2) the CNN's Arctic relevance is that same onset value
read through a 3-month SST proxy; (3) the Pacific relevance that LRP highlights — the same pattern as in LB22 and in
Li et al. (2026) — is priced at zero by occlusion; (4) the "CNN given the ice state" comparison was never fair because
the scalar was frozen by the optimiser. What survives is a small, quantified physical result: from the state at onset,
about 11 % of the following decade's trend variance is foreseeable (ice extent/volume ≈ 9 %, IPO ≈ 2.5 %), the
Pacific–Arctic bridge exists in CESM2-LE with the model's sign at ~2 % of interannual variance, and ~89 % is
decade-integrated circulation with no memory in the ocean surface. The paper is the audit; the checklist is the
contribution; the 11 % ledger is the geophysical result.

## Title

**When explainable AI finds precursors that are not there: an audit of decadal Arctic sea-ice slowdown prediction from sea surface temperature**

Alternatives: *Regression to the mean, proxies, and relevance: what a CNN learned about Arctic sea-ice slowdowns and
what it did not* · *Persistence baselines and occlusion tests for XAI precursor claims: decadal Arctic sea-ice slowdowns
in CESM2-LE*

## Abstract (≤ 250 words; AMS)

Explainable artificial intelligence (XAI) is increasingly used to locate the sources of predictability learned by neural
networks trained on climate model output. We audit one such application: a convolutional neural network (CNN) trained on
the 100-member CESM2 Large Ensemble to classify decadal slowdowns in the decline of September Arctic sea-ice extent (SIE)
from summer sea surface temperature (SST) maps, whose relevance maps highlighted Arctic and tropical-Pacific SST as
precursors. Five controls remove the result in turn. Defining the slowdown from a trend window that begins at the onset
year makes any predictor of the onset ice state skilful by construction (R² ≈ 0.25 for white noise); starting the window
one year later reduces the ice-state skill from R² 0.28 to 0.075. The CNN's Arctic relevance is this onset value read
through the SST field, and zeroing all SST outside the Arctic does not reduce skill. The tropical-Pacific relevance,
similar to that reported for global-temperature slowdowns, contributes ≤ 0.01 AUROC under occlusion, while a logistic
regression on the onset SIE anomaly outperforms every CNN configuration. A scalar ice-state input concatenated to the
network was frozen at the learning rate used, so the intended "beyond the ice state" test was uninformative until warm-
started. What remains is small and quantified: ≈ 11 % of the following decade's trend variance is foreseeable from the
onset state (ice extent and volume ≈ 9 %, Interdecadal Pacific Oscillation ≈ 2.5 %), the Pacific–Arctic teleconnection
is present with the model's sign at ~2 % of interannual variance, and the remainder has no memory in the ocean surface.
We distil the controls into a checklist for XAI-based precursor claims.

## Significance statement (≤ 120 words)

Neural networks combined with explainability methods are now routinely used to claim that ocean patterns precede
climate events. We show, for decadal slowdowns in Arctic sea-ice decline, how such a claim can arise from the way the
target is defined, from proxies for the initial state, and from the fact that explainability maps show where a network
looks rather than what it uses. Reanalysed with simple baselines and controls, the ocean surface adds almost nothing to
what the ice itself already tells us, and most of a decade's ice trend is not foreseeable. The controls are general and
cheap; we recommend them before any relevance map is read as a physical precursor.

## Key messages (for the Conclusions)

1. Any trend-slowdown label whose window contains the onset year is predictable from the onset state by construction (R² ≈ 0.25); the window must start after the predictors.
2. Persistence/state baselines first: a one-variable logistic regression outperformed a CNN on global SST maps in every configuration tested.
3. Relevance ≠ skill: LRP placed relevance in the tropical Pacific here, in LB22, and in Li et al. (2026); occlusion priced it at zero here. Report both.
4. Hybrid inputs need checking: a scalar concatenated to a CNN head moved by ~0.05 over training at lr 10⁻⁴ and was effectively ignored; warm-start at the linear fit.
5. Training diagnostics belong in the paper: a flat validation curve and epoch-1 ensemble members are findings about the signal, not housekeeping.
6. Geophysics: ~11 % of the following decade's SIE trend is foreseeable from the onset state; the Pacific bridge exists in CESM2-LE (model sign, ~2 % interannual); ~89 % is circulation without ocean-surface memory. Companion to Hoffman et al. (2025), which located interannual SIE predictability in the ice.

---

## Section skeleton and what carries over

| AIES section | content | from the GRL draft |
|---|---|---|
| **1. Introduction** | (a) Arctic SIE variability, slowdowns, proposed drivers — Pacific teleconnection (Ding, Baxter), Atlantic (Li et al. 2026), nonstationarity (B&BW); (b) ML + XAI for predictability (LB22 lineage; Mamalakis et al. on XAI trust); (c) the question: does the ocean surface carry information about the coming decade beyond the ice state, and does XAI locate it correctly; (d) what this paper does: audit. | **Keep** §1 ¶1–2 nearly verbatim (mechanism review is good). **Rewrite** ¶3 (the "we assess predictability… origins of skill" framing) as the audit question. Add Li et al., Mamalakis, Hoffman 2025. |
| **2. Data** | CESM2-LE (100 members, two forcing groups), SIE/aice/hi, SST; observations (NSIDC, OISST primary, ERSST check). | **Keep** §2.1 and §2.2 (already revised: group demeaning, OISST). Add `hi` → volume. |
| **3. Slowdown definition** | Relative labels (group-mean trend, pooled σ, onset cap); the *onset-year-in-window* problem stated up front with the white-noise result; the offset definition used from here on. Fig. 1 (definition) + Fig. 2 (arithmetic). | **Keep** §2.3 text on the relative definition (tracked-change version). **Add** the offset paragraph — this is new and central. |
| **4. Models and evaluation** | 4.1 CNN (architecture, splits, seeds, class weights, early stopping); 4.2 scalar baselines (logistic; SIE, volume, indices); 4.3 XAI: LRP-z and region occlusion; 4.4 continuous target and residual analysis; 4.5 the aux-scalar warm start; 4.6 metrics (AUROC/AUPRC/F1, R², member-block CIs, event-level rates). | **Keep** §2.4 CNN + LRP paragraphs. **Rewrite** §2.5 (climate-state conditioning) into the residual framework. **New**: baselines, occlusion, offset target, warm start, metrics. |
| **5. Results: the original claim** | The v1 result reproduced on relative labels: CNN F1/AUROC; LRP TP composite (Fig. 4a); phase dependence — presented as the starting point, not the finding. | **Keep** the *description* of Fig. 2 and the composite from §3.2, reframed as "what the network showed". Drop §3.1 skill-vs-random framing. |
| **6. Results: the audit** | 6.1 Baselines (Fig. 3, offset labels): logistic SIE ≥ every CNN. 6.2 Label construction (Fig. 2): 0.28 → 0.075 → 0.04; white noise 0.25. 6.3 Occlusion vs relevance (Fig. 4): Arctic −0.10, extra-Arctic +0.01, Pacific ≤ 0.007; openwater, lag-1. 6.4 The frozen scalar and the warm-started rerun (Fig. 7). 6.5 Training diagnostics (Fig. 7). 6.6 Sensitivity: 12 label definitions, forcing groups, 3/5/10-yr windows, RILE tail. | **New** throughout; reuse Fig. S5/S17 material. |
| **7. Results: what is foreseeable** | The ledger (Fig. 5): ice 0.075, + volume 0.087, + IPO 0.111; concurrent ≈ 0; residual–SST maps; timescale-matched Pacific index (ENSO at 3 yr, IPO at 10 yr). The bridge in CESM2-LE (Fig. 6): interannual link, model sign, nonstationarity, 2 % of year-ahead variance. | **New**; reuse the physical-interpretation sentences from §4 ¶2–3 (ENSO/IPO) with corrected sizes. |
| **8. Observations** | Arctic index under all forced references × products (Fig. 8a, = S19); what the retrained CNN says about 2016–2025 (Fig. 8b); the model-vs-observed sign discrepancy as a limit on transfer. | **Keep** §2.2/§3.3 mechanics and the 6.1 robustness text; **drop** the "end of the pause" forecast; **keep** §4 ¶5 (sign discrepancy) — it is now supported by Fig. 6e. |
| **9. Discussion** | Why GMST (LB22) and SIE differ (direct heat uptake vs atmospheric bridge; integrated OHC vs one season of SST; baseline asymmetry as a general point). Relation to Li et al. (sub-seasonal Atlantic; same missing baseline). Where the 89 % is (summer circulation without memory; Z200 as attribution, not prediction). What would carry memory (subsurface Atlantic heat; not SST). Relation to Hoffman et al. 2025. Limits: one model; CESM2's weak, opposite-sign teleconnection. | **Keep** §4 ¶6 (LB22 comparison) reworked; **keep** the Z200/U200 sentence (¶4) — it is now explained; **drop** ¶7 "predictable component associated with Arctic Ocean preconditioning". |
| **10. Conclusions + checklist** | Key messages 1–6; a boxed checklist for XAI precursor claims (label window after predictors; persistence baseline; occlusion beside relevance; check scalar inputs are used; report learning curves; state-dependent sign checks before observational transfer). | **New.** |
| Appendix A | Regression to the mean in trend windows: derivation corr(y₀, slope) = x₀/√Σx² = −0.50 for w = 10. | New, half a page. |
| Appendix B | Warm start of the output layer; the synthetic demonstration. | New. |

**Cannot be kept as written:** the title; Key Points; abstract; Plain Language Summary; §3.1 (skill vs random); §3.2's
causal language ("SST patterns contributing to predictions"); §3.3's forecast; §4 ¶1 and ¶7 (Arctic preconditioning as
the finding); Fig. 3 / S14 (P(TP|phase)) as evidence.

**Kept nearly verbatim:** §1 ¶1–2; §2.1; §2.2 (OISST version); §2.3 relative definition; §2.4 CNN and LRP description;
§4 ¶4 (Z200/U200), ¶5 (sign discrepancy), ¶6 (LB22 comparison, reworked); Open Research; Acknowledgments.

## Supplementary material (AMS "Supplemental Material", same class)

See `docs/FIGURES.md` for the file-level list. Text S1 (relative label construction detail), Text S2 (block splits and
member bootstrap), Text S3 (occlusion regions), Text S4 (OISST processing), Table S1 (all configurations × metrics),
Table S2 (label-sensitivity sweep), Table S3 (residual ΔR² by window).
