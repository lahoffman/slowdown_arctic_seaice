# Decadal-prediction ML studies (Barnes group) — applicability to the coupling result

Audited 2026-09-15 against one question: does the input state at the initial time enter the target's definition
(the trend-window coupling of Eq. coupling, `corr(b, y0) = -0.71 … -0.50`), and was the trivial "current state"
predictor scored as a baseline? Full reads of the PDFs in `manuscript_aies/refs/` (not tracked).

| Paper | Input (time) | Target | Window starts | Coupling? | State baseline? | Verdict |
|---|---|---|---|---|---|---|
| Labe & Barnes 2022 (JAMES) | annual OHC100 map, year t | 10-yr GMT trend slowdown, t…t+9 | **t** | **yes** (state is first point of the fitted trend) | IPO logistic only; no GMT-state baseline | exposed — our §9 test: ¾ coupling, ¼ real (IPO) |
| Gordon, Barnes & Hurrell 2021 (GRL) | OHC100 maps at −38, −34, −30 mo | PDO sign change within −30…0 mo (binary) | after input | no (event, not a statistic containing the input) | none (chance only) | ambiguous: label conditioned on input amplitude (weak PDO → flips); no \|PDO\| baseline |
| Gordon & Barnes 2022 (GRL; 2022b is its preprint) | OHC100/300/700 maps, 60-mo backward mean ending t | 60-mo mean SST anomaly, t+12…t+72 (and t+36…t+96) | t+12 | no | **persistence** (instantaneous SST) — beaten everywhere | clean; lead decay mild (real ocean memory) |
| Rader & Barnes 2023 (GRL) | 5-yr mean SST, t−4…t | 5-yr mean N. Atlantic SST, t+1…t+5 | t+1 | no | "mean target evolution" ≈ damped persistence (skill ≈0.04 vs 0.25 for the mask) | clean |
| Fernandez & Barnes 2025 (arXiv) | annual tas, years i, i−1, i−2 | tas in single year i+L, L=1…10 | i+1 | no | **none** (climatology + uninitialised CMIP6 + DCPP) | clean of coupling; skill vs trailing climatology is trend-inflated, no persistence shown |

## What this means for us

1. **LB22 is the exception in its own group, not the rule.** Every other decadal ML design from the Barnes
   group predicts a forward mean or a single future year, i.e. it uses what we call the offset window. Our
   `_off1` labels put the sea-ice problem into the same convention as Gordon22 / Rader23, so the comparison
   is now like with like. This is the strongest way to say the paper's point without blame: the trend-window
   label is an unusual construction, and the moment it is written as a forward mean (as the group's other
   papers do) the coupling disappears.
2. **The recurring gap is the state baseline.** Even the clean designs mostly lack a regression on the state
   the input pictures (Gordon21: no |PDO| baseline although the paper itself notes weak-PDO samples drive the
   skill; Fernandez25: no persistence or damped persistence; Rader23 only a proxy). Gordon22 is the one that
   beats an explicit persistence baseline. That supports checklist rule (ii) as the general one, independent
   of the coupling.
3. **Gordon22's regression-with-uncertainty target is the template for a clean sea-ice version** (continuous
   forward-mean target, state-dependent confidence). Our continuous-target CNN on `_off1` is the analogue;
   if we ever extend, the Gaussian-output head is the natural next step and the citation is ready.
4. **Rader23 / Fernandez25 (learned-mask analogs)** are relevant as the alternative when the memory is in
   the state: an analog forecast conditioned on the ice state is the honest version of "the map predicts".
5. Fernandez25's trend-inflated skill (reference climatology degrades with lead) is a different pitfall —
   not ours — but worth one clause in "A familiar problem": climatology-relative skill on a trending series
   rewards trend-following, which is why our anomalies are relative to the ensemble mean.

## Text placed in the manuscript (2026-09-15, review markup)

- Introduction, new paragraph: "Other decadal machine-learning studies from the same group target forward
  means or single future years (Gordon21, Gordon22, Rader23, Fernandez25) and are not exposed to the
  trend-window coupling; what recurs across them, and across this literature generally, is the absence of a
  baseline on the state that the input pictures."
- Discussion "Portability" paragraph; Conclusions rule (vii).
- `references.bib`: Gordon21, Gordon22, Rader23, Fernandez25, Meehl11.
