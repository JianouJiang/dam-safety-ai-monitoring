# STATISTICIAN REVIEW 002

**Reviewer:** Statistician-Fisher (Ronald Fisher)
**Date:** 2026-03-06
**Paper:** Physics-Informed Anomaly Detection for Dam Safety: A Hybrid DMK Framework
**Status:** Code substantially revised; 5-seed experiment in progress (seed 1/5 complete)
**Previous review:** STATISTICIAN_001_REVIEW.md (Score: 3/10)

---

## Summary

The Worker has made excellent progress addressing the three critical statistical flaws from STATISTICIAN_001. The test-set threshold leakage has been eliminated via a proper three-way split with validation-based threshold selection. Multi-seed experiments (5 seeds) with full reproducibility controls are running. McNemar's test is implemented for pairwise comparisons. The anomaly rate has been reduced to a realistic 6.8%. Event-level metrics are now computed properly. The LSTM-AE baseline has its own hyperparameters. These are substantial, correct improvements.

However, the early results from seed 42 reveal a **new critical problem**: the physics layer contributes nothing to the DS fusion. The "No Physics" ablation produces identical F1/precision/recall/FAR to the full framework (both F1=0.895). This means the PINN's anomaly signal is being entirely suppressed by the threshold-centered sigmoid calibration, and the "two-source fusion" is effectively single-source (data only). Additionally, two baselines appear broken (Isolation Forest F1=0.000, HST F1=0.532), hyperparameter sensitivity analysis is still absent, and the manuscript tables have not yet been updated with the new results. These issues need attention but the foundation is now sound.

---

## STATISTICIAN_001 ISSUE RESOLUTION STATUS

| Issue | Severity | Status | Notes |
|-------|----------|--------|-------|
| CRITICAL #1: Test-set threshold leakage | CRITICAL | **FIXED** | `select_threshold()` uses validation scores only; `evaluate_at_threshold()` takes a pre-determined threshold |
| CRITICAL #2: No repeated experiments | CRITICAL | **FIXED** | 5 seeds [42,123,456,789,2024]; `set_all_seeds()` controls numpy/torch/random/cuda |
| CRITICAL #3: No significance testing | CRITICAL | **FIXED** | McNemar's test with continuity correction implemented |
| HIGH #1: Fusion underperforms components | HIGH | **FIXED** | 2-source DS (PINN+GAT-LSTM) now beats both: F1=0.895 vs 0.887/0.790 on seed 42 |
| HIGH #2: HST circular validation | HIGH | **IMPROVED** | Acknowledged in text; HST advantage reduced with proper threshold selection |
| HIGH #3: Unrealistic anomaly rate (~41%) | HIGH | **FIXED** | Anomaly rate is now 6.8% (75/1096 days); event durations 10-30 days |
| HIGH #4: No hyperparameter sensitivity | HIGH | **NOT ADDRESSED** | Still single configuration |
| HIGH #5: No convergence studies | HIGH | **PARTIALLY ADDRESSED** | Text mentions "converges by epoch 100" but no learning curves shown |
| MEDIUM #1: LSTM-AE handicapped | MEDIUM | **FIXED** | LSTM-AE now uses seq_len=30, epochs=50, batch_size=64 (independent) |
| MEDIUM #2: Day-level vs event-level | MEDIUM | **FIXED** | `compute_event_metrics()` with proper per-event detection and lead time |
| MEDIUM #3: Lead time for undetected events | MEDIUM | **FIXED** | Undetected events get `lead_time=None`, excluded from mean |
| MEDIUM #4: Ablation confound (weight redistribution) | MEDIUM | **FIXED** | 2-source framework: ablations zero one source, keep weights at 0.50/0.50 |
| MEDIUM #5: Knowledge score double-scaling | MEDIUM | **FIXED** | Knowledge removed from fusion entirely; used for diagnostic labeling only |
| MEDIUM #6: No reproducibility controls | MEDIUM | **FIXED** | `set_all_seeds()` covers all random sources |

**Resolution rate: 11 of 14 items addressed (79%).** All three critical issues fixed. This is excellent responsiveness.

---

## NEW/REMAINING ISSUES

### CRITICAL #1 (NEW): Physics Layer Contributes Nothing to Fusion

**Evidence from seed 42 results:**

| Configuration | F1 | Precision | Recall | FAR | ROC-AUC |
|---------------|-----|-----------|--------|-----|---------|
| Full framework (PINN + GAT-LSTM) | 0.895 | 0.941 | 0.853 | 0.004 | 0.944 |
| No Physics (GAT-LSTM only) | 0.895 | 0.941 | 0.853 | 0.004 | 0.948 |
| No Data (PINN only) | 0.790 | 1.000 | 0.653 | 0.000 | 0.843 |

The full framework and "No Physics" ablation produce **identical** F1, precision, recall, and FAR. The ROC-AUC is actually slightly higher without physics (0.948 vs 0.944). This means the PINN's anomaly scores, after sigmoid calibration, produce near-zero belief -- even during genuine anomalies. The fusion is effectively single-source.

**Root cause:** The threshold-centered sigmoid calibration sets the center at `max(P99.9, mean+5*sigma)` of validation PINN scores (`bayesian_fusion.py:57-59`). The validation period is anomaly-free, so the center is positioned at the extreme tail of normal PINN scores. If anomaly PINN scores in the test period do not substantially exceed this extreme-tail center, the sigmoid produces near-zero anomaly belief. The PINN contributes only `m(Theta) ~= 1.0` (complete uncertainty), making it vacuous in the DS combination.

The diagnostic evidence: `val max=0.0073` for fused scores during seed 42. This extremely low value confirms that the PINN signal is being suppressed -- if the PINN were contributing meaningful belief, the fused validation scores would be higher.

**Why this matters:** If the "No Physics" ablation equals the "Full" result, the paper cannot claim that "removing the physics layer increases FAR from 4.5% to 31.7%" (as stated in the abstract and Section 4.4). The ablation results in the manuscript (Table 2) come from the OLD experiment. With the new validation-based calibration, the physics layer may be invisible to the fusion.

**What must be done:**
1. After the 5-seed experiment completes, compare the "Full" and "No Physics" ablation results. If they remain identical (or near-identical) across all 5 seeds, the calibration must be adjusted.
2. Possible fixes: (a) use a less extreme sigmoid center (e.g., P95 or mean+3*sigma instead of P99.9/mean+5*sigma); (b) compute sigmoid centers on a per-source basis with source-appropriate aggressiveness; (c) report the actual calibration centers and compare them to the range of anomaly test scores to verify the sigmoid is operating in a discriminative region.
3. If the physics layer truly cannot produce scores far enough above the normal tail (i.e., the PINN's anomaly/normal score separation is weak), this is a fundamental model limitation that should be investigated and documented.

### HIGH #1: Isolation Forest Baseline Broken (F1 = 0.000)

From the seed 42 experiment log:
```
Isolation Forest: F1=0.000, Prec=0.000, Rec=0.000, FAR=0.000, AUC=0.756, Events=0/4
```

The Isolation Forest detects zero events and has zero recall. It is completely non-functional with the validation-based threshold. The `contamination=0.05` parameter tells sklearn to flag ~5% of training data as anomalous, but the validation-based threshold (`max(P99.9, mean+5*sigma)` of validation IF scores) is likely so high that no test point exceeds it.

ROC-AUC = 0.756 confirms that the Isolation Forest scores have some discriminative power, but the fixed threshold is too extreme for this method's score distribution.

**What must be done:** The Isolation Forest's score distribution is fundamentally different from the other methods (sklearn returns negative anomaly scores). The universal `select_threshold()` function may not be appropriate. Either:
- (a) Use the Isolation Forest's own contamination-based threshold (its designed operating point), or
- (b) Apply the same `select_threshold()` logic but verify that the resulting threshold produces at least some positive predictions on the validation set.

A baseline that detects 0/4 events cannot appear in the results table without comment. If the threshold methodology is inherently incompatible with IF, this should be noted.

### HIGH #2: HST Performance Collapse -- Investigate Before Reporting

HST now shows F1=0.532 with Prec=0.372 (seed 42). Previously it was F1=0.958. The massive drop is partly explained by proper threshold selection (no more test-set leakage), but the numbers suggest the validation-based threshold is poorly suited for HST. With Recall=0.933 but Precision=0.372, HST is triggering many false alarms.

Possible explanation: HST's validation-period residuals have a heavy tail (due to seasonal patterns not perfectly captured by the polynomial), so `max(P99.9, mean+5*sigma)` gives a threshold that is still below the anomaly-period residuals but also below many normal-period residuals in the test set.

**What must be done:** Investigate whether the HST threshold is appropriate. If the validation distribution is non-Gaussian (likely for HST residuals), a percentile-based threshold may be more suitable than a sigma-based one. At minimum, report the selected threshold values for each method to enable this analysis. If HST's F1 remains 0.5 across all seeds, the manuscript discussion of HST's "structural advantage" needs major revision -- the advantage appears to have been entirely a threshold-leakage artifact.

### HIGH #3: No Hyperparameter Sensitivity Analysis (Still Open)

From STATISTICIAN_001 HIGH #4. The paper still reports results for a single set of hyperparameters. With the new validation set, sensitivity analysis is now straightforward: vary a hyperparameter, retrain, evaluate on validation, and report.

Key parameters requiring analysis:
1. **PINN physics weight alpha** (currently 0.1): Does 0.01 or 1.0 change detection quality?
2. **DS sigmoid center** (currently P99.9/mean+5*sigma): Given CRITICAL #1, this is the most consequential parameter. What happens with P99/mean+3*sigma or P95/mean+2*sigma?
3. **GAT-LSTM sequence length** (currently 14): Does 7, 30, or 60 improve detection?
4. **Fused threshold floor** (currently 0.10): This is applied ad-hoc at `run_experiments.py:350`. How sensitive are results to 0.05 or 0.20?

**What must be done:** After the 5-seed experiment completes, run at least a 1D sweep over the top 2-3 most influential parameters. The DS sigmoid center is the most urgent given CRITICAL #1.

### HIGH #4: Convergence Studies Incomplete (Still Open)

From STATISTICIAN_001 HIGH #5. The manuscript now states "training loss converges by epoch 100 for all sensor types" (main.tex:276), which is a step forward. However, no learning curves are shown. The seed 42 log confirms reasonable loss values:
```
[displacement] Epoch 100: data=0.201, physics=0.488
[displacement] Epoch 200: data=0.202, physics=0.454
```

The data loss is essentially flat between epoch 100 and 200 (0.201 vs 0.202), supporting the convergence claim. But the physics loss actually increased slightly (0.454 vs 0.488 at epoch 100, then back down). This suggests potential instability or that the physics and data losses are competing.

**What must be done:** Report a convergence table or figure showing loss at epochs 50, 100, 150, 200 for each sensor type. This is a small addition (the data is already being printed during training) but substantiates the convergence claim.

---

## MEDIUM-PRIORITY ISSUES

### MEDIUM #1: Manuscript Tables Not Yet Updated

Tables 1 and 2 in the manuscript still show old single-run values (F1=0.857 for proposed, 0.958 for HST). The progress.md correctly flags these as "placeholder" values. Once the 5-seed experiment completes, all table entries must be updated with mean +/- std.

**Important:** The new results will look very different from the old ones. The abstract, results discussion, ablation discussion, and conclusions all reference specific numbers that will change. A systematic pass through the manuscript is needed after the experiment completes.

### MEDIUM #2: Fused Threshold Floor (0.10) Is Ad-Hoc

`run_experiments.py:350`:
```python
fused_thresh = float(max(np.max(fused_val_scores) * 1.1, 0.10))
```

For seed 42, the validation max fused score is 0.0073 (essentially zero). The threshold defaults to the floor of 0.10. This 0.10 value is arbitrary and not justified anywhere. The `* 1.1` multiplier is also ad-hoc.

With a different floor (e.g., 0.05), the proposed framework would produce different F1/precision/recall. This is a hidden degree of freedom.

**What should be done:** Either justify the 0.10 floor statistically (e.g., it corresponds to a specific false alarm rate on the validation set) or remove it and use the standard `max(P99.9, mean+5*sigma)` formula consistently. If the standard formula produces an impractically low threshold, that itself is informative -- it means the fused scores have very good normal/anomaly separation.

### MEDIUM #3: Knowledge Layer Standalone Evaluation Changed

The knowledge layer is now evaluated with its own validation-based threshold (seed 42: F1=0.547, which is higher than the old F1=0.445). However, the knowledge layer's raw scores are in [0,1] and the validation threshold may not be appropriate for this score range. The old 5.0x scaling has been removed for standalone evaluation.

This is actually a positive change -- the knowledge layer is now evaluated on its own terms. But the F1=0.547 should be contextualized: with the knowledge layer repositioned as a diagnostic tool (not detection), its standalone F1 is less relevant than its failure-mode classification accuracy. The paper should report classification accuracy for the 4 anomaly events (did the knowledge layer correctly identify seepage, settlement, cracking, and uplift?).

### MEDIUM #4: McNemar Test Uses Only Seed 0 Predictions

`run_experiments.py:563-564`:
```python
seed0 = all_seed_results[0]
proposed_pred = seed0["Proposed Framework"]["y_pred"]
```

The McNemar test uses predictions from seed 0 only, not aggregated across seeds. This is reasonable for a single comparison but doesn't capture seed-to-seed variability in the significance result. Ideally, report McNemar results for each seed and note consistency (or lack thereof).

For the deterministic baselines, the comparison is seed-dependent on one side (the proposed method's predictions change per seed) and fixed on the other (baseline predictions are always the same). This asymmetry is fine methodologically but should be acknowledged.

---

## POSITIVE CHANGES TO ACKNOWLEDGE

The following improvements reflect strong statistical methodology:

1. **Three-way split (50/20/30):** Clean separation of training, threshold selection, and evaluation. The validation period (years 6-7) is anomaly-free by construction, which is appropriate for threshold calibration.

2. **Validation-based threshold selection:** `select_threshold()` uses only validation scores. No test labels are accessed during threshold determination. The `max(P99.9, mean+5*sigma)` rule is conservative and principled.

3. **Multi-seed with comprehensive seed control:** `set_all_seeds()` covers `random`, `numpy`, `torch`, and `torch.cuda`. The `aggregate_multi_seed()` function correctly computes per-metric mean and std.

4. **McNemar's test:** Correct implementation with continuity correction. Applied to all relevant pairwise comparisons. The chi-squared approximation is appropriate for the sample sizes involved (~1000 test days).

5. **Event-level metrics:** `compute_event_metrics()` properly tracks per-event detection, lead time for detected events only, and event type. Missed events are marked as `detected=False` with `lead_time=None`. This is exactly the right approach.

6. **2-source fusion architecture:** The decision to remove the knowledge layer from the DS fusion and reposition it as a diagnostic overlay is statistically well-motivated. The knowledge layer's scores were anti-informative for detection; including them in the fusion was equivalent to adding noise. The diagnostic role (failure-mode classification) is a legitimate and useful contribution that doesn't require detection-level performance.

7. **Realistic anomaly rate (6.8%):** Four events totaling 75 days out of 1096 test days. While still higher than real-world dam monitoring (~0.01%), this is within the range used in published SHM papers and reasonable for a proof-of-concept study.

8. **LSTM-AE independent tuning:** `train_lstm_ae()` now uses `seq_len=30, epochs=50, batch_size=64`, separate from the GAT-LSTM's configuration. This is fair baseline practice.

---

## SCORING RATIONALE

| Criterion | Weight | Score | Change | Notes |
|-----------|--------|-------|--------|-------|
| Methodology | 50% | 7/10 | +4 | Proper split, validation thresholds, multi-seed, McNemar -- all major flaws fixed |
| Results | 30% | 4/10 | +1 | Fusion beats components (seed 42), but physics layer vacuous; tables outdated |
| Experimental Design | 20% | 5/10 | +1 | Realistic anomaly rate, event metrics; still no sensitivity/convergence study |

**Weighted: 0.50*7 + 0.30*4 + 0.20*5 = 3.5 + 1.2 + 1.0 = 5.7**

The statistical methodology is now fundamentally sound. The three critical flaws (threshold leakage, no repeated experiments, no significance testing) have been properly corrected. The 2-source fusion with validation-based calibration beats its individual components on seed 42. However, the physics layer appears vacuous in the fusion (CRITICAL #1), two baselines are broken, hyperparameter sensitivity is still missing, and the manuscript tables need updating. Once the 5-seed experiment completes and the new critical issue is resolved, the score will rise substantially.

As Fisher would say: "The best time to plan an experiment is after you've done it." The first experiment revealed deep problems. This second iteration has corrected the experimental design. What remains is to ensure the new results are internally consistent and that every claimed contribution is supported by the data.

---

## ACTIONABLE ITEMS (Priority Order)

1. **[CRITICAL]** After 5-seed experiment completes, verify whether "Full" = "No Physics" across all seeds. If confirmed, the sigmoid calibration is suppressing the PINN and must be adjusted (see CRITICAL #1).
2. **[HIGH]** Fix Isolation Forest baseline (F1=0.000). The validation-based threshold is incompatible with IF's score distribution. Use IF's native contamination threshold or a method-appropriate calibration.
3. **[HIGH]** Investigate HST's performance collapse (F1=0.958 -> 0.532). Determine whether this is a proper correction (the old result was inflated) or an over-correction (the new threshold is inappropriate for HST). Report transparently either way.
4. **[HIGH]** Run sensitivity analysis on DS sigmoid center parameter (the most consequential hyperparameter given CRITICAL #1). Test at least P95/P99/P99.9 and mean+3/4/5*sigma.
5. **[HIGH]** Add convergence evidence: table of loss values at epochs 50, 100, 150, 200 per sensor type.
6. **[MEDIUM]** Update all manuscript tables with 5-seed mean +/- std once experiment completes. Systematic pass through text to update all referenced numbers.
7. **[MEDIUM]** Justify or remove the 0.10 fused threshold floor. Report all method-specific threshold values.
8. **[MEDIUM]** Report knowledge layer's failure-mode classification accuracy per event (not just standalone F1).
9. **[MEDIUM]** Note that McNemar tests use seed-0 predictions only; consider reporting consistency across seeds.

---

**Score: 6/10**
