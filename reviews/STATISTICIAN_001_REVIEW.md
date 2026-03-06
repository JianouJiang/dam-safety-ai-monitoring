# STATISTICIAN REVIEW 001

**Reviewer:** Statistician-Fisher (Ronald Fisher)
**Date:** 2026-03-05
**Paper:** Physics-Informed Anomaly Detection for Dam Safety: A Hybrid Data-Mechanism-Knowledge Framework
**Status:** Full draft complete with experiments run

---

## Summary

The paper presents a three-layer hybrid anomaly detection framework (PINN + GAT-LSTM + Knowledge + Dempster-Shafer fusion) validated on synthetic dam monitoring data. Experiments have been run and a complete manuscript exists. However, the experimental methodology contains **several critical statistical flaws** that invalidate or severely weaken the reported results. The most damaging: threshold optimization on the test set (data leakage), absence of any confidence intervals or significance testing, single-run stochastic experiments with no repetition, and a fusion framework that underperforms its own components without statistical justification. These are not peripheral issues -- they strike at the core of every quantitative claim in the paper.

---

## CRITICAL ISSUES

### CRITICAL #1: Test-Set Threshold Optimization (Data Leakage)

**File:** `codes/models/run_experiments.py:82-99`

This is the single most damaging statistical error in the paper. The `evaluate_method` function selects the detection threshold that **maximizes F1 on the test set itself**:

```python
# run_experiments.py:86-99
score_range = np.linspace(np.percentile(scores, 50), np.percentile(scores, 99.5), 50)
percentile_thresholds = np.percentile(scores[y_binary == 0], [80, 85, 90, 92, 95, 97, 99])
all_thresholds = np.unique(np.concatenate([score_range, percentile_thresholds]))
for thresh in all_thresholds:
    y_pred = (scores > thresh).astype(int)
    f1 = f1_score(y_binary, y_pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
```

This searches ~57 thresholds and picks the one giving the highest F1 **using the test labels**. Every reported F1, precision, and recall is inflated by this optimization. The percentile thresholds at line 88 explicitly condition on normal-class test samples (`scores[y_binary == 0]`), which is direct label leakage.

**What should be done:** Determine the threshold on a held-out **validation set** (e.g., split the 7-year training period into 5 years training + 2 years validation with synthetic anomaly injections), then evaluate on the untouched test set at that fixed threshold. Alternatively, report metrics that are threshold-independent (ROC-AUC, PR-AUC) as primary metrics and use a principled threshold rule (e.g., 95th percentile of training scores, or Youden's J on validation data).

**Impact:** ALL threshold-dependent metrics in Table 1 and Table 2 are unreliable. The ROC-AUC and PR-AUC values are the only metrics not affected by this issue.

### CRITICAL #2: No Confidence Intervals, No Repeated Experiments

Every result in the paper is a single-run point estimate. The PINN, GAT-LSTM, and LSTM-AE are all stochastic models (random weight initialization, random mini-batch sampling, random subset selection in PINN training). The data generation itself uses `np.random.seed(42)` (`generate_dam_data.py:28`), but PyTorch seeds are **not set** anywhere in the training code.

- No repeated runs with different random seeds
- No bootstrap confidence intervals on any metric
- No standard errors reported
- No way to assess whether F1 = 0.857 vs F1 = 0.876 is a real difference or random variation

A single stochastic run can easily vary by 2-5% in F1 depending on initialization. The claimed differences between methods (e.g., DMK 0.857 vs GAT-LSTM 0.876) are well within this noise band.

**What should be done:** Run all stochastic methods at least 5 times (preferably 10) with different random seeds. Report mean and standard deviation for all metrics. Report confidence intervals (e.g., bootstrap 95% CI on F1 differences).

### CRITICAL #3: No Significance Testing for Method Comparisons

The paper compares 11 methods and draws conclusions such as "the physics constraints are essential" and "the GAT-LSTM captures temporal patterns that the physics and knowledge layers miss." None of these claims are supported by any statistical test.

Required tests:
- **McNemar's test** or **paired bootstrap** for pairwise comparison of methods (e.g., proposed vs LSTM-AE)
- **Friedman test** with post-hoc Nemenyi test if comparing multiple methods simultaneously
- **DeLong test** for comparing ROC-AUC values between methods

Without these, the difference between the proposed framework (F1 = 0.857) and the LSTM-AE baseline (F1 = 0.921) cannot be established as statistically meaningful -- and in this case, the baseline is *winning*.

---

## HIGH-PRIORITY ISSUES

### HIGH #1: Proposed Framework Underperforms Components -- The Central Statistical Contradiction

The most embarrassing result in the paper, from a statistical standpoint:

| Method | F1 | ROC-AUC |
|--------|-----|---------|
| PINN (Physics only) | 0.863 | 0.924 |
| GAT-LSTM (Data only) | 0.876 | 0.952 |
| **Proposed (DMK fusion)** | **0.857** | **0.935** |

The fused framework scores **lower** on F1 than both of its main components individually. The DS fusion is actively degrading performance. The paper's abstract claims F1 = 0.863 and 0.876 for components and 0.857 for the whole, yet frames this as a contribution. This is the opposite of what a fusion framework should achieve.

The paper attempts to justify this by arguing for "interpretability" and "low false alarm rate," but the FAR is also worse (0.045 vs 0.031 for both components individually). The only metric where fusion might help is the Belief-Plausibility interval, but this is never quantitatively evaluated against alternatives.

**What should be done:** Either (a) fix the fusion so it actually improves performance (likely a calibration issue with the knowledge layer dragging down the combination), or (b) perform rigorous statistical testing to determine whether the F1 difference (0.857 vs 0.876) is significant, and if so, honestly acknowledge that fusion hurts aggregate performance on this dataset while providing other benefits. Do not present an underperforming fusion as a contribution without transparently confronting this result.

### HIGH #2: Circular Validation with HST Baseline

The synthetic data is generated using an HST-family model (`generate_dam_data.py:118-152`). The HST baseline (`baselines.py:38-84`) fits the same functional form. This gives the HST baseline a structural advantage -- it is being evaluated on data drawn from its own model class.

The HST result (F1 = 0.958, ROC-AUC = 0.982) is therefore not a meaningful benchmark. The paper acknowledges this in Section 4.4, but the HST result still appears in Table 1 with the best values bolded, creating a misleading visual comparison. A reader scanning the table will conclude the proposed method loses badly to a simple baseline.

**What should be done:** Either (a) generate synthetic data from a different model class (e.g., finite element simulation, not HST polynomial), (b) exclude HST from the main results table and discuss it separately as a known-confounded comparison, or (c) add a second synthetic dataset generated by a non-HST process to demonstrate that the proposed method outperforms HST on data not from the HST model family.

### HIGH #3: Extreme Class Imbalance Mismatch

The test set anomaly rate is approximately **41%** (4 events spanning ~450 anomaly days out of ~1096 test days). This is unrealistic for dam safety monitoring, where genuine anomalies represent < 0.1% of operating time. The high anomaly rate makes the detection problem much easier than real-world conditions and inflates all detection metrics.

More importantly, the Isolation Forest is configured with `contamination=0.05` (`baselines.py:90`), expecting 5% anomalies. With the true rate at 41%, this severely handicaps the Isolation Forest baseline. Similarly, the threshold method uses a 3-sigma rule calibrated on normal data, which is appropriate for low anomaly rates but suboptimal for 41%.

**What should be done:** (a) Reduce anomaly event durations to achieve a more realistic anomaly rate (1-5%), (b) configure baselines with contamination/thresholds appropriate for the actual anomaly rate, or (c) at minimum, report results at multiple anomaly rate levels by varying the evaluation window.

### HIGH #4: No Hyperparameter Sensitivity Analysis

The paper reports results for a single set of hyperparameters. Critical hyperparameters that are never varied:

| Parameter | Value | Code Location |
|-----------|-------|---------------|
| PINN physics weight alpha | 0.1 | `pinn_seepage.py:51` |
| DS sigmoid center c | 2.0 | `bayesian_fusion.py:62` |
| DS source reliabilities | 0.4/0.35/0.25 | `bayesian_fusion.py:37-39` |
| GAT hidden dim | 16 | `gat_lstm.py:181` |
| LSTM hidden dim | 64 | `gat_lstm.py:181` |
| Sequence length | 14 | `run_experiments.py:238` |
| Knowledge window | 30 | `run_experiments.py:253` |
| Knowledge score scaling | 5.0 | `bayesian_fusion.py:144` |
| Fuzzy thresholds (all rules) | 0.3 | `knowledge_layer.py:37-43` |

Were these values tuned? If so, on what data? If they were tuned on the test set (even informally), this is additional leakage. If not tuned, how sensitive are the results?

**What should be done:** Conduct sensitivity analysis for at least the top 3-4 most influential hyperparameters (alpha, c, source reliabilities, knowledge scaling). Report F1/AUC as a function of each parameter.

### HIGH #5: No Convergence Studies

For any computational paper, convergence with respect to model capacity and training duration must be demonstrated:

- **PINN training convergence:** 200 epochs with 5000 random samples per epoch (`pinn_seepage.py:200`). Is 200 epochs sufficient? Is 5000/epoch a sufficient sample of the ~250,000 total training points? The paper reports no learning curves.
- **GAT-LSTM convergence:** 30 epochs. Is performance stable? No training/validation loss curves shown.
- **LSTM-AE convergence:** 30 epochs (overridden from default 80 in `run_experiments.py:329`). Why reduced from the default? Does this hurt the baseline?
- **Sequence length convergence:** seq_len=14 chosen without justification. How does performance change with seq_len in {7, 14, 30, 60}?

**What should be done:** Report learning curves for all trained models. Show F1/AUC vs epochs to confirm convergence. Test at least 2-3 values of key architectural choices (hidden dim, seq_len, n_heads).

---

## MEDIUM-PRIORITY ISSUES

### MEDIUM #1: LSTM-AE Baseline Handicapped by Configuration

The LSTM-AE baseline (`baselines.py:129-226`) is initialized with `seq_len=30, epochs=80, batch_size=64`, but the experiment runner overrides these to `seq_len=14, epochs=30, batch_size=256` (`run_experiments.py:329-330`). The LSTM-AE is forced to use the GAT-LSTM's hyperparameters rather than its own optimized settings.

For a fair comparison, each baseline should be individually tuned (or at minimum, use its own default settings). Reducing the LSTM-AE from 80 to 30 epochs and changing batch size from 64 to 256 may underfit the model, artificially depressing baseline performance.

**What should be done:** Either use each baseline's default hyperparameters or tune all methods (including baselines) using the same validation protocol.

### MEDIUM #2: Day-Level vs Event-Level Evaluation

The evaluation treats each day independently. A 180-day seepage event that is correctly detected on 160 of 180 days contributes 160 true positives and 20 false negatives. But from an operational standpoint, the relevant question is: was the event detected at all, and how quickly?

Day-level evaluation is biased toward long events (which contribute many TPs) and against short events. The thermal crack event (60 days) has much less weight than the seepage increase (180 days) in the aggregate metrics.

**What should be done:** Report both day-level and event-level metrics. For event-level: did each of the 4 events get detected? At what lead time? This would be a 4-event table, much more informative for practitioners than aggregate F1.

### MEDIUM #3: Detection Lead Time Methodology Flawed

The `compute_detection_lead_times` function (`run_experiments.py:143-162`) has a logical error: when an event is NOT detected, it records the full event duration as the "lead time":

```python
if len(detected_idx) > 0:
    lead_times.append(detected_idx[0])
else:
    lead_times.append(i - event_start)  # BUG: undetected event counted as lead time
```

For missed events, the lead time should be marked as infinity or excluded from the mean. Including the event duration pulls the mean lead time in an unpredictable direction. This makes the Knowledge Layer's "23.0 d" lead time particularly suspicious -- it likely reflects event durations rather than actual detection latency.

**What should be done:** Separate "detected" from "not detected" events. Report detection rate (fraction of events detected) and mean lead time (computed only over detected events). An undetected event is a miss, not a "late detection."

### MEDIUM #4: Ablation Study Design Confound

The ablation studies change two things simultaneously: the removed layer's scores are zeroed AND the remaining layers' reliability weights are redistributed:

- Full: physics=0.4, data=0.35, knowledge=0.25
- No Physics: data=0.55, knowledge=0.45 (weights changed)
- No Knowledge: physics=0.5, data=0.5 (weights changed)
- No Data: physics=0.5, knowledge=0.5 (weights changed)

This confounds the layer removal with the weight redistribution. A cleaner ablation would keep the original weights and simply zero the removed layer (the DS framework handles vacuous evidence naturally).

**What should be done:** Run ablations with original weights (simply zeroing the removed layer's scores, keeping reliability weights at 0.4/0.35/0.25). The vacuous source will contribute m(Theta)=1.0, which DS handles correctly.

### MEDIUM #5: Knowledge Score Double-Handling

The knowledge score undergoes two separate scaling operations:

1. In `run_experiments.py:257`: `knowledge_scores_scaled = knowledge_test_scores * 5.0` for standalone evaluation
2. In `bayesian_fusion.py:144`: `knowledge_scaled = knowledge_score * 5.0` inside the fusion

For standalone evaluation and fusion, the scaling is consistent (both use 5.0x). However, the 5.0 factor is entirely arbitrary (`run_experiments.py:257` comment: "Scale for evaluation"). This introduces a hidden degree of freedom. If changed to 3.0 or 8.0, the fusion results would change.

**What should be done:** Use Platt scaling (logistic calibration) or isotonic regression to map all methods' raw scores to a common probability scale, calibrated on a validation set.

### MEDIUM #6: No Reproducibility Controls for Stochastic Training

- `generate_dam_data.py:28`: `np.random.seed(42)` -- only NumPy seed set
- No `torch.manual_seed()` anywhere in PINN, GAT-LSTM, or LSTM-AE training
- No `torch.backends.cudnn.deterministic = True`
- No `torch.use_deterministic_algorithms(True)`
- No documentation of software versions

Results are not reproducible even with the same code. Different PyTorch initializations will give different results on each run.

**What should be done:** Set all random seeds (NumPy, PyTorch, Python random) at the start of `run_experiments.py`. Document PyTorch version. Even better: run multiple seeds to get error bars (see CRITICAL #2).

---

## ADDITIONAL OBSERVATIONS

### On the Anomaly Injection Design

The four anomaly types are injected with well-separated temporal windows (no overlap), known magnitudes, and gradually ramping signatures. Real dam anomalies are rarely this clean. The detection task on this synthetic data is substantially easier than real-world conditions because:

1. Anomalies are temporally isolated (no concurrent failure modes)
2. Magnitudes are large relative to noise (8m head increase vs 0.3m noise std)
3. Ramp profiles are smooth (sigmoid, exponential) with no intermittency
4. No missing data, sensor drift, or calibration shifts in the synthetic data

The paper should explicitly acknowledge that the reported detection performance represents an upper bound on real-world performance.

### On the ROC-AUC Values

The ROC-AUC values are the most reliable metrics in the paper (not affected by threshold leakage). Focusing on these:

| Method | ROC-AUC |
|--------|---------|
| HST | 0.982 (confounded) |
| LSTM-AE | 0.965 |
| GAT-LSTM | 0.952 |
| DMK | 0.935 |
| PINN | 0.924 |
| Threshold | 0.924 |

Even by ROC-AUC, the LSTM-AE baseline (0.965) outperforms the proposed framework (0.935) by 3 percentage points. Without significance testing, we cannot know if this is meaningful, but the direction is concerning: the proposed three-layer framework with novel fusion loses to a standard LSTM autoencoder on the paper's own evaluation.

---

## SCORING RATIONALE

| Criterion | Weight | Score | Notes |
|-----------|--------|-------|-------|
| Methodology | 50% | 3/10 | Test-set leakage invalidates F1/precision/recall; no validation split |
| Results | 30% | 3/10 | No CIs, no significance tests, fusion underperforms components |
| Experimental Design | 20% | 4/10 | Circular HST validation, unrealistic anomaly rate, no convergence study |

The experimental methodology has fundamental statistical errors (threshold leakage, no repeated runs, no significance tests) that must be corrected before any quantitative claim can be trusted. The most reliable metric (ROC-AUC) shows the proposed framework losing to a standard baseline. The fusion framework degrades rather than improves component performance. These are not minor issues -- they undermine the paper's entire quantitative narrative.

As Fisher would say: "To consult the statistician after an experiment is finished is often merely to ask him to conduct a post-mortem examination. He can perhaps say what the experiment died of." The experiment did not die, but the evaluation methodology has wounds that need urgent treatment.

---

## ACTIONABLE ITEMS (Priority Order)

1. **[CRITICAL]** Fix threshold selection: use validation set, not test set. Re-run all evaluations with fixed thresholds.
2. **[CRITICAL]** Run all stochastic methods 5-10 times with different seeds. Report mean +/- std for all metrics.
3. **[CRITICAL]** Add pairwise significance tests (McNemar or paired bootstrap) for key comparisons.
4. **[HIGH]** Address fusion underperformance: either fix it or transparently acknowledge it with statistical testing.
5. **[HIGH]** Fix circular HST validation: generate non-HST synthetic data or exclude HST from direct comparison.
6. **[HIGH]** Reduce anomaly rate to realistic levels (1-5%) and reconfigure baselines accordingly.
7. **[HIGH]** Add hyperparameter sensitivity analysis for top 4-5 parameters.
8. **[HIGH]** Add convergence studies: learning curves, F1 vs epochs, F1 vs seq_len.
9. **[MEDIUM]** Fix detection lead time for undetected events (mark as miss, not duration).
10. **[MEDIUM]** Report event-level detection results alongside day-level metrics.
11. **[MEDIUM]** Use consistent ablation design (keep original weights, zero the removed layer only).
12. **[MEDIUM]** Set all random seeds for reproducibility. Document software versions.
13. **[MEDIUM]** Replace arbitrary 5.0x knowledge scaling with principled calibration.
14. **[MEDIUM]** Tune baselines individually rather than forcing proposed method's hyperparameters.

---

**Score: 3/10**
