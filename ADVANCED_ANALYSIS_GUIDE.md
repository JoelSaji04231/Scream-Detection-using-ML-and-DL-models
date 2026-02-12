# Advanced Model Analysis Suite

## Overview

This document describes the comprehensive advanced analysis tools added to your Scream Detection project. These tools provide statistical significance testing, ablation studies, cross-validation, error analysis, and resource consumption monitoring.

---

## Quick Start: Running the Analyses

### 1. **Statistical Significance Testing & Model Comparison**
```bash
python compare_models_advanced.py
```
**Outputs:**
- `svm_cnn_comparison_YYYYMMDD_HHMMSS.json` - Detailed comparison metrics
- `svm_cnn_comparison_report_YYYYMMDD_HHMMSS.txt` - Statistical test results

**What it does:**
- ✓ Paired t-test comparing SVM vs CNN accuracy
- ✓ McNemar's test for error pattern differences
- ✓ Two-proportion z-tests for FPR and FNR
- ✓ Cohen's Kappa agreement analysis
- ✓ Side-by-side performance comparison

---

### 2. **10-Fold Cross-Validation**
```bash
python cross_validation_analysis.py
```
**Outputs:**
- `cv_report_SVM_YYYYMMDD_HHMMSS.txt` - Detailed CV statistics
- `cv_summary_YYYYMMDD_HHMMSS.csv` - Summary table

**What it does:**
- ✓ Stratified k-fold CV with configurable folds
- ✓ Mean Accuracy ± Std Dev for stability assessment
- ✓ Train-test gap analysis (overfitting detection)
- ✓ Confidence intervals for all metrics
- ✓ Fold-by-fold breakdown

**Key Metrics:**
- Mean test accuracy with standard deviation
- Train-test gap showing potential overfitting
- Min/max scores across folds
- 95% confidence intervals

---

### 3. **Comprehensive Ablation Study**
```bash
python ablation_study.py
```
**Outputs:**
- `ablation_report_SVM_YYYYMMDD_HHMMSS.txt` - Detailed report
- `ablation_results_SVM_YYYYMMDD_HHMMSS.csv` - Feature results
- `feature_importance_YYYYMMDD_HHMMSS.png` - Visualization

**What it does:**
- ✓ Leave-one-out feature ablation
- ✓ Feature group removal analysis
- ✓ Permutation-based importance
- ✓ Pairwise feature interactions
- ✓ Identifies critical vs negligible features

**Interpretation Guide:**
- **Importance > 0.05**: Critical feature - removing causes significant accuracy drop
- **Importance 0.01-0.05**: Helpful feature - provides measurable contribution
- **Importance < 0.01**: Negligible - minimal impact on accuracy

---

### 4. **Comprehensive Model Analysis**
```bash
python model_analysis.py
```
**Outputs:**
- `analysis_results_YYYYMMDD_HHMMSS.json` - All metrics in JSON
- `analysis_report_YYYYMMDD_HHMMSS.txt` - Summary report

**Includes:**
- All cross-validation metrics
- Statistical significance tests
- Ablation study results
- Error pattern analysis
- Resource consumption metrics

---

## Detailed Feature Descriptions

### A. STATISTICAL SIGNIFICANCE TESTING

#### Paired t-test
Tests if the difference between two models' accuracy is statistically significant.

**Null Hypothesis (H₀):** Both models have equal accuracy
**Alternative (H₁):** Models have different accuracy

**Interpretation:**
- `p-value < 0.05`: Significant difference (reject H₀)
- `p-value ≥ 0.05`: No significant difference (fail to reject H₀)

**Example Output:**
```
PAIRED T-TEST: Comparing Model Accuracy
SVM Accuracy:  0.8750
CNN Accuracy:  0.8920
Difference:    +0.0170
t-statistic: 2.1450
p-value:     0.0382
✓ RESULT: REJECT null hypothesis (p < 0.05)
  There IS a statistically significant difference
  CNN is significantly better
```

#### McNemar's Test
Tests if two models make significantly different types of errors.

**Formula:** χ² = (a - b)² / (a + b)

Where:
- `a`: # cases SVM correct, CNN wrong
- `b`: # cases CNN correct, SVM wrong

**Interpretation:**
- `p-value < 0.05`: Models make different errors
- `p-value ≥ 0.05`: Models make similar errors

---

### B. CROSS-VALIDATION ANALYSIS

#### 10-Fold Stratified Cross-Validation
Divides data into 10 folds while preserving class distribution.

**Reported Metrics:**
```
Accuracy:  0.8520 ± 0.0340
Precision: 0.8640 ± 0.0295
Recall:    0.8310 ± 0.0410
F1-Score:  0.8470 ± 0.0340
ROC-AUC:   0.9120 ± 0.0220
```

**Train-Test Gap Analysis:**
```
Metric         Gap      Status
Accuracy      +0.0150  GOOD (< 5%)
Precision     +0.0220  GOOD
Recall        +0.0180  GOOD
```

**Interpretation:**
- Gap `< 0.05`: Model generalizes well
- Gap `0.05-0.10`: Moderate overfitting
- Gap `> 0.10`: Significant overfitting

---

### C. ABLATION STUDY

#### Feature Importance Analysis
Removes each feature individually and measures accuracy drop.

**Key Insights:**
1. **What it shows:** How much each feature contributes to accuracy
2. **Why it matters:** Identifies which features are truly valuable
3. **Application:** Feature selection, cost reduction, deployment optimization

**Example Results:**
```
Feature                  Accuracy    Drop        Status
mfcc_1                   0.8410      0.0110      ✓ CRITICAL
spectral_centroid        0.8390      0.0130      ✓ CRITICAL
zero_crossing_rate       0.8490      0.0030      • HELPFUL
rms_energy              0.8490      0.0020      • negligible
```

#### Feature Group Analysis
Removes categories of related features to understand their collective impact.

**Groups Analyzed:**
- MFCC coefficients
- Spectral features (centroid, rolloff, bandwidth)
- Zero Crossing Rate (ZCR)
- RMS Energy
- Chroma features
- Temporal features (deltas)
- Contrast features

**Use Case:** If removing all MFCC features causes large accuracy drop, MFCCs are collectively important even if individual MFCCs aren't.

#### Permutation Importance
Random shuffling of feature values to measure importance without retraining.

**Advantages:**
- Model-agnostic (works with any model)
- Measures real-world feature importance
- Accounts for feature interactions

---

### D. ERROR PATTERN ANALYSIS

#### Confusion Matrix Breakdown
```
True Negatives (TN):  127  (correctly identified non-screams)
False Positives (FP): 8    (falsely identified as scream)
False Negatives (FN): 5    (missed screams)
True Positives (TP):  98   (correctly identified screams)
```

#### Error Rates
- **FPR (False Positive Rate):** FP/(FP+TN) = 8/135 = 0.059
  - "What fraction of non-screams are falsely flagged?"
  - Lower is better for reducing false alarms

- **FNR (False Negative Rate):** FN/(FN+TP) = 5/103 = 0.049
  - "What fraction of screams are missed?"
  - Lower is better for detection completeness

#### Detailed Error Investigation
The analysis examines feature characteristics of misclassified samples:

**False Negatives (Missed Screams):**
- Typically lower amplitude or whispering screams
- May have unusual spectral characteristics
- Could be other high-pitched sounds mistaken for non-screams

**False Positives (False Alarms):**
- Might be cats, sirens, or other high-energy sounds
- Temporal patterns similar to screams
- May have similar spectral content

---

### E. RESOURCE CONSUMPTION ANALYSIS

#### CPU & Memory Monitoring
```
CPU Utilization:
  Mean: 35.42% ± 12.15%
  Max:  67.89%

Memory Utilization:
  Mean: 42.15% ± 8.32%
  Max:  58.92%
```

#### GPU Monitoring (if available)
```
GPU Information:
  Device: NVIDIA GeForce RTX 3060
  Total Memory: 12.00 GB
  
Allocated Memory: 4.2341 GB
Reserved Memory: 4.5000 GB
Max Memory Used: 4.2550 GB
```

#### Inference Time
```
Time per sample: 2.1450 ms
Samples per second: 466.26
Total inference time: 23.45 seconds
```

**Edge Device Considerations:**
- Mobile phones typically have <2GB RAM
- IoT devices may have <512MB RAM
- Time per sample should be <10ms for real-time processing
- Energy consumption is critical for battery-powered devices

#### Energy Estimation
```
CPU TDP: 65W
CPU Utilization: 35.42%
Estimated Power: 23.02W
Estimated Energy: 538.54 Joules (0.149780 Wh)
```

**Energy Budget Examples:**
- IoT sensor (100mAh @ 3.7V): 0.37 Wh
- Mobile phone battery (3000mAh @ 3.7V): 11.1 Wh
- Your model uses: 0.15 Wh per inference

**Cost Impact:**
- \$0.015 per 1000 inferences (@ \$0.10/kWh)
- \$0.45 per 30,000 inferences
- Optimization can save 50-70% with quantization

---

## Understanding the Reports

### Cross-Validation Report
```
┌─────────────────────────────────────────────────────┐
│ 10-FOLD STRATIFIED CROSS-VALIDATION REPORT         │
├─────────────────────────────────────────────────────┤
│ Accuracy:  0.8520 ± 0.0340                         │
│ Precision: 0.8640 ± 0.0295                         │
│ Recall:    0.8310 ± 0.0410                         │
│ F1-Score:  0.8470 ± 0.0340                         │
├─────────────────────────────────────────────────────┤
│ Train-Test Gap Analysis:                           │
│   Accuracy Drop: 0.0150  (GOOD: < 5%)              │
│   Indicates the model generalizes well             │
└─────────────────────────────────────────────────────┘
```

### Ablation Study Report
```
Top Features Contributing to Accuracy:
1. MFCC_1         (importance: 0.0340)  CRITICAL
2. RMS_Energy     (importance: 0.0280)  CRITICAL
3. ZCR_Mean       (importance: 0.0190)  CRITICAL
4. Chromatic_1    (importance: 0.0090)  HELPFUL
...

Recommendation: Focus on MFCC, RMS, and ZCR features
```

### Statistical Significance Report
```
Comparing SVM vs CNN:
├─ Paired t-test
│  ├─ SVM Accuracy: 87.50%
│  ├─ CNN Accuracy: 89.20%
│  └─ p-value: 0.0382 ✓ SIGNIFICANT
├─ McNemar's Test
│  ├─ Error patterns differ: YES
│  └─ p-value: 0.0156 ✓ SIGNIFICANT
└─ Conclusion: CNN is statistically significantly better
```

---

## Integration with Your Workflow

### Recommendation Order:
1. **Start:** `python cross_validation_analysis.py`
   - Understand baseline model stability

2. **Then:** `compare_models_advanced.py`
   - See if CNN is truly better than SVM
   - Statistical proof replaces opinion

3. **Then:** `ablation_study.py`
   - Which features matter most?
   - Can you reduce computational cost?

4. **Then:** `model_analysis.py`
   - Complete analysis with everything
   - Generate summary report

5. **Finally:** Deploy with confidence
   - You have statistical evidence of performance
   - You know resource requirements
   - You understand error modes

---

## Advanced Usage

### Running Custom Cross-Validation
```python
from cross_validation_analysis import CrossValidationAnalysis

cv = CrossValidationAnalysis(
    X=X_train,
    y=y_train,
    model=your_model,
    cv_folds=5,  # Instead of 10
    stratified=True
)
results = cv.run_cross_validation()
```

### Running Ablation on Specific Features
```python
from ablation_study import FeatureAblationStudy

subset_features = feature_names[:20]  # Only first 20 features
ablation = FeatureAblationStudy(
    X=X_train[:, :20],
    y=y_train,
    model=your_model,
    feature_names=subset_features
)
results = ablation.ablate_single_features()
```

### Custom Statistical Test
```python
from compare_models_advanced import ModelComparison
from scipy import stats

comparison = ModelComparison(
    y_true=y_test,
    svm_preds=svm_predictions,
    svm_probas=svm_probabilities,
    cnn_preds=cnn_predictions,
    cnn_probas=cnn_probabilities
)

# Run paired t-test
t_test = comparison.paired_t_test()

# Run McNemar's test
mcnemar = comparison.mcnemar_test()
```

---

## Output Files Explained

| File | Format | Contents |
|------|--------|----------|
| `cv_report_*.txt` | Text | Cross-validation statistics |
| `cv_summary_*.csv` | CSV | Summary table all models |
| `svm_cnn_comparison_*.json` | JSON | All metrics from comparison |
| `ablation_report_*.txt` | Text | Feature importance breakdown |
| `ablation_results_*.csv` | CSV | Feature-by-feature results |
| `feature_importance_*.png` | PNG | Bar chart of top features |
| `analysis_results_*.json` | JSON | Complete analysis results |
| `analysis_report_*.txt` | Text | Final summary report |

---

## Interpreting Key Metrics

### Accuracy Metrics
- **Accuracy:** Overall correctness (useful when classes balanced)
- **Precision:** Of predicted screams, how many are real? (avoid false alarms)
- **Recall:** Of real screams, how many detected? (avoid missed detections)
- **F1-Score:** Harmonic mean of precision and recall

**Trade-off:** Increasing recall (catching all screams) often decreases precision (more false alarms)

### Effect Sizes
- **|t-statistic| > 2.0:** Moderate effect
- **|t-statistic| > 3.0:** Strong effect
- **p-value < 0.01:** Very strong evidence
- **p-value < 0.05:** Strong evidence (standard threshold)

### Cross-Validation Interpretation
- **Std Dev < 0.03:** Very stable model
- **Std Dev 0.03-0.05:** Stable model
- **Std Dev > 0.05:** Unstable (may need more data)

---

## Troubleshooting

### "File not found: train_test_split_esc50.pkl"
**Solution:** Run `python train_ml.py` first to generate training data splits

### "SVM model not found"
**Solution:** Run `python train_ml.py` first to train the SVM model

### "CNN model not found"
**Solution:** Run `python train_cnn.py` first to train the CNN model

### High memory usage during ablation
**Solution:** Use fewer CV folds: `cv_folds=3` instead of `cv_folds=5`

### Slow cross-validation
**Solution:** Use parallel processing:
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)  # Use all cores
```

---

## Citations & References

**Statistical Tests Implemented:**
- Student's t-test: Paired comparison of means
- McNemar's test: Paired categorical data comparison
- Two-proportion z-test: Comparing error rates
- Cohen's Kappa: Inter-rater agreement measure

**Cross-Validation:**
- Stratified K-Fold: Maintains class distribution in each fold
- 10-fold standard: Reduces variance while maintaining reasonable computational cost

**Feature Importance:**
- Leave-one-out: Direct measurement of feature contribution
- Permutation importance: Model-agnostic metric
- Interaction analysis: Reveals feature synergies

---

## Best Practices

1. **Always use stratified CV** when classes are imbalanced
2. **Report mean ± std dev** rather than single test set accuracy
3. **Test statistical significance** before claiming one model is "better"
4. **Interpret p-values correctly**
   - p < 0.05 means statistically significant (not practically significant)
   - Always consider effect size and practical importance
5. **Use cross-validation for feature selection** to avoid overfitting
6. **Consider computational cost** in production deployment
7. **Monitor resource consumption** for edge devices
8. **Validate findings** on separate independent test set

---

## Questions?

For any questions about these analysis tools, review the code comments in:
- `compare_models_advanced.py` - Statistical significance
- `cross_validation_analysis.py` - CV methodology
- `ablation_study.py` - Feature importance
- `model_analysis.py` - Complete integration

Each file has detailed docstrings and inline comments explaining the methodology.
