# Quick Start Guide: Running Advanced Analyses

## TL;DR - Get Started in 2 Minutes

### Prerequisites
Make sure your models are trained:
```bash
python train_ml.py      # Trains SVM and creates train/test split
python train_cnn.py     # Trains CNN
```

### Run All Analyses At Once
```bash
python master_analysis.py all
```

Or use interactive menu:
```bash
python master_analysis.py
```

---

## What Gets Generated?

After running all analyses, you'll get these files in your `models/` directory:

### Cross-Validation Results
- `cv_report_SVM_*.txt` - Shows Mean Accuracy ± Std Dev
- `cv_summary_*.csv` - Summary table of metrics

### SVM vs CNN Comparison
- `svm_cnn_comparison_*.json` - All comparison metrics
- `svm_cnn_comparison_report_*.txt` - Statistical test results

### Feature Importance
- `ablation_report_SVM_*.txt` - Feature rankings
- `ablation_results_*.csv` - Detailed feature analysis
- `feature_importance_*.png` - Visualization of top features

### Resource Usage
- `analysis_report_*.txt` - CPU, memory, energy consumption

### Master Report
- `MASTER_REPORT_*.txt` - Combined summary of all analyses

---

## Step-by-Step Usage

### Option 1: Run Everything (Recommended for First Time)
```bash
python master_analysis.py all
```
**Time:** ~5-10 minutes (depending on your machine)
**Output:** Single master report + all detailed reports

### Option 2: Interactive Menu
```bash
python master_analysis.py
```
**Choose from menu:**
1. Cross-validation only
2. SVM vs CNN comparison only
3. Ablation study only
4. Complete analysis (all of above)
5. Quit

### Option 3: Run Individual Scripts

#### 1. Cross-Validation (10-fold, Mean ± Std Dev)
```bash
python cross_validation_analysis.py
```
**Output**: Accuracy, Precision, Recall with confidence intervals
**Why run it**: Verify your model is stable and reliable

#### 2. Statistical Significance Test
```bash
python compare_models_advanced.py
```
**Output**: Paired t-test p-value (is CNN really better than SVM?)
**Why run it**: Prove the difference isn't just luck

#### 3. Feature Importance Analysis
```bash
python ablation_study.py
```
**Output**: Which features matter most?
**Why run it**: Optimize extraction pipeline, reduce costs

#### 4. Everything (with error analysis)
```bash
python model_analysis.py
```
**Output**: All metrics including error patterns
**Why run it**: Complete audit before deployment

---

## Understanding Key Results

### Cross-Validation Report Shows:
```
Accuracy: 0.8520 ± 0.0340
          ^      ^^       ^^
          mean   stddev
```
- **0.8520** = Average accuracy across 10 folds
- **0.0340** = This varies ± 3.4% between folds
- **Lower std dev** = More stable (better)

### Paired t-Test Shows:
```
p-value: 0.0382  ✓ SIGNIFICANT
```
- **p < 0.05** means the difference is real, not random luck
- **p ≥ 0.05** means the difference could be by chance

### Ablation Study Shows:
```
mfcc_1           0.0110  ✓ CRITICAL
spectral_centroid 0.0130  ✓ CRITICAL
zero_crossing_rate 0.0030  • helpful
```
- **Drop > 0.05** = Essential feature
- **Drop 0.01-0.05** = Helpful
- **Drop < 0.01** = Can remove to save computation

---

## Common Questions

### Q: What's the difference between these files?
| File | Best For |
|------|----------|
| `cross_validation_analysis.py` | Model reliability/stability |
| `compare_models_advanced.py` | Choosing between SVM & CNN |
| `ablation_study.py` | Feature optimization |
| `model_analysis.py` | Everything (complete audit) |
| `master_analysis.py` | Running all at once |

### Q: How long does each analysis take?
- Cross-validation: 2-3 minutes
- Comparison: 2-3 minutes
- Ablation study: 5-10 minutes (depends on # features)
- All together: ~10-15 minutes

### Q: Which should I run first?
Recommended order:
1. `cross_validation_analysis.py` - Check stability
2. `compare_models_advanced.py` - Choose best model
3. `ablation_study.py` - Optimize features
4. `master_analysis.py` - Final report before deployment

### Q: Do I need both SVM and CNN trained?
- For individual analyses: No (use whichever model you have)
- For comparison: Yes (needs both models)
- For ablation: No (analyzes features, not architecture)

### Q: Can I modify the analyses?
Yes! Edit the Python files directly:
```python
# Example: Change from 10-fold to 5-fold CV
cv_folds=5  # Changed from cv_folds=10

# Example: Get top 30 features instead of 20
ablation_df.head(30)  # Changed from .head(20)
```

---

## Interpreting Results: Cheat Sheet

### What the metrics mean:

**Accuracy** 
- Overall: Did the model predict correctly?
- 90% = 9 out of 10 predictions correct
- Not adjusted for class imbalance

**Precision**
- Of predicted screams, how many are real?
- 95% = 95% of "scream" predictions are correct
- Matters for reducing false alarms

**Recall**
- Of actual screams, how many detected?
- 85% = 85% of real screams detected
- Matters for catching all emergencies

**F1-Score**
- Balance between precision and recall
- Use when you care about both equally

**ROC-AUC**
- Probability model ranks correct prediction higher
- 0.5 = random guessing
- 1.0 = perfect ranking

### Ablation Importance Values:

| Value | Meaning | Action |
|-------|---------|--------|
| > 0.10 | CRITICAL | Must keep |
| 0.05-0.10 | IMPORTANT | Keep |
| 0.01-0.05 | HELPFUL | Consider keeping |
| < 0.01 | NEGLIGIBLE | Can remove |

### Statistical Significance:

| p-value | Meaning |
|---------|---------|
| < 0.001 | Extremely strong evidence |
| < 0.01 | Very strong evidence |
| < 0.05 | Strong evidence (standard threshold) |
| ≥ 0.05 | Not significant (could be luck) |

---

## Troubleshooting

**"FileNotFoundError: svm_esc50_pipeline.pkl not found"**
→ Run: `python train_ml.py`

**"FileNotFoundError: cnn_spectrogram.pth not found"**
→ Run: `python train_cnn.py`

**"ValueError: negative dimension size"**
→ Check that your spectrogram size is 128x128

**Analysis takes too long?**
→ Reduce CV folds: Change `cv_folds=5` or `cv_folds=3`

**Out of memory?**
→ Run analyses individually instead of all at once

**Need GPU acceleration?**
→ It's automatic if CUDA is available, check output shows your GPU

---

## Next Steps After Analysis

1. **Read the master report** (`MASTER_REPORT_*.txt`)
2. **Review statistical significance** - Is your model truly better?
3. **Check feature importance** - Which features to prioritize?
4. **Consider resource usage** - Will it run on edge devices?
5. **Plan deployment** - Use insights for production implementation

---

## Example Workflow

```bash
# Step 1: Train models (if not already)
python train_ml.py
python train_cnn.py

# Step 2: Run quick verification (2-3 min) 
python cross_validation_analysis.py

# Step 3: Compare models (2-3 min)
python compare_models_advanced.py

# Step 4: Optimize features (5-10 min)
python ablation_study.py

# Step 5: Review results
# Look in models/ directory for outputs
# Read MASTER_REPORT_*.txt for summary

# Step 6: Deploy with confidence!
# You now have:
# ✓ Statistical proof of performance
# ✓ Feature importance analysis
# ✓ Model comparison results
# ✓ Stability metrics
```

---

## Output Examples

### Cross-Validation Output:
```
Accuracy:  0.8520 ± 0.0340
Precision: 0.8640 ± 0.0295
Recall:    0.8310 ± 0.0410
F1-Score:  0.8470 ± 0.0340
ROC-AUC:   0.9120 ± 0.0220
```

### Comparison Output:
```
SVM Accuracy:  0.8750
CNN Accuracy:  0.8920
Difference:    +0.0170
p-value:       0.0382 ✓ SIGNIFICANT
```

### Ablation Output:
```
Top Features:
1. mfcc_1              importance: 0.0340
2. spectral_centroid   importance: 0.0280
3. zero_crossing_rate  importance: 0.0190
```

---

## Tips for Better Results

1. **Ensure balanced train/test split** - Check class distribution
2. **Use stratified CV** - Automatic in our tools
3. **Try multiple runs** - Ablation study has random components
4. **Monitor computational resources** - Use resource analysis
5. **Save historical reports** - Track model improvements over time

---

## Documentation Reference

For detailed explanations, see: `ADVANCED_ANALYSIS_GUIDE.md`

For code details, see docstrings in:
- `compare_models_advanced.py`
- `cross_validation_analysis.py`
- `ablation_study.py`
- `model_analysis.py`

---

## Quick Command Reference

```bash
# Run everything
python master_analysis.py all

# Interactive menu
python master_analysis.py

# Individual analyses
python cross_validation_analysis.py
python compare_models_advanced.py
python ablation_study.py
python model_analysis.py

# Check latest results
ls -lt models/*.txt models/*.csv models/*.json | head -20
```

---

**Ready to run? Start with:**
```bash
python master_analysis.py all
```

✓ Easy
✓ Fast  
✓ Comprehensive
✓ Print-ready reports
