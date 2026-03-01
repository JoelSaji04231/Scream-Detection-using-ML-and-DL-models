# Scream Detection Using Machine Learning and Deep Learning

A comprehensive machine learning and deep learning project for detecting screams in audio using traditional ML models (SVM, Random Forest) and deep learning models (CNN, CNN-LSTM). Includes advanced statistical analysis, cross-validation, ablation studies, and resource consumption monitoring.

## 🎯 Project Overview

This project implements state-of-the-art scream detection using:
- **Machine Learning:** SVM, Random Forest, Logistic Regression with ESC-50 dataset and Kaggle scream detection dataset
- **Deep Learning:** CNN and CNN-LSTM on spectrograms and mel-spectrograms
- **Advanced Analysis:** Statistical significance testing, 10-fold CV, feature ablation, error analysis, resource monitoring

## 📊 Key Features

### Models Implemented
- ✅ **Support Vector Machine (SVM)** - ESC-50 enhanced
- ✅ **Random Forest Classifier** - With class weighting
- ✅ **Logistic Regression** - Baseline comparison
- ✅ **CNN (Spectrogram)** - Deep learning baseline
- ✅ **CNN-LSTM** - Temporal sequence modeling
- ✅ **CNN-Mel Spectrogram** - Alternative feature representation

### Advanced Analysis Suite
- ✅ **Statistical Significance Testing**
  - Paired t-test (comparing model accuracy)
  - McNemar's test (error pattern comparison)
  - Two-proportion z-tests (FPR/FNR comparison)
  - Cohen's Kappa (model agreement)

- ✅ **Cross-Validation Analysis**
  - 10-fold stratified CV with stability metrics
  - Mean ± Std Dev reporting
  - Confidence intervals
  - Train-test gap analysis (overfitting detection)

- ✅ **Feature Ablation Study**
  - Leave-one-out feature importance
  - Feature group removal analysis
  - Permutation-based importance
  - Feature interaction detection

- ✅ **Error Pattern Analysis**
  - Confusion matrix breakdown
  - False Negative/Positive detailed analysis
  - Feature characteristics of errors
  - Statistical error comparison

- ✅ **Resource Consumption Monitoring**
  - CPU/GPU utilization tracking
  - Memory usage monitoring
  - Inference time per sample
  - Energy consumption estimation

## 📁 Project Structure

```
.
├── README.md                          # This file
├── QUICK_START.md                     # Quick start guide (2 minutes)
├── ADVANCED_ANALYSIS_GUIDE.md         # Detailed analysis documentation
│
├── Data Files
├── audio_features.csv                 # Extracted features
├── prediction_logs.csv                # Prediction history
│
├── Core Training Scripts
├── train_ml.py                        # Train SVM, Random Forest, Logistic Regression
├── train_cnn.py                       # Train CNN (spectrogram-based)
├── train_cnn_lstm.py                  # Train CNN-LSTM model
├── train_cnn_mel.py                   # Train CNN on mel-spectrograms
│
├── Feature Extraction & Analysis
├── extract_features.py                # Audio feature extraction pipeline
├── analyze_audio.py                   # Audio analysis and visualization
│
├── Inference & Real-Time
├── inference_svm.py                   # SVM inference module
├── inference_cnn.py                   # CNN inference module
├── main.py                            # Dual-model real-time prediction
├── prediction_logger.py               # Prediction logging utility
├── compare_models.py                  # Basic model comparison
│
├── Advanced Analysis Tools (NEW)
├── master_analysis.py                 # Orchestrator for all analyses
├── cross_validation_analysis.py       # 10-fold CV with detailed metrics
├── compare_models_advanced.py         # Statistical significance testing
├── ablation_study.py                  # Feature importance analysis
├── model_analysis.py                  # Comprehensive model analysis
│
├── Models Directory (models/)
├── svm_esc50_pipeline.pkl            # Trained SVM model
├── train_test_split_esc50.pkl        # Train/test data split
├── cnn_spectrogram.pth               # Trained CNN model
├── cnn_lstm_model.pth                # Trained CNN-LSTM model
│
├── Data Directory (data/)
├── cnn_spectrograms.npy              # Spectrogram data
├── cnn_mel_spectrograms.npy          # Mel-spectrogram data
├── cnn_labels.npy                    # Labels for CNN
│
├── Audio Data (Converted_Separately/)
├── scream/                           # Scream audio samples
├── non_scream/                       # Non-scream audio samples
│
└── ESC-50 Dataset
    └── ESC-50-master/                # Environmental Sound Classification dataset
```

## ⚙️ Installation

### Requirements
```
Python 3.8+
CUDA 11.0+ (optional, for GPU acceleration)
```

### Dependencies
```bash
pip install pandas numpy librosa scikit-learn torch joblib matplotlib seaborn scipy psutil
```

Or from requirements.txt (ESC-50):
```bash
cd ESC-50-master
pip install -r requirements.txt
cd ..
```

## 🚀 Quick Start (2 Minutes)

### Step 1: Train Models (First Time Only)
```bash
# Train ML models (SVM, Random Forest, Logistic Regression)
python train_ml.py

# Train CNN
python train_cnn.py

# Optional: Train CNN-LSTM
python train_cnn_lstm.py
```

### Step 2: Run Advanced Analyses
```bash
# Run all analyses with interactive menu
python master_analysis.py

# Or run all analyses automatically
python master_analysis.py all
```

### Step 3: View Results
Check the `models/` directory for:
- `cv_report_*.txt` - Cross-validation results
- `svm_cnn_comparison_*.txt` - Statistical comparison
- `ablation_report_*.txt` - Feature importance
- `MASTER_REPORT_*.txt` - Combined summary

### Step 4: Real-Time Prediction
```bash
# Record 5 seconds of audio and get dual-model predictions
python main.py
```

## 📊 Running Individual Analyses

### 1. Cross-Validation (Model Stability)
```bash
python cross_validation_analysis.py
```
**Output:** Mean Accuracy ± Std Dev across 10 folds
**Time:** ~2-3 minutes

### 2. Statistical Significance Testing (SVM vs CNN)
```bash
python compare_models_advanced.py
```
**Output:** 
- Paired t-test p-value (is difference significant?)
- McNemar's test (different error patterns?)
- Cohen's Kappa (model agreement)
**Time:** ~2-3 minutes

### 3. Ablation Study (Feature Importance)
```bash
python ablation_study.py
```
**Output:**
- Feature importance rankings
- Feature group analysis
- Permutation-based importance
- Visualization chart
**Time:** ~5-10 minutes

### 4. Complete Model Analysis
```bash
python model_analysis.py
```
**Output:** All metrics combined with resource consumption
**Time:** ~10-15 minutes

## 🔬 Understanding the Results

### Cross-Validation Output
```
Accuracy:  0.8520 ± 0.0340
Precision: 0.8640 ± 0.0295
Recall:    0.8310 ± 0.0410
F1-Score:  0.8470 ± 0.0340
ROC-AUC:   0.9120 ± 0.0220
```
- **Std Dev < 0.03:** Model is very stable
- **Std Dev > 0.05:** Model may be unstable
- **Lower std dev = more reliable predictions**

### Statistical Significance Output
```
Paired t-test:
  SVM Accuracy:  0.8750
  CNN Accuracy:  0.8920
  p-value:       0.0382  ✓ SIGNIFICANT
  
McNemar's Test:
  p-value:       0.0156  ✓ SIGNIFICANT
  Models make different errors
```
- **p < 0.05:** Statistically significant (real difference)
- **p ≥ 0.05:** No significant difference (could be luck)

### Ablation Study Output
```
Feature                 Importance   Status
mfcc_1                  0.0340       ✓ CRITICAL
spectral_centroid       0.0280       ✓ CRITICAL  
zero_crossing_rate      0.0030       • HELPFUL
rms_energy             0.0020       negligible
```
- **> 0.05:** Critical feature
- **0.01-0.05:** Helpful feature
- **< 0.01:** Can be removed

### Resource Consumption Output
```
CPU Mean:           35.42% ± 12.15%
Memory Mean:        42.15% ± 8.32%
Time per sample:    2.1450 ms
Energy per inference: 0.1498 Wh
```
- **Time < 10ms:** Suitable for real-time
- **Energy < 1 Wh:** Suitable for edge devices
- **Memory < 1GB:** Deployable on IoT

## 📈 Model Comparison Results

### Latest Cross-Validation Results (SVM Model)
**Generated:** February 13, 2026

| Metric | Value | Stability |
|--------|-------|-----------|
| **Accuracy** | **96.54% ± 0.66%** | Excellent (σ < 0.01) |
| **Precision** | **92.41% ± 1.77%** | Excellent (σ < 0.02) |
| **Recall** | **96.76% ± 0.56%** | Excellent (σ < 0.01) |
| **F1-Score** | **94.53% ± 1.00%** | Excellent (σ < 0.01) |
| **ROC-AUC** | **99.18% ± 0.23%** | Excellent (σ < 0.01) |

### Historical Model Comparison

| Metric | SVM | CNN | CNN-LSTM |
|--------|-----|-----|----------|
| Accuracy | 87.5% | 89.2% | 88.8% |
| Precision | 86.4% | 90.1% | 89.5% |
| Recall | 83.1% | 85.4% | 84.2% |
| F1-Score | 84.7% | 87.6% | 86.8% |
| Training Time | < 1 min | 5-10 min | 10-15 min |
| Inference Time | 0.5 ms | 2.1 ms | 3.2 ms |

### Latest Feature Ablation Study Results (February 13, 2026)

**Top 5 Most Critical Features:**

| Feature | Importance | Impact on Accuracy |
|---------|------------|-------------------|
| **mfcc2_mean** | 0.003413 | -0.66% without |
| **mfcc10_mean** | 0.002194 | -0.48% without |
| **mfcc13_mean** | 0.002194 | -0.48% without |
| **mfcc12_mean** | 0.001707 | -0.42% without |
| **mfcc11_mean** | 0.001462 | -0.38% without |

**Feature Group Importance:**

| Group | Feature Count | Accuracy Without | Importance |
|-------|---------------|------------------|------------|
| **MFCC** | 26 | 92.03% | 0.0427 (Critical) |
| **Spectral** | 4 | 94.37% | 0.0193 (High) |
| **RMS Energy** | 2 | 95.86% | 0.0044 (Moderate) |
| **Zero Crossing Rate** | 2 | 96.20% | 0.0010 (Low) |
| **Spectral Rolloff** | 2 | 96.34% | -0.0005 (Negligible) |

**Key Insights:**
- MFCC features are **critical** for scream detection (4.3% accuracy drop without them)
- Spectral features provide **high** importance (1.9% accuracy drop)
- Energy-based features have **moderate** impact
- ZCR and spectral rolloff can be removed with minimal impact

## 🎯 Use Cases

### 1. Emergency Detection System
```bash
# Monitor continuous audio stream
python main.py
# Real-time dual-model consensus for high confidence
```

### 2. Feature Optimization
```bash
# Which audio features matter most?
python ablation_study.py
# Remove negligible features to reduce computation
```

### 3. Model Selection
```bash
# Statistically prove which model is better
python compare_models_advanced.py
# See p-values, not just raw numbers
```

### 4. Reliability Assessment
```bash
# How stable are predictions across different data?
python cross_validation_analysis.py
# Mean ± Std Dev shows consistency
```

### 5. Error Analysis
```bash
# Why does the model fail?
python model_analysis.py
# Understand false negatives vs false positives
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [QUICK_START.md](QUICK_START.md) | Get started in 2 minutes |
| [ADVANCED_ANALYSIS_GUIDE.md](ADVANCED_ANALYSIS_GUIDE.md) | Detailed methodology (300+ lines) |
| [code docstrings](cross_validation_analysis.py) | Implementation details |

## 🔧 Advanced Features

### Custom Cross-Validation
```python
from cross_validation_analysis import CrossValidationAnalysis

cv = CrossValidationAnalysis(
    X=X_train, y=y_train, model=your_model,
    cv_folds=5  # Instead of 10
)
results = cv.run_cross_validation()
```

### Custom Ablation Study
```python
from ablation_study import FeatureAblationStudy

ablation = FeatureAblationStudy(
    X=X_train, y=y_train, model=model,
    feature_names=feature_names, cv_folds=3
)
importance_df = ablation.ablate_single_features()
```

### Batch Analysis
```bash
# Run analyses and save timestamped results
python master_analysis.py all
# Later: compare results_20260212_143022 vs results_20260212_160500
```

## 📊 Output Files

| File | Format | Description |
|------|--------|-------------|
| `cv_report_*.txt` | Text | 10-fold CV statistics |
| `cv_summary_*.csv` | CSV | Summary metrics table |
| `svm_cnn_comparison_*.json` | JSON | Comparison metrics |
| `ablation_report_*.txt` | Text | Feature importance breakdown |
| `ablation_results_*.csv` | CSV | Feature-by-feature analysis |
| `feature_importance_*.png` | PNG | Feature importance chart |
| `analysis_report_*.txt` | Text | Resource consumption |
| `MASTER_REPORT_*.txt` | Text | Complete summary report |

## 🐛 Troubleshooting

### Models Not Found
```bash
# Train models first
python train_ml.py
python train_cnn.py
```

### Out of Memory During Ablation
```python
# Reduce CV folds in code
cv_folds=3  # Change from cv_folds=5
```

### GPU Not Being Used
```python
# CNN automatically detects and uses CUDA
# Check output: "Using device: cuda"
```

### Feature Extraction Fails
```bash
# Ensure Converted_Separately/ has scream/ and non_scream/ subdirectories
python extract_features.py
```

## 📖 Citation

If you use this project, please cite:
```bibtex
@software{scream_detection_2026,
  title={Scream Detection Using ML and DL},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo}
}
```

## 🤝 Contributing

Contributions welcome! Please:
1. Test changes on your dataset
2. Run full analysis suite
3. Update documentation
4. Ensure backward compatibility

## 📄 License

This project uses two datasets:

1. **ESC-50 Dataset** - Environmental Sound Classification dataset licensed under Creative Commons
2. **Scream Detection Dataset** - Audio dataset of scream and non-scream samples from [Kaggle](https://www.kaggle.com/datasets/aananehsansiam/audio-dataset-of-scream-and-non-scream) by aananehsansiam, stored in `Converted_Separately/` directory

All audio data and trained models are provided under their respective original licenses. Please refer to the Kaggle dataset page for specific licensing terms of the scream detection dataset.

## 📬 Contact & Support

For questions about:
- **Model training:** See docstrings in `train_*.py`
- **Feature extraction:** See `extract_features.py`
- **Analysis methodology:** See `ADVANCED_ANALYSIS_GUIDE.md`
- **Quick help:** See `QUICK_START.md`

## 🎓 Learning Resources

### Concepts Covered
- **Feature Engineering:** MFCC, Spectral features, ZCR, RMS
- **Machine Learning:** SVM, Random Forest, class weighting
- **Deep Learning:** CNN, LSTM, batch normalization, dropout
- **Statistical Testing:** t-tests, McNemar's test, p-values
- **Model Validation:** Cross-validation, stratification, confidence intervals

### Key Papers
- Audio classification using MFCCs and spectrograms
- Environmental Sound Classification (ESC-50)
- Deep learning for audio analysis

## 🚀 Future Enhancements

- [ ] Real-time inference server (FastAPI)
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Attention mechanisms for interpretability
- [ ] Active learning for label efficiency
- [ ] Ensemble methods combining all models
- [ ] Online learning for model updates

## 📊 Performance Benchmarks

Tested on:
- CPU: Intel i7/Ryzen 5+ (recent generation)
- GPU: NVIDIA GTX 1660 or better (optional)
- RAM: 8GB+ recommended

**Typical execution times:**
- Feature extraction: 2-5 minutes
- Model training: 5-20 minutes (depends on model)
- Cross-validation: 2-3 minutes
- Ablation study: 5-10 minutes
- Full analysis suite: 20-30 minutes

## 📈 Version History

### v2.1 (February 28, 2026)
- ✅ **Inference Robustness:** Fixed `inference_svm.py` to support both dictionary and `sklearn.Pipeline` formats.
- ✅ **CNN Fixes:** Corrected `inference_cnn.py` architecture and binary classification logic (sigmoid vs softmax).
- ✅ **Logger Stability:** Fixed `main.py` logging calls to match `PredictionLogger` signature.
- ✅ **Improved Error Handling:** Added fallback mechanisms for missing model metadata.

### v2.0
- ✅ Core ML/DL models
- ✅ Advanced statistical analysis
- ✅ Cross-validation and ablation studies
- ✅ Comprehensive documentation
- ✅ Resource monitoring

---

**Last Updated:** February 28, 2026

**For quick start:** Read [QUICK_START.md](QUICK_START.md) (5 min read)

**For deep dive:** Read [ADVANCED_ANALYSIS_GUIDE.md](ADVANCED_ANALYSIS_GUIDE.md) (15 min read)

**Ready to run analyses?**
```bash
python master_analysis.py all
```
