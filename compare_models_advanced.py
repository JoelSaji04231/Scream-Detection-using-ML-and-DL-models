"""
Advanced SVM vs CNN Comparison with Statistical Significance Testing
Includes:
- Side-by-side performance metrics
- Paired t-test and McNemar's test
- Cross-model error analysis
- Confidence calibration
"""

import pandas as pd
import numpy as np
import librosa
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# SECTION 1: LOAD CNN MODEL (must match training architecture)
# ============================================================================

class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.25)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dropout6 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout5(x)
        
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout6(x)
        
        x = self.fc3(x)
        return x


# ============================================================================
# SECTION 2: COMPARISON METRICS
# ============================================================================

class ModelComparison:
    """Compare performance of two models with statistical tests"""
    
    def __init__(self, y_true, svm_preds, svm_probas, cnn_preds, cnn_probas):
        self.y_true = y_true
        self.svm_preds = svm_preds
        self.svm_probas = svm_probas
        self.cnn_preds = cnn_preds
        self.cnn_probas = cnn_probas
        
    def calculate_metrics(self, predictions, probas, model_name):
        """Calculate detailed metrics for a model"""
        accuracy = accuracy_score(self.y_true, predictions)
        precision = precision_score(self.y_true, predictions, zero_division=0)
        recall = recall_score(self.y_true, predictions, zero_division=0)
        f1 = f1_score(self.y_true, predictions, zero_division=0)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(self.y_true, probas)
        except:
            roc_auc = None
        
        # Matthews Correlation Coefficient
        from sklearn.metrics import matthews_corrcoef
        mcc = matthews_corrcoef(self.y_true, predictions)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_true, predictions, labels=[0, 1]).ravel()
        
        metrics = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'mcc': mcc,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
        
        return metrics
    
    def print_comparison_table(self):
        """Print side-by-side comparison"""
        svm_metrics = self.calculate_metrics(self.svm_preds, self.svm_probas, 'SVM')
        cnn_metrics = self.calculate_metrics(self.cnn_preds, self.cnn_probas, 'CNN')
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc',
                          'specificity', 'sensitivity', 'fpr', 'fnr']
        
        print(f"\n{'Metric':<20} {'SVM':<20} {'CNN':<20} {'Difference':<15}")
        print("-" * 80)
        
        for metric in metrics_to_show:
            svm_val = svm_metrics[metric]
            cnn_val = cnn_metrics[metric]
            
            if svm_val is not None and cnn_val is not None:
                diff = cnn_val - svm_val
                svm_str = f"{svm_val:.4f}"
                cnn_str = f"{cnn_val:.4f}"
                diff_str = f"{diff:+.4f}"
                
                print(f"{metric:<20} {svm_str:<20} {cnn_str:<20} {diff_str:<15}")
        
        # Confusion matrices
        print("\n" + "-"*80)
        print("CONFUSION MATRICES")
        print("-"*80)
        
        print(f"\nSVM:")
        print(f"  TN: {svm_metrics['tn']:<6} FP: {svm_metrics['fp']:<6}")
        print(f"  FN: {svm_metrics['fn']:<6} TP: {svm_metrics['tp']:<6}")
        
        print(f"\nCNN:")
        print(f"  TN: {cnn_metrics['tn']:<6} FP: {cnn_metrics['fp']:<6}")
        print(f"  FN: {cnn_metrics['fn']:<6} TP: {cnn_metrics['tp']:<6}")
        
        return svm_metrics, cnn_metrics
    
    def paired_t_test(self):
        """
        Paired t-test: tests if the difference in accuracy is statistically significant
        H0: Models have equal accuracy
        H1: Models have different accuracy
        """
        print("\n" + "="*80)
        print("PAIRED T-TEST: Comparing Model Accuracy")
        print("="*80)
        
        # Create binary correct/incorrect
        svm_correct = (self.svm_preds == self.y_true).astype(int)
        cnn_correct = (self.cnn_preds == self.y_true).astype(int)
        
        svm_acc = svm_correct.mean()
        cnn_acc = cnn_correct.mean()
        diff = cnn_acc - svm_acc
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(svm_correct, cnn_correct)
        
        print(f"\nNull Hypothesis (H0): SVM and CNN have equal accuracy")
        print(f"Alternative (H1): SVM and CNN have different accuracy")
        
        print(f"\nResults:")
        print(f"  SVM Accuracy:  {svm_acc:.4f}")
        print(f"  CNN Accuracy:  {cnn_acc:.4f}")
        print(f"  Difference:    {diff:+.4f} ({'CNN better' if diff > 0 else 'SVM better'})")
        
        print(f"\n  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.6f}")
        print(f"  α = 0.05")
        
        if p_value < 0.05:
            print(f"\n  ✓ RESULT: REJECT null hypothesis (p < 0.05)")
            print(f"    There IS a statistically significant difference")
            better = "CNN" if diff > 0 else "SVM"
            print(f"    {better} is significantly better")
        else:
            print(f"\n  ✗ RESULT: FAIL TO REJECT null hypothesis (p ≥ 0.05)")
            print(f"    No statistically significant difference detected")
            print(f"    The observed difference could be due to random chance")
        
        return {
            'svm_accuracy': svm_acc,
            'cnn_accuracy': cnn_acc,
            'difference': diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def mcnemar_test(self):
        """
        McNemar's Test: tests if models make significantly different errors
        H0: Models make the same types of errors
        H1: Models make different types of errors
        """
        print("\n" + "="*80)
        print("McNEMAR'S TEST: Comparing Error Patterns")
        print("="*80)
        
        svm_correct = (self.svm_preds == self.y_true)
        cnn_correct = (self.cnn_preds == self.y_true)
        
        # Contingency table
        both_correct = np.sum(svm_correct & cnn_correct)
        both_wrong = np.sum(~svm_correct & ~cnn_correct)
        svm_correct_cnn_wrong = np.sum(svm_correct & ~cnn_correct)  # a
        svm_wrong_cnn_correct = np.sum(~svm_correct & cnn_correct)  # b
        
        print(f"\nNull Hypothesis (H0): SVM and CNN make the same errors")
        print(f"Alternative (H1): SVM and CNN make different errors")
        
        print(f"\nContingency Table:")
        print(f"  Both Correct:              {both_correct}")
        print(f"  Both Wrong:                {both_wrong}")
        print(f"  SVM Correct, CNN Wrong:    {svm_correct_cnn_wrong}")
        print(f"  CNN Correct, SVM Wrong:    {svm_wrong_cnn_correct}")
        
        # McNemar's test
        if (svm_correct_cnn_wrong + svm_wrong_cnn_correct) > 0:
            numerator = (svm_correct_cnn_wrong - svm_wrong_cnn_correct) ** 2
            denominator = svm_correct_cnn_wrong + svm_wrong_cnn_correct
            chi2_stat = numerator / denominator
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
            
            print(f"\nResults:")
            print(f"  χ² statistic: {chi2_stat:.4f}")
            print(f"  p-value:      {p_value:.6f}")
            print(f"  α = 0.05")
            
            if p_value < 0.05:
                print(f"\n  ✓ RESULT: REJECT null hypothesis (p < 0.05)")
                print(f"    Models make significantly DIFFERENT error patterns")
            else:
                print(f"\n  ✗ RESULT: FAIL TO REJECT null hypothesis (p ≥ 0.05)")
                print(f"    Models make SIMILAR error patterns")
            
            return {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'both_correct': both_correct,
                'both_wrong': both_wrong,
                'svm_c_cnn_w': svm_correct_cnn_wrong,
                'cnn_c_svm_w': svm_wrong_cnn_correct,
                'significant': p_value < 0.05
            }
        else:
            print(f"\n  ⚠ Not enough disagreement for McNemar's test")
            return None
    
    def proportions_test(self):
        """
        Two-proportion z-test: tests if FPR or FNR differs significantly
        """
        print("\n" + "="*80)
        print("TWO-PROPORTION Z-TEST: FPR and FNR Comparison")
        print("="*80)
        
        svm_metrics = self.calculate_metrics(self.svm_preds, self.svm_probas, 'SVM')
        cnn_metrics = self.calculate_metrics(self.cnn_preds, self.cnn_probas, 'CNN')
        
        # FPR test
        svm_fp_total = svm_metrics['fp'] + svm_metrics['tn']
        cnn_fp_total = cnn_metrics['fp'] + cnn_metrics['tn']
        
        if svm_fp_total > 0 and cnn_fp_total > 0:
            p1 = svm_metrics['fp'] / svm_fp_total
            p2 = cnn_metrics['fp'] / cnn_fp_total
            
            p_pool = (svm_metrics['fp'] + cnn_metrics['fp']) / (svm_fp_total + cnn_fp_total)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/svm_fp_total + 1/cnn_fp_total))
            z_fpr = (p1 - p2) / se if se > 0 else 0
            p_value_fpr = 2 * (1 - stats.norm.cdf(abs(z_fpr)))
            
            print(f"\nFalse Positive Rate:")
            print(f"  SVM FPR: {p1:.4f}")
            print(f"  CNN FPR: {p2:.4f}")
            print(f"  z-statistic: {z_fpr:.4f}")
            print(f"  p-value: {p_value_fpr:.6f}")
        
        # FNR test
        svm_fn_total = svm_metrics['fn'] + svm_metrics['tp']
        cnn_fn_total = cnn_metrics['fn'] + cnn_metrics['tp']
        
        if svm_fn_total > 0 and cnn_fn_total > 0:
            p1 = svm_metrics['fn'] / svm_fn_total
            p2 = cnn_metrics['fn'] / cnn_fn_total
            
            p_pool = (svm_metrics['fn'] + cnn_metrics['fn']) / (svm_fn_total + cnn_fn_total)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/svm_fn_total + 1/cnn_fn_total))
            z_fnr = (p1 - p2) / se if se > 0 else 0
            p_value_fnr = 2 * (1 - stats.norm.cdf(abs(z_fnr)))
            
            print(f"\nFalse Negative Rate:")
            print(f"  SVM FNR: {p1:.4f}")
            print(f"  CNN FNR: {p2:.4f}")
            print(f"  z-statistic: {z_fnr:.4f}")
            print(f"  p-value: {p_value_fnr:.6f}")


# ============================================================================
# SECTION 3: CONSENSUS ANALYSIS
# ============================================================================

class ConsensusAnalysis:
    """Analyze agreement between models"""
    
    def __init__(self, svm_preds, cnn_preds, y_true):
        self.svm_preds = svm_preds
        self.cnn_preds = cnn_preds
        self.y_true = y_true
        
    def calculate_agreement(self):
        """Calculate Cohen's kappa and agreement percentage"""
        from sklearn.metrics import cohen_kappa_score
        
        agreement = (self.svm_preds == self.cnn_preds).sum() / len(self.svm_preds)
        kappa = cohen_kappa_score(self.svm_preds, self.cnn_preds)
        
        print("\n" + "="*80)
        print("MODEL CONSENSUS ANALYSIS")
        print("="*80)
        
        print(f"\nAgreement Statistics:")
        print(f"  Agreement rate: {agreement:.4f} ({agreement*100:.2f}%)")
        print(f"  Cohen's Kappa: {kappa:.4f}")
        
        if kappa >= 0.81:
            print(f"  Interpretation: Almost Perfect Agreement")
        elif kappa >= 0.61:
            print(f"  Interpretation: Substantial Agreement")
        elif kappa >= 0.41:
            print(f"  Interpretation: Moderate Agreement")
        elif kappa >= 0.21:
            print(f"  Interpretation: Fair Agreement")
        else:
            print(f"  Interpretation: Slight Agreement")
        
        # Analyze disagreements
        disagreement = self.svm_preds != self.cnn_preds
        disagree_indices = np.where(disagreement)[0]
        
        print(f"\nDisagreement Analysis:")
        print(f"  Total disagreements: {len(disagree_indices)} ({(1-agreement)*100:.2f}%)")
        
        # Categorize disagreements
        svm_correct_cnn_wrong = (self.svm_preds[disagreement] == self.y_true[disagreement]).sum()
        cnn_correct_svm_wrong = (self.cnn_preds[disagreement] == self.y_true[disagreement]).sum()
        
        print(f"  SVM correct, CNN wrong: {svm_correct_cnn_wrong}")
        print(f"  CNN correct, SVM wrong: {cnn_correct_svm_wrong}")
        
        return {
            'agreement': agreement,
            'kappa': kappa,
            'disagreements': len(disagree_indices),
            'svm_better': svm_correct_cnn_wrong,
            'cnn_better': cnn_correct_svm_wrong
        }


# ============================================================================
# MAIN COMPARISON FUNCTION
# ============================================================================

def run_advanced_comparison():
    """Run comprehensive model comparison"""
    print("\n" + "="*80)
    print("ADVANCED SVM vs CNN COMPARISON WITH STATISTICAL SIGNIFICANCE")
    print("="*80)
    
    print("\nLoading models and data...")
    
    # Load SVM
    try:
        svm_pipeline = joblib.load("models/svm_esc50_pipeline.pkl")
        train_test_data = joblib.load("models/train_test_split_esc50.pkl")
        print("✓ SVM model loaded")
    except:
        print("✗ SVM model not found. Please train SVM first (python train_ml.py)")
        return
    
    # Load CNN
    try:
        cnn_model = SpectrogramCNN().to(device)
        cnn_model.load_state_dict(torch.load("models/cnn_spectrogram.pth", map_location=device))
        cnn_model.eval()
        print("✓ CNN model loaded")
    except:
        print("✗ CNN model not found. Please train CNN first (python train_cnn.py)")
        return
    
    # Load test data
    X_test = train_test_data['X_test']
    y_test = train_test_data['y_test']
    
    # Convert labels to numeric
    y_test_numeric = (y_test == 'scream').astype(int) if isinstance(y_test[0], str) else y_test
    
    print(f"✓ Data loaded: {len(X_test)} test samples")
    
    # ============ SVM PREDICTIONS ============
    print("\nGenerating SVM predictions...")
    svm_preds = svm_pipeline.predict(X_test)
    svm_probas = svm_pipeline.predict_proba(X_test)[:, 1]
    print(f"✓ SVM predictions generated")
    
    # ============ CNN PREDICTIONS ============
    print("Generating CNN predictions...")
    cnn_preds = []
    cnn_probas = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), 32):
            batch_X = X_test.iloc[i:i+32].values
            
            # Reshape for CNN (batch_size, 1, 128, 128)
            batch_X = torch.FloatTensor(batch_X).reshape(-1, 1, 128, 128).to(device)
            
            outputs = cnn_model(batch_X)
            probas = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probas > 0.5).astype(int)
            
            cnn_probas.extend(probas)
            cnn_preds.extend(preds)
    
    cnn_preds = np.array(cnn_preds)
    cnn_probas = np.array(cnn_probas)
    print(f"✓ CNN predictions generated")
    
    # ============ COMPARISON ============
    comparison = ModelComparison(y_test_numeric, svm_preds, svm_probas, cnn_preds, cnn_probas)
    
    # Print comparison table
    svm_metrics, cnn_metrics = comparison.print_comparison_table()
    
    # Statistical tests
    t_test_result = comparison.paired_t_test()
    mcnemar_result = comparison.mcnemar_test()
    comparison.proportions_test()
    
    # Consensus analysis
    consensus = ConsensusAnalysis(svm_preds, cnn_preds, y_test_numeric)
    consensus_result = consensus.calculate_agreement()
    
    # ============ SAVE RESULTS ============
    print("\n" + "="*80)
    print("SAVING COMPARISON RESULTS")
    print("="*80)
    
    results = {
        'svm_metrics': svm_metrics,
        'cnn_metrics': cnn_metrics,
        't_test': t_test_result,
        'mcnemar_test': mcnemar_result,
        'consensus': consensus_result,
        'timestamp': datetime.now().isoformat()
    }
    
    # Convert to JSON-serializable
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(item) for item in obj]
        return obj
    
    results = convert_types(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = f"models/svm_cnn_comparison_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {json_path}")
    
    # Save text report
    report_path = f"models/svm_cnn_comparison_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SVM vs CNN STATISTICAL COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*80 + "\n")
        
        f.write("\nPAIRED T-TEST RESULTS:\n")
        f.write(f"  SVM Accuracy: {t_test_result['svm_accuracy']:.4f}\n")
        f.write(f"  CNN Accuracy: {t_test_result['cnn_accuracy']:.4f}\n")
        f.write(f"  Difference: {t_test_result['difference']:+.4f}\n")
        f.write(f"  p-value: {t_test_result['p_value']:.6f}\n")
        f.write(f"  Significant: {'YES' if t_test_result['significant'] else 'NO'}\n")
        
        if mcnemar_result:
            f.write("\nMcNEMAR'S TEST RESULTS:\n")
            f.write(f"  χ² statistic: {mcnemar_result.get('chi2_statistic', 'N/A')}\n")
            f.write(f"  p-value: {mcnemar_result.get('p_value', 'N/A')}\n")
        
        f.write("\nCONSENSUS:\n")
        f.write(f"  Agreement: {consensus_result['agreement']:.4f}\n")
        f.write(f"  Cohen's Kappa: {consensus_result['kappa']:.4f}\n")
    
    print(f"✓ Report saved to {report_path}")


if __name__ == "__main__":
    run_advanced_comparison()
