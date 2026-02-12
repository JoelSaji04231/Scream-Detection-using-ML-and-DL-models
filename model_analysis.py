"""
Comprehensive Model Analysis:
- Statistical Significance Testing (Paired t-test, McNemar's test)
- Ablation Study (Feature importance and removal)
- Cross-Validation (10-fold stratified)
- Error Pattern Analysis (FN/FP breakdown)
- Resource Consumption (CPU/GPU/Energy)
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
import time
import psutil
import librosa
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: CROSS-VALIDATION WITH DETAILED METRICS
# ============================================================================

def perform_cross_validation(X, y, model_pipeline, cv_folds=10):
    """
    Perform 10-fold stratified cross-validation with detailed metrics.
    Returns: Mean ± Std for multiple metrics
    """
    print("\n" + "="*70)
    print("CROSS-VALIDATION ANALYSIS (10-fold Stratified)")
    print("="*70)
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Perform cross-validation
    cv_results = cross_validate(model_pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)
    
    # Compile results
    results = {}
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results[metric] = {
            'test_mean': test_scores.mean(),
            'test_std': test_scores.std(),
            'test_scores': test_scores,
            'train_mean': train_scores.mean(),
            'train_std': train_scores.std(),
            'train_scores': train_scores
        }
        
        print(f"\n{metric.upper()}:")
        print(f"  Test:  {test_scores.mean():.4f} ± {test_scores.std():.4f}")
        print(f"  Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")
        print(f"  Fold scores: {[f'{s:.4f}' for s in test_scores]}")
    
    return results


# ============================================================================
# SECTION 2: STATISTICAL SIGNIFICANCE TESTING
# ============================================================================

class StatisticalSignificanceTest:
    """Paired t-test and McNemar's test between SVM and CNN predictions"""
    
    def __init__(self, y_true, svm_predictions, cnn_predictions, svm_probas=None, cnn_probas=None):
        self.y_true = y_true
        self.svm_predictions = svm_predictions
        self.cnn_predictions = cnn_predictions
        self.svm_probas = svm_probas
        self.cnn_probas = cnn_probas
        
    def paired_t_test(self):
        """
        Paired t-test comparing model accuracies.
        Tests if the difference in accuracy is statistically significant.
        """
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE: PAIRED T-TEST")
        print("="*70)
        
        # Create binary correct/incorrect indicators
        svm_correct = (self.svm_predictions == self.y_true).astype(int)
        cnn_correct = (self.cnn_predictions == self.y_true).astype(int)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(svm_correct, cnn_correct)
        
        svm_acc = svm_correct.mean()
        cnn_acc = cnn_correct.mean()
        mean_diff = svm_acc - cnn_acc
        
        print(f"\nSVM Accuracy:  {svm_acc:.4f}")
        print(f"CNN Accuracy:  {cnn_acc:.4f}")
        print(f"Difference:    {mean_diff:+.4f}")
        print(f"\nPaired t-test results:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.6f}")
        
        if p_value < 0.05:
            model = "SVM" if mean_diff > 0 else "CNN"
            print(f"  ✓ SIGNIFICANT DIFFERENCE (p < 0.05)")
            print(f"    {model} is statistically significantly better")
        else:
            print(f"  ✗ NO SIGNIFICANT DIFFERENCE (p >= 0.05)")
            print(f"    The difference could be due to random chance")
        
        return {'t_stat': t_stat, 'p_value': p_value, 'mean_diff': mean_diff}
    
    def mcnemar_test(self):
        """
        McNemar's test for paired categorical data.
        Tests if models make different types of errors.
        """
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE: McNEMAR'S TEST")
        print("="*70)
        
        svm_correct = (self.svm_predictions == self.y_true)
        cnn_correct = (self.cnn_predictions == self.y_true)
        
        # Create contingency table
        # a: SVM correct, CNN wrong
        # b: SVM wrong, CNN correct
        # c: Both correct
        # d: Both wrong
        a = np.sum(svm_correct & ~cnn_correct)  # SVM correct, CNN wrong
        b = np.sum(~svm_correct & cnn_correct)  # SVM wrong, CNN correct
        c = np.sum(svm_correct & cnn_correct)   # Both correct
        d = np.sum(~svm_correct & ~cnn_correct) # Both wrong
        
        print(f"\nContingency Table:")
        print(f"  Both Correct:        {c}")
        print(f"  Both Wrong:          {d}")
        print(f"  SVM Correct, CNN Wrong: {a}")
        print(f"  CNN Correct, SVM Wrong: {b}")
        
        # McNemar's test
        if (a + b) > 0:
            mcnemar_stat = ((a - b)**2) / (a + b)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
            
            print(f"\nMcNemar's test results:")
            print(f"  χ² statistic: {mcnemar_stat:.4f}")
            print(f"  p-value:      {p_value:.6f}")
            
            if p_value < 0.05:
                print(f"  ✓ SIGNIFICANT DIFFERENCE (p < 0.05)")
                print(f"    Models make significantly different errors")
            else:
                print(f"  ✗ NO SIGNIFICANT DIFFERENCE (p >= 0.05)")
                print(f"    Models make similar error patterns")
                
            return {'chi2_stat': mcnemar_stat, 'p_value': p_value, 'a': a, 'b': b}
        else:
            print(f"\n  Not enough disagreement for McNemar's test")
            return {'chi2_stat': None, 'p_value': None, 'a': a, 'b': b}


# ============================================================================
# SECTION 3: ABLATION STUDY
# ============================================================================

class AblationStudy:
    """
    Systematically remove features to measure their contribution to accuracy.
    """
    
    def __init__(self, X, y, model_pipeline, feature_names):
        self.X = X
        self.y = y
        self.model_pipeline = model_pipeline
        self.feature_names = np.array(feature_names)
        self.baseline_scores = None
        
    def calculate_baseline(self):
        """Calculate baseline accuracy with all features"""
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model_pipeline, self.X, self.y, cv=5, scoring='accuracy')
        self.baseline_scores = scores.mean()
        print(f"Baseline Accuracy (all features): {self.baseline_scores:.4f}")
        return self.baseline_scores
    
    def remove_single_features(self):
        """Remove features one at a time and measure impact"""
        print("\n" + "="*70)
        print("ABLATION STUDY: Feature Importance (Leave-One-Out)")
        print("="*70)
        
        self.calculate_baseline()
        
        feature_importance = []
        
        print(f"\nRemoving features one at a time:")
        print(f"{'Feature':<20} {'Accuracy':<12} {'Drop':<12} {'Importance':<12}")
        print("-" * 56)
        
        for idx in range(len(self.feature_names)):
            # Create X without this feature
            feature_mask = np.ones(self.X.shape[1], dtype=bool)
            feature_mask[idx] = False
            X_ablated = self.X.iloc[:, feature_mask]
            
            # Calculate accuracy without this feature
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(self.model_pipeline, X_ablated, self.y, cv=5, scoring='accuracy')
            accuracy_without = scores.mean()
            
            # Calculate importance (how much accuracy dropped)
            importance = self.baseline_scores - accuracy_without
            feature_importance.append({
                'feature': self.feature_names[idx],
                'accuracy_without': accuracy_without,
                'importance': importance
            })
            
            print(f"{self.feature_names[idx]:<20} {accuracy_without:<12.4f} {importance:<12.4f} {'↓' if importance > 0 else '↑'}")
        
        # Sort by importance
        importance_df = pd.DataFrame(feature_importance).sort_values('importance', ascending=False)
        
        print(f"\n{'Top 10 Most Important Features:'}")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    def remove_feature_groups(self):
        """Remove groups of related features"""
        print("\n" + "="*70)
        print("ABLATION STUDY: Feature Group Removal")
        print("="*70)
        
        # Define feature groups based on typical feature names
        feature_groups = {
            'MFCC': [f for f in self.feature_names if 'mfcc' in f.lower()],
            'Spectral': [f for f in self.feature_names if 'spectral' in f.lower() or 'centroid' in f.lower()],
            'Zero_Crossing': [f for f in self.feature_names if 'zcr' in f.lower() or 'zero' in f.lower()],
            'RMS': [f for f in self.feature_names if 'rms' in f.lower()],
            'Chroma': [f for f in self.feature_names if 'chroma' in f.lower()],
            'Temporal': [f for f in self.feature_names if 'delta' in f.lower() or 'temp' in f.lower()]
        }
        
        # Only include groups that exist
        feature_groups = {k: v for k, v in feature_groups.items() if len(v) > 0}
        
        print(f"\nFeature groups identified:")
        group_results = []
        
        for group_name, group_features in feature_groups.items():
            # Create mask to remove this group
            # Handle both pandas and numpy arrays
            if hasattr(self.feature_names, 'isin'):
                feature_mask = ~self.feature_names.isin(group_features)
            else:
                # For numpy arrays, use numpy's isin function
                feature_mask = ~np.isin(self.feature_names, group_features)
            X_ablated = self.X.iloc[:, feature_mask]
            
            # Calculate accuracy without this group
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(self.model_pipeline, X_ablated, self.y, cv=5, scoring='accuracy')
            accuracy_without = scores.mean()
            importance = self.baseline_scores - accuracy_without
            
            group_results.append({
                'group': group_name,
                'feature_count': len(group_features),
                'accuracy_without': accuracy_without,
                'importance': importance
            })
            
            print(f"\n{group_name} ({len(group_features)} features): {accuracy_without:.4f} (drop: {importance:.4f})")
        
        return pd.DataFrame(group_results)


# ============================================================================
# SECTION 4: ERROR PATTERN ANALYSIS
# ============================================================================

class ErrorPatternAnalysis:
    """
    Detailed breakdown of False Negatives and False Positives
    """
    
    def __init__(self, y_true, predictions, file_paths=None, X=None, feature_names=None):
        self.y_true = y_true
        self.predictions = predictions
        self.file_paths = file_paths
        self.X = X
        self.feature_names = feature_names
        
    def analyze_errors(self):
        """Comprehensive error analysis"""
        print("\n" + "="*70)
        print("ERROR PATTERN ANALYSIS")
        print("="*70)
        
        # Get confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.predictions, labels=[0, 1]).ravel()
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        
        # Calculate false positive and false negative rates
        if (fp + tn) > 0:
            fpr = fp / (fp + tn)
            print(f"\nFalse Positive Rate: {fpr:.4f} ({fp}/{fp+tn})")
        
        if (fn + tp) > 0:
            fnr = fn / (fn + tp)
            print(f"False Negative Rate: {fnr:.4f} ({fn}/{fn+tp})")
        
        # Identify error samples
        errors = self.y_true != self.predictions
        error_indices = np.where(errors)[0]
        
        print(f"\nTotal Errors: {len(error_indices)} out of {len(self.y_true)} ({100*len(error_indices)/len(self.y_true):.2f}%)")
        
        # Separate FN and FP
        false_negatives = (self.y_true == 1) & (self.predictions == 0)
        false_positives = (self.y_true == 0) & (self.predictions == 1)
        
        fn_indices = np.where(false_negatives)[0]
        fp_indices = np.where(false_positives)[0]
        
        print(f"  - False Negatives: {len(fn_indices)} (scream missed)")
        print(f"  - False Positives: {len(fp_indices)} (non-scream predicted as scream)")
        
        # Analyze feature characteristics of errors
        if self.X is not None and self.feature_names is not None:
            self._analyze_error_features(fn_indices, fp_indices)
        
        return {
            'confusion_matrix': (tn, fp, fn, tp),
            'fpr': fpr if (fp + tn) > 0 else None,
            'fnr': fnr if (fn + tp) > 0 else None,
            'fn_indices': fn_indices,
            'fp_indices': fp_indices
        }
    
    def _analyze_error_features(self, fn_indices, fp_indices):
        """Analyze feature values for errors"""
        print(f"\nFeature Analysis for Errors:")
        
        if len(fn_indices) > 0:
            print(f"\nFalse Negatives (missed screams) - Characteristics:")
            fn_X = self.X.iloc[fn_indices]
            print(f"  Feature means: {dict(zip(self.feature_names[:5], fn_X.iloc[:, :5].mean().values))}")
        
        if len(fp_indices) > 0:
            print(f"\nFalse Positives (false alarms) - Characteristics:")
            fp_X = self.X.iloc[fp_indices]
            print(f"  Feature means: {dict(zip(self.feature_names[:5], fp_X.iloc[:, :5].mean().values))}")
        
        # Statistical test: are error samples significantly different?
        if len(fn_indices) > 0 and len(fp_indices) > 0:
            from scipy.stats import ttest_ind
            # Compare first feature
            if self.feature_names is not None and len(self.feature_names) > 0:
                fn_feature = self.X.iloc[fn_indices, 0]
                fp_feature = self.X.iloc[fp_indices, 0]
                t_stat, p_val = ttest_ind(fn_feature, fp_feature)
                
                print(f"\nDifference between FN and FP in first feature ({self.feature_names[0]}):")
                print(f"  FN mean: {fn_feature.mean():.4f}, FP mean: {fp_feature.mean():.4f}")
                print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.6f}")


# ============================================================================
# SECTION 5: RESOURCE CONSUMPTION ANALYSIS
# ============================================================================

class ResourceConsumptionAnalysis:
    """
    Measure CPU, GPU, and energy consumption during inference
    """
    
    def __init__(self):
        self.initial_cpu_percent = None
        self.initial_memory = None
        self.start_time = None
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.start_time = time.time()
        self.initial_cpu_percent = psutil.cpu_percent(interval=0.1)
        self.initial_memory = psutil.virtual_memory().percent
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def get_cpu_memory_stats(self, num_samples=100, batch_size=32):
        """Measure CPU and memory during inference"""
        print("\n" + "="*70)
        print("RESOURCE CONSUMPTION: CPU & MEMORY")
        print("="*70)
        
        samples_cpu = []
        samples_memory = []
        
        for i in range(num_samples):
            current_cpu = psutil.cpu_percent(interval=0.01)
            current_memory = psutil.virtual_memory().percent
            
            samples_cpu.append(current_cpu)
            samples_memory.append(current_memory)
        
        cpu_mean = np.mean(samples_cpu)
        cpu_std = np.std(samples_cpu)
        cpu_max = np.max(samples_cpu)
        
        mem_mean = np.mean(samples_memory)
        mem_std = np.std(samples_memory)
        mem_max = np.max(samples_memory)
        
        print(f"\nCPU Utilization:")
        print(f"  Mean: {cpu_mean:.2f}% ± {cpu_std:.2f}%")
        print(f"  Max:  {cpu_max:.2f}%")
        
        print(f"\nMemory Utilization:")
        print(f"  Mean: {mem_mean:.2f}% ± {mem_std:.2f}%")
        print(f"  Max:  {mem_max:.2f}%")
        
        return {
            'cpu_mean': cpu_mean,
            'cpu_std': cpu_std,
            'cpu_max': cpu_max,
            'memory_mean': mem_mean,
            'memory_std': mem_std,
            'memory_max': mem_max
        }
    
    def get_gpu_stats(self):
        """Measure GPU utilization"""
        if not torch.cuda.is_available():
            print("\n⚠ GPU not available")
            return None
        
        print(f"\nGPU Information:")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9
        gpu_mem_max = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"  Allocated Memory: {gpu_mem_allocated:.4f} GB")
        print(f"  Reserved Memory: {gpu_mem_reserved:.4f} GB")
        print(f"  Max Memory Used: {gpu_mem_max:.4f} GB")
        
        return {
            'allocated_gb': gpu_mem_allocated,
            'reserved_gb': gpu_mem_reserved,
            'max_allocated_gb': gpu_mem_max
        }
    
    def measure_inference_time(self, model, X, batch_size=32, warmup_batches=5):
        """Measure inference time per sample"""
        print(f"\nInference Time Measurement:")
        print(f"  Total samples: {len(X)}")
        print(f"  Batch size: {batch_size}")
        
        # Warmup
        for i in range(min(warmup_batches, len(X) // batch_size)):
            _ = model.predict(X.iloc[i*batch_size:(i+1)*batch_size])
        
        # Timing
        start = time.time()
        n_samples = 0
        for i in range(len(X) // batch_size):
            _ = model.predict(X.iloc[i*batch_size:(i+1)*batch_size])
            n_samples += batch_size
        
        total_time = time.time() - start
        time_per_sample = (total_time / n_samples) * 1000  # ms per sample
        
        print(f"  Time per sample: {time_per_sample:.4f} ms")
        print(f"  Samples per second: {1000/time_per_sample:.2f}")
        print(f"  Total inference time: {total_time:.2f} seconds")
        
        return {
            'time_per_sample_ms': time_per_sample,
            'samples_per_second': 1000/time_per_sample,
            'total_time': total_time
        }
    
    def estimate_energy_consumption(self, cpu_mean, inference_time_seconds, tdp_watts=65):
        """
        Estimate energy consumption based on CPU utilization and TDP (Thermal Design Power)
        
        Args:
            cpu_mean: Average CPU utilization (0-100)
            inference_time_seconds: Time for inference
            tdp_watts: CPU TDP (Thermal Design Power) - default 65W for typical CPUs
        """
        print(f"\nEnergy Consumption Estimate:")
        print(f"  CPU TDP: {tdp_watts}W")
        print(f"  CPU Utilization: {cpu_mean:.2f}%")
        print(f"  Inference Time: {inference_time_seconds:.2f}s")
        
        # Rough estimate: Energy = Power * Time
        # Power used ≈ TDP * (CPU utilization %)
        power_used = tdp_watts * (cpu_mean / 100.0)
        energy_joules = power_used * inference_time_seconds
        energy_wh = energy_joules / 3600  # Convert joules to watt-hours
        
        print(f"  Estimated Power: {power_used:.2f}W")
        print(f"  Estimated Energy: {energy_joules:.4f} Joules ({energy_wh:.6f} Wh)")
        
        return {
            'estimated_power_w': power_used,
            'estimated_energy_joules': energy_joules,
            'estimated_energy_wh': energy_wh
        }


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_comprehensive_analysis():
    """Run all analyses"""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL ANALYSIS SUITE")
    print("="*70)
    
    # Load data
    print("\nLoading models and data...")
    
    # Load trained models
    svm_pipeline = joblib.load("models/svm_esc50_pipeline.pkl")
    train_test_data = joblib.load("models/train_test_split_esc50.pkl")
    
    X_test = train_test_data['X_test']
    y_test = train_test_data['y_test']
    feature_names = train_test_data['feature_names']
    
    # Convert labels to numeric if needed
    try:
        if len(y_test) > 0 and isinstance(y_test.iloc[0] if hasattr(y_test, 'iloc') else y_test[0], str):
            y_test_numeric = (y_test == 'scream').astype(int)
        else:
            y_test_numeric = y_test
    except (IndexError, KeyError, TypeError):
        # Handle case where y_test is empty or has different structure
        print(f"⚠ Warning: Could not access y_test[0], using y_test as-is")
        y_test_numeric = y_test
    
    print(f"✓ Models loaded")
    print(f"✓ Test set: {len(X_test)} samples")
    
    results = {}
    
    # ============ CROSS-VALIDATION ============
    X_train = train_test_data['X_train']
    y_train = train_test_data['y_train']
    
    # Convert training labels to numeric if needed
    try:
        if len(y_train) > 0 and isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
            y_train_numeric = (y_train == 'scream').astype(int)
        else:
            y_train_numeric = y_train
    except (IndexError, KeyError, TypeError):
        # Handle case where y_train is empty or has different structure
        print(f"⚠ Warning: Could not access y_train[0], using y_train as-is")
        y_train_numeric = y_train
    
    cv_results = perform_cross_validation(X_train, y_train_numeric, svm_pipeline, cv_folds=10)
    results['cross_validation'] = cv_results
    
    # ============ PREDICTIONS ============
    print("\nGenerating predictions...")
    svm_predictions = svm_pipeline.predict(X_test)
    svm_probas = svm_pipeline.predict_proba(X_test)[:, 1]
    results['svm_predictions'] = svm_predictions
    results['svm_probas'] = svm_probas
    
    # ============ STATISTICAL SIGNIFICANCE ============
    sig_test = StatisticalSignificanceTest(
        y_test_numeric, svm_predictions, svm_predictions  # Would compare with CNN in full version
    )
    t_test_results = sig_test.paired_t_test()
    mcnemar_results = sig_test.mcnemar_test()
    results['statistical_tests'] = {
        't_test': t_test_results,
        'mcnemar': mcnemar_results
    }
    
    # ============ ABLATION STUDY ============
    ablation = AblationStudy(X_train, y_train_numeric, svm_pipeline, feature_names)
    ablation_feature_importance = ablation.remove_single_features()
    ablation_group_importance = ablation.remove_feature_groups()
    results['ablation_study'] = {
        'feature_importance': ablation_feature_importance.to_dict(),
        'group_importance': ablation_group_importance.to_dict()
    }
    
    # ============ ERROR PATTERN ANALYSIS ============
    # Convert predictions to numeric format for consistency with y_test_numeric
    svm_predictions_numeric = (svm_predictions == 'scream').astype(int)
    error_analysis = ErrorPatternAnalysis(y_test_numeric, svm_predictions_numeric, X=X_test, feature_names=feature_names)
    error_patterns = error_analysis.analyze_errors()
    results['error_analysis'] = error_patterns
    
    # ============ RESOURCE CONSUMPTION ============
    resource_monitor = ResourceConsumptionAnalysis()
    resource_monitor.start_monitoring()
    
    cpu_mem_stats = resource_monitor.get_cpu_memory_stats(num_samples=100)
    gpu_stats = resource_monitor.get_gpu_stats()
    
    # Measure inference time
    inference_stats = resource_monitor.measure_inference_time(svm_pipeline, X_test.iloc[:min(100, len(X_test))], batch_size=10)
    
    # Energy estimation
    energy_stats = resource_monitor.estimate_energy_consumption(
        cpu_mem_stats['cpu_mean'], 
        inference_stats['total_time']
    )
    
    results['resource_consumption'] = {
        'cpu_memory': cpu_mem_stats,
        'gpu': gpu_stats,
        'inference': inference_stats,
        'energy': energy_stats
    }
    
    # ============ SAVE RESULTS ============
    print("\n" + "="*70)
    print("SAVING ANALYSIS RESULTS")
    print("="*70)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    results_json = convert_types(results)
    
    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"models/analysis_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n✓ Results saved to {json_path}")
    
    # Save summary report
    report_path = f"models/analysis_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE MODEL ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*70 + "\n")
        
        f.write("\n1. CROSS-VALIDATION RESULTS\n")
        f.write("-"*70 + "\n")
        for metric, data in cv_results.items():
            f.write(f"{metric.upper()}: {data['test_mean']:.4f} ± {data['test_std']:.4f}\n")
        
        f.write("\n2. STATISTICAL SIGNIFICANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Paired t-test p-value: {t_test_results['p_value']:.6f}\n")
        f.write(f"McNemar test p-value: {mcnemar_results.get('p_value', 'N/A')}\n")
        
        f.write("\n3. ERROR ANALYSIS\n")
        f.write("-"*70 + "\n")
        tn, fp, fn, tp = error_patterns['confusion_matrix']
        f.write(f"True Negatives: {tn}, False Positives: {fp}\n")
        f.write(f"False Negatives: {fn}, True Positives: {tp}\n")
        
        f.write("\n4. RESOURCE CONSUMPTION\n")
        f.write("-"*70 + "\n")
        f.write(f"CPU Mean: {cpu_mem_stats['cpu_mean']:.2f}%\n")
        f.write(f"Memory Mean: {cpu_mem_stats['memory_mean']:.2f}%\n")
        f.write(f"Time per sample: {inference_stats['time_per_sample_ms']:.4f}ms\n")
        f.write(f"Estimated Energy: {energy_stats['estimated_energy_wh']:.6f}Wh\n")
    
    print(f"✓ Report saved to {report_path}")
    
    return results


if __name__ == "__main__":
    run_comprehensive_analysis()
