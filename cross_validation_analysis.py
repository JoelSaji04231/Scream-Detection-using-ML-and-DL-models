"""
Advanced Cross-Validation Module
Implements 10-fold stratified cross-validation with detailed metrics reporting
and fold-by-fold analysis for both SVM and CNN models
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')


class CrossValidationAnalysis:
    """Comprehensive cross-validation with 10-fold stratified split"""
    
    def __init__(self, X, y, model, cv_folds=10, stratified=True):
        """
        Args:
            X: Feature matrix
            y: Target labels
            model: Fitted model or pipeline
            cv_folds: Number of folds (default=10)
            stratified: Use stratified k-fold to preserve class distribution
        """
        self.X = X
        self.y = y
        self.model = model
        self.cv_folds = cv_folds
        self.stratified = stratified
        self.fold_results = []
        self.fold_predictions = []
        
    def run_cross_validation(self):
        """Run stratified k-fold cross-validation"""
        print("\n" + "="*80)
        print(f"{self.cv_folds}-FOLD STRATIFIED CROSS-VALIDATION")
        print("="*80)
        
        if self.stratified:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
        }
        
        # Run cross-validation
        cv_results = cross_validate(
            self.model, self.X, self.y,
            cv=cv, scoring=scoring,
            return_train_score=True,
            return_estimator=False  # Set True if you need each fold's estimator
        )
        
        # Store fold-by-fold results
        fold_data = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            fold_data[metric] = {
                'test_scores': test_scores,
                'train_scores': train_scores,
                'test_mean': test_scores.mean(),
                'test_std': test_scores.std(),
                'test_min': test_scores.min(),
                'test_max': test_scores.max(),
                'train_mean': train_scores.mean(),
                'train_std': train_scores.std(),
            }
        
        self._print_results(fold_data)
        self._print_fold_details(cv_results, scoring)
        
        return fold_data
    
    def _print_results(self, fold_data):
        """Print summary statistics"""
        print(f"\nSummary Statistics (Test Set):")
        print(f"\n{'Metric':<15} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
        print("-" * 63)
        
        for metric, data in fold_data.items():
            print(f"{metric:<15} {data['test_mean']:.4f}    {data['test_std']:.4f}    "
                  f"{data['test_min']:.4f}    {data['test_max']:.4f}")
        
        # Print train-test gap
        print(f"\n\nTrain-Test Gap (Overfitting Indicator):")
        print(f"{'Metric':<15} {'Gap':<12} {'Status':<20}")
        print("-" * 47)
        
        for metric, data in fold_data.items():
            gap = data['train_mean'] - data['test_mean']
            status = "GOOD" if gap < 0.05 else ("MODERATE" if gap < 0.10 else "OVERFITTING")
            print(f"{metric:<15} {gap:+.4f}    {status:<20}")
    
    def _print_fold_details(self, cv_results, scoring):
        """Print fold-by-fold breakdown"""
        print(f"\n\nFold-by-fold Results:")
        print(f"\n{'Fold':<6}", end='')
        for metric in scoring.keys():
            print(f" {metric:<12}", end='')
        print()
        print("-" * (6 + len(scoring) * 13))
        
        for fold_idx in range(self.cv_folds):
            print(f"{fold_idx+1:<6}", end='')
            for metric in scoring.keys():
                score = cv_results[f'test_{metric}'][fold_idx]
                print(f" {score:<12.4f}", end='')
            print()
    
    def get_fold_predictions(self):
        """Get predictions for each fold"""
        if self.stratified:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        all_predictions = np.zeros_like(self.y)
        all_probas = np.zeros((len(self.y), 2)) if hasattr(self.model, 'predict_proba') else None
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y)):
            X_train_fold, X_test_fold = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train_fold = self.y.iloc[train_idx]
            
            # Clone and fit model on training fold
            from sklearn.base import clone
            model_fold = clone(self.model)
            model_fold.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            all_predictions[test_idx] = model_fold.predict(X_test_fold)
            if all_probas is not None:
                all_probas[test_idx] = model_fold.predict_proba(X_test_fold)
        
        return all_predictions, all_probas


class CrossValidationComparison:
    """Compare two models using cross-validation"""
    
    def __init__(self, X, y, model1, model2, model1_name, model2_name, cv_folds=10):
        self.X = X
        self.y = y
        self.model1 = model1
        self.model2 = model2
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.cv_folds = cv_folds
        
    def compare(self):
        """Compare two models using cross-validation"""
        print("\n" + "="*80)
        print(f"CROSS-VALIDATION COMPARISON: {self.model1_name} vs {self.model2_name}")
        print("="*80)
        
        # Run CV for both models
        cv1 = CrossValidationAnalysis(self.X, self.y, self.model1, cv_folds=self.cv_folds)
        results1 = cv1.run_cross_validation()
        
        cv2 = CrossValidationAnalysis(self.X, self.y, self.model2, cv_folds=self.cv_folds)
        results2 = cv2.run_cross_validation()
        
        # Compare results
        print(f"\n\nDirect Comparison:")
        print(f"\n{'Metric':<15} {self.model1_name:<15} {self.model2_name:<15} {'Difference':<15}")
        print("-" * 62)
        
        for metric in results1.keys():
            score1 = results1[metric]['test_mean']
            score2 = results2[metric]['test_mean']
            diff = score2 - score1
            
            print(f"{metric:<15} {score1:.4f}         {score2:.4f}         {diff:+.4f}")
        
        # Statistical test (paired t-test on fold scores)
        from scipy import stats
        
        print(f"\n\nStatistical Significance Test (Paired t-test):")
        print(f"\nTesting if {self.model2_name} is significantly better than {self.model1_name}")
        
        accuracy_diff = results2['accuracy']['test_scores'] - results1['accuracy']['test_scores']
        t_stat, p_value = stats.ttest_rel(accuracy_diff, np.zeros_like(accuracy_diff))
        
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            better = self.model2_name if np.mean(accuracy_diff) > 0 else self.model1_name
            print(f"  ✓ SIGNIFICANT (p < 0.05) - {better} is significantly better")
        else:
            print(f"  ✗ NOT SIGNIFICANT (p ≥ 0.05) - No significant difference")
        
        return results1, results2


class CrossValidationReporter:
    """Generate detailed reports from cross-validation results"""
    
    def __init__(self, cv_analysis, model_name):
        self.cv_analysis = cv_analysis
        self.model_name = model_name
    
    def generate_report(self, output_dir="models"):
        """Generate comprehensive CV report"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"cv_report_{self.model_name}_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"CROSS-VALIDATION REPORT: {self.model_name.upper()}\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("="*80 + "\n")
            
            f.write(f"\nConfiguration:\n")
            f.write(f"  CV Folds: {self.cv_analysis.cv_folds}\n")
            f.write(f"  Stratified: {self.cv_analysis.stratified}\n")
            f.write(f"  Sample Size: {len(self.cv_analysis.X)}\n")
            f.write(f"  Number of Features: {self.cv_analysis.X.shape[1]}\n")
            
            # Run analysis and save results
            results = self.cv_analysis.run_cross_validation()
            
            f.write(f"\n\nDetailed Metrics:\n")
            for metric, data in results.items():
                f.write(f"\n{metric.upper()}:\n")
                f.write(f"  Test Set - Mean: {data['test_mean']:.4f}, Std: {data['test_std']:.4f}\n")
                f.write(f"  Test Set - Min: {data['test_min']:.4f}, Max: {data['test_max']:.4f}\n")
                f.write(f"  Train Set - Mean: {data['train_mean']:.4f}, Std: {data['train_std']:.4f}\n")
                f.write(f"  Train-Test Gap: {data['train_mean'] - data['test_mean']:+.4f}\n")
        
        print(f"\n✓ Report saved to {report_path}")
        return report_path


# ============================================================================
# UTILITY: 10-Fold CV Summary Table
# ============================================================================

def generate_cv_summary_table(results_dict, output_path="models/cv_summary.csv"):
    """
    Generate a CSV summary of cross-validation results for multiple models
    
    Args:
        results_dict: Dictionary of {model_name: cv_results}
    """
    summary_data = []
    
    for model_name, results in results_dict.items():
        for metric, data in results.items():
            summary_data.append({
                'Model': model_name,
                'Metric': metric,
                'Mean': data['test_mean'],
                'Std': data['test_std'],
                'Min': data['test_min'],
                'Max': data['test_max'],
                'CI_Lower': data['test_mean'] - 1.96 * data['test_std'],
                'CI_Upper': data['test_mean'] + 1.96 * data['test_std']
            })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_path, index=False)
    print(f"✓ Summary table saved to {output_path}")
    
    return df_summary


# ============================================================================
# MAIN FUNCTION: Run 10-fold CV on SVM
# ============================================================================

def run_svm_cross_validation():
    """Run 10-fold cross-validation on SVM model"""
    print("\n" + "="*80)
    print("SVM MODEL: 10-FOLD STRATIFIED CROSS-VALIDATION")
    print("="*80)
    
    # Load data
    try:
        train_test_data = joblib.load("models/train_test_split_esc50.pkl")
        print("✓ Data loaded")
    except:
        print("✗ Data file not found. Please run train_ml.py first")
        return
    
    # Load SVM
    try:
        svm_pipeline = joblib.load("models/svm_esc50_pipeline.pkl")
        print("✓ SVM model loaded")
    except:
        print("✗ SVM model not found. Please run train_ml.py first")
        return
    
    # Use training data for cross-validation
    X_train = train_test_data['X_train']
    y_train = train_test_data['y_train']
    
    # Convert labels to numeric if needed
    y_train_numeric = (y_train == 'scream').astype(int) if isinstance(y_train[0], str) else y_train
    
    # Run CV
    cv_analysis = CrossValidationAnalysis(
        X_train, y_train_numeric, svm_pipeline,
        cv_folds=10, stratified=True
    )
    cv_results = cv_analysis.run_cross_validation()
    
    # Generate report
    reporter = CrossValidationReporter(cv_analysis, "SVM")
    reporter.generate_report()
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_dict = {
        'SVM': cv_results
    }
    
    # Save summary table
    df_summary = generate_cv_summary_table(results_dict, f"models/cv_summary_{timestamp}.csv")
    print(f"\n{df_summary.to_string()}")
    
    return cv_results


if __name__ == "__main__":
    run_svm_cross_validation()
