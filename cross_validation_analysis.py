"""
Cross-Validation Analysis with Stability Metrics
10-fold stratified cross-validation for model reliability assessment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, make_scorer
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CROSS-VALIDATION ANALYSIS WITH STABILITY METRICS")
print("10-fold stratified cross-validation for model reliability")
print("="*60)

class CrossValidationAnalysis:
    def __init__(self, X, y, model, cv_folds=10, random_state=42):
        self.X = X
        self.y = y
        self.model = model
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}
        
    def run_cross_validation(self):
        """Run comprehensive cross-validation analysis"""
        print(f"\nRunning {self.cv_folds}-fold stratified cross-validation...")
        
        # Define custom scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1': make_scorer(f1_score, average='weighted'),
            'roc_auc': make_scorer(roc_auc_score, average='weighted', needs_proba=True),
            'mcc': make_scorer(matthews_corrcoef)
        }
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Run cross-validation
        cv_results = cross_validate(
            self.model, self.X, self.y, 
            cv=skf, scoring=scoring, 
            return_train_score=True, n_jobs=-1
        )
        
        # Process results
        self.process_cv_results(cv_results)
        
        return self.results
    
    def process_cv_results(self, cv_results):
        """Process and analyze cross-validation results"""
        print("\nProcessing cross-validation results...")
        
        # Extract scores
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc']
        
        for metric in metrics:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            # Calculate statistics
            test_mean = np.mean(test_scores)
            test_std = np.std(test_scores, ddof=1)
            train_mean = np.mean(train_scores)
            train_std = np.std(train_scores, ddof=1)
            
            # Calculate confidence intervals (95%)
            ci_lower, ci_upper = self.calculate_ci(test_scores)
            
            # Calculate coefficient of variation (CV)
            cv_coefficient = (test_std / test_mean) * 100 if test_mean > 0 else 0
            
            # Calculate train-test gap (overfitting indicator)
            train_test_gap = train_mean - test_mean
            
            self.results[metric] = {
                'test_mean': test_mean,
                'test_std': test_std,
                'test_ci_lower': ci_lower,
                'test_ci_upper': ci_upper,
                'train_mean': train_mean,
                'train_std': train_std,
                'cv_coefficient': cv_coefficient,
                'train_test_gap': train_test_gap,
                'scores': test_scores
            }
            
            print(f"\n{metric.upper()}:")
            print(f"  Test:  {test_mean:.4f} ± {test_std:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"  Train: {train_mean:.4f} ± {train_std:.4f}")
            print(f"  CV:    {cv_coefficient:.2f}% (stability)")
            print(f"  Gap:   {train_test_gap:.4f} (overfitting)")
    
    def calculate_ci(self, scores, confidence=0.95):
        """Calculate confidence interval"""
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        se = std / np.sqrt(n)
        
        # t-distribution critical value
        from scipy.stats import t
        alpha = 1 - confidence
        t_critical = t.ppf(1 - alpha/2, n-1)
        
        ci_lower = mean - t_critical * se
        ci_upper = mean + t_critical * se
        
        return ci_lower, ci_upper
    
    def stability_assessment(self):
        """Assess model stability based on CV results"""
        print("\n" + "="*60)
        print("STABILITY ASSESSMENT")
        print("="*60)
        
        stability_report = {}
        
        for metric, results in self.results.items():
            if metric == 'scores':
                continue
                
            cv_coefficient = results['cv_coefficient']
            train_test_gap = results['train_test_gap']
            
            # Stability rating based on CV coefficient
            if cv_coefficient < 3:
                stability = "Excellent"
            elif cv_coefficient < 5:
                stability = "Good"
            elif cv_coefficient < 10:
                stability = "Fair"
            else:
                stability = "Poor"
            
            # Overfitting assessment
            if abs(train_test_gap) < 0.02:
                overfitting = "None"
            elif abs(train_test_gap) < 0.05:
                overfitting = "Minimal"
            elif abs(train_test_gap) < 0.10:
                overfitting = "Moderate"
            else:
                overfitting = "High"
            
            stability_report[metric] = {
                'stability': stability,
                'cv_coefficient': cv_coefficient,
                'overfitting': overfitting,
                'train_test_gap': train_test_gap
            }
            
            print(f"\n{metric.upper()}:")
            print(f"  Stability: {stability} (CV: {cv_coefficient:.2f}%)")
            print(f"  Overfitting: {overfitting} (Gap: {train_test_gap:.4f})")
        
        return stability_report
    
    def generate_report(self):
        """Generate comprehensive CV report"""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        # Stability assessment
        stability_report = self.stability_assessment()
        
        # Overall assessment
        print("\nOVERALL MODEL ASSESSMENT:")
        
        # Calculate average CV coefficient
        cv_coefficients = [results['cv_coefficient'] for metric, results in self.results.items() 
                          if metric != 'scores']
        avg_cv = np.mean(cv_coefficients)
        
        # Calculate average train-test gap
        train_test_gaps = [abs(results['train_test_gap']) for metric, results in self.results.items() 
                          if metric != 'scores']
        avg_gap = np.mean(train_test_gaps)
        
        # Overall stability rating
        if avg_cv < 5:
            overall_stability = "Very Stable"
        elif avg_cv < 10:
            overall_stability = "Stable"
        elif avg_cv < 15:
            overall_stability = "Moderately Stable"
        else:
            overall_stability = "Unstable"
        
        # Overall overfitting assessment
        if avg_gap < 0.03:
            overall_overfitting = "Minimal"
        elif avg_gap < 0.07:
            overall_overfitting = "Moderate"
        else:
            overall_overfitting = "High"
        
        print(f"Overall Stability: {overall_stability} (Avg CV: {avg_cv:.2f}%)")
        print(f"Overall Overfitting: {overall_overfitting} (Avg Gap: {avg_gap:.4f})")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        if avg_cv > 10:
            print("⚠ High variability detected. Consider:")
            print("  - Increasing training data")
            print("  - Feature selection/reduction")
            print("  - Regularization techniques")
        
        if avg_gap > 0.05:
            print("⚠ Overfitting detected. Consider:")
            print("  - Reducing model complexity")
            print("  - Increasing regularization")
            print("  - Early stopping")
        
        if avg_cv <= 5 and avg_gap <= 0.03:
            print("✓ Model shows excellent stability and minimal overfitting")
            print("✓ Ready for production deployment")
        
        return {
            'overall_stability': overall_stability,
            'overall_overfitting': overall_overfitting,
            'avg_cv_coefficient': avg_cv,
            'avg_train_test_gap': avg_gap,
            'stability_report': stability_report
        }

# ===== MAIN EXECUTION =====

def main():
    print("\nLoading data...")
    
    # Load training data
    try:
        train_test_data = joblib.load("models/train_test_split_esc50.pkl")
        X_train = train_test_data['X_train']
        y_train = train_test_data['y_train']
        print(f"✓ Training data loaded: {len(X_train)} samples")
    except FileNotFoundError:
        print("✗ Training data not found. Please run train_ml.py first.")
        return
    
    # Create SVM model (best performing typically)
    print("\nCreating SVM model...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
    
    # Run cross-validation analysis
    print("\nRunning cross-validation analysis...")
    cv_analyzer = CrossValidationAnalysis(X_train, y_train, svm_model, cv_folds=10)
    
    # Run CV
    cv_results = cv_analyzer.run_cross_validation()
    
    # Generate comprehensive report
    overall_assessment = cv_analyzer.generate_report()
    
    # Save results
    print("\nSaving results...")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = f"models/cv_report_SVM_{timestamp}.txt"
    with open(results_file, 'w') as f:
        f.write("CROSS-VALIDATION ANALYSIS REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Model: SVM with RBF kernel\n")
        f.write(f"CV Folds: 10 (Stratified)\n\n")
        
        # Write detailed metrics
        for metric, results in cv_results.items():
            if metric == 'scores':
                continue
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Test Mean ± Std: {results['test_mean']:.4f} ± {results['test_std']:.4f}\n")
            f.write(f"  95% CI: [{results['test_ci_lower']:.4f}, {results['test_ci_upper']:.4f}]\n")
            f.write(f"  Train Mean: {results['train_mean']:.4f}\n")
            f.write(f"  CV Coefficient: {results['cv_coefficient']:.2f}%\n")
            f.write(f"  Train-Test Gap: {results['train_test_gap']:.4f}\n\n")
        
        # Write overall assessment
        f.write("OVERALL ASSESSMENT:\n")
        f.write(f"Overall Stability: {overall_assessment['overall_stability']}\n")
        f.write(f"Overall Overfitting: {overall_assessment['overall_overfitting']}\n")
        f.write(f"Average CV Coefficient: {overall_assessment['avg_cv_coefficient']:.2f}%\n")
        f.write(f"Average Train-Test Gap: {overall_assessment['avg_train_test_gap']:.4f}\n")
    
    # Save summary CSV
    summary_file = f"models/cv_summary_SVM_{timestamp}.csv"
    summary_data = []
    for metric, results in cv_results.items():
        if metric == 'scores':
            continue
        summary_data.append({
            'Metric': metric.upper(),
            'Test_Mean': results['test_mean'],
            'Test_Std': results['test_std'],
            'Test_CI_Lower': results['test_ci_lower'],
            'Test_CI_Upper': results['test_ci_upper'],
            'CV_Coefficient': results['cv_coefficient'],
            'Train_Test_Gap': results['train_test_gap']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    
    print(f"✓ Detailed report saved: {results_file}")
    print(f"✓ Summary saved: {summary_file}")
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()