"""
Advanced Statistical Significance Testing for Model Comparison
Compares SVM vs CNN with proper statistical tests
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import stats
from scipy.stats import ttest_rel, chi2_contingency, norm
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ADVANCED STATISTICAL SIGNIFICANCE TESTING")
print("Comparing SVM vs CNN with proper statistical tests")
print("="*60)

class StatisticalModelComparison:
    def __init__(self, svm_model, cnn_model, X_test, y_test, device='cpu'):
        self.svm_model = svm_model
        self.cnn_model = cnn_model
        self.X_test = X_test
        self.y_test = y_test
        self.device = device
        self.results = {}
        
    def get_svm_predictions(self, X_test):
        """Get SVM predictions and probabilities"""
        scaler = self.svm_model['scaler']
        model = self.svm_model['model']
        
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        
        return predictions, probabilities
    
    def get_cnn_predictions(self, X_test):
        """Get CNN predictions and probabilities"""
        self.cnn_model.eval()
        
        with torch.no_grad():
            if isinstance(X_test, np.ndarray):
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            else:
                X_test_tensor = X_test.to(self.device)
                
            outputs = self.cnn_model(X_test_tensor)
            probabilities = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
        return predictions, probabilities
    
    def paired_t_test(self, svm_scores, cnn_scores):
        """Paired t-test for accuracy comparison"""
        print("\n1. PAIRED T-TEST (Accuracy Comparison)")
        print("-" * 40)
        
        # Check if we have the same number of samples
        if len(svm_scores) != len(cnn_scores):
            print(f"✗ Different sample sizes: SVM={len(svm_scores)}, CNN={len(cnn_scores)}")
            return None
            
        # Calculate differences
        differences = svm_scores - cnn_scores
        
        # Paired t-test
        t_stat, p_value = ttest_rel(svm_scores, cnn_scores)
        
        # Effect size (Cohen's d for paired samples)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Confidence interval for mean difference
        n = len(differences)
        se_diff = std_diff / np.sqrt(n)
        t_critical = stats.t.ppf(0.975, n-1)  # 95% CI
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        print(f"SVM Mean Accuracy:  {np.mean(svm_scores):.4f} ± {np.std(svm_scores):.4f}")
        print(f"CNN Mean Accuracy:  {np.mean(cnn_scores):.4f} ± {np.std(cnn_scores):.4f}")
        print(f"Mean Difference:    {mean_diff:.4f}")
        print(f"95% CI for Diff:    [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"t-statistic:        {t_stat:.4f}")
        print(f"p-value:            {p_value:.4f}")
        print(f"Cohen's d:          {cohens_d:.4f}")
        
        # Interpretation
        if p_value < 0.001:
            significance = "*** HIGHLY SIGNIFICANT ***"
        elif p_value < 0.01:
            significance = "** VERY SIGNIFICANT **"
        elif p_value < 0.05:
            significance = "* SIGNIFICANT *"
        else:
            significance = "NOT SIGNIFICANT"
            
        print(f"Result:             {significance}")
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_size = "Negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "Small"
        elif abs(cohens_d) < 0.8:
            effect_size = "Medium"
        else:
            effect_size = "Large"
            
        print(f"Effect Size:        {effect_size}")
        
        self.results['paired_t_test'] = {
            'svm_mean': np.mean(svm_scores),
            'svm_std': np.std(svm_scores),
            'cnn_mean': np.mean(cnn_scores),
            'cnn_std': np.std(cnn_scores),
            'mean_difference': mean_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significance': significance,
            'effect_size': effect_size
        }
        
        return self.results['paired_t_test']
    
    def mcnemars_test(self, svm_predictions, cnn_predictions):
        """McNemar's test for error pattern comparison"""
        print("\n2. MCNEMAR'S TEST (Error Pattern Comparison)")
        print("-" * 40)
        
        # Create contingency table
        # Rows: SVM Correct/Incorrect, Columns: CNN Correct/Incorrect
        contingency_table = confusion_matrix(svm_predictions == self.y_test, 
                                           cnn_predictions == self.y_test)
        
        # McNemar's test
        # H0: Both models have the same error patterns
        # H1: Models have different error patterns
        
        # Extract the discordant pairs (b and c cells)
        b = contingency_table[0, 1]  # SVM correct, CNN incorrect
        c = contingency_table[1, 0]  # SVM incorrect, CNN correct
        
        # McNemar's statistic
        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1)**2 / (b + c)  # With continuity correction
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat = 0
            p_value = 1.0
        
        print(f"SVM Correct, CNN Incorrect: {b}")
        print(f"SVM Incorrect, CNN Correct: {c}")
        print(f"McNemar's χ²: {mcnemar_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        # Interpretation
        if p_value < 0.05:
            result = "Models have SIGNIFICANTLY different error patterns"
        else:
            result = "Models have similar error patterns"
            
        print(f"Result: {result}")
        
        self.results['mcnemars_test'] = {
            'svm_cnn_incorrect': b,
            'cnn_svm_incorrect': c,
            'mcnemar_statistic': mcnemar_stat,
            'p_value': p_value,
            'result': result
        }
        
        return self.results['mcnemars_test']
    
    def two_proportion_z_test(self, svm_predictions, cnn_predictions):
        """Two-proportion z-test for FPR and FNR comparison"""
        print("\n3. TWO-PROPORTION Z-TEST (FPR/FNR Comparison)")
        print("-" * 40)
        
        # Calculate confusion matrices
        svm_cm = confusion_matrix(self.y_test, svm_predictions)
        cnn_cm = confusion_matrix(self.y_test, cnn_predictions)
        
        # Calculate rates
        def calculate_rates(cm):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            return fpr, fnr, tn, fp, fn, tp
        
        svm_fpr, svm_fnr, svm_tn, svm_fp, svm_fn, svm_tp = calculate_rates(svm_cm)
        cnn_fpr, cnn_fnr, cnn_tn, cnn_fp, cnn_fn, cnn_tp = calculate_rates(cnn_cm)
        
        print(f"SVM - FPR: {svm_fpr:.4f}, FNR: {svm_fnr:.4f}")
        print(f"CNN - FPR: {cnn_fpr:.4f}, FNR: {cnn_fnr:.4f}")
        
        # Two-proportion z-test for FPR
        n1_svm = svm_fp + svm_tn
        n2_cnn = cnn_fp + cnn_tn
        
        if n1_svm > 0 and n2_cnn > 0:
            p1 = svm_fpr
            p2 = cnn_fpr
            n1 = n1_svm
            n2 = n2_cnn
            
            p_pooled = (svm_fp + cnn_fp) / (n1 + n2)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            
            if se > 0:
                z_fpr = (p1 - p2) / se
                p_value_fpr = 2 * (1 - norm.cdf(abs(z_fpr)))
            else:
                z_fpr = 0
                p_value_fpr = 1.0
        else:
            z_fpr = 0
            p_value_fpr = 1.0
        
        # Two-proportion z-test for FNR
        n1_svm = svm_fn + svm_tp
        n2_cnn = cnn_fn + cnn_tp
        
        if n1_svm > 0 and n2_cnn > 0:
            p1 = svm_fnr
            p2 = cnn_fnr
            n1 = n1_svm
            n2 = n2_cnn
            
            p_pooled = (svm_fn + cnn_fn) / (n1 + n2)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            
            if se > 0:
                z_fnr = (p1 - p2) / se
                p_value_fnr = 2 * (1 - norm.cdf(abs(z_fnr)))
            else:
                z_fnr = 0
                p_value_fnr = 1.0
        else:
            z_fnr = 0
            p_value_fnr = 1.0
        
        print(f"FPR Z-test: z={z_fpr:.4f}, p={p_value_fpr:.4f}")
        print(f"FNR Z-test: z={z_fnr:.4f}, p={p_value_fnr:.4f}")
        
        # Interpretation
        if p_value_fpr < 0.05:
            fpr_result = "SIGNIFICANT difference in FPR"
        else:
            fpr_result = "No significant difference in FPR"
            
        if p_value_fnr < 0.05:
            fnr_result = "SIGNIFICANT difference in FNR"
        else:
            fnr_result = "No significant difference in FNR"
            
        print(f"FPR Result: {fpr_result}")
        print(f"FNR Result: {fnr_result}")
        
        self.results['two_proportion_z_test'] = {
            'svm_fpr': svm_fpr,
            'svm_fnr': svm_fnr,
            'cnn_fpr': cnn_fpr,
            'cnn_fnr': cnn_fnr,
            'z_fpr': z_fpr,
            'p_value_fpr': p_value_fpr,
            'z_fnr': z_fnr,
            'p_value_fnr': p_value_fnr,
            'fpr_result': fpr_result,
            'fnr_result': fnr_result
        }
        
        return self.results['two_proportion_z_test']
    
    def cohens_kappa(self, svm_predictions, cnn_predictions):
        """Cohen's Kappa for inter-rater agreement"""
        print("\n4. COHEN'S KAPPA (Inter-rater Agreement)")
        print("-" * 40)
        
        # Calculate observed agreement
        observed_agreement = np.mean(svm_predictions == cnn_predictions)
        
        # Calculate expected agreement
        svm_probs = np.bincount(svm_predictions) / len(svm_predictions)
        cnn_probs = np.bincount(cnn_predictions) / len(cnn_predictions)
        expected_agreement = np.sum(svm_probs * cnn_probs)
        
        # Calculate Cohen's Kappa
        if (1 - expected_agreement) > 0:
            kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        else:
            kappa = 0
        
        # Standard error of kappa
        n = len(svm_predictions)
        se_kappa = np.sqrt(observed_agreement * (1 - observed_agreement) / (n * (1 - expected_agreement)**2))
        
        # 95% confidence interval
        z_critical = 1.96
        kappa_ci_lower = kappa - z_critical * se_kappa
        kappa_ci_upper = kappa + z_critical * se_kappa
        
        print(f"Observed Agreement: {observed_agreement:.4f}")
        print(f"Expected Agreement: {expected_agreement:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print(f"95% CI: [{kappa_ci_lower:.4f}, {kappa_ci_upper:.4f}]")
        
        # Interpretation
        if kappa < 0:
            agreement_level = "Poor"
        elif kappa < 0.20:
            agreement_level = "Slight"
        elif kappa < 0.40:
            agreement_level = "Fair"
        elif kappa < 0.60:
            agreement_level = "Moderate"
        elif kappa < 0.80:
            agreement_level = "Substantial"
        else:
            agreement_level = "Almost Perfect"
            
        print(f"Agreement Level: {agreement_level}")
        
        self.results['cohens_kappa'] = {
            'observed_agreement': observed_agreement,
            'expected_agreement': expected_agreement,
            'kappa': kappa,
            'ci_lower': kappa_ci_lower,
            'ci_upper': kappa_ci_upper,
            'agreement_level': agreement_level
        }
        
        return self.results['cohens_kappa']
    
    def run_all_tests(self):
        """Run all statistical tests"""
        print("\n" + "="*60)
        print("RUNNING ALL STATISTICAL TESTS")
        print("="*60)
        
        # Get predictions
        print("\nGetting model predictions...")
        svm_predictions, svm_probabilities = self.get_svm_predictions(self.X_test)
        cnn_predictions, cnn_probabilities = self.get_cnn_predictions(self.X_test)
        
        # Calculate individual accuracies
        svm_accuracy = accuracy_score(self.y_test, svm_predictions)
        cnn_accuracy = accuracy_score(self.y_test, cnn_predictions)
        
        print(f"\nSVM Accuracy: {svm_accuracy:.4f}")
        print(f"CNN Accuracy: {cnn_accuracy:.4f}")
        
        # Run all tests
        self.paired_t_test(np.array([svm_accuracy]), np.array([cnn_accuracy]))
        self.mcnemars_test(svm_predictions, cnn_predictions)
        self.two_proportion_z_test(svm_predictions, cnn_predictions)
        self.cohens_kappa(svm_predictions, cnn_predictions)
        
        return self.results

# ===== MAIN EXECUTION =====

def main():
    # Load models and data
    print("\nLoading models and data...")
    
    try:
        # Load SVM pipeline
        svm_pipeline = joblib.load("models/svm_esc50_pipeline.pkl")
        print("✓ SVM model loaded")
    except FileNotFoundError:
        print("✗ SVM model not found. Please run train_ml.py first.")
        return
    
    try:
        # Load CNN model
        from train_cnn import SpectrogramCNN
        cnn_model = SpectrogramCNN().to(device)
        cnn_model.load_state_dict(torch.load("models/cnn_spectrogram.pth", map_location=device))
        cnn_model.eval()
        print("✓ CNN model loaded")
    except FileNotFoundError:
        print("✗ CNN model not found. Please run train_cnn.py first.")
        return
    
    # Load test data
    try:
        # Load CNN test data
        X_test_cnn = np.load("data/cnn_spectrograms.npy")
        y_test_cnn = np.load("data/cnn_labels.npy")
        
        # Load SVM test data
        train_test_data = joblib.load("models/train_test_split_esc50.pkl")
        X_test_svm = train_test_data['X_test']
        y_test_svm = train_test_data['y_test']
        
        print(f"✓ Test data loaded")
        print(f"  - SVM test samples: {len(X_test_svm)}")
        print(f"  - CNN test samples: {len(X_test_cnn)}")
        
    except FileNotFoundError as e:
        print(f"✗ Test data not found: {e}")
        print("Please ensure you have run the training scripts first.")
        return
    
    # Run statistical comparison
    print("\nRunning statistical comparison...")
    
    # For this comparison, we'll use a subset to ensure balanced comparison
    min_samples = min(len(X_test_svm), len(X_test_cnn))
    
    # Take random subsets for fair comparison
    np.random.seed(42)
    svm_indices = np.random.choice(len(X_test_svm), min_samples, replace=False)
    cnn_indices = np.random.choice(len(X_test_cnn), min_samples, replace=False)
    
    X_test_svm_balanced = X_test_svm[svm_indices]
    y_test_balanced = y_test_svm[svm_indices]
    X_test_cnn_balanced = X_test_cnn[cnn_indices]
    
    # Convert CNN data to tensor
    X_test_cnn_tensor = torch.FloatTensor(X_test_cnn_balanced).to(device)
    
    # Create comparison object
    comparator = StatisticalModelComparison(
        svm_pipeline, cnn_model, 
        X_test_svm_balanced, y_test_balanced, device
    )
    
    # Run all tests
    results = comparator.run_all_tests()
    
    # Save results
    print("\nSaving results...")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"models/svm_cnn_comparison_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON SUMMARY")
    print("="*60)
    
    if 'paired_t_test' in results and results['paired_t_test']:
        t_test = results['paired_t_test']
        print(f"Paired t-test: {t_test['significance']} (p={t_test['p_value']:.4f})")
        print(f"Effect size: {t_test['effect_size']} (Cohen's d={t_test['cohens_d']:.4f})")
    
    if 'mcnemars_test' in results and results['mcnemars_test']:
        mcnemar = results['mcnemars_test']
        print(f"McNemar's test: {mcnemar['result']} (p={mcnemar['p_value']:.4f})")
    
    if 'cohens_kappa' in results and results['cohens_kappa']:
        kappa = results['cohens_kappa']
        print(f"Cohen's Kappa: {kappa['agreement_level']} (κ={kappa['kappa']:.4f})")
    
    print("\n✓ Statistical comparison complete!")

if __name__ == "__main__":
    main()