"""
Comprehensive Ablation Study Module
Systematically removes features to measure their contribution to model accuracy.
Includes:
- Leave-One-Out feature importance
- Feature group removal analysis
- Permutation-based importance
- Visualization of feature contributions
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


class FeatureAblationStudy:
    """Leave-one-out feature importance analysis"""
    
    def __init__(self, X, y, model, feature_names, cv_folds=5):
        """
        Args:
            X: Feature matrix
            y: Target labels
            model: Fitted model or pipeline
            feature_names: Names of features
            cv_folds: Cross-validation folds for stability
        """
        self.X = X
        self.y = y
        self.model = model
        self.feature_names = np.array(feature_names)
        self.cv_folds = cv_folds
        self.baseline_score = None
        self.ablation_results = []
        
    def calculate_baseline(self):
        """Calculate baseline accuracy with all features"""
        scores = cross_val_score(
            self.model, self.X, self.y,
            cv=self.cv_folds, scoring='accuracy'
        )
        self.baseline_score = scores.mean()
        
        print("\n" + "="*80)
        print("BASELINE PERFORMANCE (All Features)")
        print("="*80)
        print(f"\nBaseline Accuracy: {self.baseline_score:.4f}")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std:  {scores.std():.4f}")
        print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
        
        return self.baseline_score
    
    def ablate_single_features(self):
        """Remove features one at a time and measure impact"""
        print("\n" + "="*80)
        print("FEATURE ABLATION: Leave-One-Out Analysis")
        print("="*80)
        
        self.calculate_baseline()
        
        print(f"\nRemoving features one at a time...")
        print(f"{'Feature':<30} {'Accuracy':<12} {'Drop':<12} {'Importance':<12}")
        print("-" * 66)
        
        for idx in range(len(self.feature_names)):
            # Create X without this feature
            feature_mask = np.ones(self.X.shape[1], dtype=bool)
            feature_mask[idx] = False
            X_ablated = self.X.iloc[:, feature_mask]
            
            # Calculate accuracy without this feature
            scores = cross_val_score(
                self.model, X_ablated, self.y,
                cv=self.cv_folds, scoring='accuracy'
            )
            accuracy_without = scores.mean()
            
            # Calculate importance (how much accuracy dropped)
            importance = self.baseline_score - accuracy_without
            
            self.ablation_results.append({
                'feature': self.feature_names[idx],
                'idx': idx,
                'accuracy_without': accuracy_without,
                'importance': importance,
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            })
            
            status = "✓ CRITICAL" if importance > 0.05 else ("• HELPFUL" if importance > 0.01 else "  negligible")
            print(f"{self.feature_names[idx]:<30} {accuracy_without:<12.4f} "
                  f"{importance:<12.4f} {status:<12}")
        
        # Sort by importance
        importance_df = pd.DataFrame(self.ablation_results).sort_values('importance', ascending=False)
        
        print(f"\n\nTop 15 Most Important Features:")
        print(importance_df[['feature', 'importance', 'accuracy_without']].head(15).to_string(index=False))
        
        return importance_df
    
    def ablate_feature_groups(self):
        """Remove groups of related features"""
        print("\n" + "="*80)
        print("FEATURE GROUP ABLATION")
        print("="*80)
        
        # Define feature groups based on typical feature names
        feature_groups = {
            'MFCC': [i for i, f in enumerate(self.feature_names) if 'mfcc' in f.lower()],
            'Spectral': [i for i, f in enumerate(self.feature_names) if 'spectral' in f.lower() or 'centroid' in f.lower()],
            'Spectral Rolloff': [i for i, f in enumerate(self.feature_names) if 'rolloff' in f.lower()],
            'Spectral Bandwidth': [i for i, f in enumerate(self.feature_names) if 'bandwidth' in f.lower()],
            'Zero Crossing Rate': [i for i, f in enumerate(self.feature_names) if 'zcr' in f.lower() or 'zero' in f.lower()],
            'RMS Energy': [i for i, f in enumerate(self.feature_names) if 'rms' in f.lower()],
            'Chroma': [i for i, f in enumerate(self.feature_names) if 'chroma' in f.lower()],
            'Temporal': [i for i, f in enumerate(self.feature_names) if 'delta' in f.lower() or 'temp' in f.lower()],
            'Contrast': [i for i, f in enumerate(self.feature_names) if 'contrast' in f.lower()],
            'Tempogram': [i for i, f in enumerate(self.feature_names) if 'tempogram' in f.lower()]
        }
        
        # Only include groups that exist
        feature_groups = {k: v for k, v in feature_groups.items() if len(v) > 0}
        
        print(f"\n{'Group':<25} {'Count':<8} {'Accuracy':<12} {'Drop':<12} {'Status':<15}")
        print("-" * 72)
        
        group_results = []
        
        for group_name, indices in sorted(feature_groups.items(), key=lambda x: len(x[1]), reverse=True):
            # Create mask without this group
            feature_mask = np.ones(self.X.shape[1], dtype=bool)
            for idx in indices:
                feature_mask[idx] = False
            
            X_ablated = self.X.iloc[:, feature_mask]
            
            # Calculate accuracy without this group
            scores = cross_val_score(
                self.model, X_ablated, self.y,
                cv=self.cv_folds, scoring='accuracy'
            )
            accuracy_without = scores.mean()
            importance = self.baseline_score - accuracy_without
            
            status = "⚠ CRITICAL" if importance > 0.10 else ("✓ HELPFUL" if importance > 0.02 else "  negligible")
            
            group_results.append({
                'group': group_name,
                'feature_count': len(indices),
                'accuracy_without': accuracy_without,
                'importance': importance,
                'std': scores.std()
            })
            
            print(f"{group_name:<25} {len(indices):<8} {accuracy_without:<12.4f} "
                  f"{importance:<12.4f} {status:<15}")
        
        return pd.DataFrame(group_results).sort_values('importance', ascending=False)
    
    def permutation_importance_analysis(self):
        """Calculate permutation-based feature importance"""
        print("\n" + "="*80)
        print("PERMUTATION-BASED FEATURE IMPORTANCE")
        print("="*80)
        print("\nCalculating permutation importance (this may take a while)...")
        
        # Fit model on full data first
        from sklearn.base import clone
        model_fitted = clone(self.model)
        model_fitted.fit(self.X, self.y)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model_fitted, self.X, self.y,
            n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Create results dataframe
        perm_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std,
            'min': perm_importance.importances[:, :].min(axis=1),
            'max': perm_importance.importances[:, :].max(axis=1)
        }).sort_values('importance', ascending=False)
        
        print(f"\n{'Feature':<30} {'Importance':<15} {'Std Dev':<12}")
        print("-" * 57)
        
        for _, row in perm_df.head(15).iterrows():
            print(f"{row['feature']:<30} {row['importance']:<15.6f} {row['std']:<12.6f}")
        
        return perm_df


class FeatureInteractionAnalysis:
    """Analyze interactions between features"""
    
    def __init__(self, X, y, model, feature_names, cv_folds=5):
        self.X = X
        self.y = y
        self.model = model
        self.feature_names = np.array(feature_names)
        self.cv_folds = cv_folds
    
    def pairwise_feature_interaction(self, top_n=20):
        """Test if removing pairs of features reveals interactions"""
        print("\n" + "="*80)
        print("PAIRWISE FEATURE INTERACTION ANALYSIS")
        print("="*80)
        
        baseline = cross_val_score(
            self.model, self.X, self.y,
            cv=self.cv_folds, scoring='accuracy'
        ).mean()
        
        print(f"\nBaseline accuracy: {baseline:.4f}")
        print(f"Analyzing top {top_n} feature pairs...")
        
        # Calculate single feature importance first
        single_importance = []
        for idx in range(len(self.feature_names)):
            mask = np.ones(self.X.shape[1], dtype=bool)
            mask[idx] = False
            score = cross_val_score(
                self.model, self.X.iloc[:, mask], self.y,
                cv=self.cv_folds, scoring='accuracy'
            ).mean()
            single_importance.append(baseline - score)
        
        # Get top features
        top_indices = np.argsort(single_importance)[-top_n:]
        
        interactions = []
        
        for i, idx1 in enumerate(top_indices):
            for idx2 in top_indices[i+1:]:
                # Remove both features
                mask = np.ones(self.X.shape[1], dtype=bool)
                mask[idx1] = False
                mask[idx2] = False
                
                score_both = cross_val_score(
                    self.model, self.X.iloc[:, mask], self.y,
                    cv=self.cv_folds, scoring='accuracy'
                ).mean()
                
                drop_both = baseline - score_both
                expected_drop = single_importance[idx1] + single_importance[idx2]
                interaction = drop_both - expected_drop
                
                if abs(interaction) > 0.01:  # Only report non-negligible interactions
                    interactions.append({
                        'feature1': self.feature_names[idx1],
                        'feature2': self.feature_names[idx2],
                        'interaction': interaction,
                        'synergy': 'positive' if interaction > 0 else 'negative'
                    })
        
        # Handle empty interactions list
        if not interactions:
            print(f"\nNo significant feature interactions detected")
            return pd.DataFrame(columns=['feature1', 'feature2', 'interaction', 'synergy'])
        
        interactions_df = pd.DataFrame(interactions).sort_values('interaction', key=abs, ascending=False)
        
        if len(interactions_df) > 0:
            print(f"\nTop Feature Interactions:")
            print(interactions_df.head(10).to_string(index=False))
        else:
            print(f"\nNo significant feature interactions detected")
        
        return interactions_df


class AblationStudyReporter:
    """Generate reports and visualizations"""
    
    def __init__(self, ablation_df, group_df=None, perm_df=None):
        self.ablation_df = ablation_df
        self.group_df = group_df
        self.perm_df = perm_df
    
    def generate_report(self, model_name="SVM", output_dir="models"):
        """Generate detailed ablation study report"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"ablation_report_{model_name}_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"ABLATION STUDY REPORT: {model_name.upper()}\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("="*80 + "\n")
            
            f.write(f"\nFEATURE-BY-FEATURE ANALYSIS:\n")
            f.write(f"Top 20 Most Important Features\n")
            f.write("-"*80 + "\n")
            f.write(self.ablation_df.head(20)[['feature', 'importance', 'accuracy_without']].to_string(index=False))
            
            if self.group_df is not None:
                f.write(f"\n\nFEATURE GROUP ANALYSIS:\n")
                f.write(f"Feature Groups by Importance\n")
                f.write("-"*80 + "\n")
                f.write(self.group_df.to_string(index=False))
            
            if self.perm_df is not None:
                f.write(f"\n\nPERMUTATION-BASED IMPORTANCE:\n")
                f.write(f"Top 20 Features by Permutation Importance\n")
                f.write("-"*80 + "\n")
                f.write(self.perm_df.head(20)[['feature', 'importance', 'std']].to_string(index=False))
        
        print(f"\n✓ Report saved to {report_path}")
        
        # Save CSV files
        csv_path = os.path.join(output_dir, f"ablation_results_{model_name}_{timestamp}.csv")
        self.ablation_df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to {csv_path}")
        
        return report_path
    
    def plot_feature_importance(self, top_n=20, output_dir="images"):
        """Create visualization of feature importance"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        top_features = self.ablation_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['#d62728' if x > 0.05 else '#1f77b4' for x in top_features['importance']]
        ax.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance (Accuracy Drop)')
        ax.set_title(f'Top {top_n} Most Important Features')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"feature_importance_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {plot_path}")
        plt.close()
        
        return plot_path


# ============================================================================
# MAIN FUNCTION: Run Complete Ablation Study
# ============================================================================

def run_complete_ablation_study():
    """Run comprehensive ablation study on SVM model"""
    print("\n" + "="*80)
    print("COMPLETE ABLATION STUDY: SVM MODEL")
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
    
    X_train = train_test_data['X_train']
    y_train = train_test_data['y_train']
    feature_names = train_test_data['feature_names']
    
    # Convert labels to numeric
    y_train_numeric = (y_train == 'scream').astype(int) if isinstance(y_train[0], str) else y_train
    
    # ============ ABLATION STUDY ============
    ablation = FeatureAblationStudy(X_train, y_train_numeric, svm_pipeline, feature_names, cv_folds=5)
    ablation_results = ablation.ablate_single_features()
    group_results = ablation.ablate_feature_groups()
    perm_results = ablation.permutation_importance_analysis()
    
    # ============ INTERACTION ANALYSIS ============
    interaction = FeatureInteractionAnalysis(X_train, y_train_numeric, svm_pipeline, feature_names, cv_folds=5)
    interaction_results = interaction.pairwise_feature_interaction(top_n=15)
    
    # ============ GENERATE REPORTS ============
    reporter = AblationStudyReporter(ablation_results, group_results, perm_results)
    reporter.generate_report(model_name="SVM")
    reporter.plot_feature_importance(top_n=20)
    
    print(f"\n✓ Ablation study complete!")
    
    return {
        'ablation': ablation_results,
        'groups': group_results,
        'permutation': perm_results,
        'interactions': interaction_results
    }


if __name__ == "__main__":
    run_complete_ablation_study()
