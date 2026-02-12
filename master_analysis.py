"""
MASTER ANALYSIS SCRIPT
Orchestrates all advanced analyses in sequence and generates a comprehensive report.
Runs:
1. Cross-Validation Analysis
2. SVM vs CNN Comparison (with statistical tests)
3. Ablation Study
4. Error Pattern Analysis
5. Resource Consumption Monitoring
6. Final Comprehensive Report
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}\n")


def print_status(message, status='info'):
    """Print status message with appropriate color"""
    if status == 'success':
        print(f"{Colors.GREEN}✓{Colors.ENDC} {message}")
    elif status == 'error':
        print(f"{Colors.RED}✗{Colors.ENDC} {message}")
    elif status == 'warning':
        print(f"{Colors.YELLOW}⚠{Colors.ENDC} {message}")
    else:
        print(f"{Colors.BLUE}❯{Colors.ENDC} {message}")


def run_analysis_script(script_name, description):
    """Run one of the analysis scripts"""
    print_section(description)
    
    try:
        # Check if script exists
        if not os.path.exists(script_name):
            print_status(f"Script {script_name} not found!", 'error')
            return False
        
        # Run the script
        print_status(f"Starting: {description}", 'info')
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print_status(f"Completed: {description}", 'success')
            return True
        else:
            print_status(f"Failed: {description}", 'error')
            return False
            
    except Exception as e:
        print_status(f"Error running {script_name}: {str(e)}", 'error')
        return False


def verify_requirements():
    """Verify that all required files exist"""
    print_section("VERIFYING REQUIREMENTS")
    
    required_files = [
        'models/svm_esc50_pipeline.pkl',
        'models/train_test_split_esc50.pkl',
        'models/cnn_spectrogram.pth'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print_status(f"Found: {file}", 'success')
        else:
            print_status(f"Missing: {file}", 'error')
            all_exist = False
    
    if not all_exist:
        print_status("\nRequired files missing. Please train models first:", 'error')
        print(f"  1. python train_ml.py      (generates SVM and train_test_split)")
        print(f"  2. python train_cnn.py     (generates CNN model)")
        return False
    
    print_status("All required files found!", 'success')
    return True


def get_available_scripts():
    """Get list of available analysis scripts"""
    scripts = [
        ('cross_validation_analysis.py', '1. CROSS-VALIDATION ANALYSIS (10-fold stratified)'),
        ('compare_models_advanced.py', '2. SVM vs CNN COMPARISON (Statistical significance)'),
        ('ablation_study.py', '3. ABLATION STUDY (Feature importance analysis)'),
        ('model_analysis.py', '4. COMPREHENSIVE MODEL ANALYSIS (All metrics)'),
    ]
    
    available = []
    for script, description in scripts:
        if os.path.exists(script):
            available.append((script, description))
    
    return available


def interactive_menu():
    """Interactive menu for selecting analyses"""
    print_section("ADVANCED MODEL ANALYSIS SUITE")
    
    print("Select analyses to run:")
    print()
    
    scripts = get_available_scripts()
    
    for idx, (script, description) in enumerate(scripts, 1):
        print(f"{idx}. {description}")
    print(f"{len(scripts) + 1}. RUN ALL")
    print(f"{len(scripts) + 2}. QUIT")
    
    print()
    while True:
        try:
            choice = int(input(f"Enter choice (1-{len(scripts) + 2}): "))
            if 1 <= choice <= len(scripts) + 2:
                return choice, scripts
            else:
                print_status("Invalid choice. Please try again.", 'warning')
        except ValueError:
            print_status("Invalid input. Please enter a number.", 'warning')


def get_latest_results(pattern):
    """Find the latest result file matching a pattern"""
    import glob
    files = glob.glob(pattern)
    if files:
        # Sort by modification time and return the latest
        return max(files, key=os.path.getctime)
    return None


def generate_master_report():
    """Generate a master report combining all results"""
    print_section("GENERATING MASTER REPORT")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find latest result files
    cv_report = get_latest_results("models/cv_report_*.txt")
    comparison_json = get_latest_results("models/svm_cnn_comparison_*.json")
    ablation_csv = get_latest_results("models/ablation_results_*.csv")
    analysis_json = get_latest_results("models/analysis_results_*.json")
    
    report_path = f"models/MASTER_REPORT_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE ADVANCED ANALYSIS MASTER REPORT\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*80 + "\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n")
        
        # Cross-validation summary
        if cv_report:
            f.write(f"\n1. CROSS-VALIDATION RESULTS\n")
            f.write(f"   Source: {cv_report}\n")
            f.write(f"   This analysis shows model stability across 10 different data splits.\n")
            f.write(f"   Key metric: Mean Accuracy ± Std Dev indicates reliability.\n")
            try:
                with open(cv_report, 'r') as cv_f:
                    content = cv_f.read()
                    # Extract key metrics
                    lines = content.split('\n')
                    for line in lines:
                        if 'Accuracy' in line and ('Mean' in line or 'Std' in line):
                            f.write(f"   {line.strip()}\n")
            except:
                pass
        
        # Comparison summary
        if comparison_json:
            f.write(f"\n2. SVM vs CNN STATISTICAL COMPARISON\n")
            f.write(f"   Source: {comparison_json}\n")
            try:
                with open(comparison_json, 'r') as comp_f:
                    data = json.load(comp_f)
                    if 't_test' in data:
                        t_test = data['t_test']
                        f.write(f"   Paired t-test p-value: {t_test.get('p_value', 'N/A')}\n")
                        f.write(f"   Significant difference: {'YES' if t_test.get('significant') else 'NO'}\n")
                    if 'consensus' in data:
                        consensus = data['consensus']
                        f.write(f"   Model agreement: {consensus.get('agreement', 'N/A')}\n")
            except:
                pass
        
        # Ablation summary
        if ablation_csv:
            f.write(f"\n3. FEATURE ABLATION STUDY\n")
            f.write(f"   Source: {ablation_csv}\n")
            try:
                df = pd.read_csv(ablation_csv)
                critical = df[df['importance'] > 0.05]
                f.write(f"   Critical features: {len(critical)}\n")
                if len(critical) > 0:
                    f.write(f"   Top 5 critical features:\n")
                    for idx, row in critical.head(5).iterrows():
                        f.write(f"      - {row['feature']}: importance={row['importance']:.4f}\n")
            except:
                pass
        
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        
        f.write("\n1. MODEL SELECTION:\n")
        f.write("   - Consider using the statistically better-performing model\n")
        f.write("   - Review SVM vs CNN comparison for practical implications\n")
        f.write("   - Consider inference time and resource requirements\n")
        
        f.write("\n2. FEATURE OPTIMIZATION:\n")
        f.write("   - Ablation study identifies which features provide most value\n")
        f.write("   - Removing negligible features reduces computational cost\n")
        f.write("   - Critical features should be prioritized in extraction/processing\n")
        
        f.write("\n3. MODEL RELIABILITY:\n")
        f.write("   - Cross-validation results show consistency across data splits\n")
        f.write("   - Low std dev indicates stable, reliable predictions\n")
        f.write("   - Train-test gap indicates potential overfitting\n")
        
        f.write("\n4. DEPLOYMENT CONSIDERATIONS:\n")
        f.write("   - Review resource consumption metrics\n")
        f.write("   - Ensure inference time meets requirements\n")
        f.write("   - Plan for error handling based on error pattern analysis\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FILE LOCATIONS\n")
        f.write("="*80 + "\n")
        
        if cv_report:
            f.write(f"\nCross-Validation: {cv_report}\n")
        if comparison_json:
            f.write(f"Comparison: {comparison_json}\n")
        if ablation_csv:
            f.write(f"Ablation Study: {ablation_csv}\n")
        if analysis_json:
            f.write(f"Complete Analysis: {analysis_json}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NEXT STEPS\n")
        f.write("="*80 + "\n")
        
        f.write("\n1. Review detailed reports in models/ directory\n")
        f.write("2. Use insights from ablation study for feature optimization\n")
        f.write("3. Validate findings on independent test set\n")
        f.write("4. Deploy with confidence in statistical evidence of performance\n")
    
    print_status(f"Master report generated: {report_path}", 'success')
    return report_path


def run_all_analyses():
    """Run all available analyses in sequence"""
    print_section("RUNNING ALL ANALYSES")
    
    if not verify_requirements():
        print_status("\nCannot proceed without required model files.", 'error')
        return False
    
    scripts = get_available_scripts()
    results = []
    start_time = time.time()
    
    for idx, (script, description) in enumerate(scripts, 1):
        print(f"\n[{idx}/{len(scripts)}] Running analysis...")
        success = run_analysis_script(script, description)
        results.append({'script': script, 'description': description, 'success': success})
        
        if not success:
            print_status(f"Analysis failed: {script}", 'warning')
    
    total_time = time.time() - start_time
    
    # Generate master report
    print()
    master_report = generate_master_report()
    
    # Summary
    print_section("ANALYSIS COMPLETE")
    
    successful = sum(1 for r in results if r['success'])
    print_status(f"Completed: {successful}/{len(results)} analyses", 'success')
    print_status(f"Total time: {total_time/60:.1f} minutes", 'info')
    print_status(f"Master report: {master_report}", 'success')
    
    # Print summary
    print("\nAnalysis Results:")
    for result in results:
        status = '✓' if result['success'] else '✗'
        print(f"  {status} {result['description']}")
    
    return True


def main():
    """Main entry point"""
    print_section("ADVANCED SCREAM DETECTION MODEL ANALYSIS SUITE")
    
    print("This suite provides:")
    print("  • 10-fold cross-validation with stability metrics")
    print("  • Statistical significance testing (paired t-test, McNemar's test)")
    print("  • Feature importance analysis via ablation study")
    print("  • Error pattern analysis and diagnostics")
    print("  • Resource consumption monitoring")
    print()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'all':
            print_status("Running all analyses...", 'info')
            run_all_analyses()
            return
        elif sys.argv[1].lower() == '--help':
            print("Usage:")
            print("  python master_analysis.py          (interactive menu)")
            print("  python master_analysis.py all      (run all analyses)")
            return
    
    # Interactive menu
    while True:
        choice, scripts = interactive_menu()
        
        if choice == len(scripts) + 2:  # QUIT
            print_status("Exiting...", 'info')
            break
        elif choice == len(scripts) + 1:  # RUN ALL
            print_status("Running all analyses...", 'info')
            if run_all_analyses():
                if input("\nRun again? (y/n): ").lower() != 'y':
                    print_status("Exiting...", 'info')
                    break
        else:  # Run specific analysis
            script, description = scripts[choice - 1]
            if run_analysis_script(script, description):
                if input("\nRun another analysis? (y/n): ").lower() != 'y':
                    print_status("Exiting...", 'info')
                    break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
except Exception as e:
        print(f"\n{Colors.RED}Error: {str(e)}{Colors.ENDC}")
