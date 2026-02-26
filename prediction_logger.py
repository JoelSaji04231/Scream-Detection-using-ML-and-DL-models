"""
Prediction Logging Utility
Logs predictions from both SVM and CNN models for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

class PredictionLogger:
    def __init__(self, log_file="prediction_logs.csv"):
        """Initialize prediction logger"""
        self.log_file = log_file
        self.columns = [
            'timestamp', 'audio_source', 'cnn_prediction', 'cnn_confidence', 
            'cnn_prediction_time', 'svm_prediction', 'svm_confidence', 
            'svm_prediction_time', 'consensus', 'avg_confidence'
        ]
        
        # Create log file if it doesn't exist
        if not os.path.exists(self.log_file):
            self.create_empty_log()
    
    def create_empty_log(self):
        """Create empty log file with headers"""
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.log_file, index=False)
        print(f"✓ Created empty prediction log: {self.log_file}")
    
    def log_prediction(self, audio_source, cnn_results, svm_results):
        """Log a prediction from both models"""
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract CNN results
        cnn_prediction = cnn_results.get('prediction', 'unknown')
        cnn_confidence = cnn_results.get('confidence', 0.0)
        cnn_prediction_time = cnn_results.get('prediction_time', 0.0)
        
        # Extract SVM results
        svm_prediction = svm_results.get('prediction', 'unknown')
        svm_confidence = svm_results.get('confidence', 0.0)
        svm_prediction_time = svm_results.get('prediction_time', 0.0)
        
        # Calculate consensus
        consensus = cnn_prediction == svm_prediction
        
        # Calculate average confidence
        avg_confidence = (cnn_confidence + svm_confidence) / 2
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'audio_source': audio_source,
            'cnn_prediction': cnn_prediction,
            'cnn_confidence': cnn_confidence,
            'cnn_prediction_time': cnn_prediction_time,
            'svm_prediction': svm_prediction,
            'svm_confidence': svm_confidence,
            'svm_prediction_time': svm_prediction_time,
            'consensus': consensus,
            'avg_confidence': avg_confidence
        }
        
        # Append to log file
        try:
            df = pd.DataFrame([log_entry])
            df.to_csv(self.log_file, mode='a', header=False, index=False)
            return True
        except Exception as e:
            print(f"✗ Error logging prediction: {e}")
            return False
    
    def log_single_prediction(self, model_name, audio_source, prediction, confidence, prediction_time):
        """Log a prediction from a single model"""
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create log entry with missing values for other model
        if model_name.lower() == 'cnn':
            log_entry = {
                'timestamp': timestamp,
                'audio_source': audio_source,
                'cnn_prediction': prediction,
                'cnn_confidence': confidence,
                'cnn_prediction_time': prediction_time,
                'svm_prediction': 'unknown',
                'svm_confidence': 0.0,
                'svm_prediction_time': 0.0,
                'consensus': False,
                'avg_confidence': confidence
            }
        elif model_name.lower() == 'svm':
            log_entry = {
                'timestamp': timestamp,
                'audio_source': audio_source,
                'cnn_prediction': 'unknown',
                'cnn_confidence': 0.0,
                'cnn_prediction_time': 0.0,
                'svm_prediction': prediction,
                'svm_confidence': confidence,
                'svm_prediction_time': prediction_time,
                'consensus': False,
                'avg_confidence': confidence
            }
        else:
            print(f"✗ Unknown model name: {model_name}")
            return False
        
        # Append to log file
        try:
            df = pd.DataFrame([log_entry])
            df.to_csv(self.log_file, mode='a', header=False, index=False)
            return True
        except Exception as e:
            print(f"✗ Error logging prediction: {e}")
            return False
    
    def get_logs(self, limit=None):
        """Get all logs or limited number of logs"""
        try:
            df = pd.read_csv(self.log_file)
            if limit and len(df) > limit:
                return df.tail(limit)
            return df
        except Exception as e:
            print(f"✗ Error reading logs: {e}")
            return pd.DataFrame(columns=self.columns)
    
    def get_consensus_stats(self):
        """Get consensus statistics"""
        df = self.get_logs()
        if df.empty:
            return {"error": "No logs available"}
        
        total_predictions = len(df)
        consensus_predictions = df['consensus'].sum()
        consensus_rate = consensus_predictions / total_predictions if total_predictions > 0 else 0
        
        # High confidence predictions (avg confidence > 0.8)
        high_confidence = df[df['avg_confidence'] > 0.8]
        high_confidence_consensus = high_confidence['consensus'].sum() if not high_confidence.empty else 0
        high_confidence_rate = high_confidence_consensus / len(high_confidence) if len(high_confidence) > 0 else 0
        
        return {
            'total_predictions': total_predictions,
            'consensus_predictions': consensus_predictions,
            'consensus_rate': consensus_rate,
            'high_confidence_predictions': len(high_confidence),
            'high_confidence_consensus_rate': high_confidence_rate
        }
    
    def get_model_comparison_stats(self):
        """Get model comparison statistics"""
        df = self.get_logs()
        if df.empty:
            return {"error": "No logs available"}
        
        # Filter out rows where both models made predictions
        complete_predictions = df[(df['cnn_prediction'] != 'unknown') & (df['svm_prediction'] != 'unknown')]
        
        if complete_predictions.empty:
            return {"error": "No complete dual-model predictions available"}
        
        # CNN statistics
        cnn_correct = complete_predictions[complete_predictions['cnn_prediction'] == 'scream']
        cnn_accuracy = len(cnn_correct) / len(complete_predictions) if len(complete_predictions) > 0 else 0
        cnn_avg_confidence = complete_predictions['cnn_confidence'].mean()
        cnn_avg_time = complete_predictions['cnn_prediction_time'].mean()
        
        # SVM statistics
        svm_correct = complete_predictions[complete_predictions['svm_prediction'] == 'scream']
        svm_accuracy = len(svm_correct) / len(complete_predictions) if len(complete_predictions) > 0 else 0
        svm_avg_confidence = complete_predictions['svm_confidence'].mean()
        svm_avg_time = complete_predictions['svm_prediction_time'].mean()
        
        return {
            'cnn_accuracy': cnn_accuracy,
            'cnn_avg_confidence': cnn_avg_confidence,
            'cnn_avg_time': cnn_avg_time,
            'svm_accuracy': svm_accuracy,
            'svm_avg_confidence': svm_avg_confidence,
            'svm_avg_time': svm_avg_time,
            'total_dual_predictions': len(complete_predictions)
        }
    
    def export_logs(self, format='json', filename=None):
        """Export logs to different formats"""
        df = self.get_logs()
        if df.empty:
            print("✗ No logs to export")
            return False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_logs_export_{timestamp}"
        
        try:
            if format.lower() == 'json':
                df.to_json(f"{filename}.json", orient='records', indent=2)
                print(f"✓ Logs exported to {filename}.json")
            elif format.lower() == 'xlsx':
                df.to_excel(f"{filename}.xlsx", index=False)
                print(f"✓ Logs exported to {filename}.xlsx")
            elif format.lower() == 'csv':
                df.to_csv(f"{filename}.csv", index=False)
                print(f"✓ Logs exported to {filename}.csv")
            else:
                print(f"✗ Unsupported format: {format}")
                return False
            return True
        except Exception as e:
            print(f"✗ Error exporting logs: {e}")
            return False
    
    def clear_logs(self):
        """Clear all logs"""
        try:
            self.create_empty_log()
            print("✓ Prediction logs cleared")
            return True
        except Exception as e:
            print(f"✗ Error clearing logs: {e}")
            return False
    
    def get_recent_predictions(self, hours=24):
        """Get predictions from the last N hours"""
        df = self.get_logs()
        if df.empty:
            return df
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by time
        cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
        recent_df = df[df['timestamp'] > cutoff_time]
        
        return recent_df

# ===== MAIN EXECUTION =====

def test_logger():
    """Test the prediction logger"""
    print("Testing Prediction Logger")
    print("="*40)
    
    # Initialize logger
    logger = PredictionLogger("test_prediction_logs.csv")
    
    # Test logging dual predictions
    print("\n1. Testing dual prediction logging...")
    cnn_results = {
        'prediction': 'scream',
        'confidence': 0.85,
        'prediction_time': 2.5
    }
    svm_results = {
        'prediction': 'scream',
        'confidence': 0.92,
        'prediction_time': 0.8
    }
    
    success = logger.log_prediction("test_audio.wav", cnn_results, svm_results)
    print(f"✓ Dual prediction logged: {success}")
    
    # Test logging single predictions
    print("\n2. Testing single prediction logging...")
    success = logger.log_single_prediction("CNN", "test2.wav", "non_scream", 0.75, 2.1)
    print(f"✓ CNN single prediction logged: {success}")
    
    success = logger.log_single_prediction("SVM", "test3.wav", "scream", 0.88, 0.5)
    print(f"✓ SVM single prediction logged: {success}")
    
    # Test statistics
    print("\n3. Testing statistics...")
    consensus_stats = logger.get_consensus_stats()
    print("Consensus Statistics:")
    for key, value in consensus_stats.items():
        print(f"  {key}: {value}")
    
    # Test model comparison
    print("\n4. Testing model comparison...")
    comparison_stats = logger.get_model_comparison_stats()
    print("Model Comparison Statistics:")
    for key, value in comparison_stats.items():
        print(f"  {key}: {value}")
    
    # Test export
    print("\n5. Testing export...")
    success = logger.export_logs('json', 'test_export')
    print(f"✓ Export test: {success}")
    
    # Show recent logs
    print("\n6. Recent logs:")
    recent_logs = logger.get_logs(limit=5)
    if not recent_logs.empty:
        print(recent_logs.to_string())
    else:
        print("No recent logs")
    
    print("\n✓ Prediction logger test complete!")

if __name__ == "__main__":
    test_logger()