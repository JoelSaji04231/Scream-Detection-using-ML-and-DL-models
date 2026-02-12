import csv
import os
from datetime import datetime


class PredictionLogger:
    """Logger for recording model predictions with timestamps"""
    
    def __init__(self, log_file="prediction_logs.csv"):
        self.log_file = log_file
        self.headers = [
            'timestamp',
            'audio_source',
            'cnn_prediction',
            'cnn_confidence',
            'cnn_prediction_time',
            'svm_prediction',
            'svm_confidence',
            'svm_prediction_time',
            'consensus',
            'avg_confidence'
        ]
        
        # Create log file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            self._create_log_file()
    
    def _create_log_file(self):
        """Create a new log file with headers"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        print(f"Created new log file: {self.log_file}")
    
    def log_prediction(self, results, audio_source="live_recording"):
        """
        Log prediction results from both models
        
        Args:
            results: List of dictionaries containing prediction results from both models
                     [cnn_result, svm_result]
            audio_source: Source of audio (e.g., "live_recording" or file path)
        """
        if results is None or len(results) < 2:
            print("Warning: Invalid results, skipping log entry")
            return
        
        cnn_result = results[0]
        svm_result = results[1]
        
        # Determine consensus
        if cnn_result['prediction'] == svm_result['prediction']:
            if cnn_result['prediction'] == 'SCREAM':
                consensus = 'BOTH_SCREAM'
            else:
                consensus = 'BOTH_NON_SCREAM'
        else:
            consensus = 'DISAGREE'
        
        # Calculate average confidence
        avg_confidence = (cnn_result['confidence'] + svm_result['confidence']) / 2
        
        # Create log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = [
            timestamp,
            audio_source,
            cnn_result['prediction'],
            f"{cnn_result['confidence']:.4f}",
            f"{cnn_result['prediction_time']:.4f}",
            svm_result['prediction'],
            f"{svm_result['confidence']:.4f}",
            f"{svm_result['prediction_time']:.4f}",
            consensus,
            f"{avg_confidence:.4f}"
        ]
        
        # Append to log file
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_entry)
            print(f"\n✓ Prediction logged to {self.log_file}")
        except Exception as e:
            print(f"\nError writing to log file: {e}")
    
    def get_recent_logs(self, n=10):
        """Get the n most recent log entries"""
        try:
            if not os.path.exists(self.log_file):
                return []
            
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                logs = list(reader)
                return logs[-n:] if len(logs) > n else logs
        except Exception as e:
            print(f"Error reading log file: {e}")
            return []
    
    def print_summary(self):
        """Print a summary of recent predictions"""
        logs = self.get_recent_logs(10)
        
        if not logs:
            print("\nNo logs found.")
            return
        
        print("\n" + "="*80)
        print("RECENT PREDICTIONS SUMMARY (Last 10)")
        print("="*80)
        
        for log in logs:
            print(f"\nTime: {log['timestamp']}")
            print(f"Source: {log['audio_source']}")
            print(f"CNN: {log['cnn_prediction']} ({float(log['cnn_confidence'])*100:.2f}%)")
            print(f"SVM: {log['svm_prediction']} ({float(log['svm_confidence'])*100:.2f}%)")
            print(f"Consensus: {log['consensus']}")
            print("-" * 80)
