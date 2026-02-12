import sounddevice as sd
from scipy.io.wavfile import write
import os
import sys
import io
from contextlib import redirect_stdout
from prediction_logger import PredictionLogger

# Initialize prediction logger
logger = PredictionLogger("prediction_logs.csv")

# Suppress loading messages
f = io.StringIO()

print("Loading models...")
print("1. Loading CNN model...")
with redirect_stdout(f):
    from inference_cnn import quick_predict as cnn_predict
print("CNN model loaded")

print("2. Loading SVM model...")
with redirect_stdout(f):
    from inference_svm import quick_predict as svm_predict
print("SVM model loaded")


def compare_models(duration=5, sample_rate=22050, temp_file="temp_audio.wav"):
    """Record audio and predict using CNN and SVM (ESC-50)"""
    
    print("\nCRIME ALERT - DUAL MODEL COMPARISON")
    print("Recording audio for 5 seconds...")
    
    try:
        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype='float32')
        sd.wait()
        
        print("\nRecording finished.\n")
        
        # Save to temporary file
        write(temp_file, sample_rate, audio_data)
        
        # Predict with both models
        print("Running predictions...")
        
        f = io.StringIO()
        
        with redirect_stdout(f):
            cnn_result = cnn_predict(temp_file)
        print("CNN model prediction complete")
        
        with redirect_stdout(f):
            svm_result = svm_predict(temp_file)
        print("SVM model prediction complete")
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Display results
        print("\nPREDICTION RESULTS")
        
        results = [
            {**cnn_result, 'model': 'CNN (Spectrogram)'},
            {**svm_result, 'model': 'SVM'}
        ]
        
        for result in results:
            print(f"\n{result['model'].upper()}")
            print(f"Prediction:         {result['prediction']}")
            print(f"Confidence:         {result['confidence']*100:.2f}%")
            print(f"Prediction Time:    {result['prediction_time']:.4f} seconds")
        
        # Show consensus
        if cnn_result['prediction'] == svm_result['prediction']:
            if cnn_result['prediction'] == 'SCREAM':
                print("\n" + "="*60)
                print("CONSENSUS: SCREAM DETECTED BY BOTH MODELS!")
                print("="*60)
            else:
                print("\n" + "="*60)
                print("CONSENSUS: Non-scream audio (both models agree)")
                print("="*60)
        else:
            print("\n" + "="*60)
            print("MIXED PREDICTIONS - Models disagree")
            print("="*60)
        
        # Log the prediction
        logger.log_prediction(results, audio_source="live_recording")
        
        return results
        
    except Exception as e:
        print(f"\nError during prediction: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return None


def predict_from_file(audio_file_path):
    """Predict from an existing audio file using CNN and SVM (ESC-50)"""
    
    if not os.path.exists(audio_file_path):
        print(f"Error: File '{audio_file_path}' not found!")
        return None
    
    print(f"ANALYZING AUDIO FILE")
    print(f"File: {os.path.basename(audio_file_path)}")
    print(f"Path: {audio_file_path}\n")
    
    try:
        # Predict with both models
        print("Running predictions...")
        
        f = io.StringIO()
        
        with redirect_stdout(f):
            cnn_result = cnn_predict(audio_file_path)
        print("CNN model prediction complete")
        
        with redirect_stdout(f):
            svm_result = svm_predict(audio_file_path)
        print("SVM model prediction complete")
        
        # Display results
        print("\nPREDICTION RESULTS")
        
        results = [
            {**cnn_result, 'model': 'CNN (Spectrogram)'},
            {**svm_result, 'model': 'SVM (ESC-50)'}
        ]
        
        for result in results:
            print(f"\n{result['model'].upper()}")
            print(f"Prediction:         {result['prediction']}")
            print(f"Confidence:         {result['confidence']*100:.2f}%")
            print(f"Prediction Time:    {result['prediction_time']:.4f} seconds")
        
        # Show consensus
        if cnn_result['prediction'] == svm_result['prediction']:
            if cnn_result['prediction'] == 'SCREAM':
                print("\n" + "="*60)
                print("CONSENSUS: SCREAM DETECTED BY BOTH MODELS!")
                print("="*60)
            else:
                print("\n" + "="*60)
                print("CONSENSUS: Non-scream audio (both models agree)")
                print("="*60)
        else:
            print("\n" + "="*60)
            print("MIXED PREDICTIONS - Models disagree")
            avg_conf = (cnn_result['confidence'] + svm_result['confidence']) / 2
            print(f"Average confidence: {avg_conf*100:.2f}%")
            print("="*60)
        
        # Log the prediction
        logger.log_prediction(results, audio_source=audio_file_path)
        
        return results
        
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Crime Alert - Audio Scream Detection')
    parser.add_argument('--file', '-f', type=str, help='Path to audio file to analyze')
    parser.add_argument('--live', '-l', action='store_true', help='Record live audio (default if no file specified)')
    parser.add_argument('--duration', '-d', type=int, default=5, help='Recording duration in seconds (default: 5)')
    
    args = parser.parse_args()
    
    if args.file:
        results = predict_from_file(args.file)
    else:
        if args.live or len(sys.argv) == 1:
            results = compare_models(duration=args.duration)
        else:
            parser.print_help()
