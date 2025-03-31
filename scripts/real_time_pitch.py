import librosa
import numpy as np
import sounddevice as sd
import joblib
import time

# Load the trained model and label encoder
model_filename = r"E:\Real-Time-Pitch-Detection\models\pitch_model.pkl"
label_encoder_filename = r"E:\Real-Time-Pitch-Detection\models\label_encoder.pkl"

model = joblib.load(model_filename)
label_encoder = joblib.load(label_encoder_filename)

# Audio recording parameters
duration = 2  # Duration to record in seconds
sampling_rate = 22050  # Sampling rate

# Extract features from audio
def extract_features(audio, sr):
    # Extract 15 MFCC features to match the model
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=15)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

# Real-time pitch detection
print("🎤 Starting real-time pitch detection. Speak or sing into the microphone...")

while True:
    print("Recording...")
    audio_data = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is done
    audio_data = audio_data.flatten()

    # Extract features from recorded audio
    features = extract_features(audio_data, sampling_rate)

    # Check feature shape
    if features.shape[1] == 15:
        # Predict the pitch
        predicted_label = model.predict(features)
        predicted_pitch = label_encoder.inverse_transform(predicted_label)

        print(f"🎶 Detected Pitch: {predicted_pitch[0]}")
    else:
        print(f"⚠️ Feature mismatch! Expected 15 features but got {features.shape[1]}.")

    # Wait for a few seconds before the next prediction
    time.sleep(1)
