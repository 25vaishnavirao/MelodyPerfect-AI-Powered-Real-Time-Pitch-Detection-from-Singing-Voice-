import os
import librosa
import numpy as np
import pandas as pd

# Set the correct dataset path
dataset_path = r"E:\Real-Time-Pitch-Detection\datasets"

print("Starting feature extraction...")

# Check if datasets folder exists
if not os.path.exists(dataset_path):
    print(f"Error: datasets folder not found at {dataset_path}!")
    exit()

features = []
labels = []

# Process each folder inside the datasets directory
for pitch_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, pitch_folder)

    if os.path.isdir(folder_path):
        print(f"Processing folder: {pitch_folder}")  # Debug print

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                print(f"Processing file: {file_path}")  # Debug print

                try:
                    # Load audio file
                    y, sr = librosa.load(file_path, sr=22050)

                    # Extract features
                    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

                    # Store features and labels
                    features.append(np.hstack([mfccs, spectral_centroid, zero_crossing_rate]))
                    labels.append(pitch_folder)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Convert to DataFrame and save as CSV
df = pd.DataFrame(features)
df["label"] = labels

# Define output CSV path
csv_output_path = r"E:\Real-Time-Pitch-Detection\pitch_features.csv"
df.to_csv(csv_output_path, index=False)

print(f"Feature extraction completed. CSV saved as {csv_output_path}")
