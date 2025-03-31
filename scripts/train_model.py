import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib  # To save the trained model
import os  # To handle file paths and directories

# Load the dataset
csv_path = r"E:\Real-Time-Pitch-Detection\pitch_features.csv"  # Update this path if needed
df = pd.read_csv(csv_path)

# Separate features and labels
X = df.iloc[:, :-1].values  # Feature columns (all except last)
y = df["label"].values  # Target labels

# Encode labels into numerical format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Create the models directory if it doesn't exist
model_dir = r"E:\Real-Time-Pitch-Detection\models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model
model_filename = os.path.join(model_dir, "pitch_model.pkl")
try:
    joblib.dump(model, model_filename)
    print(f"Model saved successfully at {model_filename}")
except Exception as e:
    print(f"Error saving model: {e}")

# Save the label encoder to decode labels later
label_filename = os.path.join(model_dir, "label_encoder.pkl")
try:
    joblib.dump(label_encoder, label_filename)
    print(f"Label encoder saved successfully at {label_filename}")
except Exception as e:
    print(f"Error saving label encoder: {e}")
