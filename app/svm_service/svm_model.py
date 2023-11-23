import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import joblib

# Function to extract features from audio files
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Function to load the dataset
def load_dataset(data_path):
    labels = []
    features = []
    genres = os.listdir(data_path)

    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        for filename in os.listdir(genre_path):
            file_path = os.path.join(genre_path, filename)
            label = genres.index(genre)
            feature = extract_features(file_path)

            if feature is not None:
                labels.append(label)
                features.append(feature)

    return np.array(features), np.array(labels)

# Load dataset
data_path = "./../data/genres_original/Data"  # Replace with the actual path to your dataset
X, y = load_dataset(data_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create SVM model
model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1))
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the model to a file
model_filename = "svm_model.joblib"
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")
