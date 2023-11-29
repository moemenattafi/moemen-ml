# # app.py
# from flask import Flask, render_template, request, jsonify
# import librosa
# import joblib

# app = Flask(__name__)

# # Load SVM model
# with open('svm_model.joblib', 'rb') as model_file:
#    svm_model = joblib.load(model_file)
#    #print("🚀 ~ file: app.py:11 ~ svm_model:", svm_model)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get file from the request
#     file = request.files['file']
#     print("🚀 ~ file: app.py:21 ~ request:", request)

#     # Convert the file using librosa

#     # You can modify this part based on your model input requirements
#     data, sr = librosa.load(file)

#     # Use the loaded SVM model for prediction
#     result = svm_model.predict(data.reshape(1, -1))

#     # Return the result as JSON
#     return jsonify({'result': result[0]})


# if __name__ == '__main__':
#     app.run(debug=True,host="0.0.0.0")
from flask import Flask, request,render_template, jsonify
import joblib
import librosa
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("svm_model.joblib")

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        raise RuntimeError(f"Error extracting features from {file_path}: {e}")

# Function to map a label to a genre
def get_genre_from_label(label):
    # Replace with your actual genre names
    genres = ["blues", "classical","metal","disco","hiphop","jazz","country","pop","reggae","rock"]
    return genres[label]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file
        file = request.files['file']

        # Save the file temporarily
        file_path = "temp.wav"
        file.save(file_path)

        # Extract features from the uploaded file
        features = extract_features(file_path)

        # Make a prediction
        prediction = model.predict([features])

        # Get the predicted genre
        predicted_genre = get_genre_from_label(prediction[0])

        # Return the result as JSON
        result = {'prediction': predicted_genre}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
