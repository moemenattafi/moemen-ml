from flask import Flask, request,render_template, jsonify
import tenserflow as tf
import joblib
import librosa
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("svm_model.joblib")
model = tf.keras.models.load_model('best_model.h5')

# Function to extract features from an audio file
def extract_mel_spectrogram(audio_data, sample_rate):
    try:
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=sample_rate, res_type='kaiser_fast')
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram
    except Exception as e:
        raise RuntimeError(f"Error extracting mel spectrogram from audio data: {e}")

# Function to preprocess mel spectrogram for VGG16 model
def preprocess_mel_spectrogram(mel_spectrogram):
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    mel_spectrogram = mel_spectrogram / 255.0  # Normalize to [0, 1]
    return mel_spectrogram

@app.route('/predict_genre', methods=['POST'])
def predict_genre():
    try:
        # Get the audio data from the request
        audio_data = request.files['file'].read()
        sample_rate = int(request.form['sample_rate'])  # Assuming sample_rate is included in the form data

        # Extract mel spectrogram from the audio data
        mel_spectrogram = extract_mel_spectrogram(audio_data, sample_rate)

        # Preprocess mel spectrogram for VGG16 model
        input_data = preprocess_mel_spectrogram(mel_spectrogram)

        # Make a prediction using the VGG16 model
        prediction = model.predict(input_data)

        # Get the predicted genre
        predicted_genre = get_genre_from_label(np.argmax(prediction))

        # Return the result as JSON
        result = {'prediction': predicted_genre}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

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
    print(label)
    genres = ["metal","hiphop","disco","blues","rock","classical","country","pop","reggae","jazz"]
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
