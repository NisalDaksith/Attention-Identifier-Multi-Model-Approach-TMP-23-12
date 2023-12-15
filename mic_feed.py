import time
import numpy as np
import librosa
from keras.models import load_model
import joblib
from pydub import AudioSegment
import glob
import os
from pydub import AudioSegment
import datetime

def split_audio(audio_path, chunk_length):
    audio = AudioSegment.from_wav(audio_path)
    length_audio = len(audio)
    chunks = []
    start_time = datetime.datetime.now()  # Assume recording started now, adjust if needed
    
    for i in range(0, length_audio, chunk_length * 1000):  # chunk_length in milliseconds
        end_time = start_time + datetime.timedelta(seconds=chunk_length)
        
        chunk_name = str(end_time.strftime('%Y-%m-%d %H-%M-%S')) + ".wav"
        chunks.append((chunk_name, audio[i:i + chunk_length * 1000]))
        
        start_time = end_time
    return chunks


model = load_model('noise_detection_model.h5')

# Load the scaler used during training
scaler = joblib.load('scaler_noise.pkl')  # Make sure to provide the correct scaler file name

# Define a function to extract features from a single audio file
def extract_features(file_path):
    X, s_rate = librosa.load(file_path, res_type='kaiser_fast')
    mf = np.mean(librosa.feature.mfcc(y=X, sr=s_rate).T, axis=0)
    try:
        t = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=s_rate).T, axis=0)
    except:
        t = np.zeros(6)  # Handle case when tonnetz extraction fails
    m = np.mean(librosa.feature.melspectrogram(y=X, sr=s_rate).T, axis=0)
    s = np.abs(librosa.stft(X))
    c = np.mean(librosa.feature.chroma_stft(S=s, sr=s_rate).T, axis=0)
    features = np.concatenate((m, mf, t, c), axis=0)
    return features

# Normalize function
def normalize_features(features, scaler):
    normalized_features = scaler.transform([features])
    return normalized_features

def convert_mp3_to_wav(input_path, output_path):
    audio = AudioSegment.from_mp3(input_path)
    audio.export(output_path, format="wav")

def get_uploaded_audio_files():
    # Check if the "uploads" directory exists
    if not os.path.exists('uploads'):
        return []

    # Use glob to find all audio files in the "uploads" directory
    audio_files = glob.glob('uploads/*.*')
    # Extract only the file names (without the path)
    audio_files = [os.path.basename(file) for file in audio_files]

    return audio_files

# Define a function to get the duration of a WAV file
def get_wav_duration(file_path):
    audio, _ = librosa.load(file_path, sr=None)
    duration = len(audio) / _  # Calculate duration in seconds
    return duration

# Modify the predict_audio function
def predict_audio(file_path, threshold=0.5, length_threshold=20):
    # Check the length of the WAV file
    duration = get_wav_duration(file_path)

    # if duration < length_threshold:
    #     return "Normal"
    
    # Extract features from the uploaded audio file
    audio_features = extract_features(file_path)

    # Normalize the features using the loaded scaler
    normalized_features = normalize_features(audio_features, scaler)

    # Reshape the features for prediction
    normalized_features = normalized_features.reshape(1, -1)

    # Make a prediction using the model
    prediction = model.predict(normalized_features)
    
    # Check if the predicted probability for the "Normal" class is above the threshold
    if prediction[0, 1] >= threshold:
        predicted_class = "Normal"
    else:
        predicted_class = "Noisy"

    return predicted_class

def prediction_thread():
    while True:
        audio_files = get_uploaded_audio_files()

        if audio_files:
            for audio_file in audio_files:
                print("Predicting for:", audio_file)
                audio_file_path = os.path.join('uploads', audio_file)
                predicted_class = predict_audio(audio_file_path)
                print(f"Predicted class for {audio_file}: {predicted_class}")

                # Remove the processed audio file
                os.remove(audio_file_path)

        # time.sleep(10)  # Sleep for 10 seconds
