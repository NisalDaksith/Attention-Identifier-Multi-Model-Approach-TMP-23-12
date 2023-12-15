import os
import time
import numpy as np
from flask import Flask, request, jsonify
import librosa
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import joblib
from threading import Thread
from pydub import AudioSegment
import glob
from flask_cors import CORS
from flask import Flask, render_template, request, Response

app = Flask(__name__)
CORS(app) 

# @app.route('/')
# def index():
#     return render_template('index_Noise.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded."})

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({"error": "No selected audio file."})

    if audio_file:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        audio_file_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(audio_file_path)

        # If the uploaded file is in MP3 format, convert it to WAV
        if audio_file_path.lower().endswith('.mp3'):
            wav_filename = audio_file_path.replace('.mp3', '.wav')
            convert_mp3_to_wav(audio_file_path, wav_filename)
            os.remove(audio_file_path)  # Remove the original MP3 file
            audio_file_path = wav_filename

        # Calculate the predicted class for the uploaded audio
        predicted_class = predict_audio(audio_file_path)

        # Remove the processed audio file
        os.remove(audio_file_path)

        return jsonify({"message": "Audio file uploaded successfully.", "predicted_class": predicted_class})

    
if __name__ == '__main__':
    prediction_thread = Thread(target=prediction_thread)
    prediction_thread.daemon = True
    prediction_thread.start()
    app.run(debug=False, use_reloader=False)