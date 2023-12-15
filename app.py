from flask import Flask, render_template, request, Response, jsonify
import pandas as pd
from threading import Thread
import os
from functools import reduce

from video_feed_emo import generate_frames
from video_feed_emo import emotion_data
from video_feed_emo import attention_data

from mic_feed import convert_mp3_to_wav, predict_audio, prediction_thread, split_audio
app = Flask(__name__)

# Key Prediction
from app_key import get_key_predictions, stop_key_scheduler, start_key_scheduler
app.add_url_rule('/get_key_predictions', 'get_key_predictions', get_key_predictions, methods=['GET'])
app.add_url_rule('/stop_key_scheduler', 'stop_key_scheduler', stop_key_scheduler)
app.add_url_rule('/start_key_scheduler', 'start_key_scheduler', start_key_scheduler)


AUDIO_CHUNK_LENGTH = 60


def process_chunks(chunks):
    results = []
    for chunk_name, chunk in chunks:
        # Save the chunk with its timestamped filename
        chunk.export(chunk_name, format="wav")

        # Predict the chunk
        predicted_class = predict_audio(chunk_name)

        # Append the result
        results.append(
            (chunk_name.replace('.wav', '').replace(':', '-'), predicted_class))

        #  remove the saved chunk
        os.remove(chunk_name)
    return results


@app.route('/')
def index():
    return render_template('index_emo.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed_awake')
# def video_feed_awake():
#     return Response(generate_frames_awake(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/end_session_emo')
def end_session_emo():
    session_data = pd.DataFrame(emotion_data, columns=["Time", "Emotion"])
    session_data.to_csv("emotions.csv", index=False)
    return "emotion data saved to 'emotions.csv'"


@app.route('/end_session_awake')
def end_session_awake():
    print(attention_data)
    attention_df = pd.DataFrame(attention_data, columns=["Time", "Attention"])
    attention_df.to_csv("attention_status.csv", index=False)
    return "attention data saved to 'attention_status.csv'"


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

        chunk_length = AUDIO_CHUNK_LENGTH
        audio_chunks = split_audio(audio_file_path, chunk_length)

        # Calculate the predicted class for the uploaded audio
        # predicted_class = predict_audio(audio_file_path)
        predicted_results = process_chunks(audio_chunks)

        # Remove the processed audio file
        os.remove(audio_file_path)

        # predicted_results to disk
        df = pd.DataFrame(predicted_results, columns=["Time", "Noise_Status"])
        df.to_csv("Noice_results.csv", index=False)

        return jsonify({"message": "Audio file uploaded successfully.", "predicted_class": predicted_results})


@app.route('/get_predictions')
def get_predictions():
    attention_df = pd.read_csv("attention_status.csv")
    emotion_df = pd.read_csv("emotions.csv")
    noise_df = pd.read_csv("Noice_results.csv")
    key_df = pd.read_csv("key_predictions.csv")

    attention_df['Time'] = attention_df['Time'].apply(lambda x: x[:-2]+'00')
    emotion_df['Time'] = emotion_df['Time'].apply(lambda x: x[:-2]+'00')
    noise_df['Time'] = noise_df['Time'].apply(lambda x: x[:-2]+'00')
    key_df['Time'] = key_df['Time'].apply(lambda x: x[:-2]+'00')

    # rename column name key_df
    key_df.rename(columns={'Prediction': 'Key_Status'}, inplace=True)

    noise_df['Time'] = pd.to_datetime(
        noise_df['Time'], format='%Y-%m-%d %H-%M-%S')
    attention_df['Time'] = pd.to_datetime(
        attention_df['Time'], format='%Y-%m-%d %H:%M:%S')
    emotion_df['Time'] = pd.to_datetime(
        emotion_df['Time'], format='%Y-%m-%d %H:%M:%S')
    key_df['Time'] = pd.to_datetime(key_df['Time'], format='%Y-%m-%d %H:%M:%S')

    # merged_df = pd.merge(emotion_df, attention_df, noise_df, key_df,  on='Time', how='outer').sort_values(by='Time')

    dfs = [emotion_df, attention_df, noise_df, key_df]
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['Time'],
                                                    how='outer'), dfs).sort_values(by='Time').reset_index(drop=True)
    df_html = merged_df.to_html(classes='dataframe table table-striped')
    return df_html


if __name__ == "__main__":
    prediction_thread = Thread(target=prediction_thread)
    prediction_thread.daemon = True
    prediction_thread.start()
    app.run(debug=False, use_reloader=False)
