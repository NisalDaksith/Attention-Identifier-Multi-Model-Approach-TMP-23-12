from flask import Flask, render_template, request, Response
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import time
from video_feed_emo import generate_frames
from video_feed_emo import emotion_data

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index_emo.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/end_session')
def end_session():
    session_data = pd.DataFrame(emotion_data, columns=["Time", "Emotion"])
    session_data.to_csv("emotions.csv", index=False)
    return session_data.to_html()

if __name__ == "__main__":
    app.run(debug=True)

