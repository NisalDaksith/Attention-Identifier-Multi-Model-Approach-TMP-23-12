from flask import Flask, render_template, request, Response
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd  
from datetime import datetime
import time
from video_feed_awake import generate_frames
from video_feed_awake import attention_data

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index_drowsy.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/end_session_awake')
def end_session():
    
    attention_df = pd.DataFrame(attention_data, columns=["Status", "Time"])
    
    # Save the data to a CSV file named "attention_status.csv"
    attention_df.to_csv("attention_status.csv", index=False)
    
    return "Session data saved to 'attention_status.csv'"

if __name__ == "__main__":
    app.run(debug=True)
