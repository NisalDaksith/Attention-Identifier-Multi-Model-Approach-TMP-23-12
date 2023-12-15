import cv2
import tensorflow as tf
import numpy as np
import pandas as pd  # Import pandas library
from datetime import datetime
import time

# Load the attention level detection model
attention_model = tf.keras.models.load_model('Drowsiness_Identifier.h5')

# Define the Haarcascade classifier for face and eye detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)

# Create a list to store attention status data
attention_data = []

def generate_frames_awake():
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame = process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_frame_awake(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in eyes:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyess = eye_cascade.detectMultiScale(roi_gray)

        if len(eyess) == 0:
            # Eyes not detected, consider it as inattentive
            cv2.putText(frame, "Inattentive", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Append attention status data to the list
            attention_data.append(("Inattentive", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        else:
            # Eyes detected, consider it as attentive
            cv2.putText(frame, "Attentive", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Append attention status data to the list
            attention_data.append(("Attentive", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return frame