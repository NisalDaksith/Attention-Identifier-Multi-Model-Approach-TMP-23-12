import cv2
import tensorflow as tf
import numpy as np
from datetime import datetime
import time
import threading


INTERVAL = 60

# Load the emotion detection model
emotion_identifier = tf.keras.models.load_model('emotion_identifier_model_2.h5')

# Define the Haarcascade classifier for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define the emotion classes
emotion_classes = ["Attentive", "Displeasure", "Bored", "Shocked"]


# Load the attention level detection model
attention_model = tf.keras.models.load_model('Drowsiness_Identifier.h5')

# Define the Haarcascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Global variables for capturing and processing frames
video_capture = cv2.VideoCapture(0)
emotion_data = []
attention_data = []

# Initialize the last_emotion_process_time outside the function
last_emotion_process_time = 0


def process_frame_awake(frame):
    print("Double Check")
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
            attention_data.append(( datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Inattentive",))
        
        elif len(eyes) == 1:
            # Eyes detected, consider it as attentive
            cv2.putText(frame, "Attentive", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Append attention status data to the list
            attention_data.append(( datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Attentive",))

        # else:
        #     # Eyes detected, consider it as attentive
        #     cv2.putText(frame, "Attentive", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     # Append attention status data to the list
        #     attention_data.append(("Attentive", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return frame

def process_frame_emo(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        # No face detected, consider it as "Distracted"
        #current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        emotion_data.append((current_time, "Distracted"))
    else:
        for x, y, w, h in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            facess = faceCascade.detectMultiScale(roi_gray)

            if len(facess) == 0:
                # Face detected in the frame, but no facial features, consider it as "Distracted"
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                emotion_data.append((current_time, "Distracted"))
            else:
                for (ex, ey, ew, eh) in facess:
                    face_roi = roi_color[ey:ey+eh, ex:ex+ew]

                    final_1 = cv2.resize(face_roi, (224, 224))
                    final_2 = np.expand_dims(final_1, axis=0)
                    final_3 = final_2 / 255.0

                    predict = emotion_identifier.predict(final_3)
                    predicted_class = np.argmax(predict)

                    emotion_label = emotion_classes[predicted_class]

                    # Record the emotion and timestamp
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    emotion_data.append((current_time, emotion_label))

                    # Draw the emotion label and rectangle around the detected face
                    cv2.putText(frame, f"Emotion: {emotion_label}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

def generate_frames():
    global last_emotion_process_time  # Declare last_emotion_process_time as global

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        current_time = time.time()
        time_elapsed = current_time - last_emotion_process_time

        #Adjusting Time
        if time_elapsed >= INTERVAL:
            t1 = threading.Thread(target=process_frame_awake, args=(frame,))
            t2 = threading.Thread(target=process_frame_emo, args=(frame,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            # frame = process_frame_awake(frame)
            # frame = process_frame_emo(frame)
            last_emotion_process_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
