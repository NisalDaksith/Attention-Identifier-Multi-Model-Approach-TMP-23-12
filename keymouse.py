import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener, Key
import time

loaded_model = load_model('keystroke_model.h5')
scaler_x = joblib.load('./scalers/scaler_x.pkl')
scaler_y = joblib.load('./scalers/scaler_y.pkl')
global df
df = pd.DataFrame()  # Initialize df outside functions for scope access


def capture_and_predict():
    # global df  # Access the global df

    # Initialize counts
    mouse_movement_count = 0
    mouse_click_count = 0
    keyboard_letter_key_count = 0
    keyboard_number_key_count = 0
    keyboard_arrow_counts = [0, 0, 0, 0]  # [Up, Left, Down, Right]

    def on_mouse_click(x, y, button, pressed):
        nonlocal mouse_movement_count, mouse_click_count

        # Count mouse movements
        mouse_movement_count += 1

        if pressed:
            # Count mouse clicks
            mouse_click_count += 1

    def on_key_press(key):
        nonlocal keyboard_letter_key_count, keyboard_number_key_count, keyboard_arrow_counts

        try:
            if hasattr(key, 'name'):
                if key.name.startswith('left'):
                    keyboard_arrow_counts[1] += 1
                elif key.name.startswith('right'):
                    keyboard_arrow_counts[3] += 1
                elif key.name.startswith('up'):
                    keyboard_arrow_counts[0] += 1
                elif key.name.startswith('down'):
                    keyboard_arrow_counts[2] += 1
            elif key.char and key.char.isalpha():
                keyboard_letter_key_count += 1
            elif key.char and key.char.isdigit():
                keyboard_number_key_count += 1
        except AttributeError:
            pass

        if key == Key.esc:
            return False

    with MouseListener(on_click=on_mouse_click) as mouse_listener, KeyboardListener(on_press=on_key_press) as keyboard_listener:
        start_time = time.time()
        while time.time() - start_time < 10:  # Listen for events for 5 minutes
            pass

    # Assign the counts to specific indices
    load_data = [0] * 9
    load_data[0] = 1
    load_data[1] = mouse_movement_count
    load_data[2] = keyboard_arrow_counts[0]  # Up arrow count
    load_data[3] = keyboard_arrow_counts[1]  # Left arrow count
    load_data[4] = keyboard_arrow_counts[2]  # Down arrow count
    load_data[5] = keyboard_arrow_counts[3]  # Right arrow count
    load_data[6] = mouse_click_count
    load_data[7] = keyboard_letter_key_count
    load_data[8] = keyboard_number_key_count

    # Display the counts_array 
    load_data = np.array(load_data)

    # Preprocessing real-time data
    real_time_data_scaled = scaler_x.transform(load_data.reshape(1, -1))

    # Makeing predictions
    predictions = loaded_model.predict(real_time_data_scaled)
    #print(predictions)

    # Inverse transform predictions to get the original scale
    scaler_y = joblib.load('./scalers/scaler_y.pkl')
    predictions_original = scaler_y.inverse_transform(predictions)

    # Printing the predicted class based on the thresholds used in your original code
    value = round(predictions[0][0], 2)

    if value < 1.00:
        predicted_class = "Gaming"
    elif 1 < value <= 2.5:
        predicted_class = "Attentive"
    elif 2.6 < value < 3:
        predicted_class = "Unrelated"
    else:
        predicted_class = "Social"

    # Store predictions in the dataframe
    current_time = datetime.now()
    #df = pd.DataFrame({'Timestamp': [current_time], 'Prediction': [predictions]})
    global df
    df = df.append({'Timestamp': current_time, 'Prediction': predicted_class}, ignore_index=True)
    return df

scheduler = BackgroundScheduler()

# Scheduler to run the capture_and_predict function every 5 minutes
scheduler.add_job(capture_and_predict, 'interval', minutes=1) 