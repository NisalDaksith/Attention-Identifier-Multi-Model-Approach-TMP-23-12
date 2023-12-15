from flask import Flask, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener, Key
import time
import logging


SCHEDULER_INTERVAL = 1  # in minutes


app = Flask(__name__)

loaded_key_model = load_model('keystroke_model.h5')
scaler_x = joblib.load('./scalers/scaler_x.pkl')
scaler_y = joblib.load('./scalers/scaler_y.pkl')
key_df = pd.DataFrame(columns=['Time', 'Prediction'])

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def capture_and_predict():

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
    predictions = loaded_key_model.predict(real_time_data_scaled)
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
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #key_df = pd.DataFrame({'Time': [current_time], 'Prediction': [predictions]})
    
    key_df_dictionary = pd.DataFrame([{'Time': current_time, 'Prediction': predicted_class}])

    global key_df
    key_df = pd.concat([key_df, key_df_dictionary], ignore_index=True)

    # print(key_df, 'classsssss',predicted_class)


    return key_df


@app.route('/get_key_predictions', methods=['GET'])
def get_key_predictions():
    # Return the predictions in JSON format
    print(key_df.to_dict(orient='records'))
    return jsonify(key_df.to_dict(orient='records'))

# stop the scheduler
@app.route('/stop_key_scheduler')
def stop_key_scheduler():
    scheduler.shutdown()
    try:
        key_df.to_csv('key_predictions.csv', index=False)
    except:
        pass
    return "Key scheduler is stopped"

# start the scheduler
@app.route('/start_key_scheduler')
def start_key_scheduler():
    try:
        scheduler.start()
        return "Scheduler is started"
    except:
        return "Error starting scheduler"



scheduler = BackgroundScheduler()
# Scheduler to run the capture_and_predict function every 1 minutes
scheduler.add_job(capture_and_predict, 'interval', minutes=SCHEDULER_INTERVAL)

if __name__ == '__main__':
    # Start the scheduler after the app runs
    print("Starting key scheduler...")
    # scheduler.start()
    app.run(debug=False, use_reloader=True)