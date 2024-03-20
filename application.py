import csv 
import os 
from pyOpenBCI import OpenBCIGanglion
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras import regularizers, layers


# Load the trained model
num_classes = 6

model = models.Sequential([
    layers.InputLayer(input_shape=(500, 4)),  # Define input shape here
    layers.Conv1D(64, 5),
    layers.Dropout(0.2),
    layers.MaxPooling1D(3),
    layers.Conv1D(128, 5, activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling1D(3),
    layers.Conv1D(256, 5, activation='relu'),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
model.load_weights('model_weights.h5')
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

x_value = 0
global_count = 0
index = 0
labels = ['blink', 'close_hand', 'eyeMovement', 'jaw', 'normal', 'think_right']


def preprocess_data(file_path):
    data = []
    df = pd.read_csv(file_path, header=None)
    # Preprocess using FFT
    fft_result = np.fft.fft(df.iloc[:, 1:], axis=0)
    df_fft = pd.DataFrame(np.abs(fft_result))
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_fft)
    data_combined = data_scaled.reshape(df_fft.shape[0], -1)
    nan_mask = np.isnan(data_combined)  # Check for NaN values
    inf_mask = np.isinf(data_combined)  # Check for infinity values
    zero_std_mask = (scaler.scale_ == 0)  # Check for zero standard deviation
    
    # Set NaN, infinity, and zero standard deviation values to 0
    data_scaled[nan_mask] = 0
    data_scaled[inf_mask] = 0
    data_scaled[:, zero_std_mask] = 0
    data_scaled_reshaped = data_scaled.reshape(df_fft.shape)
    data.append(data_scaled_reshaped)
    if not data:
        raise ValueError("No data loaded. Check the file path or data format.")
    return np.array(data)


def print_raw_to_csv(sample, csv_file_path='data.csv'):
    global x_value, global_count, index
    global_count += 1
    count = 0
    
    csv_file_path = 'data' + '.csv'
    data = sample.channels_data

    # Check if the CSV file exists
    file_exists = os.path.exists(csv_file_path)

    # Open the CSV file in 'a+' (append and read) mode
    with open(csv_file_path, 'a+', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header only if the file is newly created
        #if not file_exists:
        #    csv_writer.writerow(["x_value", "channel1", "channel2", "channel3", "channel4"])

        # Write the data to the CSV file
        csv_writer.writerow([x_value] + list(data))
        #csv_writer.writerow(list(data))

        # Print the data for verification (optional)
        #print("Data written to CSV:", [x_value] + list(data))
        x_value += 1

        if global_count == 500:
            global_count = 0
            x_value = 0
            index += 1
            data, labels = preprocess_data(csv_file_path)
            
            #data_combined = data.reshape(data.shape[0], -1)
            
            #scaler = StandardScaler()
            #data_scaled = scaler.fit_transform(data_combined)
            #nan_mask = np.isnan(data_scaled)  # Check for NaN values
            #inf_mask = np.isinf(data_scaled)  # Check for infinity values
            #zero_std_mask = (scaler.scale_ == 0)  # Check for zero standard deviation
            
            #data_scaled[nan_mask] = 0
            #data_scaled[inf_mask] = 0
            #data_scaled[:, zero_std_mask] = 0
            #data_scaled_reshaped = data_scaled.reshape(data.shape)

            
            print("Shape of preprocessed data:", data_scaled_reshaped.shape)
            predictions = model.predict(data_scaled_reshaped)
            predicted_class = np.argmax(predictions[0])
            print("Predicted class:", predicted_class)
            os.remove('data.csv')
            input("Press 1 to Start Again: ")
            


# Replace 'E9:10:76:85:71:ED' with the actual Bluetooth MAC address of your OpenBCI Ganglion
board = OpenBCIGanglion(mac='E9:10:76:85:71:ED')

if os.path.exists('data.csv'):
    os.remove('data.csv')

board.start_stream(print_raw_to_csv)
