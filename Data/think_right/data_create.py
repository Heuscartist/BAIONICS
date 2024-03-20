import csv 
import os 
from pyOpenBCI import OpenBCIGanglion

x_value = 0
global_count = 0
index = 0

def print_raw_to_csv(sample, csv_file_path='data.csv'):
    global x_value, global_count, index
    global_count += 1
    count = 0
    
    csv_file_path = 'data' + str(index) + '.csv'
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
        print("Data written to CSV:", [x_value] + list(data))
        x_value += 1

        if global_count == 500:
            global_count = 0
            x_value = 0
            index += 1
            input("Press 1 to Start Again: ")

# Replace 'E9:10:76:85:71:ED' with the actual Bluetooth MAC address of your OpenBCI Ganglion
board = OpenBCIGanglion(mac='E9:10:76:85:71:ED')

if os.path.exists('data.csv'):
    os.remove('data.csv')

board.start_stream(print_raw_to_csv)

