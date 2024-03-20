import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_raw_and_fft(data):
    num_channels = data.shape[1] - 1  # Excluding the index column
    time = data.iloc[:, 0]  # Assuming the first column is the time/index

    fig, axs = plt.subplots(num_channels, 2, figsize=(10, 2*num_channels))

    for i in range(num_channels):
        channel_data = data.iloc[:, i + 1]  # Extracting data for each channel
        axs[i, 0].plot(time, channel_data)
        axs[i, 0].set_title(f'Channel {i+1} Raw Data')
        axs[i, 0].set_xlabel('Time')
        axs[i, 0].set_ylabel('Amplitude')

        # Calculate the FFT
        fft_result = np.fft.fft(channel_data)
        fft_freq = np.fft.fftfreq(len(channel_data), d=1.0/len(channel_data))

        # Define a cutoff frequency (10 Hz in this case)
        cutoff_frequency = 10

        # Apply the high-pass filter
        fft_result_filtered = fft_result * (np.abs(fft_freq) >= cutoff_frequency)

        axs[i, 1].plot(fft_freq, np.abs(fft_result_filtered))
        axs[i, 1].set_title(f'Channel {i+1} FFT (High-pass at {cutoff_frequency} Hz)')
        axs[i, 1].set_xlabel('Frequency (Hz)')
        axs[i, 1].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

# Load the data without header
data = pd.read_csv("data0.csv", header=None)

# Plot raw data and FFT
plot_raw_and_fft(data)

