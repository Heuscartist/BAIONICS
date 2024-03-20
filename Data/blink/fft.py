import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data without header
data = pd.read_csv("data1_processed.csv", header=None)

# Extract the signal from the DataFrame
signal = data.iloc[:, 1]  # Assuming the first column is the index

# Calculate the FFT
fft_result = np.fft.fft(signal)
fft_freq = np.fft.fftfreq(len(signal), d=1.0/len(signal))

# Define a cutoff frequency (10 Hz in this case)
cutoff_frequency = 10

# Apply the high-pass filter
fft_result_filtered = fft_result * (np.abs(fft_freq) >= cutoff_frequency)

# Plot the original and filtered FFT results
plt.figure(figsize=(10, 6))

plt.plot(fft_freq, np.abs(fft_result_filtered), label=f'Filtered Signal (High-pass at {cutoff_frequency} Hz)', linestyle='dashed')
plt.title('FFT of the Signal with High-pass Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

