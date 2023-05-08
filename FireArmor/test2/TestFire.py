#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 10:53:27 2023

@author: hugo
"""


from hmmlearn import hmm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
import skfuzzy as fuzz
import random

# Collect syscall data from file
def collect_syscall_data(file_path):
    with open(file_path) as f:
        syscall_data = np.array([float(line.strip()) for line in f])
    return syscall_data

# Compute frequency spectrum of syscall data
def compute_spectrum(syscall_data):
    n = len(syscall_data)
    freqs = np.fft.fftfreq(n)
    mask = freqs > 0
    fft_vals = fft(syscall_data)
    ##fft_vals = 2.0/n * np.abs(fft_vals[0:n//2])
    return freqs[mask], fft_vals[mask]

# Fuzzy logic for anomaly detection
def fuzzy_logic(fft_vals):
    x = np.arange(len(fft_vals))
    mfx = fuzz.trimf(x, [0, len(fft_vals)//2, len(fft_vals)])
    thresh_low = fuzz.interp_membership(x, mfx, 5)
    thresh_high = fuzz.interp_membership(x, mfx, len(fft_vals) - 5)
    return thresh_low, thresh_high

# Hysteresis for anomaly detection
def hysteresis(syscall_data, thresh_low, thresh_high):
    anomalies = []
    anomaly = False
    for i, val in enumerate(syscall_data):
        if val > thresh_high:
            anomaly = True
        elif val < thresh_low:
            anomaly = False
        if anomaly:
            anomalies.append(i)
    return anomalies



# Generate 50 random numbers between 0 and 1
syscall_data = [random.uniform(0, 1) for _ in range(50)]

# Add 10 anomalies between 1 and 2
for _ in range(10):
    idx = random.randint(0, 49)
    syscall_data[idx] = random.uniform(1, 2)

# Write the data to file
with open("syscall_data.txt", "w") as f:
    for val in syscall_data:
        f.write(f"{val}\n")

# Collect syscall data
syscall_data = collect_syscall_data('syscall_data.txt')

# Compute frequency spectrum
freqs, fft_vals = compute_spectrum(syscall_data)

# Apply fuzzy logic for anomaly detection
thresh_low, thresh_high = fuzzy_logic(fft_vals)

# Apply hysteresis for anomaly detection
anomalies = hysteresis(fft_vals, thresh_low, thresh_high)

# Plot the results
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(freqs, fft_vals, label='Spectrum')
ax.axhline(thresh_low, color='red', label='Low Threshold')
ax.axhline(thresh_high, color='green', label='High Threshold')
ax.scatter(freqs[anomalies], fft_vals[anomalies], color='black', label='Anomalies')
ax.set_xlabel('Frequency')
ax.set_ylabel('Magnitude')
ax.legend()
plt.show()


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
import numpy as np

# Générer des données aléatoires
data = np.random.rand(100)  # 100 nombres aléatoires, vous pouvez ajuster la taille selon vos besoins

# Enregistrer les données dans le fichier syscall_data.txt
np.savetxt("syscall_data.txt", data)
data = pd.read_csv("syscall_data.txt", header=None)
values = data.values.flatten()

# Fourier Analysis
fft = np.fft.fft(values)
frequencies = np.fft.fftfreq(len(values))
amplitudes = np.abs(fft)
plt.plot(frequencies, amplitudes)
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("Fourier Analysis")
plt.show()

# Blurred Logic
window_size = 5
threshold = 0.5
blurred_values = np.convolve(values, np.ones(window_size)/window_size, mode='same')
blurred_anomalies = np.abs(values - blurred_values) > threshold
plt.plot(values, label="Original Data")
plt.plot(blurred_values, label="Blurred Data")
plt.plot(blurred_anomalies * blurred_values, 'r.', label="Anomalies")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Blurred Logic")
plt.legend()
plt.show()

# Hysteresis
lower_threshold = 0.1
upper_threshold = 0.3
state = 0
hysteresis_anomalies = []
for value, blurred_value in zip(values, blurred_values):
    if state == 0:
        if blurred_value > upper_threshold:
            state = 1
    elif state == 1:
        if blurred_value < lower_threshold:
            state = 0
            hysteresis_anomalies.append(value)
hysteresis_anomalies = np.array(hysteresis_anomalies)
plt.plot(values, label="Original Data")
plt.plot(blurred_values, label="Blurred Data")
plt.plot(hysteresis_anomalies, 'r.', label="Anomalies")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Hysteresis")
plt.legend()
plt.show()
"""
