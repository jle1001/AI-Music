import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot_basic_components(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file)

    # Data specifications
    headers = ['Filename', 'Type', 'Duration (seconds)', 'Sample Rate (hz)']
    data = [os.path.basename(audio_file), audio_file.split(".")[-1], round(librosa.get_duration(path=audio_file)), sr]

    # Find the longest string in the headers
    max_header_width = max(len(header) for header in headers)

    # Print the data
    for header, value in zip(headers, data):
        print(f"{header:{max_header_width}}: {value}")

    # Separate the harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Create a time array in seconds
    time = np.arange(len(y)) / sr

    # Plot the input sound signal
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, y)
    plt.title('Input sound signal')
    plt.xlabel('Time (s)')

    # Plot the harmonic component
    plt.subplot(3, 1, 2)
    plt.plot(time[:len(y_harmonic)], y_harmonic)
    plt.title('Harmonic component')
    plt.xlabel('Time (s)')

    # Plot the percussive component
    plt.subplot(3, 1, 3)
    plt.plot(time[:len(y_percussive)], y_percussive)
    plt.title('Percussive component')
    plt.xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python librosa_basic_components.py [audio_file]')
        sys.exit(1)

    audio_file = sys.argv[1]
    plot_basic_components(audio_file)