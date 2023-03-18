import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot_spectograms(audio_file):
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

    # Compute spectogram
    spectogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Compute spectrogram for harmonic component and percussive component separately
    S_harmonic = librosa.feature.melspectrogram(y=y_harmonic)
    S_percussive = librosa.feature.melspectrogram(y=y_percussive)

    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    plt.figure(figsize=(10, 6))
    
    # Plot MFCCs
    plt.subplot(4, 1, 1)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')

    # Plotting harmonic spectrogram 
    plt.subplot(4, 1, 2)
    librosa.display.specshow(librosa.power_to_db(spectogram), x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectogram (Total)')

    # Plotting harmonic spectrogram 
    plt.subplot(4, 1, 3)
    librosa.display.specshow(librosa.power_to_db(S_harmonic), x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram (Harmonic)')
     
    # Plotting percussive spectrogram 
    plt.subplot(4, 1, 4)
    librosa.display.specshow(librosa.power_to_db(S_percussive), x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram (Percussive)')
     
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python librosa_advanced_components.py [audio_file]')
        sys.exit(1)

    audio_file = sys.argv[1]
    plot_spectograms(audio_file)