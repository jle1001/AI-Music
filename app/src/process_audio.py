import librosa
import librosa.display
import matplotlib.pyplot as plt

def show_waveform(audio):
    y, sr = librosa.load(audio)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y=y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()