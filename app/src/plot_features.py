import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

UPLOAD_TRACK = './app/upload/upload.mp3'
y, sr = librosa.load(UPLOAD_TRACK)

def show_waveform(audio=UPLOAD_TRACK):
    """Display the waveform of an audio file.

    Args:
        audio (str): Path to the audio file. Default is 'upload/upload.mp3'.
    Returns:
        str: Base64-encoded image data in PNG format.
    """
    fig = plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y=y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('ascii')
    return f'data:image/png;base64,{data}'

def show_spectogram(audio=UPLOAD_TRACK):
    """Display the Mel-frequency spectrogram of an audio file.
    Args:
        audio (str): Path to the audio file. Default is 'upload/upload.mp3'.
    Returns:
        str: Base64-encoded image data in PNG format.
    """
    fig = plt.figure(figsize=(12, 4))
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    img = librosa.display.specshow(spec_db, x_axis='time', y_axis='mel', sr=sr)
    plt.title('Mel-frequency spectrogram')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('ascii')
    return f'data:image/png;base64,{data}'

def show_chromagram(audio=UPLOAD_TRACK):
    """Display the Chromagram of an audio file.
    Args:
        audio (str): Path to the audio file. Default is 'upload/upload.mp3'.
    Returns:
        str: Base64-encoded image data in PNG format.
    """
    fig = plt.figure(figsize=(12, 4))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', cmap='coolwarm')
    plt.title('Chromagram')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('ascii')
    return f'data:image/png;base64,{data}'

def show_MFCC(audio=UPLOAD_TRACK):
    """Display the Mel-frequency cepstral coefficients (MFCC) of an audio file.
    Args:
        audio (str): Path to the audio file. Default is 'upload/upload.mp3'.
    Returns:
        str: Base64-encoded image data in PNG format.
    """
    fig = plt.figure(figsize=(12, 4))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    img = librosa.display.specshow(mfccs, x_axis='time')
    plt.title('MFCC')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('ascii')
    return f'data:image/png;base64,{data}'