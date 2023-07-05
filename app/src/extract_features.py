import librosa
import numpy as np
import sys
import pandas as pd
import pathlib

# Change numpy print options to store the entire MFCC array.
np.set_printoptions(threshold=sys.maxsize)

# Data paths
data_path = pathlib.Path('data/raw/')
track_genres = pd.read_csv('data/processed/track_genres.csv', delimiter=',')

def extract_mfccs():
    """Extracts Mel-frequency cepstral coefficients (MFCCs) from audio files and saves them with corresponding track genre information."""

    # Dictionary to store MFCCs for each audio track.
    mfccs = dict()

    # Iterate over all files and folders in the directory and its subdirectories
    for item in data_path.glob('**/*'):
        # Check if the item is a file
        if item.is_file():
            try:
                # Loads the file
                y, sr = librosa.load(item)
                if librosa.get_duration(y=y, sr=sr) < 25:
                    continue
                # Uses the first 25 seconds
                y = y[:(25 * sr)]
                # Extract 10 MFCCs coeficients
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)

                # Normalize MFCC features for the entire song
                mfcc_normalized = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)
                mfccs[int(item.stem)] = mfcc_normalized
            except Exception as e:
                print(f"Error loading file: {e}")
                continue

    # Merges track_genres and MFCCs on one DataFrame
    track_genres_mfcc = pd.DataFrame({'track_id': mfccs.keys(), 'mfcc': mfccs.values()})
    track_genres_mfcc = pd.merge(track_genres, track_genres_mfcc, on='track_id', how='inner')
    print(track_genres_mfcc.head())

    # Save as object. CSV files converts lists to str and loses precision.
    track_genres_mfcc.to_pickle('data/processed/track_genres_mfcc-SMALL.pkl')

def extract_spectograms():
    """Extracts spectograms from audio files and saves them with corresponding track genre information."""

    # Dictionary to store spectograms for each audio track.
    spectograms = dict()

    # Iterate over all files and folders in the directory and its subdirectories
    for item in data_path.glob('**/*'):
        # Check if the item is a file
        if item.is_file():
            try:
                # Loads the file
                y, sr = librosa.load(item)
                if librosa.get_duration(y=y, sr=sr) < 25:
                    continue
                # Uses the first 25 seconds
                y = y[:(25 * sr)]
                # Extract spectogram
                spectogram = librosa.feature.melspectrogram(y=y, sr=sr)

                # Normalize Spectogram features
                spectogram_normalized = (spectogram - np.mean(spectogram)) / np.std(spectogram)
                spectograms[int(item.stem)] = spectogram_normalized.ravel()
            except Exception as e:
                print(f"Error loading file: {e}")
                continue

    # Merges track_genres and spectograms on one DataFrame
    track_genres_spectograms = pd.DataFrame({'track_id': spectograms.keys(), 'spectogram': spectograms.values()})
    track_genres_spectograms = pd.merge(track_genres, track_genres_spectograms, on='track_id', how='inner')
    print(track_genres_spectograms.head())

    # Save as object. CSV files converts lists to str and loses precision.
    track_genres_spectograms.to_pickle('data/processed/track_genres_spectograms-SMALL.pkl')

def extract_chromagrams():
    """Extracts chromagrams from audio files and saves them with corresponding track genre information."""

    # Dictionary to store chromagrams for each audio track.
    chromagrams = dict()

    # Iterate over all files and folders in the directory and its subdirectories
    for item in data_path.glob('**/*'):
        # Check if the item is a file
        if item.is_file():
            try:
                # Loads the file
                y, sr = librosa.load(item)
                if librosa.get_duration(y=y, sr=sr) < 25:
                    continue
                # Uses the first 25 seconds
                y = y[:(25 * sr)]
                # Extract chromagram
                chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

                # Normalize Chromagram features
                chromagram_normalized = (chromagram - np.mean(chromagram)) / np.std(chromagram)
                chromagrams[int(item.stem)] = chromagram_normalized.ravel()
            except Exception as e:
                print(f"Error loading file: {e}")
                continue
    # Merges track_genres and chromagrams on one DataFrame
    track_genres_chromagrams = pd.DataFrame({'track_id': chromagrams.keys(), 'chromagrams': chromagrams.values()})
    track_genres_chromagrams = pd.merge(track_genres, track_genres_chromagrams, on='track_id', how='inner')
    print(track_genres_chromagrams.head())

    # Save as object. CSV files converts lists to str and loses precision.
    track_genres_chromagrams.to_pickle('data/processed/track_genres_chromagrams-SMALL.pkl')

extract_mfccs()
# extract_spectograms()
# extract_chromagrams()