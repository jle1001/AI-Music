import librosa
import numpy as np
import sys
import pandas as pd
import pathlib

# Change numpy print options to store the entire MFCC array.
np.set_printoptions(threshold=sys.maxsize)

data_path = pathlib.Path('data/raw/')
track_genres = pd.read_csv('data/processed/track_genres.csv', delimiter=',')

# print(track_genres.head())

def extract_mfccs():
    mfccs = dict()

    # Iterate over all files and folders in the directory and its subdirectories
    for item in data_path.glob('**/*'):
        # Check if the item is a file
        if item.is_file():
            try:
                y, sr = librosa.load(item)
                if librosa.get_duration(y=y, sr=sr) < 25:
                    continue
                y = y[:(25 * sr)]
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)

                # Normalize MFCC features for the entire song
                mfcc_normalized = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)

                mfccs[int(item.stem)] = mfcc_normalized
            except Exception as e:
                print(f"Error loading file: {e}")
                continue

    track_genres_mfcc = pd.DataFrame({'track_id': mfccs.keys(), 'mfcc': mfccs.values()})
    track_genres_mfcc = pd.merge(track_genres, track_genres_mfcc, on='track_id', how='inner')
    print(track_genres_mfcc.head())

    # Save as object. CSV files converts lists to str and loses precision.
    track_genres_mfcc.to_pickle('data/processed/track_genres_mfcc-SMALL.pkl')

def extract_spectograms():
    spectograms = dict()

    # Iterate over all files and folders in the directory and its subdirectories
    for item in data_path.glob('**/*'):
        # Check if the item is a file
        if item.is_file():
            try:
                y, sr = librosa.load(item)
                if librosa.get_duration(y=y, sr=sr) < 25:
                    continue
                y = y[:(25 * sr)]
                spectogram = librosa.feature.melspectrogram(y=y, sr=sr)

                # Normalize Spectogram features
                spectogram_normalized = (spectogram - np.mean(spectogram)) / np.std(spectogram)

                spectograms[int(item.stem)] = spectogram_normalized.ravel()
                # print(spectograms)
            except Exception as e:
                print(f"Error loading file: {e}")
                continue

    track_genres_spectograms = pd.DataFrame({'track_id': spectograms.keys(), 'spectogram': spectograms.values()})
    track_genres_spectograms = pd.merge(track_genres, track_genres_spectograms, on='track_id', how='inner')
    print(track_genres_spectograms.head())

    # Save as object. CSV files converts lists to str and loses precision.
    track_genres_spectograms.to_pickle('data/processed/track_genres_spectograms.pkl')

def extract_chromagrams():
    chromagrams = dict()

    # Iterate over all files and folders in the directory and its subdirectories
    for item in data_path.glob('**/*'):
        # Check if the item is a file
        if item.is_file():
            try:
                y, sr = librosa.load(item)
                if librosa.get_duration(y=y, sr=sr) < 25:
                    continue
                y = y[:(25 * sr)]
                chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

                # Normalize Chromagram features
                chromagram_normalized = (chromagram - np.mean(chromagram)) / np.std(chromagram)

                chromagrams[int(item.stem)] = chromagram_normalized.ravel()
                # print(chromagrams)
            except Exception as e:
                print(f"Error loading file: {e}")
                continue

    track_genres_chromagrams = pd.DataFrame({'track_id': chromagrams.keys(), 'chromagrams': chromagrams.values()})
    track_genres_chromagrams = pd.merge(track_genres, track_genres_chromagrams, on='track_id', how='inner')
    print(track_genres_chromagrams.head())

    # Save as object. CSV files converts lists to str and loses precision.
    track_genres_chromagrams.to_pickle('data/processed/track_genres_chromagrams.pkl')

extract_mfccs()