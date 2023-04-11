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

mfccs = dict()

# Iterate over all files and folders in the directory and its subdirectories
for item in data_path.glob('**/*'):
    # Check if the item is a file
    if item.is_file():
        # print(item)
        y, sr = librosa.load(item)
        y = y[:(25 * sr)]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        mfccs[int(item.stem)] = mfcc
        # print(mfccs)

track_genres_mfcc = pd.DataFrame({'track_id': mfccs.keys(), 'mfcc': mfccs.values()})
track_genres_mfcc = pd.merge(track_genres, track_genres_mfcc, on='track_id', how='inner')
# print(track_genres_mfcc.head())

track_genres_mfcc.to_csv('data/processed/track_genres_mfcc.csv', index=False)