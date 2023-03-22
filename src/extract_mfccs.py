import librosa
import numpy as np
import pandas as pd

data = pd.read_csv('data/processed/labels/labels.csv')
audio_files = data['filename'].values
print(audio_files)
genres = data['genre'].values

mfccs = []
mfccs_shape = []
labels = []

for file, genre in zip(audio_files, genres):
    y, sr = librosa.load(f'data/raw/{file}')
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    print(mfcc.shape)
    mfccs_shape.append(mfcc.shape)
    mfccs.append(np.ravel(mfcc)[:4000])
    labels.append(genre)

for i in mfccs:
    print("ITEM: ", i)
mfccs_array = np.array(mfccs)
labels_array = np.array(labels)

print(labels_array)

np.save('data/processed/mfccs/mfccs.npy', mfccs_array)
np.save('data/processed/labels/labels.npy', labels_array)
np.savetxt('data/processed/shapes/mfccs.csv', mfccs_shape, delimiter=",")