import librosa
import numpy as np
import pandas as pd
import pathlib

data_path = pathlib.Path('data/raw/')

# Iterate over all files and folders in the directory and its subdirectories
for item in data_path.glob('**/*'):
    # Check if the item is a file
    if item.is_file():
        print(item.absolute())

# genres = data['genre'].values

# mfccs = []
# mfccs_shape = []
# labels = []

# for file, genre in zip(audio_files, genres):
#     y, sr = librosa.load(f'data/raw/{file}')
#     mfcc = librosa.feature.mfcc(y=y, sr=sr)
#     print(mfcc.shape)
#     mfccs_shape.append(mfcc.shape)
#     mfccs.append(np.ravel(mfcc)[:4000])
#     labels.append(genre)

# for i in mfccs:
#     print("ITEM: ", i)
# mfccs_array = np.array(mfccs)
# labels_array = np.array(labels)

# print(labels_array)

# np.save('data/processed/mfccs/mfccs.npy', mfccs_array)
# np.save('data/processed/labels/labels.npy', labels_array)
# np.savetxt('data/processed/shapes/mfccs.csv', mfccs_shape, delimiter=",")