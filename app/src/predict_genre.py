import tensorflow as tf
import numpy as np
import pandas as pd
import os
import librosa

UPLOAD_TRACK = './app/static/upload/upload.mp3'
GENRES_LIST = './data/raw_genres.csv'

# Load trained model
# initial_model = tf.keras.models.load_model('app/models/model_6L_RMS.h5')
# simple_conv_model = tf.keras.models.load_model('app/models/model_11L_RMS.h5')
# conv_model = tf.keras.models.load_model('app/models/model_7L_RMS.h5')

def predict(audio=UPLOAD_TRACK, n_model=3):
    """Predicts the genre of an audio track using its MFCCs features into a trained deep learning model.

    Args:
        audio: Path to the audio file to be analyzed.
        n_model: Model number.
    
    Returns:
        Top three predicted genres for the given audio track."""
    if n_model == "1":
        model = tf.keras.models.load_model('app/models/model_8L_Ada.h5')
    elif n_model == "2":
        model = tf.keras.models.load_model('app/models/model_11L_Ada.h5')
    elif n_model == "3":
        model = tf.keras.models.load_model('app/models/model_17L_Ada.h5')
    else:
        return

    # Extract MFCC of uploaded track
    y, sr = librosa.load(audio)
    y = y[:(25 * sr)]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfcc = tf.reshape(mfcc.ravel(), shape=(1, 1077, 10))

    # Predict output
    genres = []
    predictions = model.predict(mfcc)[0]
    print(predictions)
    # print(np.argmax(model.predict(mfcc)[0]))
    top_3_predictions = np.argsort(model.predict(mfcc)[0])[-3:][::-1]

    # Returns the top three predicted genres
    for i in top_3_predictions:
        genres.append((get_genre_name(i), predictions[i] * 100))

    # Removes upload audio file from disk due to legal reasons.
    os.remove(UPLOAD_TRACK)

    return ', '.join(f'{genre_name}: {genre_prob:.2f}%' for genre_name, genre_prob in genres)

def get_model(n_model=1):
    """ Retrieve the name of the model type based on the number.

    The function maps input number to model name:
    "1" -> 'Dense'
    "2" -> 'Simple Convolutional'
    "3" -> 'Convolutional'

    Args:
        n_model: number of the model. Default value is 1.

    Returns:
        str: The name of the corresponding model. """
    model = []
    if n_model == "1":
        model = f'Dense'
    elif n_model == "2":
        model = f'Simple Convolutional'
    elif n_model == "3":
        model = f'Convolutional'

    return model

def get_genre_name(genre_number):
    """Returns the genre name from the identifier number.

    Args:
        genre_number: Genre identifier number.

    Returns:
        The name of the genre in string format."""
    genres_list = pd.read_csv(GENRES_LIST)
    return genres_list[genres_list["genre_id"] == genre_number]["genre_handle"].item()

# print(get_genre_name(12))
# predict()