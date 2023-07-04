import tensorflow as tf
import numpy as np
import pandas as pd
import librosa

UPLOAD_TRACK = './app/static/upload/upload.mp3'
GENRES_LIST = './data/raw_genres.csv'

# Load trained model
# initial_model = tf.keras.models.load_model('app/models/model_6L_RMS.h5')
# simple_conv_model = tf.keras.models.load_model('app/models/model_11L_RMS.h5')
# conv_model = tf.keras.models.load_model('app/models/model_7L_RMS.h5')

def predict(audio=UPLOAD_TRACK, n_model=3):
    # TODO: Documentate this function
    # model = []
    # if n_model == "1":
    #     model = initial_model
    # elif n_model == "2":
    #     model = simple_conv_model
    # elif n_model == "3":
    #     model = conv_model

    model = tf.keras.models.load_model('app/models/model_9L_RMS.h5')

    # Extract MFCC of uploaded track
    y, sr = librosa.load(audio)
    y = y[:(25 * sr)]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfcc = tf.reshape(mfcc.ravel(), shape=(1,10770))

    # Predict output
    genres = []
    print(model.predict(mfcc)[0])
    print(np.argmax(model.predict(mfcc)[0]))
    print(f"{np.argsort(model.predict(mfcc)[0])[-3:][::-1]}")

    # return get_genre_name(np.argmax(model.predict(mfcc)[0]))
    for i in np.argsort(model.predict(mfcc)[0])[-3:][::-1]:
        genres.append(get_genre_name(i))
    return f'{genres[0]}, {genres[1]}, {genres[2]}'

def get_model(n_model=1):
    """ Retrieve the name of the model type based on the number.

    The function maps input number to model name:
    "1" -> 'Initial'
    "2" -> 'Simple Convolutional'
    "3" -> 'Convolutional'

    Args:
        n_model: number of the model. Default value is 1.

    Returns:
        str: The name of the corresponding model. """
    model = []
    if n_model == "1":
        model = f'Initial'
    elif n_model == "2":
        model = f'Simple Convolutional'
    elif n_model == "3":
        model = f'Convolutional'

    return model

def get_genre_name(genre_number):
    genres_list = pd.read_csv(GENRES_LIST)
    return genres_list[genres_list["genre_id"] == genre_number]["genre_handle"].item()

# print(get_genre_name(12))
# predict()