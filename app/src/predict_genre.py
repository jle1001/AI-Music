import tensorflow as tf
import numpy as np
import pandas as pd
import librosa

UPLOAD_TRACK = './app/upload/upload.mp3'
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

    model = tf.keras.models.load_model('app/models/model_22L_RMS.h5')

    # Extract MFCC of uploaded track
    y, sr = librosa.load(audio)
    y = y[:(25 * sr)]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfcc = tf.reshape(mfcc.ravel(), shape=(1,10770))

    # Predict output
    print(model.predict(mfcc)[0])
    print(np.argmax(model.predict(mfcc)[0]))
    print(f"{np.argsort(model.predict(mfcc)[0])[-3:][::-1]}")
    return get_genre_name(np.argmax(model.predict(mfcc)[0]))

def get_model(n_model=1):
    # TODO: Documentate this function
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