import tensorflow as tf
import numpy as np
import librosa

UPLOAD_TRACK = './app/upload/upload.mp3'

# Load trained model
model = tf.keras.models.load_model('app/models/initial_model.h5')

def predict(audio=UPLOAD_TRACK):
    # Extract MFCC of uploaded track
    y, sr = librosa.load(audio)
    y = y[:(25 * sr)]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfcc = tf.reshape(mfcc.ravel(), shape=(1,10770))

    # Predict output
    print(model.predict(mfcc)[0])
    return np.argmax(model.predict(mfcc)[0])