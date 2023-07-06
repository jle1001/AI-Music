import unittest
import librosa
import numpy as np
import tensorflow as tf

class TestModelPrediction(unittest.TestCase):
    TEST_TRACK = './app/tests/test_audio.ogg'
    def setUp(self):
        # Load the trained model
        self.model = tf.keras.models.load_model('app/models/model_8L_Ada.h5')
        self.target_styles = [38, 15, 12, 10, 17, 25, 21]

    def test_prediction_quantity(self):
        y, sr = librosa.load(self.TEST_TRACK)
        y = y[:(25 * sr)]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        mfcc = tf.reshape(mfcc.ravel(), shape=(1, 1077, 10))
        predictions = self.model.predict(mfcc)[0]
        # Check that the number of genres predicted is correct.
        self.assertEqual(len(predictions), max(self.target_styles) + 1)

if __name__ == '__main__':
    unittest.main()