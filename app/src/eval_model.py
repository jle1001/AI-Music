import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np

# Load the model
model = tf.keras.models.load_model('model.h5')