import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler
from ast import literal_eval

# Load MFCC features
track_genres_mfcc = pd.read_pickle('data/processed/track_genres_mfcc.pkl')

# Obtain X and y sets
X = np.vstack(track_genres_mfcc['mfcc'])
print(X)

y = track_genres_mfcc['genre_top'].values
y = tf.convert_to_tensor(y, dtype=tf.int16)
print(y)

# # Normalization
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Split data into train, validation and test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# print(X_train)

# TF model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

# Compilation
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=16)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Loss: {loss}')
print(f'Accuracy: {acc}')