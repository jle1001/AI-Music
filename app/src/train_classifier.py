import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler
from keras.optimizers import Adam

# Load MFCC features
track_genres_mfcc = pd.read_pickle('data/processed/track_genres_mfcc.pkl')

# Obtain X and y sets
X = np.array(track_genres_mfcc['mfcc'])
y = np.array(track_genres_mfcc['genre_top'])

# print(len(np.unique(y)))
# Normalization
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Split data into train, validation and test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(X_train)
print(y_train)

# TF model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(164, activation='softmax')
])

# Compilation
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Traning and test set to tensor
X_train = np.vstack(X_train)
y_train = np.vstack(y_train)
X_val = np.vstack(X_val)
y_val = np.vstack(y_val)

X_test = np.vstack(X_test)
y_test = np.vstack(y_test)

# Training
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=5)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Loss: {loss}')
print(f'Accuracy: {acc}')

# Save model
model.save('app/models/initial_model.h5')