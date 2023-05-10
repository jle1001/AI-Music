import time
import uuid
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# Load MFCC features
track_genres_mfcc = pd.read_pickle('data/processed/track_genres_mfcc.pkl')

# Obtain X and y sets
X = np.array(track_genres_mfcc['mfcc'])
y = np.array(track_genres_mfcc['genres'])
converted_y = []
for i in y:
    if i == "[]":
        converted_y.append(0)
        continue
    converted_list = [int(num) for num in i.strip('[]').split(',')]
    converted_y.append(converted_list[0])
print(converted_y)

# print(len(np.unique(y)))
# Normalization
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Split data into train, validation and test.
X_train, X_test, y_train, y_test = train_test_split(X, converted_y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# print(X_train)
# print(y_train)

initial_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(164, activation='softmax')
])

simple_conv_model = tf.keras.Sequential([
    tf.keras.layers.Reshape((10770, 1), input_shape=(None, 10770)),
    tf.keras.layers.Conv1D(64, (3), activation='relu'),
    tf.keras.layers.MaxPooling1D((2)),
    tf.keras.layers.Conv1D(128, (3), activation='relu'),
    tf.keras.layers.MaxPooling1D((2)),
    tf.keras.layers.Conv1D(64, (3), activation='relu'),
    tf.keras.layers.MaxPooling1D((2)),
    tf.keras.layers.Conv1D(64, (3), activation='relu'),
    tf.keras.layers.MaxPooling1D((2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(164, activation='softmax')
])

conv_model = tf.keras.Sequential([
    tf.keras.layers.Reshape((10770, 1), input_shape=(None, 10770)),
    tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(164, activation="softmax")
])

# Traning and test set to tensor
X_train = np.vstack(X_train)
y_train = np.vstack(y_train)
X_val = np.vstack(X_val)
y_val = np.vstack(y_val)

X_test = np.vstack(X_test)
y_test = np.vstack(y_test)

def train(model=initial_model):
    # Compilation
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=15)

    # Evaluate model
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'Loss: {loss}')
    print(f'Accuracy: {acc}')

    # Generate model name
    layer_count = len(model.layers)
    optimizer_name = model.optimizer.__class__.__name__
    optimizer_name_short = optimizer_name[:3]
    uuid_string = uuid.uuid4().hex
    uuid_short = uuid_string[:8]
    model_name = f'model_{layer_count}L_{optimizer_name_short}_{uuid_short}'
    print(model_name)

    # Save model
    model.save(f'app/models/{model_name}.h5')

train(conv_model)