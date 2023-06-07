import time
import uuid
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from keras.optimizers import Adam

# Load MFCC features
track_genres_mfcc = pd.read_pickle('data/processed/track_genres_mfcc.pkl')

# Obtain X and y sets
X = np.array(track_genres_mfcc['mfcc'])
# y = np.array(track_genres_mfcc['genre_top'])
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

# Split data into train, validation and test.
X_train, X_test, y_train, y_test = train_test_split(X, converted_y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# print(X_train)
# print(y_train)

initial_model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1236, activation='softmax')
])

# simple_conv_model = tf.keras.Sequential([
#     tf.keras.layers.Reshape((10770, 1), input_shape=(None, 10770)),
#     tf.keras.layers.Conv1D(64, (3), activation='relu'),
#     tf.keras.layers.MaxPooling1D((2)),
#     tf.keras.layers.Conv1D(128, (3), activation='relu'),
#     tf.keras.layers.MaxPooling1D((2)),
#     tf.keras.layers.Conv1D(64, (3), activation='relu'),
#     tf.keras.layers.MaxPooling1D((2)),
#     tf.keras.layers.Conv1D(64, (3), activation='relu'),
#     tf.keras.layers.MaxPooling1D((2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(164, activation='softmax')
# ])

# conv_model = tf.keras.Sequential([
#     tf.keras.layers.Reshape((10770, 1), input_shape=(None, 10770)),
#     tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling1D(pool_size=2),

#     tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling1D(pool_size=2),

#     tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling1D(pool_size=2),

#     tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling1D(pool_size=2),

#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(1235, activation="softmax")
# ])

# Look to the dimensionality about the conv and simple conv models.

# Traning and test set to tensor
X_train = np.vstack(X_train)
y_train = np.vstack(y_train)
X_val = np.vstack(X_val)
y_val = np.vstack(y_val)

X_test = np.vstack(X_test)
y_test = np.vstack(y_test)

def generator(X, y, batch_size):
    # TODO: Add documentation
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size 
    indices = np.arange(num_samples)
    
    while True:
        np.random.shuffle(indices)  # Shuffle to avoid bias
        
        for i in range(num_batches):
            batch_indices = indices[i*batch_size : (i+1)*batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            yield batch_X, batch_y

def train(model):
    # TODO: Add documentation
    # Compilation
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training
    # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    # Dataset generation
    train_data = 2048
    test_data = 512
    val_data = 512
    train_dataset = tf.data.Dataset.from_generator(
        lambda: generator(X_train, y_train, train_data), 
        output_signature=(
            tf.TensorSpec(shape=(train_data, 10770), dtype=tf.float32),
            tf.TensorSpec(shape=(train_data, 1), dtype=tf.float32)
        )
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: generator(X_test, y_test, test_data), 
        output_signature=(
            tf.TensorSpec(shape=(test_data, 10770), dtype=tf.float32),
            tf.TensorSpec(shape=(test_data, 1), dtype=tf.float32)
        )
    )

    val_dataset = tf.data.Dataset.from_generator(
        lambda: generator(X_val, y_val, val_data), 
        output_signature=(
            tf.TensorSpec(shape=(val_data, 10770), dtype=tf.float32),
            tf.TensorSpec(shape=(val_data, 1), dtype=tf.float32)
        )
    )

    # Model fit with generators
    model.fit(train_dataset, 
              steps_per_epoch=128,
              validation_data=val_dataset,
              validation_steps=12,
              epochs=3)
    
    # Model evaluation with generators
    loss, acc = model.evaluate(test_dataset, steps=X_test.shape[0] // 100, verbose=2)
    print(f'Loss: {loss}')
    print(f'Accuracy: {acc}')

    # Generate model name
    layer_count = len(model.layers)
    optimizer_name = model.optimizer.__class__.__name__
    optimizer_name_short = optimizer_name[:3]
    uuid_string = uuid.uuid4().hex
    uuid_short = uuid_string[:8]
    model_name = f'model_{layer_count}L_{optimizer_name_short}_{uuid_short}'
    model_name = f'model_{layer_count}L_{optimizer_name_short}'
    print(model_name)

    # Save model
    # model.save(f'app/models/{model_name}.h5')

# train(conv_model)
# train(initial_model)