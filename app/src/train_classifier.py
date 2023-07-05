import uuid
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.regularizers import l2
import matplotlib.pyplot as plt

#######################################################################
# Python script for loading, process and train a neural network model #
#######################################################################

# Load MFCC features
track_genres_mfcc = pd.read_pickle('data/processed/track_genres_mfcc.pkl')

# Obtain X and y sets
X = track_genres_mfcc['mfcc'].apply(lambda x: np.array(x).reshape(-1, 10))
y = np.array(track_genres_mfcc['genres'])

# Extraction of the main genre of each track. This approach focuses on classifying the main genre, making the problem more manageable.
processed_y = []
for i in y:
    if i == "[]":
        processed_y.append(0)
        continue
    converted_list = [int(num) for num in i.strip('[]').split(',')]
    processed_y.append(converted_list[0])
print(np.unique(processed_y, return_counts=True))

# print(np.unique(y, return_counts=True))

# Get 8 more important music genres in the dataset.
# 38: Experimental
# 15: Electronic
# 12: Rock
# 10: Pop
# 17: Folk
# 25: Punk
# 21: Hip-Hop
converted_x = []
converted_y = []
target_styles = [38, 15, 12, 10, 17, 25, 21]
target_counts = {style: 0 for style in target_styles}

for i, j in zip(X, processed_y):
    if j not in target_styles:
        continue
    converted_y.append(j)
    converted_x.append(i)
    target_counts[j] += 1

# print(np.unique(converted_y, return_counts=True))
# print(len(converted_y))
# print(len(converted_x))

# Split data into train, validation and test.
X_train, X_test, y_train, y_test = train_test_split(converted_x, converted_y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

def get_model(n_model):
    """Function that returns a deep learning model based on the parameter input. 
    There are three types of models that can be generated: two types of Convolutional Neural Networks (CNN) and a Dense Network. 

    Args:
        n_model (int): An integer to specify the type of model to generate. 
            Accepts 1, 2 or 3, refering to the first CNN model, the second CNN model, or the Dense network respectively.

    Returns:
        model (tf.keras.Sequential): Returns a deep learning model. The output model depends on the n_model parameter."""
    
    if (n_model == 1):
        # First CNN Model.
        model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding="same"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv1D(32, 2, activation='relu', padding="same"),
            tf.keras.layers.MaxPooling1D(1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(max(converted_y) + 1, activation='softmax')
        ])
    elif (n_model == 2):
        # Second CNN Model. Includes kernel regularization to prevent overfitting.
        model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.Conv1D(128, (3), activation='relu', kernel_regularizer=l2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPooling1D((2)),

            tf.keras.layers.Conv1D(64, (3), activation='relu', kernel_regularizer=l2(0.01)),
            tf.keras.layers.Dropout(0.5), 
            tf.keras.layers.MaxPooling1D((2)),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(max(converted_y) + 1, activation='softmax')
        ])
    elif (n_model == 3):
        # Dense Model.
        model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(max(converted_y) + 1, activation="softmax")
        ])

    return model

def generator(X, y, batch_size):
    """Generates batches of training data in a random order. 
    This generator continuously generates batches of data indefinitely.

    Args:
        X: Feature matrix.
        y: Target vector.
        batch_size: The size of the batches to generate.

    Yields:
        (batch_X, batch_y): batch of features and batch of targets."""
    
    num_samples = X.shape[0]
    print(len(y))
    num_batches = num_samples // batch_size 
    print(f'{num_samples}, {num_batches}')
    indices = np.arange(num_samples)
    
    while True:
        # Shuffle to avoid bias
        np.random.shuffle(indices)  
        
        for i in range(num_batches):
            batch_indices = indices[i*batch_size : (i+1)*batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            yield batch_X, batch_y

def train(n_model):
    """Trains a neural network model and saves it and its training data to the disk.

    Args:
        n_model: A parameter to be passed to get_model() function to retrieve the model to be trained."""

    # Get the model
    model = get_model(n_model)

    # Model Compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Data to tensor so TensorFlow can work with them.
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)
    y_test_np = np.array(y_test)

    # callbacks = [EarlyStopping(patience=5, restore_best_weights=True)] # Function to stop the training in a certain point when the performance stops improving.
    history = model.fit(X_train_np, y_train_np, epochs=10, validation_split=0.2, batch_size=5)
    loss, acc = model.evaluate(X_test_np, y_test_np, verbose=2)
    print(f'Loss: {loss}')
    print(f'Accuracy: {acc}')
    model.summary()
    
    # Generate model name
    layer_count = len(model.layers)
    optimizer_name = model.optimizer.__class__.__name__
    optimizer_name_short = optimizer_name[:3]
    uuid_string = uuid.uuid4().hex
    uuid_short = uuid_string[:8]
    model_name = f'model_{layer_count}L_{optimizer_name_short}_{uuid_short}'
    model_name = f'model_{layer_count}L_{optimizer_name_short}'
    print(model_name)

    # Save model to the disk
    model.save(f'app/models/{model_name}.h5')

    # Save the training history to the disk to do analysis or create visualizations
    import pickle
    with open(f'app/models/{model_name}_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def train_generators(n_model, train_data=1024, test_data=64, val_data=64):
    """Trains a neural network model using generators saves it to the disk.

    Args:
        n_model: A parameter to be passed to get_model() function to retrieve the model to be trained.
        train_data: Batch size for the training dataset. Default is 1024.
        test_data: Batch size for the testing dataset. Default is 64.
        val_data: Batch size for the validation dataset. Default is 64."""
    
    # Get the model
    model = get_model(n_model)

    # Model Compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Dataset generation. Using tf.data.Dataset.from_generator() creates a dataset from a generator function.
        # lambda: generator(X, y, train_data) returns batches data.
        # output_signature: define the shape and type of the output from the generator function.

    train_data = train_data
    test_data = test_data
    val_data = val_data

    train_dataset = tf.data.Dataset.from_generator(
        lambda: generator(X_train, y_train, train_data), 
        output_signature=(
            tf.TensorSpec(shape=(train_data, 10, 1077), dtype=tf.float32),
            tf.TensorSpec(shape=(train_data,), dtype=tf.float32)
        )
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: generator(X_test, y_test, test_data), 
        output_signature=(
            tf.TensorSpec(shape=(test_data, 10, 1077), dtype=tf.float32),
            tf.TensorSpec(shape=(test_data,), dtype=tf.float32)
        )
    )

    val_dataset = tf.data.Dataset.from_generator(
        lambda: generator(X_val, y_val, val_data), 
        output_signature=(
            tf.TensorSpec(shape=(val_data, 10, 1077), dtype=tf.float32),
            tf.TensorSpec(shape=(val_data,), dtype=tf.float32)
        )
    )
    
    # Model fit using generators
    model.fit(train_dataset, 
              steps_per_epoch=526,
              validation_data=val_dataset,
              validation_steps=4,
              epochs=50)
    
    # Model evaluation using generators
    loss, acc = model.evaluate(test_dataset, verbose=2)
    print(f'Loss: {loss}')
    print(f'Accuracy: {acc}')

def eval_model():
    """Loads a trained model and evaluate it."""
    model = tf.keras.models.load_model('app/models/model_17L_Ada.h5')

    # Selects the class with the highest probability.
    y_pred = model.predict(np.stack(X_test))
    y_pred = np.argmax(y_pred, axis=1)

    # Provide classification metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Overall Precision: {report['weighted avg']['precision']}")
    print(f"Overall Recall: {report['weighted avg']['recall']}")
    print(f"Overall F1-score: {report['weighted avg']['f1-score']}")
    print(f"Accuracy: {report['accuracy']}")

train(1)
# eval_model()