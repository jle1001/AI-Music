import uuid
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.regularizers import l2
import matplotlib.pyplot as plt

# Load MFCC features
track_genres_mfcc = pd.read_pickle('data/processed/track_genres_mfcc.pkl')

# Obtain X and y sets
X = track_genres_mfcc['mfcc'].apply(lambda x: np.array(x).reshape(-1, 10))
y = np.array(track_genres_mfcc['genres'])

# Get only the main genre
processed_y = []
for i in y:
    if i == "[]":
        processed_y.append(0)
        continue
    converted_list = [int(num) for num in i.strip('[]').split(',')]
    processed_y.append(converted_list[0])
print(np.unique(processed_y, return_counts=True))

# y = track_genres_mfcc['genre_top']
print(np.unique(y, return_counts=True))

converted_x = []
converted_y = []
target_styles = [38, 15, 12, 10, 17, 25, 21]
target_counts = {style: 0 for style in target_styles}

for i, j in zip(X, processed_y):
    # Selection of certain genres
    if j not in target_styles:
        continue
    # if target_counts[j] >= 150:
    #     continue
    converted_y.append(j)
    converted_x.append(i)
    target_counts[j] += 1

    # if all(count == 150 for count in target_counts.values()):
    #     break
print(np.unique(converted_y, return_counts=True))
print(len(converted_y))
print(len(converted_x))

# Split data into train, validation and test.
X_train, X_test, y_train, y_test = train_test_split(converted_x, converted_y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Temp method, change later
def get_model(n_model):
    if (n_model == 1):
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
        model = tf.keras.Sequential([
            # tf.keras.layers.Reshape((10770, 1), input_shape=(None, 10770)),
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.Conv1D(128, (3), activation='relu', kernel_regularizer=l2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPooling1D((2)),
            tf.keras.layers.Conv1D(64, (3), activation='relu', kernel_regularizer=l2(0.01)),
            tf.keras.layers.Dropout(0.5),  # 50% dropout
            tf.keras.layers.MaxPooling1D((2)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(max(converted_y) + 1, activation='softmax')
        ])
    else:
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
    # TODO: Add documentation
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
    # TODO: Add documentation

    # Get the model
    model = get_model(n_model)

    # Model Compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Data to tensor
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)
    y_test_np = np.array(y_test)

    # callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
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

    # Save model
    model.save(f'app/models/{model_name}.h5')

    # Saving the history
    import pickle
    with open(f'app/models/{model_name}_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def train_generators(n_model, train_data=1024, test_data=64, val_data=64):
    # TODO: Add documentation

    # Get the model
    model = get_model(n_model)

    # Model Compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Dataset generation
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
    history = model.fit(train_dataset, 
                steps_per_epoch=526,
                validation_data=val_dataset,
                validation_steps=4,
                epochs=50)
    
    # Model evaluation using generators
    loss, acc = model.evaluate(test_dataset, verbose=2)
    print(f'Loss: {loss}')
    print(f'Accuracy: {acc}')

def eval_model():
    model = tf.keras.models.load_model('app/models/model_8L_Ada.h5')
    y_pred = model.predict(np.stack(X_test))
    y_pred = np.argmax(y_pred, axis=1)

    report = classification_report(y_test, y_pred, output_dict=True)

    # You can then access the overall recall, precision, and f1-score as follows:
    print(f"Overall Precision: {report['weighted avg']['precision']}")
    print(f"Overall Recall: {report['weighted avg']['recall']}")
    print(f"Overall F1-score: {report['weighted avg']['f1-score']}")
    print(f"Accuracy: {report['accuracy']}")

    plot_metrics(report, 'precision')
    plot_metrics(report, 'recall')
    plot_metrics(report, 'f1-score')

# Define a function to plot metrics
def plot_metrics(metrics_dict, metric):
    # Exclude overall metrics, only keep per-class metrics
    classes = [k for k in metrics_dict.keys() if isinstance(metrics_dict[k], dict)]
    values = [metrics_dict[class_label][metric] for class_label in classes]

    plt.figure(figsize=(10,5))
    plt.bar(classes, values, color='blue')
    plt.xlabel('Class')
    plt.ylabel(metric)
    plt.title(f'{metric} for different classes')

    # Save the plot to a file
    plt.savefig(f'app/plots/model_8L_{metric}_plot.png')

# Change later
train(1)
# eval_model()