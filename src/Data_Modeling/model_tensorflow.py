import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pyarrow.parquet as pq
import pyarrow as pa
import joblib
import numpy as np 
import kerastuner as kt

def train_model(train_file, validation_file, y_train_file, y_validation_file,
                batch_size=32, shuffle_buffer_size=100, epochs=10, num_trials=1):
    
    # Read parquet file
    df_train = pd.read_parquet(train_file)
    df_validation = pd.read_parquet(validation_file)

    y_train = joblib.load(y_train_file)
    y_validation = joblib.load(y_validation_file)

    y_train = np.array(y_train)
    y_validation = np.array(y_validation)

    # Convert to TensorFlow Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((df_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((df_validation, y_validation))

    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size, drop_remainder=True).prefetch(1)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    input_shape = df_train.shape[1]

    # Define the model
    def make_model(hp):
        model = Sequential()
        for i in range(hp.Int('num_layers', 2, 20)):
            model.add(Dense(units=hp.Int('units_' + str(i),
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                            activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='binary_crossentropy',
            metrics=['accuracy'])
        return model

    # Instantiate the tuner
    tuner = kt.Hyperband(
        make_model,
        objective='val_accuracy',
        max_epochs=10,
        directory='my_dir',
        project_name='predictive_maintenance')

    # Perform hypertuning
    tuner.search(train_dataset.batch(64),
                 validation_data=test_dataset.batch(64),
                 epochs=50,
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=num_trials)[0]

    # Build the model with the optimal hyperparameters and train it
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_dataset.batch(64), epochs=50, validation_data=test_dataset.batch(64))

    # Save the model
    model.save('best_model')
    
    return model

# Load the saved model
from tensorflow.keras.models import load_model
loaded_model = load_model('best_model')
loaded_model.summary()

# Call the function like this:
train_file = 'data/processed_data.parquet'
validation_file = 'data/processed_data_validation.parquet'
y_train_file = "data/y_train.pkl"
y_validation_file = "data/y_validation.pkl"
model = train_model(train_file, validation_file, y_train_file, y_validation_file)
