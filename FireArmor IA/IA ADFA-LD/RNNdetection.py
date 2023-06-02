import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import tensorflow as tf
import InputData

PREDICTIONS = {
    0: "NO ATTACK",
    1: "ATTACK"
}

ATTACK_TYPES = {
    0: "Adduser",
    1: "Hydra_FTP",
    2: "Hydra_SSH",
    3: "Java_Meterpreter",
    4: "Meterpreter",
    5: "Web_Shell"
}

# Global hyper-parameters
SEQUENCE_LENGTH = 300
EPOCHS = 20
BATCH_SIZE = 32

MODEL_PROB_PATH = "model_prob.h5"
MODEL_CLASS_PATH = "model_class.h5"
DIRECTORY = "FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/"

def save_model(model, model_name):
    """
    Save the trained model

    Args:
        model: The trained model
        model_name: The name of the model file

    Returns:
        None
    """
    model.save(model_name)

def load_model(model_name):
    """ Load the trained model 
    Args:
        model_name: The name of the model file

    Returns:
        The trained model
    """
    model = tf.keras.models.load_model(model_name)
    return model

def preprocess_data(file_path):
    """ Preprocess the trace data 
    Args:
        file_path: The path to the trace data file

    Returns:
        X_train: The training data
        X_test: The test data
        y_train: The training labels
        y_test: The test labels
    """
    data = pd.read_csv(file_path)

    # Preprocess trace sequences
    data['trace'] = data['trace'].apply(lambda x: list(map(int, x.split())))

    # Convert the traces into a 2D list
    traces = data['trace'].tolist()

    # Pad the sequences so that they are all the same length
    traces = pad_sequences(traces, maxlen=SEQUENCE_LENGTH)

    # Define the targets
    targets = data[['Adduser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']].sum(axis=1).apply(lambda x: 1 if x > 0 else 0).values

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(traces, targets, test_size=0.2, random_state=42)

    # Reshape input to be 3D [samples, timesteps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test

def build_model_prob(input_shape):
    """ Build the LSTM model for anomaly detection 
    Args:
        input_shape: The shape of the input data

    Returns:
        The LSTM model
    """
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_model_class(input_shape, num_classes):
    """ Build the LSTM model for attack classification 
    Args:
        input_shape: The shape of the input data
        num_classes: The number of classes

    Returns:
        The LSTM model
    """
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model_prob(model, X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """ Train the LSTM model for anomaly detection 
    Args:
        model: The LSTM model
        X_train: The training data
        y_train: The training labels
        X_test: The test data
        y_test: The test labels
        epochs: The number of epochs
        batch_size: The batch size

    Returns:
        The trained LSTM model
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model

def train_model_class(model, X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """ Train the LSTM model for attack classification 
    Args:
        model: The LSTM model
        X_train: The training data
        y_train: The training labels
        X_test: The test data
        y_test: The test labels
        epochs: The number of epochs
        batch_size: The batch size

    Returns:
        The trained LSTM model
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model

def predict_trace(model_prob, model_class, trace):
    """ Predict if a trace is normal or anomalous, and classify the attack type 
    Args:
        model_prob: The LSTM model for anomaly detection
        model_class: The LSTM model for attack classification
        trace: The trace to predict

    Returns:
        predicted_class_label: The predicted class label
        attack_type_label: The attack type label
        anomaly_prediction: The anomaly prediction
    """
    # Pad the sequence to match the expected length
    trace = pad_sequences([trace], maxlen=SEQUENCE_LENGTH)

    # Reshape input to be 3D [samples, timesteps, features]
    trace = np.reshape(trace, (1, trace.shape[1], 1))

    # Make anomaly prediction
    anomaly_prediction = model_prob.predict(trace)

    # Make attack classification prediction
    predicted_class_index = 1 if anomaly_prediction >= 0.33 else 0
    predicted_class_label = PREDICTIONS[predicted_class_index]

    # Get the attack type
    if predicted_class_label == "ATTACK":
        attack_type_prediction = np.argmax(model_class.predict(trace), axis=1)[0]
        attack_type_label = ATTACK_TYPES[attack_type_prediction]
    else:
        attack_type_label = "N/A"

    return predicted_class_label, attack_type_label, anomaly_prediction

def main():
    """ The main function """
    if os.path.exists(os.path.join(DIRECTORY, MODEL_PROB_PATH)) and os.path.exists(os.path.join(DIRECTORY, MODEL_CLASS_PATH)):
        # Load the models
        model_prob = load_model(os.path.join(DIRECTORY, MODEL_PROB_PATH))
        model_class = load_model(os.path.join(DIRECTORY, MODEL_CLASS_PATH))
    else:
        # Delete the old model files if they exist
        if os.path.exists(MODEL_PROB_PATH):
            os.remove(MODEL_PROB_PATH)
        if os.path.exists(os.path.join(DIRECTORY, MODEL_CLASS_PATH)):
            os.remove(os.path.join(DIRECTORY, MODEL_CLASS_PATH))

        # Preprocess the data
        file_path = 'train.csv'
        X_train, X_test, y_train, y_test = preprocess_data(file_path)

        # Create the LSTM models
        input_shape = (X_train.shape[1], 1)
        num_classes = 1
        model_prob = build_model_prob(input_shape)
        model_class = build_model_class(input_shape, num_classes)

        # Train the models
        model_prob = train_model_prob(model_prob, X_train, y_train, X_test, y_test)
        model_class = train_model_class(model_class, X_train, y_train, X_test, y_test)

        # Save the models
        save_model(model_prob, os.path.join(DIRECTORY, MODEL_PROB_PATH))
        save_model(model_class, os.path.join(DIRECTORY, MODEL_CLASS_PATH))

    # Predict a trace
    trace = InputData.readCharsFromFile("FireArmor-AI-Anomaly-Detection/FireArmor IA/IA ADFA-LD/tests/UAD-Hydra-SSH-1-2311.txt")

    predicted_class_label, attack_type_label, anomaly_prediction = predict_trace(model_prob, model_class, trace)

    print("Attack Type:", predicted_class_label)
    print("Attack Subtype:", attack_type_label)
    print("Anomaly Probability:", anomaly_prediction)

if __name__ == "__main__":
    main()