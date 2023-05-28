import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
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
sequence_length = 100
random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
epochs = 10
batch_size = 32

def dropin(X, y):
    """ Function to augment data by randomly duplicating samples """
    X_hat = []
    y_hat = []
    for i in range(len(X)):
        for j in range(np.random.randint(0, random_data_dup)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat)

def preprocess_data(file_path):
    """ Preprocess the trace data """
    data = pd.read_csv(file_path)

    # Preprocess trace sequences
    data['trace'] = data['trace'].apply(lambda x: list(map(int, x.split())))

    # Convert the traces into a 2D list
    traces = data['trace'].tolist()

    # Pad the sequences so that they are all the same length
    traces = pad_sequences(traces, maxlen=sequence_length)

    # Define the targets
    targets = data[['Adduser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']].sum(axis=1).apply(lambda x: 1 if x > 0 else 0).values

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(traces, targets, test_size=0.2, random_state=42)

    # Reshape input to be 3D [samples, timesteps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test


def build_model_prob(input_shape):
    """ Build the LSTM model for anomaly detection """
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_model_class(input_shape, num_classes):
    """ Build the LSTM model for attack classification """
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model_prob(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    """ Train the LSTM model for anomaly detection """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model

def train_model_class(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    """ Train the LSTM model for attack classification """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model


def predict_trace(model_prob, model_class, trace):
    """ Predict if a trace is normal or anomalous, and classify the attack type """
    # Pad the sequence to match the expected length
    trace = pad_sequences([trace], maxlen=sequence_length)

    # Reshape input to be 3D [samples, timesteps, features]
    trace = np.reshape(trace, (1, trace.shape[1], 1))

    # Make anomaly prediction
    anomaly_prediction = model_prob.predict(trace)
    if anomaly_prediction >= 0.5:
        is_anomaly = True
    else:
        is_anomaly = False

    # Make attack classification prediction
    class_prediction = model_class.predict(trace)
    predicted_class_index = 1 if class_prediction >= 0.5 else 0
    predicted_class_label = PREDICTIONS[predicted_class_index]

    # Get the attack type
    if predicted_class_index == 1:
        attack_type_prediction = np.argmax(model_class.predict(trace), axis=1)[0]
        attack_type_label = ATTACK_TYPES[attack_type_prediction]
    else:
        attack_type_label = "N/A"

    return is_anomaly, predicted_class_label, attack_type_label, anomaly_prediction

# Define the path to the trace data file
file_path = 'train.csv'

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(file_path)

# Create the LSTM models
input_shape = (X_train.shape[1], 1)
num_classes = 1
model_prob = build_model_prob(input_shape)
model_class = build_model_class(input_shape, num_classes)

# Train the models
model_prob = train_model_prob(model_prob, X_train, y_train, X_test, y_test, epochs, batch_size)
model_class = train_model_class(model_class, X_train, y_train, X_test, y_test, epochs, batch_size)

# Predict a trace
trace = InputData.readCharsFromFile("FireArmor IA/AI_With_ADFA/IA ADFA-LD/tests/UAD-Hydra-FTP-1-9186.txt")

is_anomaly, predicted_class_label, attack_type_label, anomaly_prediction = predict_trace(model_prob, model_class, trace)

print("Is Anomaly:", is_anomaly)
print("Attack Type:", predicted_class_label)
print("Attack Subtype:", attack_type_label)
print("Anomaly Probability:", anomaly_prediction)
