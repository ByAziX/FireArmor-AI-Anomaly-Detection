from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

import InputData 

PREDICTIONS = {
    0: "NO ATTACK",
    1: "ADD SECOND ADMIN",
    2: "HYDRA FTP BRUTEFORCE",
    3: "HYDRA SSH BRUTEFORCE",
    4: "JAVA METERPRETER",
    5: "METERPRETER",
    6: "WEB_SHELL"
}


binary_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
attack_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


def load_data(train_data_path, validation_data_path):
    try:
        train_data = pd.read_csv(train_data_path)
        validation_data = pd.read_csv(validation_data_path)
        attack_data = train_data[train_data.iloc[:, 2:].sum(axis=1) == 1]
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None
    return train_data, validation_data, attack_data

def get_X_y(df, attack_vector=None):

    if attack_vector is not None:
        traces = df["trace"].apply(lambda x: x.split())
        X = transform_attack(traces, attack_vector)
    else:
        traces = df["trace"].apply(lambda x: list(map(int, x.split())))
        X = transform_X(traces)

    y = np.array(df.iloc[:, 2:])
    return X, y


def transform_X(traces):
    X = []
    for arr in traces:
        temp = [0] * 340
        for i in arr:
            if i > 340:
                continue
            temp[i - 1] += 1
        X.append(temp)
    return np.array(X)

def transform_attack(trace,attack_vector):
    res = []
    for arr in trace:
        temp = [0]*len(attack_vector) + [350]
        for size in range(2, 6):
            for i in range(0, len(arr) - size):
                sub = arr[i: i+size]
                key = "-".join(map(str, sub))
                if key in attack_vector:
                    temp[attack_vector[key]] += 1
        temp = np.array(temp, dtype="float64")
        res.append(temp)
    
    return np.array(res)

    
def prepare_vector(trace):
    d = {}
    features = set()
    ind = 0
    for arr in trace:
        for size in range(2, 6):
            for i in range(0, len(arr) - size):
                sub = arr[i: i+size]
                key = "-".join(sub)
                if key in features:
                    if key not in d:
                        d[key] = ind
                        ind += 1
                else:
                    features.add(key)                               
    return d


def train_attack(attack_vector,attack_data):

    print("\nEntraînement de la détection d'attaque en cours")
    traces = attack_data["trace"].apply(lambda x: x.split())

    X, y = get_X_y(attack_data,attack_vector)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

   
    y_train = y_train.argmax(axis=1)
    y_test = y_test.argmax(axis=1)
     
    attack_classifier.fit(X_train, y_train)
    
    y_pred = attack_classifier.predict(X_test)
    
    print("\nPrécision de la détection d'attaque :", accuracy_score(y_test, y_pred))

    return y_test,y_pred


        
def train_binary(attack_vector,attack_data,train_data,validation_data):

    print('-' * 60)
    print("Entraînement du classifieur binaire en cours")
        
    X, y = get_X_y(train_data)
    y = y.sum(axis=1)
    X_attack, y_attack = get_X_y(attack_data)
    X_val, y_val = get_X_y(validation_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    binary_classifier.fit(X_train, y_train)

    pred_val = binary_classifier.predict(X_val)
    pred_a = binary_classifier.predict(X_attack)
    y_pred = binary_classifier.predict(X_test)
    
    pred_val = pred_val
    pred_a = pred_a
    y_pred = y_pred
    
    print("Précision globale du classifieur binaire sur les données de test :", accuracy_score(y_test, y_pred))
    print("Précision du classifieur binaire sur les attaques uniquement :", accuracy_score([1 for _ in range(len(pred_a))], pred_a))
    print("Précision du classifieur binaire sur la validation uniquement :", accuracy_score([0 for _ in range(len(pred_val))], pred_val))
    return y_test,pred_val,pred_a,y_pred



def predict(trace, attack_vector,predict_one=False):
    if isinstance(trace, str):
        trace = np.array([list(map(int, trace.split()))])
        
    X_bin = transform_X(trace)
    bp = binary_classifier.predict(X_bin)
    
    if predict_one and not bp[0]:
            print("No attack")
            return 0
    print("Attack")
    X_atk = transform_attack(trace, attack_vector)
    attack_predict = attack_classifier.predict(X_atk) + 1

    if predict_one:
        return attack_predict[0]
    return attack_predict


def testIAWithSomeAttack():
    try:
        files = {}
        file_directory = "FireArmor IA/AI_With_ADFA/IA ADFA-LD/tests/"
        files = InputData.readfilesfromAdir(file_directory)
        print(f"Loading ...")

        for filename in files:
            with open(filename) as fs:
                trace = fs.read().strip()
            print("Fichier envoyé à l'IA : ", filename)
            pred = PREDICTIONS.get(predict(trace,attack_vector, predict_one=True), "-")
            print("VERDICT:", pred)
            print('-' * 60)

    except Exception as e:
        print(e)
        print()
    


# Le reste du code reste inchangé...
if __name__ == "__main__":
    
    
    train_data_path = "train.csv"
    # train_data_path = "FireArmor IA/AI_With_ADFA/IA ADFA-LD/train_data.csv"
    validation_data_path = "FireArmor IA/AI_With_ADFA/IA ADFA-LD/validation_data.csv"
    train_data, validation_data, attack_data = load_data(train_data_path, validation_data_path)

    print('-' * 60)
    print("Training model")
    traces = attack_data["trace"].apply(lambda x: x.split())
    attack_vector = prepare_vector(traces)
    train_binary(attack_vector,attack_data,train_data,validation_data)
    train_attack(attack_vector,attack_data)
    print("Training complete")

    print('-' * 60)
    print("Testing model")
    testIAWithSomeAttack()
    print("Testing complete")
    print('-' * 60)

