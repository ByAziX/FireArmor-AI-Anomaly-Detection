from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle

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


binary_classifier = RandomForestClassifier(n_estimators=150)
attack_classifier = RandomForestClassifier(n_estimators=150)

def sauvegarder_modele(modele, fichier):
    """
    Sauvegarde le modèle binaire dans un fichier à l'aide de pickle.

    Args:
        modele (object): Le modèle binaire à sauvegarder.
        fichier (str): Le nom du fichier de sauvegarde.
    """
    with open(fichier, "wb") as file:
        pickle.dump(modele, file)

def charger_modele(fichier):
    """
    Charge le modèle binaire à partir d'un fichier sauvegardé avec pickle.

    Args:
        fichier (str): Le nom du fichier contenant le modèle sauvegardé.

    Returns:
        object: Le modèle binaire chargé.
    """
    with open(fichier, "rb") as file:
        modele = pickle.load(file)
    return modele 


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
        X = effectuer_transformation_attaque(traces, attack_vector)
        y = np.array(df.iloc[:, 2:])
    return X, y



# Transformation des données pour l'entrainement du classifieur d'attaque (Type d'attaque)
def effectuer_transformation_attaque(traces, vecteur_attaque):
    # Initialiser une liste pour stocker les résultats
    resultats = []

    # Parcourir chaque trace dans les traces
    for trace in traces:
        # Créer une matrice temporaire avec des zéros de la taille du vecteur d'attaque et ajouter 350 à la fin
        temp_matrice = [0]*len(vecteur_attaque) + [350]

        # Parcourir une plage de tailles de 2 à 5
        for taille in range(2, 6):
            # Parcourir la trace actuelle avec la taille actuelle
            for index in range(0, len(trace) - taille):
                # Créer un sous-ensemble de la trace
                sous_ensemble = trace[index: index+taille]

                # Créer une clé en joignant les éléments du sous-ensemble avec un tiret
                pattern = "-".join(map(str, sous_ensemble))

                # Si la clé est dans le vecteur d'attaque, augmenter le compte dans la matrice temporaire
                if pattern in vecteur_attaque:
                    temp_matrice[vecteur_attaque[pattern]] += 1

        # Convertir la matrice temporaire en un tableau numpy et l'ajouter aux résultats
        temp_matrice = np.array(temp_matrice, dtype="float64")
        resultats.append(temp_matrice)
    
    # Retourner les résultats comme un tableau numpy
    return np.array(resultats)


# Préparation du vecteur d'attaque pour l'entrainement du classifieur d'attaque (Type d'attaque)

def preparer_vecteur(traces):
    # Initialiser un dictionnaire pour stocker les vecteurs d'attaque
    vecteur_attaque = {}

    # Initialiser un ensemble pour stocker les caractéristiques uniques
    caracteristiques = set()

    # Initialiser un index pour suivre l'index actuel dans le vecteur d'attaque
    index = 0

    # Parcourir chaque trace dans les traces
    for trace in traces:
        # Parcourir une plage de tailles de 2 à 5
        for taille in range(2, 6):
            # Parcourir la trace actuelle avec la taille actuelle
            for i in range(0, len(trace) - taille):
                # Créer un sous-ensemble de la trace
                sous_ensemble = trace[i: i+taille]

                # Créer une clé en joignant les éléments du sous-ensemble avec un tiret
                pattern = "-".join(sous_ensemble)

                # Si la clé est dans les caractéristiques et pas dans le vecteur d'attaque, ajouter la pattern au vecteur d'attaque
                if pattern in caracteristiques:
                    if pattern not in vecteur_attaque:
                        vecteur_attaque[pattern] = index
                        index += 1
                # Sinon, si la clé n'est pas dans les caractéristiques, ajouter la pattern aux caractéristiques
                else:
                    caracteristiques.add(pattern)

    # Retourner le vecteur d'attaque
    return vecteur_attaque

# Detection d'une attaque ou non
def train_binary(attack_data,train_data,validation_data,attack_vector,train_vector,validation_vector):

    print('-' * 60)
    print("Entraînement du classifieur binaire en cours")
    
    X, y = get_X_y(train_data, train_vector)
    y = y.sum(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    binary_classifier.fit(X_train, y_train)

    # Précision du classifieur binaire
    y_pred = binary_classifier.predict(X_test)
    print("\nPrécision du classifieur binaire :", accuracy_score(y_test, y_pred))
    

    print('-' * 60)


# Detection du type d'attaque

def train_attack(attack_vector,attack_data):

    print("\nEntraînement de la détection d'attaque en cours")

    X, y = get_X_y(attack_data,attack_vector)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_train = y_train.argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    
    attack_classifier.fit(X_train, y_train)
    
    y_pred = attack_classifier.predict(X_test)
    
    print("\nPrécision de la détection d'attaque :", accuracy_score(y_test, y_pred))

    print('-' * 60)

# Prediction de la dection d'une attaque ou non et du type d'attaque

def predict(trace,train_vecteur, attack_vector, threshold=0.13):
    if isinstance(trace, str):
        trace = np.array([list(map(int, trace.split()))])
        
    X_bin = effectuer_transformation_attaque(trace, train_vecteur)
    bp = binary_classifier.predict_proba(X_bin)[:, 1]
    print("Binary prediction :", bp[0])

    if bp[0] < threshold:
        print("No attack")
        return 0
    else:
        print("Attack")
        X_atk = effectuer_transformation_attaque(trace, attack_vector)
        attack_predict = attack_classifier.predict(X_atk) + 1
        return attack_predict[0]

# Test de l'IA avec des attaques

def testIAWithSomeAttack(train_vecteur, attack_vector):
    try:
        files = {}
        file_directory = "FireArmor IA/AI_With_ADFA/IA ADFA-LD/tests/"
        files = InputData.readfilesfromAdir(file_directory)
        print(f"Loading ...")

        for filename in files:
            with open(filename) as fs:
                trace = fs.read().strip()
            print("Fichier envoyé à l'IA : ", filename)
            pred = PREDICTIONS.get(predict(trace,train_vecteur,attack_vector), "-")
            print("VERDICT:", pred)
            print('-' * 60)

    except Exception as e:
        print(e)
        print()
    
def getDataFromTetragon():
    try:
        print(f"Loading ...")
        with open("FireArmor IA/AI_With_ADFA/IA ADFA-LD/tests/attack.txt") as fs:
            trace = fs.read().strip()
        print("Fichier envoyé à l'IA : ", "attack.txt")
        pred = PREDICTIONS.get(predict(trace,attack_vector), "-")
        print("VERDICT:", pred)
        print('-' * 60)
    except:
        print("Error")
        print()
    return 0


if __name__ == "__main__":
    
    train_data_path = "train.csv"
    DataSetFolder = "FireArmor IA/AI_With_ADFA/ADFA-LD/DataSet/"
    # train_data_path = "FireArmor IA/AI_With_ADFA/IA ADFA-LD/train_data.csv"
    validation_data_path = "FireArmor IA/AI_With_ADFA/IA ADFA-LD/validation_data.csv"
    train_data, validation_data, attack_data = load_data(train_data_path, validation_data_path)

    print('-' * 60)
    print("Training model")
    traces_attack = attack_data["trace"].apply(lambda x: x.split())
    traces_train = train_data["trace"].apply(lambda x: x.split())
    traces_validation = validation_data["trace"].apply(lambda x: x.split())
    attack_vector = preparer_vecteur(traces_attack)
    train_vector = preparer_vecteur(traces_train)
    validation_vector = preparer_vecteur(traces_validation)

    train_binary(attack_data,train_data,validation_data,attack_vector,train_vector,validation_vector)
    sauvegarder_modele(binary_classifier, DataSetFolder+"binary_classifier.pkl")
    train_attack(attack_vector,attack_data)
    sauvegarder_modele(attack_classifier, DataSetFolder+"attack_classifier.pkl")
    print("Training complete")

    print('-' * 60)
    print("Testing model")
    testIAWithSomeAttack(train_vector, attack_vector)
    print("Testing complete")
    print('-' * 60)

