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


binary_classifier = RandomForestClassifier(n_estimators=150, class_weight={0: 1.0, 1: 2.0})
attack_classifier = RandomForestClassifier(n_estimators=150)



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
    else:
        traces = df["trace"].apply(lambda x: list(map(int, x.split())))
        X = transformer_donnees(traces)

    y = np.array(df.iloc[:, 2:])
    return X, y


# Fonction pour effectuer la transformation des données pour le binaire d'attaque
def transformer_donnees(traces):
    # Initialiser une liste pour stocker les résultats
    donnees_transformees = []

    # Parcourir chaque trace dans les traces
    for trace in traces:
        # Créer une liste temporaire avec 340 zéros
        temp_liste = [0] * 340

        # Parcourir chaque élément dans la trace
        for element in trace:
            # Ignorer les éléments supérieurs à 340
            if element > 340:
                continue
            # Incrémenter le compte à l'index correspondant de la liste temporaire
            temp_liste[element - 1] += 1

        # Ajouter la liste temporaire aux données transformées
        donnees_transformees.append(temp_liste)

    # Retourner les données transformées comme un tableau numpy
    return np.array(donnees_transformees)


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
                cle = "-".join(map(str, sous_ensemble))

                # Si la clé est dans le vecteur d'attaque, augmenter le compte dans la matrice temporaire
                if cle in vecteur_attaque:
                    temp_matrice[vecteur_attaque[cle]] += 1

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
                cle = "-".join(sous_ensemble)

                # Si la clé est dans les caractéristiques et pas dans le vecteur d'attaque, ajouter la clé au vecteur d'attaque
                if cle in caracteristiques:
                    if cle not in vecteur_attaque:
                        vecteur_attaque[cle] = index
                        index += 1
                # Sinon, si la clé n'est pas dans les caractéristiques, ajouter la clé aux caractéristiques
                else:
                    caracteristiques.add(cle)

    # Retourner le vecteur d'attaque
    return vecteur_attaque

# Detection d'une attaque ou non
def train_binary(attack_data,train_data,validation_data):

    print('-' * 60)
    print("Entraînement du classifieur binaire en cours")
        
    X, y = get_X_y(train_data)
    y = y.sum(axis=1)
    X_attack, y_attack = get_X_y(attack_data)
    X_val, y_val = get_X_y(validation_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    binary_classifier.fit(X_train, y_train)

    pred_val = binary_classifier.predict(X_val)
    pred_a = binary_classifier.predict(X_attack)
    y_pred = binary_classifier.predict(X_test)
    
    print("Précision globale du classifieur binaire sur les données de test :", accuracy_score(y_test, y_pred))
    print("Précision du classifieur binaire sur les attaques uniquement :", accuracy_score([1 for _ in range(len(pred_a))], pred_a))
    print("Précision du classifieur binaire sur la validation uniquement :", accuracy_score([0 for _ in range(len(pred_val))], pred_val))
    print('-' * 60)


# Detection du type d'attaque

def train_attack(attack_vector,attack_data):

    print("\nEntraînement de la détection d'attaque en cours")
    traces = attack_data["trace"].apply(lambda x: x.split())

    X, y = get_X_y(attack_data,attack_vector)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_train = y_train.argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    
    attack_classifier.fit(X_train, y_train)
    
    y_pred = attack_classifier.predict(X_test)
    
    print("\nPrécision de la détection d'attaque :", accuracy_score(y_test, y_pred))

    print('-' * 60)

# Prediction de la dection d'une attaque ou non et du type d'attaque

def predict(trace, attack_vector, threshold=0.13):
    if isinstance(trace, str):
        trace = np.array([list(map(int, trace.split()))])
        
    X_bin = transformer_donnees(trace)
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
            pred = PREDICTIONS.get(predict(trace,attack_vector), "-")
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
    # train_data_path = "FireArmor IA/AI_With_ADFA/IA ADFA-LD/train_data.csv"
    validation_data_path = "FireArmor IA/AI_With_ADFA/IA ADFA-LD/validation_data.csv"
    train_data, validation_data, attack_data = load_data(train_data_path, validation_data_path)

    print('-' * 60)
    print("Training model")
    traces = attack_data["trace"].apply(lambda x: x.split())
    attack_vector = preparer_vecteur(traces)
    train_binary(attack_data,train_data,validation_data)
    train_attack(attack_vector,attack_data)
    print("Training complete")

    print('-' * 60)
    print("Testing model")
    testIAWithSomeAttack()
    print("Testing complete")
    print('-' * 60)

