from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
import os


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


param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt']
}



binary_classifier = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1  # Utiliser tous les processeurs disponibles
)
attack_classifier = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1  # Utiliser tous les processeurs disponibles
)

TRAIN_DATA_PATH = "FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/train.csv"
DATASETFOLDER = "FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/"
VALIDATION_DATA_PATH = "FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/validation_data.csv"
ONE_TEST_ATTACK = "FireArmor-AI-Anomaly-Detection/FireArmor IA/IA ADFA-LD/tests/no_attack_and_attack.txt"
TEST_ALL_ATTACKS = "FireArmor-AI-Anomaly-Detection/FireArmor IA/IA ADFA-LD/tests/"

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

def testGridSearch(X_train, y_train):
    """
    Fonction qui permet de tester la fonction GridSearchCV de sklearn

    Args:
        X_train (numpy.ndarray): Les données d'entrainement.
        y_train (numpy.ndarray): Les labels d'entrainement.
    """
    
    global binary_classifier, param_grid
    # Utilisation de GridSearchCV pour trouver les meilleurs paramètres
    grid_search = GridSearchCV(estimator=binary_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres
    print("Meilleurs paramètres pour le classifieur binaire :", grid_search.best_params_)



def load_data(train_data_path, validation_data_path):
    """
    Fonction qui permet de charger les données d'entrainement et de validation.

    Args:
        train_data_path (str): Le chemin vers le fichier contenant les données d'entrainement.
        validation_data_path (str): Le chemin vers le fichier contenant les données de validation.

    Returns:
        tuple: Un tuple contenant les données d'entrainement, de validation et les données d'attaque.
    """

    try:
        train_data = pd.read_csv(train_data_path)
        validation_data = pd.read_csv(validation_data_path)
        attack_data = train_data[train_data.iloc[:, 2:].sum(axis=1) == 1]
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None
    return train_data, validation_data, attack_data

def get_X_y(df, attack_vector=None):
    """
    Fonction qui permet de transformer les données en un format utilisable par le classifieur.

    Args:
        df (pandas.DataFrame): Le dataframe contenant les données.
        attack_vector (dict, optional): Le vecteur d'attaque. Defaults to None.

    Returns:
        tuple: Un tuple contenant les données d'entrée et les labels.
    """

    if attack_vector is not None:
        traces = df["trace"].apply(lambda x: x.split())
        X = effectuer_transformation_attaque(traces, attack_vector)
        y = np.array(df.iloc[:, 2:])
    return X, y



def effectuer_transformation_attaque(traces, vecteur_attaque):
    """
    Fonction qui permet de transformer les données en un format utilisable par le classifieur d'attaque.

    Args:
        traces (pandas.Series): Les traces à transformer.
        vecteur_attaque (dict): Le vecteur d'attaque.

    Returns:
        numpy.ndarray: Les données transformées.

    """

    resultats = []

    for trace in traces:
        temp_matrice = [0]*len(vecteur_attaque) # + [350]

        for taille in range(2, 6):
            for index in range(0, len(trace) - taille):
                sous_ensemble = trace[index: index+taille]

                pattern = "-".join(map(str, sous_ensemble))

                if pattern in vecteur_attaque:
                    temp_matrice[vecteur_attaque[pattern]] += 1

        temp_matrice = np.array(temp_matrice, dtype="float64")
        resultats.append(temp_matrice)
    
    return np.array(resultats)



def preparer_vecteur(traces):
    """
    Fonction qui permet de préparer le vecteur d'attaque.

    Args:
        traces (pandas.Series): Les traces à transformer.

    Returns:
        dict: Le vecteur d'attaque.
    """
    vecteur_attaque = {}

    caracteristiques = set()

    index = 0

    for trace in traces:
        for taille in range(2, 6):
            for i in range(0, len(trace) - taille):
                sous_ensemble = trace[i: i+taille]

                pattern = "-".join(sous_ensemble)

                if pattern in caracteristiques:
                    if pattern not in vecteur_attaque:
                        vecteur_attaque[pattern] = index
                        index += 1
                else:
                    caracteristiques.add(pattern)

    return vecteur_attaque

def train_binary(attack_data,train_data,validation_data,train_vector):
    """
    Fonction qui permet d'entrainer le classifieur binaire. Detection d'une attaque ou non


    Args:

        attack_data (pandas.DataFrame): Les données d'attaque.
        train_data (pandas.DataFrame): Les données d'entrainement.
        validation_data (pandas.DataFrame): Les données de validation.
        attack_vector (dict): Le vecteur d'attaque.
        train_vector (dict): Le vecteur d'entrainement.
        validation_vector (dict): Le vecteur de validation.
    """
    global binary_classifier
    print('-' * 60)
    print("Entraînement du classifieur binaire en cours")
    
    X, y = get_X_y(train_data, train_vector)
    y = y.sum(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    binary_classifier.fit(X_train, y_train)

    y_pred = binary_classifier.predict(X_test)
    print("\nPrécision du classifieur binaire :", accuracy_score(y_test, y_pred))
    

    print('-' * 60)



def train_attack(attack_vector,attack_data):
    """
    Fonction qui permet d'entrainer le classifieur d'attaque.

    Args:
        attack_vector (dict): Le vecteur d'attaque.
        attack_data (pandas.DataFrame): Les données d'attaque.
    """


    print("\nEntraînement de la détection d'attaque en cours")

    X, y = get_X_y(attack_data,attack_vector)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_train = y_train.argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    
    attack_classifier.fit(X_train, y_train)
    
    y_pred = attack_classifier.predict(X_test)
    
    print("\nPrécision de la détection d'attaque :", accuracy_score(y_test, y_pred))

    print('-' * 60)


def predict(trace,train_vecteur, attack_vector, threshold=0.5):
    """
    Fonction qui permet de prédire si une attaque est en cours ou non.

    Args:
        trace (str): La trace à prédire.
        train_vecteur (dict): Le vecteur d'entrainement.
        attack_vector (dict): Le vecteur d'attaque.
        threshold (float, optional): Le seuil de prédiction. Defaults to 0.13.

    Returns:
        int: Le type d'attaque.
    """
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


def testIAWithSomeAttack(train_vecteur, attack_vector):
    """
    Fonction qui permet de tester l'IA avec des attaques.

    Args:
        train_vecteur (dict): Le vecteur d'entrainement.
        attack_vector (dict): Le vecteur d'attaque.
    """

    try:
        files = {}
        file_directory = TEST_ALL_ATTACKS
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
    
previous_syscalls = []

def predict_trace_from_file(train_vector, attack_vector):
    """
    Fonction qui lit chaque numéro d'une trace à partir d'un fichier et le prédit avec l'IA,
    en conservant en mémoire les systèmes d'appel précédents pour réaliser une trace complète.

    Args:
        train_vector (dict): Le vecteur d'entrainement.
        attack_vector (dict): Le vecteur d'attaque.

    Returns:
        list: Les prédictions pour chaque système d'appel.
    """
    global previous_syscalls
    file_path = ONE_TEST_ATTACK

    with open(file_path) as file:
        trace = file.read().strip().split()

    predictions = []

    for i, syscall in enumerate(trace):
        previous_syscalls.append(trace[i-1])  # Ajoute le système d'appel précédent à la liste des précédents
        if i > 0: 
            pred = PREDICTIONS.get(predict([previous_syscalls], train_vector, attack_vector), "-")
            print(f"Prediction : {pred}")
            predictions.append(pred)
        
        # tout les 10 on reset la liste des précédents
        #if i % 20 == 0:
            # on garde les 20 derniers
            # previous_syscalls = previous_syscalls[-10:]



    return predictions




if __name__ == "__main__":
    
    
    
    train_data, validation_data, attack_data = load_data(TRAIN_DATA_PATH, VALIDATION_DATA_PATH)

    print('-' * 60)
    print("Training model")
    traces_attack = attack_data["trace"].apply(lambda x: x.split())
    traces_train = train_data["trace"].apply(lambda x: x.split())
    traces_validation = validation_data["trace"].apply(lambda x: x.split())
    

    if os.path.exists(DATASETFOLDER+"binary_classifier.pkl") and os.path.exists(DATASETFOLDER+"attack_classifier.pkl"):
        attack_vector = charger_modele(DATASETFOLDER+"attack_vector.pkl")
        train_vector = charger_modele(DATASETFOLDER+"train_vector.pkl")
        binary_classifier = charger_modele(DATASETFOLDER+"binary_classifier.pkl")
        attack_classifier = charger_modele(DATASETFOLDER+"attack_classifier.pkl")

    else:
        attack_vector = preparer_vecteur(traces_attack)
        train_vector = preparer_vecteur(traces_train)
        sauvegarder_modele(attack_vector, DATASETFOLDER+"attack_vector.pkl")
        sauvegarder_modele(train_vector, DATASETFOLDER+"train_vector.pkl")

        train_binary(attack_data,train_data,validation_data,train_vector)
        sauvegarder_modele(binary_classifier, DATASETFOLDER+"binary_classifier.pkl")
        train_attack(attack_vector,attack_data)
        sauvegarder_modele(attack_classifier, DATASETFOLDER+"attack_classifier.pkl")
    print("Training complete")

    print('-' * 60)
    print("Testing model")
    testIAWithSomeAttack(train_vector, attack_vector)
    print("Testing complete")
    print('-' * 60)
    print("Predicting trace from system call file")
    predict_trace_from_file(train_vector, attack_vector)