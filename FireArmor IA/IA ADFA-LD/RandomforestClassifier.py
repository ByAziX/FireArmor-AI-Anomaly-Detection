from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV


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

TRAIN_DATA_PATH = "FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/train.csv"
DATASETFOLDER = "FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/"
VALIDATION_DATA_PATH = "FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/validation_data.csv"
TEST_ALL_ATTACKS = "FireArmor-AI-Anomaly-Detection/FireArmor IA/IA ADFA-LD/tests/"



binary_classifier = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    class_weight={0: 1.0, 1: 2.0}
)

attack_classifier = RandomForestClassifier(
    n_estimators=500
)

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
    Effectue une recherche des meilleurs paramètres pour le classifieur binaire.

    Args:
        X_train (numpy.ndarray): Les données d'entrainement.
        y_train (numpy.ndarray): Les étiquettes d'entrainement.

    Returns:
        object: Le classifieur binaire avec les meilleurs paramètres.
    """
    global attack_classifier, param_grid
    # Utilisation de GridSearchCV pour trouver les meilleurs paramètres
    grid_search = GridSearchCV(estimator=attack_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres
    print("Meilleurs paramètres pour le classifieur binaire :", grid_search.best_params_)




def load_data(train_data_path, validation_data_path):
    """
    Charge les données d'entrainement et de validation à partir des fichiers csv.

    Args:
        train_data_path (str): Le chemin vers le fichier csv contenant les données d'entrainement.
        validation_data_path (str): Le chemin vers le fichier csv contenant les données de validation.

    Returns:
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame: Les données d'entrainement, les données de validation et les données d'attaque.
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
    Récupère les données et les étiquettes à partir d'un dataframe.

    Args:
        df (pandas.DataFrame): Le dataframe contenant les données.
        attack_vector (list, optional): Le vecteur d'attaque. Defaults to None.

    Returns:
        numpy.ndarray, numpy.ndarray: Les données et les étiquettes.
    """

    if attack_vector is not None:
        traces = df["trace"].apply(lambda x: x.split())
        X = effectuer_transformation_attaque(traces, attack_vector)
    else:
        traces = df["trace"].apply(lambda x: list(map(int, x.split())))
        X = transformer_donnees(traces)

    y = np.array(df.iloc[:, 2:])
    return X, y


def transformer_donnees(traces):
    """
    Transforme les données en un tableau numpy.

    Args:
        traces (list): La liste des traces.

    Returns:
        numpy.ndarray: Les données transformées.

    """
    donnees_transformees = []

    for trace in traces:
        temp_liste = [0] * 340

        for element in trace:
            if element > 340:
                continue
            temp_liste[element - 1] += 1

        donnees_transformees.append(temp_liste)

    return np.array(donnees_transformees)


def effectuer_transformation_attaque(traces, vecteur_attaque):
    """
    Transforme les données en un tableau numpy.
    Transformation des données pour l'entrainement du classifieur d'attaque (Type d'attaque)


    Args:
        traces (list): La liste des traces.
        vecteur_attaque (list): Le vecteur d'attaque.

    Returns:
        numpy.ndarray: Les données transformées.
    """
    resultats = []

    for trace in traces:
        temp_matrice = [0]*len(vecteur_attaque) + [350]

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
    Prépare le vecteur d'attaque pour l'entrainement du classifieur d'attaque (Type d'attaque)

    Args:
        traces (list): La liste des traces.

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

def train_binary(attack_data,train_data,validation_data):
    """
    Entraîne le classifieur binaire.

    Args:
        attack_data (pandas.DataFrame): Les données d'attaque.
        train_data (pandas.DataFrame): Les données d'entrainement.
        validation_data (pandas.DataFrame): Les données de validation.

    Returns:
        sklearn.linear_model.LogisticRegression: Le classifieur binaire.
    """

    print('-' * 60)
    print("Entraînement du classifieur binaire en cours")
        
    X, y = get_X_y(train_data)

    y = y.sum(axis=1)
    X_attack, y_attack = get_X_y(attack_data)
    X_val, y_val = get_X_y(validation_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # testGridSearch(X_train, y_train)

    binary_classifier.fit(X_train, y_train)

    pred_val = binary_classifier.predict(X_val)
    pred_a = binary_classifier.predict(X_attack)
    y_pred = binary_classifier.predict(X_test)
    
    print("Précision globale du classifieur binaire sur les données de test :", accuracy_score(y_test, y_pred))
    print("Précision du classifieur binaire sur les attaques uniquement :", accuracy_score([1 for _ in range(len(pred_a))], pred_a))
    print("Précision du classifieur binaire sur la validation uniquement :", accuracy_score([0 for _ in range(len(pred_val))], pred_val))
    print('-' * 60)



def train_attack(attack_vector,attack_data):
    """
    Entraîne le classifieur d'attaque.

    Args:
        attack_vector (list): Le vecteur d'attaque.
        attack_data (pandas.DataFrame): Les données d'attaque.

    Returns:
        sklearn.linear_model.LogisticRegression: Le classifieur d'attaque.
    """

    print("\nEntraînement de la détection d'attaque en cours")
    traces = attack_data["trace"].apply(lambda x: x.split())

    X, y = get_X_y(attack_data,attack_vector)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # testGridSearch(X_train, y_train)
    y_train = y_train.argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    
    attack_classifier.fit(X_train, y_train)
    
    y_pred = attack_classifier.predict(X_test)
    
    print("\nPrécision de la détection d'attaque :", accuracy_score(y_test, y_pred))

    print('-' * 60)


def predict(trace, attack_vector, threshold=0.13):
    """
    Prédit si une attaque a été effectuée ou non et le type d'attaque.

    Args:
        trace (list): La trace à analyser.
        attack_vector (list): Le vecteur d'attaque.
        threshold (float, optional): Le seuil de prédiction. Defaults to 0.13.

    Returns:
        int: Le type d'attaque.
    """

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

def testIAWithSomeAttack(attack_vector):
    """
    Teste l'IA avec des attaques.

    Args:
        attack_vector (list): Le vecteur d'attaque.
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
            pred = PREDICTIONS.get(predict(trace,attack_vector), "-")
            print("VERDICT:", pred)
            print('-' * 60)

    except Exception as e:
        print(e)
        print()
    

if __name__ == "__main__":
    

    train_data, validation_data, attack_data = load_data(TRAIN_DATA_PATH, VALIDATION_DATA_PATH)

    print('-' * 60)
    print("Training model")
    traces = attack_data["trace"].apply(lambda x: x.split())
    attack_vector = preparer_vecteur(traces)
    train_binary(attack_data,train_data,validation_data)
    train_attack(attack_vector,attack_data)
    print("Training complete")

    print('-' * 60)
    print("Testing model")
    testIAWithSomeAttack(attack_vector)
    print("Testing complete")
    print('-' * 60)

