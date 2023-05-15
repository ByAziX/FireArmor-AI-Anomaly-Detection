import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report
from sklearn.base import clone
import pandas as pd
import csv
import random
import psutil
import time
import sys

def train_user_model(X, y, base_model):
    clf = clone(base_model)
    clf.fit(X, y)
    return clf

#draw the  process bar
def drawProgressBar(percent, barLen = 20):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()



def create_Random_data():

    n_samples = 100  # Remplacez par le nombre d'échantillons souhaité
    n_features = 2  # Remplacez par le nombre de caractéristiques souhaité

    data = {
        'pid': [random.randint(1, 10000) for _ in range(n_samples)],
        'nom_processus': [f'processus_{random.randint(1, 1000)}' for _ in range(n_samples)],
        'temps_creation': [random.uniform(0, 100) for _ in range(n_samples)],
        'duration': [random.uniform(0, 10000) for _ in range(n_samples)],
        'frequency': [random.uniform(0, 100) for _ in range(n_samples)],
    }

    label = np.random.randint(0, 2, n_samples)

    with open("label.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            
            # Ajouter l'en-tête au fichier CSV
            csv_writer.writerow(["etiquette"])

            # Ajouter les étiquettes au fichier CSV
            csv_writer.writerows(label.reshape(-1, 1))

    # Créer un DataFrame avec les données générées
    df = pd.DataFrame(data)

    # Sauvegarder les données dans un fichier CSV
    df.to_csv('donnees_system_calls.csv', index=False)




# etape 2 : pretraitement des données
def pretraitement_donnees():
    # Charger les données
    df = pd.read_csv('donnees_system_calls.csv')

    # Supprimer les colonnes inutiles
    df = df.drop(['pid', 'nom_processus'], axis=1)

    # supprimer temps_creation
    df = df.drop(['temps_creation'], axis=1)

    # Normaliser les données

    # Sauvegarder les données dans un fichier CSV
    df.to_csv('donnees_system_calls.csv', index=False)



    

def load_data():
    # Lire les données des fichiers CSV
    donnees_system_calls = pd.read_csv("donnees_system_calls.csv")
    label = pd.read_csv("label.csv")

    # Convertir les DataFrames en tableaux numpy
    X = np.array(donnees_system_calls)
    y = np.array(label).flatten()  # Utilisez `flatten()` pour convertir le tableau en 1D

    return X, y

# Créez des données fictif et des étiquettes
create_Random_data()
pretraitement_donnees()




# Chargez vos données (X) et étiquettes (y) ici
# X doit être un tableau numpy 2D avec les caractéristiques des system calls
# y doit être un tableau numpy 1D avec les étiquettes (0 pour normal, 1 pour anormal)
X, y = load_data()

# Divisez les données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Divisez les données d'apprentissage en sous-ensembles pour chaque utilisateur
n_users = 5  # Remplacez 5 par le nombre d'utilisateurs souhaité
n_samples = len(X_train)
samples_per_user = max(1, n_samples // n_users)
indices = np.random.permutation(n_samples)

X_train_splits = [X_train[indices[i:i+samples_per_user]] for i in range(0, n_samples, samples_per_user)]
y_train_splits = [y_train[indices[i:i+samples_per_user]] for i in range(0, n_samples, samples_per_user)]

# Réduisez le nombre d'utilisateurs si nécessaire pour éviter les sous-ensembles vides
n_users = len(X_train_splits)

# Créez un modèle de base de forêt aléatoire
base_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînez un modèle pour chaque utilisateur
user_models = [train_user_model(X_train_splits[i], y_train_splits[i], base_model) for i in range(n_users)]

# Combine les modèles des utilisateurs pour créer un modèle global en utilisant VotingClassifier
global_model = VotingClassifier(estimators=[(f'user_{i}', user_models[i]) for i in range(n_users)], voting='hard')
global_model.fit(X_train, y_train)  # Fit sur l'ensemble d'apprentissage complet pour déterminer les poids

# Testez le modèle global sur l'ensemble de test
y_pred = global_model.predict(X_test)

# Affichez les performances du modèle global
print(classification_report(y_test, y_pred))








# test real data

# Mettre le PID de ton preccesus qui tourne sur ton ordinateur ici
"""desired_pid = 86464

try:
    
    # Créer un objet Process avec le PID souhaité
    pid = psutil.Process(desired_pid)

    process_name = pid.name()

    create_time = pid.create_time()

    create_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(create_time))

    duration = time.time() - create_time

    pid.cpu_percent()

    time.sleep(1)

    cpu_percent = pid.cpu_percent()


    data = np.array([[duration, cpu_percent]])
    prediction = global_model.predict(data)

    if prediction[0] == 0:
        print("Le sys call est normal.")
    else:
        print("Le sys call est anormal.")
    
    print(f"Process name: {process_name}")
    print(f"Process creation time: {create_time_str}")
    print(f"Process duration: {duration} seconds")
    print(f"Process CPU usage: {cpu_percent}%")
    print(f"preidction: {prediction} ")


except psutil.NoSuchProcess:
    print(f"Process with PID {desired_pid} does not exist.")

    

    



"""





try:

    # Obtenez la liste de tous les processus en cours d'exécution
    process_list = psutil.process_iter()
    index = 0
    nombre_processus = 100

    # Parcourez chaque processus et affichez son nom et son ID
    for processus in process_list:
        index += 1
        if index > nombre_processus:
            break
        percent = (index+0.0)/nombre_processus
        drawProgressBar(percent)

        process_name = processus.name()

        create_time = processus.create_time()

        create_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(create_time))

        duration = time.time() - create_time

        processus.cpu_percent()

        time.sleep(1)

        cpu_percent = processus.cpu_percent()


        data = np.array([[duration, cpu_percent]])

         # Créer un DataFrame avec les données générées
        df = pd.DataFrame(data)
        
        # Sauvegarder les données dans un fichier CSV
        # ajouter les données dans le fichier a la ligne suivante
        df.to_csv('donnees_system_calls.csv', index=False, mode='a', header=False)
    
        prediction = global_model.predict(data)

        print(f"\nProcess PID: {processus.pid}")
        print(f"Process name: {process_name}")
        print(f"Process creation time: {create_time_str}")
        print(f"Process duration: {duration} seconds")
        print(f"Process CPU usage: {cpu_percent}%")


        # add to the file label
        with open('label.csv', 'a') as f:
            f.write(str(prediction[0]) + '\n')

        if prediction[0] == 0:
            print("Le sys call est normal.")
        else:
            print("Le sys call est anormal.")
    

except psutil.NoSuchProcess:
    print(f"Process with PID {desired_pid} does not exist.")



# Étape 6: Déploiement et mise à jour
# (La mise en œuvre de cette étape dépend de l'environnement et des exigences spécifiques du projet)


# Étape 7: Surveillance et maintenance





# https://towardsdatascience.com/workflow-of-a-machine-learning-project-ec1dba419b94 