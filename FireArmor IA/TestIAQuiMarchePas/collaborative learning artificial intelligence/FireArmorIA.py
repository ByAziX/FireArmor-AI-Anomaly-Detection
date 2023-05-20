import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.base import clone
import pandas as pd
import csv
import random
import psutil
import time
import sys


# import python AI.py 


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
    

def count_nb_syscall_normal_anormal(predictions):
    nb_syscall_normal = 0
    nb_syscall_anormal = 0
    for prediction in predictions:
        if prediction == 0:
            nb_syscall_normal += 1
        else:
            nb_syscall_anormal += 1
    print(f"Nombre de sys call normal: {nb_syscall_normal}")
    print(f"Nombre de sys call anormal: {nb_syscall_anormal}")
    





    

def load_data():
    # Lire les données des fichiers CSV
    donnees_system_calls = pd.read_csv("donnees_system_calls.csv")
    label = pd.read_csv("label.csv")

    # Convertir les DataFrames en tableaux numpy
    X = np.array(donnees_system_calls)
    y = np.array(label).flatten()  # Utilisez `flatten()` pour convertir le tableau en 1D

    return X, y

# Chargez vos données (X) et étiquettes (y) ici
# X doit être un tableau numpy 2D avec les caractéristiques des system calls
# y doit être un tableau numpy 1D avec les étiquettes (0 pour normal, 1 pour anormal)
X, y = load_data()

# Divisez les données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créez un modèle de base de forêt aléatoire
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînez le modèle
model.fit(X_train, y_train)

# Testez le modèle sur l'ensemble de test
y_pred = model.predict(X_test)

# Affichez les performances du modèle
print(classification_report(y_test, y_pred))





# test real data from your computer

try:

    # Obtenez la liste de tous les processus en cours d'exécution
    process_list = psutil.process_iter()
    index = 0
    nombre_processus = 10000
    predictions = []

    # Parcourez chaque processus et affichez son nom et son ID
    for processus in process_list:
        index += 1
        if index > nombre_processus:
            break
        percent = index / nombre_processus
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
    
        # prediction = global_model.predict(data)
        prediction = model.predict(data)
        predictions.append(prediction[0])

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

        count_nb_syscall_normal_anormal(predictions)

except psutil.NoSuchProcess:
    print(f"Process with PID {desired_pid} does not exist.")



# Étape 6: Déploiement et mise à jour
# (La mise en œuvre de cette étape dépend de l'environnement et des exigences spécifiques du projet)


# Étape 7: Surveillance et maintenance







# test real data from your computer
# choose a processus and test it


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
