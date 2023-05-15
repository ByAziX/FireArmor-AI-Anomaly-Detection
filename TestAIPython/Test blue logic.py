import os
import psutil
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import psutil
import time
import sys
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

# étape 1 : collecte des données
def test_donnees():
    data = {
        'pid': [random.randint(1, 10000) for _ in range(10000)],
        'nom_processus': [f'processus_{random.randint(1, 1000)}' for _ in range(10000)],
        'temps_creation': [random.uniform(0, 100) for _ in range(10000)],
        'duration': [random.uniform(0, 100) for _ in range(10000)],
        'frequency': [random.uniform(0, 100) for _ in range(10000)],
        'label': [random.choice([0, 1]) for _ in range(10000)]
    }

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

# Étape 3: Construction du modèle de logique floue

# Définir des variables linguistiques
duration = ctrl.Antecedent(np.arange(0, 101, 1), 'duration')
frequency = ctrl.Antecedent(np.arange(0, 101, 1), 'frequency')
anomaly = ctrl.Consequent(np.arange(0, 101, 1), 'anomaly')

# Créer des ensembles flous
duration['low'] = fuzz.trimf(duration.universe, [0, 0, 50])
duration['medium'] = fuzz.trimf(duration.universe, [0, 50, 100])
duration['high'] = fuzz.trimf(duration.universe, [50, 100, 100])

frequency['low'] = fuzz.trimf(frequency.universe, [0, 0, 50])
frequency['medium'] = fuzz.trimf(frequency.universe, [0, 50, 100])
frequency['high'] = fuzz.trimf(frequency.universe, [50, 100, 100])

anomaly['low'] = fuzz.trimf(anomaly.universe, [0, 0, 50])
anomaly['medium'] = fuzz.trimf(anomaly.universe, [0, 50, 100])
anomaly['high'] = fuzz.trimf(anomaly.universe, [50, 100, 100])

# Établir des règles floues
rule1 = ctrl.Rule(duration['low'] & frequency['low'], anomaly['low'])
rule2 = ctrl.Rule(duration['medium'] & frequency['low'], anomaly['medium'])
rule3 = ctrl.Rule(duration['high'] & frequency['low'], anomaly['high'])
rule4 = ctrl.Rule(duration['low'] & frequency['medium'], anomaly['low'])
rule5 = ctrl.Rule(duration['medium'] & frequency['medium'], anomaly['medium'])
rule6 = ctrl.Rule(duration['high'] & frequency['medium'], anomaly['high'])
rule7 = ctrl.Rule(duration['low'] & frequency['high'], anomaly['low'])
rule8 = ctrl.Rule(duration['medium'] & frequency['high'], anomaly['medium'])
rule9 = ctrl.Rule(duration['high'] & frequency['high'], anomaly['high'])

# Créer un système de contrôle flou
anomaly_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
anomaly_detection = ctrl.ControlSystemSimulation(anomaly_ctrl)

# Étape 4: Implémentation de l'hystérésis
def hysteresis(value, lower_threshold, upper_threshold, current_state):
    if current_state:
        if value < lower_threshold:
            return False
    else:
        if value > upper_threshold:
            return True
    return current_state

current_state = False
lower_threshold = 30
upper_threshold = 70

# Étape 5: Entraînement et évaluation du modèle
# Générer des données aléatoires pour l'entraînement et l'évaluation
"""data = [{'duration': random.uniform(0, 100), 'frequency': random.uniform(0, 100), 'label': random.choice([0, 1])} for _ in range(1000)]
df = pd.DataFrame(data)
df.to_csv('donnees_fictives.csv', index=False)
"""

# Générer les données
test_donnees()
pretraitement_donnees()

# Charger les données
df = pd.read_csv('donnees_system_calls.csv')

# Entraîner et évaluer le modèle
predictions = []
for index, row in df.iterrows():
    anomaly_detection.input['duration'] = row['duration']
    anomaly_detection.input['frequency'] = row['frequency']
    anomaly_detection.compute()
    prediction = 1 if hysteresis(anomaly_detection.output['anomaly'], lower_threshold, upper_threshold, current_state) else 0
    predictions.append(prediction)
    current_state = prediction

    percent = (index+0.0)/len(df)
    drawProgressBar(percent)

accuracy = accuracy_score(df['label'], predictions)
precision = precision_score(df['label'], predictions)
recall = recall_score(df['label'], predictions)
f1 = f1_score(df['label'], predictions)

# write prediction to csv
df['etiquettes'] = predictions
df.to_csv('etiquettes.csv', index=False)


print("\n-------------------------------------------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("-------------------------------------------")

# test real data


# Mettre le PID de ton preccesus qui tourne sur ton ordinateur ici
desired_pid = 30863

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

    anomaly_detection.input['duration'] = duration
    anomaly_detection.input['frequency'] = cpu_percent
    anomaly_detection.compute()
    print("Valeur Pour l'hysteresis :",anomaly_detection.output['anomaly'])
    prediction = 1 if hysteresis(anomaly_detection.output['anomaly'], lower_threshold, upper_threshold, current_state) else 0

    print(f"Process name: {process_name}")
    print(f"Process creation time: {create_time_str}")
    print(f"Process duration: {duration} seconds")
    print(f"Process CPU usage: {cpu_percent}%")
    print(f"preidction: {prediction} ")


except psutil.NoSuchProcess:
    print(f"Process with PID {desired_pid} does not exist.")



# Étape 6: Déploiement et mise à jour
# (La mise en œuvre de cette étape dépend de l'environnement et des exigences spécifiques du projet)


# Étape 7: Surveillance et maintenance





# https://towardsdatascience.com/workflow-of-a-machine-learning-project-ec1dba419b94