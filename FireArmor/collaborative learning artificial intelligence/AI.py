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
import csv
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


# Étape 3: Construction du modèle de logique floue

# Définir des variables linguistiques
def fuzzLogique():
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
    return anomaly_detection

# Étape 4: Implémentation de l'hystérésis
def hysteresis(value, lower_threshold, upper_threshold, current_state):
    if current_state:
        if value < lower_threshold:
            return False
    else:
        if value > upper_threshold:
            return True
    return current_state



# Étape 5: Entraînement et évaluation du modèle
# Générer des données aléatoires pour l'entraînement et l'évaluation

def run():
    current_state = False
    lower_threshold = 30
    upper_threshold = 70

    # Charger les données
    df = pd.read_csv('donnees_system_calls.csv')
    anomaly_detection = fuzzLogique()
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

        with open("etiquettes.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            
            # Ajouter l'en-tête au fichier CSV
            csv_writer.writerow(["etiquette"])

            # Ajouter les étiquettes au fichier CSV
            csv_writer.writerows(zip(predictions))



    


