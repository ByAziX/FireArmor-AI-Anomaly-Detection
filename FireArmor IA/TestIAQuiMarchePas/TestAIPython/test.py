import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from multiprocessing import Process, Pipe

def apprendre_et_communiquer(index, conn):
    # Générer des données aléatoires pour simuler des appels système
    np.random.seed(index)
    normal_syscalls = np.random.normal(0, 1, (1000, 10))
    anormal_syscalls = np.random.normal(2, 1, (100, 10))

    # Créer des étiquettes pour les appels système (0 = normal, 1 = anormal)
    y_normal = np.zeros(normal_syscalls.shape[0])
    y_anormal = np.ones(anormal_syscalls.shape[0])

    # Concaténer les données et les étiquettes
    X = np.vstack((normal_syscalls, anormal_syscalls))
    y = np.hstack((y_normal, y_anormal))

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=index)

    # Créer et entraîner un classificateur
    clf = RandomForestClassifier(random_state=index)
    clf.fit(X_train, y_train)

    # Évaluer le modèle
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    print(f"Agent {index} - Précision: {accuracy:.2f}")
    print(f"Agent {index} - Matrice de confusion:")
    print(conf_mat)

    # Envoyer les prédictions à l'autre agent
    conn.send((index, y_pred))
    conn.close()

if __name__ == '__main__':
    # Créer les connections pour la communication entre les agents
    parent_conn1, child_conn1 = Pipe()
    parent_conn2, child_conn2 = Pipe()

    # Créer et lancer les agents
    agent1 = Process(target=apprendre_et_communiquer, args=(1, child_conn1))
    agent2 = Process(target=apprendre_et_communiquer, args=(2, child_conn2))
    agent1.start()
    agent2.start()

    # Attendre les résultats des agents
    index1, y_pred1 = parent_conn1.recv()
    index2, y_pred2 = parent_conn2.recv()

    # Comparer les prédictions des agents
    accord = np.sum(y_pred1 == y_pred2) / len(y_pred1)
    print(f"Accord entre les agents {index1} et {index2}: {accord:.2f}")

    # Terminer les agents
    agent1.join()
    agent2.join()
