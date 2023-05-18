# FireArmor-AI-Anomaly-Detection

## Prérequis
  1. Python 3.7 ou supérieur
  2. Bibliothèques Python: numpy, pandas, scikit-learn, matplotlib, seaborn
  3. Dataset ADFA-LD (disponible sur le site officiel)


## Introduction
Le dataset ADFA-LD (Advanced Data Mining and Its Applications Lab Dataset) est un ensemble de données qui contient des informations sur les appels système Linux. Il a été conçu pour la recherche sur la détection d'anomalies et est souvent utilisé pour tester des modèles d'apprentissage automatique.

Dans ce tutoriel, nous utiliserons l'algorithme de forêt aléatoire (Random Forest) pour créer un modèle qui détecte les anomalies dans les systèmes d'appels.

## Étapes

### 1. Préparation des données du DataSet ADFA-LD











```
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
```
