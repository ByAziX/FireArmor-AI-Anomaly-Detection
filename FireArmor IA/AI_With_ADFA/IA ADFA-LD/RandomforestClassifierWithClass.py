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

class MyClassifier():
    def __init__(self, train_data_path, validation_data_path, rs=42, logging=True):
        self.binary_classifier = RandomForestClassifier(n_estimators=100,random_state=rs)
        self.attack_classifier = RandomForestClassifier(n_estimators=100,random_state=rs)
        
        self.train_data_path = train_data_path
        self.validation_data_path = validation_data_path
        self.attack_vector = None
        self.rs = rs
        self.metrics = {}
        self.logging = logging
        
    def load_data(self):
        try:
            self.train_data = pd.read_csv(self.train_data_path)
            self.validation_data = pd.read_csv(self.validation_data_path)
            self.attack_data = self.train_data[self.train_data.iloc[:, 2:].sum(axis=1) == 1]
        except Exception as e:
            print(f"Erreur lors du chargement des données : {e}")
            return None

    def get_X_y(self, df):
        trace = df["trace"].apply(lambda x: list(map(int, x.split())))
        X = self.transform_X(trace)
        y = np.array(df.iloc[:, 2:])
        return X, y

    def get_attack_X_y(self, df):
        traces = df["trace"].apply(lambda x: x.split())
        X = self.transform_attack(traces)
        y = np.array(df.iloc[:, 2:])
        return X, y

    def transform_X(self, traces):
        X = []
        for arr in traces:
            temp = [0] * 340
            for i in arr:
                if i > 340:
                    continue
                temp[i-1] += 1
            X.append(temp)
        return np.array(X)
    
    def prepare_vector(self, X):
        d = {}
        features = set()
        ind = 0
        for arr in X:
            for size in range(2, 6):
                for i in range(0, len(arr) - size):
                    sub = arr[i: i+size]
                    key = "-".join(sub)
                    if key in features:
                        if key not in d:
                            d[key] = ind
                            ind += 1
                    else:
                        features.add(key)                               
        return d

    def transform_attack(self, X):
        res = []
        for arr in X:
            temp = [0]*len(self.attack_vector) + [350]
            for size in range(2, 6):
                for i in range(0, len(arr) - size):
                    sub = arr[i: i+size]
                    key = "-".join(map(str, sub))
                    if key in self.attack_vector:
                        temp[self.attack_vector[key]] += 1
            temp = np.array(temp, dtype="float64")
            res.append(temp)
        
        return np.array(res)
        
    def adfa_train(self, callback=None):
        self.binary_train()
        self.attack_train()
        if callback:
            callback()

    def attack_train(self):
        if self.logging:
            print("\nEntraînement de la détection d'attaque en cours")
        traces = self.attack_data["trace"].apply(lambda x: x.split())
        self.attack_vector = self.prepare_vector(traces)

        X, y = self.get_attack_X_y(self.attack_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=self.rs, stratify=y.argmax(axis=1))

        self.metrics["y_test"] = y_test.T
        y_train = y_train.argmax(axis=1)
        y_test = y_test.argmax(axis=1)
        
        self.attack_classifier.fit(X_train, y_train)

        self.metrics["probas"] = self.attack_classifier.predict_proba(X_test).T
        self.metrics["map_x"] = X_test
        self.metrics["map_y"] = y_test
        
        y_pred = self.attack_classifier.predict(X_test)
        if self.logging:
            print("\nPrécision de la détection d'attaque :", accuracy_score(y_test, y_pred))

        self.metrics["multilabel_accuracy"] = accuracy_score(y_test, y_pred)


        
    def binary_train(self):
        if self.logging:
            print("Entraînement du classifieur binaire en cours")
            
        X, y = self.get_X_y(self.train_data)
        y = y.sum(axis=1)
        X_attack, y_attack = self.get_X_y(self.attack_data)
        X_val, y_val = self.get_X_y(self.validation_data)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=self.rs)
        self.binary_classifier.fit(X_train, y_train)

        pred_val = self.binary_classifier.predict(X_val)
        pred_a = self.binary_classifier.predict(X_attack)
        y_pred = self.binary_classifier.predict(X_test)
        
        pred_val = pred_val
        pred_a = pred_a
        y_pred = y_pred
        
        if self.logging:
            print("Précision globale du classifieur binaire sur les données de test :", accuracy_score(y_test, y_pred))
            print("Précision du classifieur binaire sur les attaques uniquement :", accuracy_score([1 for _ in range(len(pred_a))], pred_a))
            print("Précision du classifieur binaire sur la validation uniquement :", accuracy_score([0 for _ in range(len(pred_val))], pred_val))

        self.metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).ravel()
        self.metrics["binary_accuracy"] = accuracy_score(y_test, y_pred)



    def predict(self, X, predict_one=False):
        if isinstance(X, str):
            X = np.array([list(map(int, X.split()))])
            
        X_bin = self.transform_X(X)
        bp = self.binary_predict(X_bin)
        
        if predict_one and not bp[0]:
            if self.logging:
                print("No attack")
            return 0
        print("Attack")
        X_atk = self.transform_attack(X)
        attack_predict = self.attack_predict(X_atk) + 1

        if predict_one:
            return attack_predict[0]
        return attack_predict

    def binary_predict(self, X):
        return self.binary_classifier.predict(X)

    def attack_predict(self, X):
        return self.attack_classifier.predict(X)


     # Le reste du code reste inchangé...
if __name__ == "__main__":
    
    
    train_data_path = "/home/hugo/ISEN/Cours/FireArmor/FireArmor-AI-Anomaly-Detection/train.csv"
    # train_data_path = "FireArmor IA/AI_With_ADFA/IA ADFA-LD/train_data.csv"
    validation_data_path = "FireArmor IA/AI_With_ADFA/IA ADFA-LD/validation_data.csv"

    mc = MyClassifier(train_data_path, validation_data_path)
    mc.load_data()
    print("Training model")
    mc.adfa_train()
    print("Training complete")

    try:
            files = {}
            file_directory = "FireArmor IA/AI_With_ADFA/IA ADFA-LD/tests/"
            files = InputData.readfilesfromAdir(file_directory)
        
            print(f"Loading ...")

            for filename in files:
                with open(filename) as fs:
                    trace = fs.read().strip()

                pred = PREDICTIONS.get(mc.predict(trace, predict_one=True), "-")
                print("VERDICT:", pred)

    except Exception as e:
        print(e)
        print()
