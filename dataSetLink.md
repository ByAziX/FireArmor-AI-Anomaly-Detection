https://web.archive.org/web/20200925174405/https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-IDS-Datasets/




https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset


https://csr.lanl.gov/data/cyber1/


https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset






https://github.com/darkenezy/ADFA-LD-Classifier/blob/master/venv/Scripts/ml.py



https://search.brave.com/search?q=Unix+auditd+trace+syscall&source=web





Réseau:

https://www.netresec.com/?page=PcapFiles
https://www.westpoint.edu/centers-and-research/cyber-research-center/data-sets









Various machine learning algorithms can be used for intrusion detection, including:

Gaussian Naive Bayes
Decision Tree
Random Forest
Support Vector Machine
Logistic Regression
k-Nearest Neighbors


https://ieeexplore.ieee.org/document/6555301

https://www.researchgate.net/post/Were-can-I-get-a-labelled-version-of-the-ADFA-LD-dataset-for-HIDS-evaluation

Ok for any other person requesting for the labels or how to extract them. Firstly, you can still use the unlabeled form of the dataset for training your models or for your algorithms as all you need is basically the class (i.e the attack type). But if you really want to know what each field in the dataset means, then you need to find the .h file that I talked about in this thread. The numbers you see in the dataset are syscalls that were generated or called when Creech (the person who generated the dataset) ran the attacks.  So if you really need a labeled version of this dataset, you need to look at the .h file for the integer number for the dataset and see what name is given to such syscall, then you can say, count the number of system calls with this name e.g if [110 111 110 665 888 999 555 110 110 110 110] is a line in one of the files in the ADFA-LD, and you found out that "110" means "syscall X" (i.e you found out from the .h file), you can say decide to run a little code to extract something like "syscall X" = 6 from the above example, which means that there was 6 "syscall X"s in that line. Based on this, you can move on to whatever. You can see that from my explanation, you might not even need to know the syscall names before being able to use the dataset. 













En matière de détection d'anomalies dans les appels système, il est généralement préférable de ne pas ajouter automatiquement les prédictions de l'IA à l'ensemble de données d'entraînement. Voici pourquoi :

Qualité des données : En détection d'anomalies, il est crucial d'avoir des étiquettes précises pour les anomalies et les comportements normaux. Les prédictions de l'IA ne sont pas des vérités absolues et peuvent donc introduire des erreurs et des incertitudes dans l'ensemble de données.

Rétroaction (Feedback Loop) : Si vous ajoutez les prédictions de l'IA à l'ensemble de données et que l'IA fait une erreur, cette erreur peut être réintroduite dans le modèle, créant une boucle de rétroaction qui pourrait renforcer les erreurs de l'IA au fil du temps.

Évaluation du modèle : L'ajout des prédictions de l'IA à l'ensemble de données peut rendre l'évaluation de la performance du modèle plus difficile, car les erreurs du modèle peuvent être réintroduites dans le modèle.

Cependant, cela ne signifie pas que les prédictions de l'IA ne peuvent pas être utilisées pour améliorer le modèle. Par exemple, vous pourriez utiliser les prédictions de l'IA pour identifier des exemples intéressants à étiqueter manuellement, ce qui pourrait vous aider à enrichir votre ensemble de données avec des exemples pertinents.

En fin de compte, la décision d'inclure ou non les prédictions de l'IA dans votre ensemble de données d'entraînement dépend de votre contexte spécifique et des exigences de qualité des données de votre tâche.


