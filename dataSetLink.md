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