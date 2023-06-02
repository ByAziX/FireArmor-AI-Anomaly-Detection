# FireArmor-AI-Anomaly-Detection

## Prérequis
  1. Python 3.7 ou supérieur
  2. Bibliothèques Python: numpy, pandas, scikit-learn, matplotlib, seaborn
  3. Dataset ADFA-LD (disponible dans le github)


## Introduction
The ADFA-LD (Advanced Data Mining and Its Applications Lab Dataset) is a dataset containing information on Linux system calls. It has been designed for anomaly detection research and is often used to test machine learning models.

In this tutorial, we'll use the Random Forest algorithm to create a model that detects anomalies in system calls.

## Steps

### 1. Importing libraries and modules

Install the required libraries (numpy, pandas, scikit-learn, matplotlib, seaborn) before running the code.
### 2. Preparing ADFA-LD DataSet data

To launch the python scripts, go to the github repository source, as the paths are in relative path.

Before you start, make sure you've downloaded the ADFA-LD dataset. The files required for this model are train.csv and validation_data.csv. Place these files in the appropriate directory.

If you don't have the train.csv file, you can create it using the `InputData.py` script. This script is responsible for creating the `train.csv` file from the raw trace files. Make sure you have the necessary raw trace files and run the script to generate the train.csv file.

The train.csv file contains the training data required to form the anomaly detection model. Once you've created the train.csv file, you can place it in the appropriate directory and run the rest of the code to create and evaluate the model.
### 3. Code execution

You can run the Python code `RandomForestClassifierWithPattern` to create and evaluate the anomaly detection model based on the random forest algorithm. Make sure you have correctly configured the Python environment and have the required data files in the appropriate directory.

Please refer to the comments in the code for detailed information on each step and function.
