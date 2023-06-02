import os
import sys
import numpy as np


def readfilesfromAdir(dataset):
    """ read all the files from a directory

    Args:
        dataset (str): The path of the directory containing the files

    Returns:
        list: The list of the files in the directory
    """

    files = os.listdir(dataset)
    files_absolute_paths = []
    for i in files:
        files_absolute_paths.append(dataset+str(i))
    return files_absolute_paths

#this is used to read a char sequence from
def readCharsFromFile(file):
    channel_values = open(file).read().split()
    return channel_values


def get_attack_subdir(path):
    subdirectories = os.listdir(path)
    for i in range(0,len(subdirectories)):
        subdirectories[i] = path + subdirectories[i] + "/"

    return (subdirectories)


def create_file(file, dataTrain, dataAttack, sub_dir_attack):
    """ Create a file containing the data of the dataset

    Args:
        file (str): The path of the file to create
        dataTrain (list): The list of the files in the training dataset
        dataAttack (list): The list of the files in the attack dataset
        sub_dir_attack (list): The list of the subdirectories in the attack dataset
    """

    with open(file, 'w') as f:
        label = ['trace', 'Adduser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']
        index = []
        count=0
        # write the label with virgule separator 
        f.write(",%s\n" % ','.join(map(str, label)))
        for dataset in [dataTrain, dataAttack]:
            count+=1
            for fichier in dataset:
                fileContent = readCharsFromFile(fichier)
                                    # add index increment by 1 and write the sequence with virgule separator 
                index.append(index[-1]+1 if index else 0)
                f.write("%s," % index[-1])
                f.write("%s" % ' '.join(map(str, fileContent)))
                if count == 1: 
                    results = [0] * len(label[1:])
                    f.write(",%s\n" % ','.join(map(str, results)))
                        
                    # get name of the file sub_dir_attack and if the label = sub_dir_attack then 1 else 0
                    
                subdomain = ''
                results = [0] * len(label[1:])
                if count == 2:
                    # get the subdomain name
                    subdomain = fichier.split('/')[-2]
                    subdomain = subdomain.split('_')[:-1]
                    subdomain = '_'.join(subdomain)
                    # check if the subdomain is in the label
                    if subdomain in label[1:]:
                        # get the index of the subdomain in the label
                        index_subdomain = label.index(subdomain)
                        #print(subdomain,index_subdomain)
                        # set the value to 1
                        if index_subdomain:
                            results[index_subdomain-1] = 1
                            countLabel[index_subdomain-1] += 1
                            f.write(",%s\n" % ','.join(map(str, results)))


def tableauDataSet(countLabel,dataTrain,dataAttack):
    """ Create a table containing the number of files for each label

    Args:
        countLabel (list): The list of the number of files for each label
        dataTrain (list): The list of the files in the training dataset
        dataAttack (list): The list of the files in the attack dataset
    """

    label = ['Adduser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']
    print('-' * 60)
    print('{:<30s}{:<15s}'.format('Label', 'Count'))
    print('-' * 60)
    for i in range(0,len(label)):
        if countLabel[i] > 0:
            print('{:<30s}{:<15d}'.format(label[i],countLabel[i]))
    print('-' * 60)

    print('{:<30s}{:<15d}'.format('Nombre de fichiers d\'attaque : ', len(dataAttack)))
    print('{:<30s}{:<15d}'.format('Nombre de fichiers de train : ', len(dataTrain)))
    total_files = len(dataAttack) + len(dataTrain)
    print('{:<30s}{:<15d}'.format('Nombre de fichiers total : ', total_files))
    print('-' * 60)




if __name__ == "__main__":
    directory_train = "FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/Training_Data_Master/"
    directory_validation = "FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/Validation_Data_Master/"
    directory_attack = "FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/Attack_Data_Master/"
    train_files = readfilesfromAdir(directory_train)
    sub_dir_attack = get_attack_subdir(directory_attack)
    print("train.csv is created")
    countLabel = [0] * 6
    # récupère les fichiers de chaque sous-dossier
    attack_files = []
    for sub_dir in sub_dir_attack:
        attack_files.extend(readfilesfromAdir(sub_dir))

    create_file("FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/train.csv", train_files, attack_files, sub_dir_attack)

    tableauDataSet(countLabel,train_files,attack_files)