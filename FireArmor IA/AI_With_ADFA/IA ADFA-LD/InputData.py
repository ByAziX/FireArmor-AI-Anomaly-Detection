#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ ="Ktian"
import os
import sys
import numpy as np


def readfilesfromAdir(dataset):
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


def get_all_call_sequences(directory, loggin=False):
    
    allthelist = []

    files.sort()
    for eachfile in files:
        if not eachfile.endswith("DS_Store"):
            allthelist.append(readCharsFromFile(eachfile))
            if loggin:
                print ("The file "+ str(eachfile) + " is read")
                print(allthelist[-1])
        else:
            print ("Skip the file "+ str(eachfile))

    
    if loggin:
        elements = []
        for item in allthelist:
            for key in item:
                if key not in elements:
                    elements.append(key)
        elements = map(int,elements)
        elements = sorted(elements)
        print ("The total unique elements:")
        print (elements)

        print ("The maximum number of elements:")
        print (max(elements))

        #print ("The length elements:")
        #print (len(elements))
        print (len(allthelist))

        #clean the all list data set
        _max = 0
        for i in range(0,len(allthelist)):
            _max = max(_max,len(allthelist[i]))
            allthelist[i] = map(int,allthelist[i])
            print(allthelist[i])


        print ("The maximum length of a sequence is that {}".format(_max))

    return (allthelist)


def create_file(file, dataTrain, dataAttack,filesTrain, sub_dir_attack ):
    with open(file, 'w') as f:
        label = ['trace', 'Adduser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']
        index = []
        # write the label with virgule separator 
        f.write(",%s\n" % ','.join(map(str, label)))
        for dataset in [dataTrain, dataAttack]:
            for item in dataset: 
                results = []  # Create an empty list to store the results
                # add index increment by 1 and write the sequence with virgule separator 
                index.append(index[-1]+1 if index else 0)
                f.write("%s," % index[-1])
                f.write("%s" % ' '.join(map(str, item)))
                    
                # get name of the file sub_dir_attack and if the label = sub_dir_attack then 1 else 0
                for i in range(0,len(sub_dir_attack)):
                    # get the subdomain name
                    subdomain = sub_dir_attack[i].split('/')[-2]
                    subdomain = subdomain.split('_')[0]
                    if subdomain in label[1:]:
                            results.append(1)
                    else:
                            results.append(0)

                # write the label with virgule separator
                f.write(",%s\n" % ','.join(map(str, results)))

                





if __name__ == "__main__":
    directory_train = "FireArmor IA/AI_With_ADFA/ADFA-LD/Training_Data_Master/"
    directory_validation = "FireArmor IA/AI_With_ADFA/ADFA-LD/Validation_Data_Master/"
    directory_attack ="FireArmor IA/AI_With_ADFA/ADFA-LD/Attack_Data_Master/"
    

    files = readfilesfromAdir(directory_train)
    train = get_all_call_sequences(files)
    sub_dir_attack = get_attack_subdir(directory_attack)
    attack = get_all_call_sequences(sub_dir_attack)
    create_file("train.csv", train, attack, files, sub_dir_attack)
    
    

