
import numpy as np
import io_helper


random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function


def dropin(X, y):
    """
    The name suggests the inverse of dropout, i.e. adding more samples. See Data Augmentation section at
    http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
    :param X: Each row is a training sequence
    :param y: Tne target we train and will later predict
    :return: new augmented X, y
    """
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_hat = []
    y_hat = []
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0, random_data_dup)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat)



def preprocess():

    arrayfile = "/home/hugo/ISEN/Cours/FireArmor/FireArmor-AI-Anomaly-Detection/array_test.pickle"
   
    array = io_helper.loadfrompickle(arrayfile)
    print(len(array))
    array = array.reshape(-1, 1)
    x_train = array[:,:-1]
    y_train = array[:,-1]

    print(len(x_train))
    print(len(y_train))


    print ("The train data size is that ")
    print (x_train.shape)
    print (y_train.shape)
    return (x_train,y_train)


if __name__ =="__main__":

    preprocess()