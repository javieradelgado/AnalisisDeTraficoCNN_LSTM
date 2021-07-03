from __future__ import print_function
import os
# -*- coding: utf-8 -*-

import warnings

import multiprocessing
from multiprocessing import Pool

import numpy as np # Biblioteca de funciones matematicas de alto nivel
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
import keras

import keras

warnings.catch_warnings()
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.preprocessing import sequence
from keras.models import Sequential # necesario para poder generar la red neuronal
from keras.layers import Dense, Dropout, Activation, Lambda # Tipos de capa, hacen lo siguiente:
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten, LSTM
from keras.callbacks import CSVLogger # para guardar los datos en un excel
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

"""### ESTAS SON LAS CAPAS MAS IMPORTANTES
* **Dense**: La capa que mas se utiliza, de la siguiente manera ::  output = activation(dot(input, kernel)+bias) dot representa la operacion punto a punto de todos los inputs y sus correspondientes pesos
* **Dropout**: Se usa para resolver el problema de over-fitting, se intenta eliminar el ruido en esta capa
* **Activation**: capa de activacion
* **Lambda**: sirve para transformar los datos de entrada usando una expresión o una función
* **Embedding**: esta capa sirve para convertir a vectores de tamaño fijo
* **Convolution1D**: capa donde se realiza la convolucion
* **MaxPooling1D**: capa donde se realiza la operacion de pooling
* **Flatten**: capa donde se realiza la operacion de flatten (se puede usar para poner los datos en 1 sola dimension)
"""

from keras.datasets import imdb # un dataset incluido en keras
from keras import backend as K # importas el backend (Tensorflow, Theano, etc)
import pandas as pd # pandas es una libreria extension de numpy usada para manipulacion y analisis de datos, para manipular tablas numericas y series temporales

from keras.utils.np_utils import to_categorical # sirve para convertir vectores de enteros a una matriz de clases binaria, por ejemplo:

""" a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
 
                     [1, 0, 0, 0]
                     [0, 1, 0, 0]
                     [0, 0, 1, 0]
                     [0, 0, 0, 1]
"""

import h5py # para almacenar un monton de datos numericos y dar facilidades de manipulacion para datos de Numpy

from sklearn.preprocessing import Normalizer # Para normalizar los datos
from sklearn.model_selection import train_test_split #para hacer la separacion entre datos de test y train
from sklearn.preprocessing import OneHotEncoder #para convertir los datos de entrada

def Prediccion(option, input_N, ruta):
    if (option==0):
      #Cargamos los resultados obtenidos en la etapa de train
      print("loading Infilteration weigths...")
      Infilteration.load_weights(ruta)    
      print("Infilteration weigths loaded") 
      #predecimos la naturaleza de los paquetes de test
      y_pred_infilt = Infilteration.predict_classes(input_N)
      print("\nInfilteration prediction: ", y_pred_infilt)
      archivoInfilt = open('InfiltOutput.txt','w')
      for element in y_pred_infilt:
        archivoInfilt.write((str)(element))
      archivoInfilt.close() #Cierras el archivo.

    elif(option==1):
      #Cargamos los resultados obtenidos en la etapa de train
      print("loading Bruteforce weigths...")
      Bruteforce.load_weights(ruta)
      print("Bruteforce weigths loaded")
      #predecimos la naturaleza de los paquetes de test
      y_pred_brute = Bruteforce.predict_classes(input_N)
      print("\nBruteforce prediction: ", y_pred_brute)
      archivoBrute = open('BruteOutput.txt','w')
      for element in y_pred_brute:
        archivoBrute.write((str)(element))
      archivoBrute.close() #Cierras el archivo.

    elif(option==2):
      #Cargamos los resultados obtenidos en la etapa de train
      print("loading DDoS weigths...")
      DDoS.load_weights(ruta)
      print("DDoS weigths loaded")
      #predecimos la naturaleza de los paquetes de test
      y_pred_dos = DDoS.predict_classes(input_N)
      print("\nDDoS prediction: ", y_pred_dos)
      archivoDoS = open('DoSOutput.txt','w')
      for element in y_pred_dos:
        archivoDoS.write((str)(element))
      archivoDoS.close() #Cierras el archivo.

pids = []
pids.append(os.getpid())

if __name__ == '__main__' and os.getpid()==pids[0]:

    
    #print("Father pid is ", os.getpid())
    #pids.append(os.getpid())

    print("Number of CPUs: ", multiprocessing.cpu_count())

    print("importing dataset...")
    dataset = pd.read_csv('C:/Users/jadelsot/Desktop/TFG/dataset/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv') # lectura de datos
    print("dataset imported...")

    print("dataset containing:")
    print(dataset["Label"].value_counts())

    print("dataset shape:")
    print(dataset.shape) #comprobamos el tamaño

    #Eliminamos los datos mal introducidos
    dataset = dataset.drop(dataset[dataset['Dst Port']=='Dst Port'].index)

    """## Eliminamos la columnaTimestamp"""

    #Eliminamos la columna
    dataset = dataset.drop(['Timestamp'], axis=1)

    n=0
    for column in dataset:
        column
        if column != 'Label':
            dataset[column] = dataset[column].astype(float)

    #revisamos cuantos valores puede tener la ultima columna, osea, los tipos de flujo
    Labels = dataset['Label'].unique()
    print("Dataset con Labels:")
    print(Labels) #para asignar nombres a las diferentes metricas en un futuro

    Y = dataset["Label"]

    # Replacing infinite and nan 
    dataset.replace([np.inf, -np.inf], -1, inplace=True) 
    dataset.replace([np.nan, -np.nan], -1, inplace=True)


    #ahora que hemos conseguido transformar estas columnas a numeros, podemos empezar con la red neuronal
    #necesitamos un grupo de train y otro de test para la red neuronal, los crearemos con train_test_split
    X=dataset.iloc[:, 0:78]
    X.head()

    Y = lb_make.fit_transform(Y)
    Labels = lb_make.inverse_transform(Y)
    Labels = list(lb_make.classes_)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    #volvemos a copiar su codigo:

    scaler = Normalizer().fit(X) # Normalizamos los datos
    testT = scaler.transform(X) # Asi se representan los datos

    # reshape input to be [samples, time steps, features]
    X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1)) # cambias la forma de X_train para que sea del mismo tamaño que trainX

    print("dataset shape after croping:")
    print(X_test.shape) #comprobamos el tamaño

DoS_attacks =["Bening", 'DoS attacks-SlowHTTPTest', 'DoS attacks-Hulk', 'Label', 'DoS attacks-GoldenEye', 'DoS attacks-Slowloris', 'DDOS attack-LOIC-UDP', 'DDOS attack-HOIC']
BruteForce_attacks =["Bening", 'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection','FTP-BruteForce', 'SSH-Bruteforce']
Infilteration_attacks=["Bening", "Infilteration"]

    #creamos la red neuronal
lstm_output_size = 70

Infilteration = Sequential()
Infilteration.add(Convolution1D(64, 3, activation="relu",input_shape=(78, 1), padding = 'same'))
    #######
    #   Añadimos la primera capa de Convolution1D, los diferentes parametros indican lo siguiente:
    #       64 --> numero de filtros
    #       3 --> tamaño del filtro (3,1)
    #       border_mode = "same" --> este parametro sirve para que el output sea del mismo tamaño que elinput
    #       activation = "relu" --> Tipo de funcion de activacion de neuronas que vamos a usar
    #       input_shape = (79, 1) --> tamaño de la entrada, hay 79 features
    #######
Infilteration.add(Convolution1D(64, 3, activation="relu", padding = 'same'))
Infilteration.add(MaxPooling1D(pool_size=(2))) # capa donde se lleva a cabo el pooling, se queda con el maximo de cada 2
Infilteration.add(Convolution1D(128, 3,  activation="relu", padding = 'same'))
Infilteration.add(Convolution1D(128, 3,  activation="relu", padding = 'same'))
Infilteration.add(MaxPooling1D(pool_size=(2)))
Infilteration.add(LSTM(lstm_output_size)) # Se añade una LSTM como segunda red
Infilteration.add(Dropout(0.1)) #
Infilteration.add(Dense(Infilteration_attacks.shape[0], activation="softmax")) # capa fully conected para decision final, usamos softmax porque con ella los valores finales tienen mas relacion con los valores
    # anteriores y no solo con 1

Bruteforce = Sequential()
Bruteforce.add(Convolution1D(64, 3, activation="relu",input_shape=(78, 1), padding = 'same'))
    #######
    #   Añadimos la primera capa de Convolution1D, los diferentes parametros indican lo siguiente:
    #       64 --> numero de filtros
    #       3 --> tamaño del filtro (3,1)
    #       border_mode = "same" --> este parametro sirve para que el output sea del mismo tamaño que elinput
    #       activation = "relu" --> Tipo de funcion de activacion de neuronas que vamos a usar
    #       input_shape = (79, 1) --> tamaño de la entrada, hay 79 features
    #######
Bruteforce.add(Convolution1D(64, 3, activation="relu", padding = 'same'))
Bruteforce.add(MaxPooling1D(pool_size=(2))) # capa donde se lleva a cabo el pooling, se queda con el maximo de cada 2
Bruteforce.add(Convolution1D(128, 3,  activation="relu", padding = 'same'))
Bruteforce.add(Convolution1D(128, 3,  activation="relu", padding = 'same'))
Bruteforce.add(MaxPooling1D(pool_size=(2)))
Bruteforce.add(LSTM(lstm_output_size)) # Se añade una LSTM como segunda red
Bruteforce.add(Dropout(0.1)) #
Bruteforce.add(Dense(BruteForce_attacks.shape[0], activation="softmax")) # capa fully conected para decision final, usamos softmax porque con ella los valores finales tienen mas relacion con los valores
# anteriores y no solo con 1

DDoS = Sequential()
DDoS.add(Convolution1D(64, 3, activation="relu",input_shape=(78, 1), padding = 'same'))
    #######
    #   Añadimos la primera capa de Convolution1D, los diferentes parametros indican lo siguiente:
    #       64 --> numero de filtros
    #       3 --> tamaño del filtro (3,1)
    #       border_mode = "same" --> este parametro sirve para que el output sea del mismo tamaño que elinput
    #       activation = "relu" --> Tipo de funcion de activacion de neuronas que vamos a usar
    #       input_shape = (79, 1) --> tamaño de la entrada, hay 79 features
    #######
DDoS.add(Convolution1D(64, 3, activation="relu", padding = 'same'))
DDoS.add(MaxPooling1D(pool_size=(2))) # capa donde se lleva a cabo el pooling, se queda con el maximo de cada 2
DDoS.add(Convolution1D(128, 3,  activation="relu", padding = 'same'))
DDoS.add(Convolution1D(128, 3,  activation="relu", padding = 'same'))
DDoS.add(MaxPooling1D(pool_size=(2)))
DDoS.add(LSTM(lstm_output_size)) # Se añade una LSTM como segunda red
DDoS.add(Dropout(0.1)) #
DDoS.add(Dense(DoS_attacks.shape[0], activation="softmax")) # capa fully conected para decision final, usamos softmax porque con ella los valores finales tienen mas relacion con los valores
# anteriores y no solo con 1

if __name__ == '__main__' and os.getpid()==pids[0]:

    """# TEST"""

    print(__name__)
    salidas = []


    salidas =[]
    networks = ["C:/Users/jadelsot/Desktop/TFG/resultados/cuarto_entrenamiento/Infilteration/checkpoints/checkpoint-12.hdf5", "C:/Users/jadelsot/Desktop/TFG/resultados/cuarto_entrenamiento/BruteForce/checkpoints/checkpoint-11.hdf5",
    "C:/Users/jadelsot/Desktop/TFG/resultados/tercer_entrenamiento/DoS_WITHOUT_Thuesday-20-02-2018/checkpoints/checkpoint-15.hdf5"]
    input_NN= X_test[142700:142900]

    #PREDICCION

    y_pred_infilt = [] * input_NN.shape[0]
    y_pred_brute = [] * input_NN.shape[0]
    y_pred_dos = [] * input_NN.shape[0]

    y_pred_infilt_text = [] * input_NN.shape[0]
    y_pred_brute_text = [] * input_NN.shape[0]
    y_pred_dos_text = [] * input_NN.shape[0]

    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=Prediccion,args=(i,input_NN, networks[i]))
        p.start()
        processes.append(p)

    # Now you can wait for the networks to finish training before executing the 
    # rest of the script
    for process in processes:
        process.join()


    # Evaluación de los resultados y conversión de estos
    archivoInfilt = open('InfiltOutput.txt','r')
    y_pred = archivoInfilt.read()
    y_pred_infilt = list(y_pred)
    archivoInfilt.close() #Cierras el archivo.

    archivoBrute = open('BruteOutput.txt','r')
    y_pred = archivoBrute.read()
    y_pred_brute = list(y_pred)
    archivoBrute.close() #Cierras el archivo.

    archivoDoS = open('DoSOutput.txt','r')
    y_pred = archivoDoS.read()
    y_pred_dos = list(y_pred)
    archivoDoS.close() #Cierras el archivo.

    # Resultados estan en list separados por espacio dentro de un mismo elemento, queremos separarlos en distintos elementos

    

    for i in range(0, input_NN.shape[0]):
        y_pred_infilt_text.append(Infilteration_attacks[int(y_pred_infilt[i])])
        y_pred_brute_text.append(BruteForce_attacks[int(y_pred_brute[i])])
        y_pred_dos_text.append(DoS_attacks[int(y_pred_dos[i])])

    # Mostramos por pantalla los resultados obtenidos
    print("\nSegun la evaluación de cada red neuronal, estos son los resultados: ")
    print("La red neuronal detectora de ataque DoS dice que estos flujos son: ", y_pred_dos_text)
    print("La red neuronal detectora de ataque BruteForce dice que estos flujos son: ", y_pred_brute_text)
    print("La red neuronal detectora de ataque Infilteration dice que estos flujos son: ", y_pred_infilt_text)

    #Sistema de votacion final
    decision_final = [] * input_NN.shape[0]

    #Para el sistema de votacion final se seguirá la siguiente aproximacion:
    #  solo se considerará benigno si todas las redes neuronales califican asi el flujo.
    #  Si solo una red neuronal lo califica como ataque, la decisión final será ese ataque en concreto
    #  Si más de una red neuronal califica el flujo como ciberataque se asumirá que este flujo corresponde al ataque que mejor porcentaje de deteccion tenga
    
    for i in range(0, input_NN.shape[0]):
        if(y_pred_dos_text[i] == y_pred_brute_text[i] == y_pred_infilt_text[i]): # Todas ven Bening
            decision_final.append(y_pred_dos_text[i])

        elif(y_pred_dos_text[i] == y_pred_brute_text[i] and y_pred_brute_text[i] != y_pred_infilt_text[i]): #Si DoS y Brute ven trafico Bening e Infilt no 
            decision_final.append(y_pred_infilt_text[i])

        elif(y_pred_infilt_text[i] == y_pred_brute_text[i] and y_pred_brute_text[i] != y_pred_dos_text[i]): #Si Infilt y Brute ven trafico Bening y DoS no 
            decision_final.append(y_pred_dos_text[i])

        elif(y_pred_dos_text[i] == y_pred_infilt_text[i] and y_pred_infilt_text[i] != y_pred_brute_text[i]): #Si DoS y Brute ven trafico Bening e Infilt no 
            decision_final.append(y_pred_brute_text[i])
        else:
            decision_final.append([y_pred_infilt_text[i],y_pred_brute_text[i], y_pred_dos_text[i]])
    
    print("\nLa decision final que se ha tomado sobre los flujos es: ", decision_final)
        
