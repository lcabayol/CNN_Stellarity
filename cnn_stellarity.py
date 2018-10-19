"""
    Code depeloved by Laura Cabayol
    email: lcabayol@ifae.es

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, RobustScaler, scale, StandardScaler,MinMaxScaler
import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten, Activation,Conv1D, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,average_precision_score
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from keras.callbacks import *


def dataset_dealer(dataset):
    """takes the initial dataset, applies a cut on magnitude and creates two differentiated datasets: one for stars and another for galaxies"""
    print('The magnitude cut is at', magCut)
    dataset = dataset[dataset.i_auto < magCut]
    dataset = random_sample(dataset)#shuffle the dataset to avoid having stars and galaxies clustered
    dataset = dataset.dropna()#avoid any object that can cause problems in the training
    dataset = dataset.reset_index(drop = True)
    Stars = dataset[dataset.Label == 1]
    Stars = Stars.reset_index(drop = False)
    Galaxies = dataset[dataset.Label == 0]
    Galaxies = Galaxies.reset_index(drop = False)
    return Stars, Galaxies


def sample_split(Galaxies, Stars, nStarsTRN, nStarsVal, nGalTRN, nGalVal):
    """#takes the star and galaxy datasets and generates a training and validation sample with the number of objects per class defined before"""
    train = pd.concat((Galaxies.loc[1:(nGalTRN+1), : ],Stars.loc[1:(nStarsTRN+1), :]), axis = 0,ignore_index ='True')
    val = pd.concat((Galaxies.loc[(nGalTRN+1): (nGalTRN + 1 + nGalVal) ,:],Stars.loc[(nStarsTRN+1): (nStarsTRN + 1 + nStarsVal) ,: ]), axis = 0, ignore_index ='True' )
    train = random_sample(train)
    val = random_sample(val)
    return train, val

def random_sample(DataFrame):
    DataFrame = shuffle(DataFrame)
    DataFrame = DataFrame.reset_index(drop = True)
    return DataFrame


def Flux_input(Fluxes):
    """ Format that the network needs"""
    Fluxes = Fluxes.reshape(-1, Fluxes.shape[1] ,1)
    Fluxes = Fluxes.astype('float32')
    return Fluxes

def CNN():
    ConvNNet = Sequential()
    ConvNNet.add(Conv1D(filters = 32, kernel_size = 10,activation='linear',input_shape=(FeaturesTRN.shape[1],1),padding='same'))
    ConvNNet.add(LeakyReLU(alpha=0.1))
    ConvNNet.add(MaxPooling1D(pool_size = 4, padding='same'))
    ConvNNet.add(Conv1D(filters =64, kernel_size = 5, activation='linear',padding='same'))
    ConvNNet.add(LeakyReLU(alpha=0.1))
    ConvNNet.add(MaxPooling1D(pool_size= 2, padding='same'))
    ConvNNet.add(Conv1D(filters = 128, kernel_size =3 , activation='linear',padding='same'))
    ConvNNet.add(LeakyReLU(alpha=0.1))
    ConvNNet.add(MaxPooling1D(pool_size= 2 ,padding='same'))
    ConvNNet.add(Flatten())
    ConvNNet.add(Dense(128, activation='linear'))
    ConvNNet.add(LeakyReLU(alpha=0.1))
    #ConvNNet.add(Dropout(0.1))
    ConvNNet.add(Dense(num_classes, activation='softmax'))
    return ConvNNet



#---------------------------------------------MAIN-------------------------------------------

#-------------------parameters--------------------
nStarsTRN = 5000                                #|
nGalTRN = 15000                                 #|
nStarsVal = 10000                               #|
nGalVal = 10000                                 #|
                                                #|
magCut = 22.5                                   #|
                                                #|
Labels = ['Label']                              #|
magnitude = ['i_auto']                          #|
coord = ['ra','dec']                            #|
magerr = ['magerr']                             #|
                                                #|
batch_size = 8                                  #|
epochs = 5                                      #|
num_classes = 2                                 #|
                                                #|
bands = ['NB455','NB465','NB475','NB485','NB495','NB505','NB515','NB525','NB535','NB545','NB555','NB565','NB575','NB585','NB595','NB605','NB615','NB625','NB635','NB645', 'NB655','NB665','NB675','NB685','NB695','NB705','NB715','NB725','NB735','NB745','NB755','NB765','NB775','NB785','NB795','NB805','NB815','NB825','NB835','NB845']
                                                #|
#-----------------------------------------------#|



print('This code classifies stars and galaxies with a ConvNet. The trainig and validation samples must be labeled:')
print(' --> stars = 1')
print(' --> galaxies = 0')
print('The dataset contains the 40NB fluxes, the truth label and i_auto and it consists of PAUs objects on the COSMOS field ')



"""This file should be a csv file containing:
- red_id,
- PAUS fluxes in the 40 NB: [NB455,...,NB845]
- i_auto
- ra
- dec
- Label: From Leauthaud et al. 2007
"""
DataPAU = pd.read_table('YourFile.csv', sep  = ',', comment = '#', header = 0)

Stars, Galaxies = dataset_dealer(DataPAU)#input:original dataset, output: star and galaxy datasets

print('The total number of stars is',Stars.shape[0])
print('The total number of galaxies is',Galaxies.shape[0])

DataTRN, DataVal = sample_split(Galaxies, Stars, nStarsTRN, nStarsVal, nGalTRN, nGalVal)


#Define input features and input labels
FeaturesTRN = DataTRN.loc[:,bands]
LabelsTRN = DataTRN.loc[:,'Label']
FeaturesVal = DataVal.loc[:,bands]
LabelsVal = DataVal.loc[:,'Label']

#input array into the format needed for the cnn
FeaturesTRN = Flux_input(FeaturesTRN.values)
FeaturesVal = Flux_input(FeaturesVal.values)

#input labels must be hot encoded: 0 --> [1,0]; 1 --> [0,1]
LabelsTRN_cat = to_categorical(LabelsTRN)
LabelsVal_cat = to_categorical(LabelsVal)



#----------------------- cnn ---------------------------
ConvNNet = CNN()

ConvNNet.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr =0.001, decay = 0.001),metrics={'output_a': 'accuracy'})#define the network: loss function, optimization method, metrics

print(ConvNNet.summary())


ConvNNet_train = ConvNNet.fit(FeaturesTRN, LabelsTRN_cat,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(FeaturesVal, LabelsVal_cat), initial_epoch = 0) #train

Stellar_flag_TRN = ConvNNet.predict(FeaturesTRN, batch_size=None, verbose=0) #validate with the network trained
Stellar_flag_Val = ConvNNet.predict(FeaturesVal, batch_size=None, verbose=0) #validate with the network trained



















































