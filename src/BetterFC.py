import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import plot_model

import matplotlib
from matplotlib import pyplot as plt

import numpy as np
from numpy import genfromtxt

import time

import DataSetConverter as dsc

class FullyConnected:
    def __init__(self):
        self.model
        self.DataSetLoader
        self.x_train
        self.y_train
        
        
        
    keras.backend.clear_session()
    
    start_time = time.time()
    
    
    time_series = np.arange(total_time, dtype="float32")
    
    
    def load_data():
        input_data = dsc.get_sensors_data()
        X,y_series = dsc.window_data(input_data)
        x_series=dsc.flatten_windowed_data(X)
        return x_series, y_series
    
    def
    
    total_time = x_series.shape[0]
    split_time = int(total_time*(9/10))
    
    time_series = np.arange(total_time, dtype="float32")
    time_train = time_series[:split_time]
    time_valid = time_series[split_time:]
    
    x_train = x_series[:split_time]
    y_train = y_series[:split_time]
    
    x_test = x_series[split_time:]
    y_test = y_series[split_time:]
    
    window_size=x_train.shape[1]
    
    def train_model(number_epochs, learning_rate, batch_size, loss_function):
        number_epochs = 30
        learning_rate=0.001
        batch=100
        loss_function='mse'
        model = Sequential([
            Dense(9, input_shape=[window_size], activation='relu'), #Numero de neuronas, tamaño datos entrada, activacion
            Dense(16, activation='relu'),
            Dense(1, activation='relu')
        ]) 
        sgd = SGD(lr=learning_rate)
        model.compile(loss=loss_function, optimizer=sgd, metrics=[loss_function])
        history = model.fit(x_train, y_train, batch_size=batch, validation_split=0.2 ,epochs=number_epochs, verbose=2) #entrenar la red #validar
    
        return model
    
    
    predictions = model.predict(x_test, verbose=0)
    np.savetxt("Predictions/predictionsFC.csv", predictions, delimiter=";")
    
    def show_plots():
        #Mostrar gráfico con la evolución del coste para cada epoch
        epochs=np.arange(number_epochs)
        fig, axs = plt.subplots(2)
        axs[0].plot(epochs,history.history['loss'],epochs,history.history['val_loss'])
        axs[0].set_ylabel('loss')
        axs[0].set_xlabel('epoch')
        axs[0].legend(['train', 'validation'], loc='upper left')
        axs[1].plot(epochs[100:],history.history['loss'][100:],epochs[100:],history.history['val_loss'][100:])
        axs[1].set_ylabel('loss')
        axs[1].set_xlabel('epoch')
        axs[1].legend(['train', 'validation'], loc='upper left')
    
    #Mostrar tiempo de ejecución
    print(f"\nTiempo de ejecución = {(time.time() - start_time)} seconds")
    
    plt.show()
    
    #Mostrar gŕafico con la predicción y el valor real.
