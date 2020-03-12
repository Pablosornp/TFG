import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda, SimpleRNN
from keras.optimizers import SGD
from keras.utils import plot_model

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

import numpy as np
from numpy import genfromtxt

import tensorflow as tf
import datetime
import time

tf.keras.backend.clear_session()

start_time = time.time()

x_series = genfromtxt('Datos_prueba/x_train_1min_dh6_100000.csv', delimiter=';')
y_series = genfromtxt('Datos_prueba/y_train_1min_dh6_100000.csv', delimiter=';')
#print(x_series.shape)
#print(y_series.shape)

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

number_epochs = 100
learning_rate=0.01

model = Sequential([
  Lambda(lambda x: tf.expand_dims(x, axis=-1),input_shape=[None]),
  #SimpleRNN(20, activation='relu' ,return_sequences=True),
  SimpleRNN(20, activation='relu'),
  Dense(1),
# Lambda(lambda x: x * 100.0)
])

sgd = SGD(lr=learning_rate)

model.compile(loss='mae', optimizer=SGD(lr=0.01), metrics=['mae'])

history = model.fit(x_train, y_train,validation_split=0.2, epochs=number_epochs, verbose=2) #entrenar la red #validar

predictions = model.predict(x_test, verbose=0)
#print(f"Predictions size = {predictions.shape}")
#print(f"Actual_values size = {y_test.shape}")
np.savetxt("Predictions/predictionsRNN.csv", predictions, delimiter=";")

#Mostrar gráfico con la evolución del coste para cada epoch
epochs=np.arange(number_epochs)
fig, axs = plt.subplots(2)
axs[0].plot(epochs,history.history['loss'],epochs,history.history['val_loss'])
axs[0].set_ylabel('loss')
axs[0].set_xlabel('epoch')
axs[0].legend(['train', 'validation'], loc='upper left')
axs[1].plot(epochs[10:],history.history['loss'][10:],epochs[10:],history.history['val_loss'][10:])
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].legend(['train', 'validation'], loc='upper left')

#Mostrar tiempo de ejecución
print(f"\nTiempo de ejecución = {(time.time() - start_time)} seconds")

plt.show()
