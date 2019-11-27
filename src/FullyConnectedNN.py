import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

import tensorflow as tf
import datetime

x_series = genfromtxt('Datos_prueba/x_train_1min_dh6_600.csv', delimiter=';')
y_series = genfromtxt('Datos_prueba/y_train_1min_dh6_600.csv', delimiter=';')

total_time = x_series.shape[0]
split_time = int(total_time*(2/3))

time = np.arange(total_time, dtype="float32")
time_train = time[:split_time]
time_valid = time[split_time:]

x_train = x_series[:split_time]
y_train = y_series[:split_time]

x_test = x_series[split_time:]
y_test = y_series[split_time:]

window_size=x_train.shape[1]

model = Sequential([
    Dense(9, input_shape=[window_size], activation='relu'), #Numero de neuronas, tamaño datos entrada, activacion
    Dense(32, activation='relu'),
    Dense(1, activation='relu')
])

#model.summary()
#plot_model(model, to_file='model.png')

sgd = SGD(lr=0.0001)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


history = model.fit(x_train, y_train, batch_size=1, validation_split=0.2 ,epochs=100, shuffle=False, verbose=2 ,callbacks=[tensorboard_callback]) #entrenar la red #validar
print(history.history.keys())

predictions = model.predict(x_test, verbose=0)
print(f"Predictions size = {predictions.shape}")
print(f"Actual_values size = {y_test.shape}")
np.savetxt("Predictions/predictions.csv", predictions, delimiter=";")

#Mostrar gráfico con la evolución del coste para cada epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Mostrar gŕafico con la predicción y el valor real.
