import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda, SimpleRNN
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.callbacks.callbacks import LearningRateScheduler

# from matplotlib import pyplot as plt

import numpy as np
from numpy import genfromtxt

import time
import DataSetLoader
import DataScaler
import Plotter

keras.backend.clear_session()

start_time = time.time()
############### Carga de datos ##############################
dsl = DataSetLoader.DataSetLoader()

# dsl.generateCSVs(False,5,60)

# x_series = genfromtxt('Datos_prueba/X_flat_stand_01m_ts5_pred60.csv', delimiter=';')
# y_series = genfromtxt('Datos_prueba/y_stand_01m_ts5_pred60.csv', delimiter=';')
# y_series = np.expand_dims(y_series, axis=1)


# x_series = genfromtxt('C:/Users/PABLO/Desktop/Datos_prueba/X_flat_filled_01m_ts5_pred60.csv', delimiter=';')
# y_series = genfromtxt('C:/Users/PABLO/Desktop/Datos_prueba/y_filled_01m_ts5_pred60.csv', delimiter=';')
# y_series = np.expand_dims(y_series, axis=1)


raw_data = dsl.get_sensors_data(True,False)
x_series, y_series = dsl.window_data(raw_data,window_size=5,lookforward=61)


(x_train, y_train),(x_test, y_test) = dsl.generateTrainTestSet(x_series,y_series,split=9/10)

input_shape=x_train.shape[1]

print(f"\nDatos cargados: {(time.time() - start_time)} seconds")

############### Escalar los datos entre 0 y 1 ##############################
ds = DataScaler.DataScaler()

x_train = ds.fit_scale_data(x_train)
y_train = ds.scale_data(y_train)

x_test = ds.scale_data(x_test)


############### Hiperparametros ##############################
learning_rate=1e-3
number_epochs = 400
batch=300
loss_function='mse'
neuron='tanh'
 
precision="mae"

# optimizer = SGD(lr=learning_rate)
# optimizer = RMSprop(lr=learning_rate,rho=0.9)
# optimizer = Adadelta(rho=0.95)
optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

############### Modelo ############################## 
model = Sequential([
  # Lambda(lambda x: tf.expand_dims(x, axis=-1),input_shape=[None]),
  # SimpleRNN(20, activation=neuron ,return_sequences=True),
  SimpleRNN(20, activation=neuron,input_shape=[x_train.shape[1],x_train.shape[2]] ,return_sequences=True),
  SimpleRNN(20, activation=neuron),
  Dense(1, activation='relu'),
])



model.compile(loss=loss_function, optimizer=optimizer, metrics=[precision])

history = model.fit(x_train, y_train, batch_size=batch, validation_split=0.2 ,epochs=number_epochs, verbose=2) #entrenar la red #validar

# lr_schedule = LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
# history = model.fit(x_train, y_train, batch_size=batch, validation_split=0.2 ,epochs=number_epochs, verbose=2, callbacks=[lr_schedule])
# #Mostrar gráfico del learning rate
# lrs = 1e-8 * (10 ** (np.arange(200) / 20))
# plt.semilogx(lrs, history.history["loss"])
# plt.axis([1e-8, 1e-2, 0, 1])
# plt.show()

# Gráfico con la evolución del coste para cada epoch
Plotter.plot_loss(number_epochs,history)

################# Predicción ################# 
predictions = model.predict(x_test, verbose=0)

# predictions= genfromtxt('Predictions/naive_prediction.csv', delimiter=';')
# predictions= genfromtxt('Predictions/moving_average_prediction.csv', delimiter=';')
# predictions= dsl.generateTestSet(predictions)
# predictions =np.expand_dims(predictions, axis=1)

### Desescalar los datos ##
predictions = ds.inverse_scale_data(predictions)

#Gráfica Predicción
Plotter.plot_prediction(y_test, predictions,start=100,end=1000)
Plotter.plot_prediction(y_test, predictions,start=600,end=800)

print(f"Overfitting = {Plotter.overfitting(history):.2f}%")
print(f"Precisión (mae) = {Plotter.mae(y_test, predictions):.3f}")
# print(f"Precisión (mape) = {DataSetLoader.mape(y_test, predictions):.3f}")

#Mostrar tiempo de ejecución
print(f"\nTiempo de ejecución = {(time.time() - start_time)} seconds")