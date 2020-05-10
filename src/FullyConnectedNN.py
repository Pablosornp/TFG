import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda, Dropout
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.regularizers import l2
from keras.callbacks.callbacks import LearningRateScheduler

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

import time
import DataSetLoader, DataScaler, Plotter

keras.backend.clear_session()


start_time = time.time()
############### Carga de datos ##############################
dsl = DataSetLoader.DataSetLoader()

# dsl.generateCSVs(False,5,60)

# x_series = genfromtxt('C:/Users/PABLO/Desktop/Datos_prueba/X_flat_stand_01m_ts5_pred60.csv', delimiter=';')
# y_series = genfromtxt('C:/Users/PABLO/Desktop/Datos_prueba/y_stand_01m_ts5_pred60.csv', delimiter=';')
# y_series = np.expand_dims(y_series, axis=1)

x_series = genfromtxt('C:/Users/PABLO/Desktop/Datos_prueba/X_flat_filled_01m_ts5_pred60.csv', delimiter=';')
y_series = genfromtxt('C:/Users/PABLO/Desktop/Datos_prueba/y_filled_01m_ts5_pred60.csv', delimiter=';')
y_series = np.expand_dims(y_series, axis=1)


(x_train, y_train),(x_test, y_test) = dsl.generateTrainTestSet(x_series,y_series,split=9/10)

input_shape=x_train.shape[1]

tiempo_carga_datos = (time.time() - start_time)
print(f"\nDatos cargados: {(time.time() - start_time):2f} seconds")

############### Escalar los datos entre 0 y 1 ##############################
ds = DataScaler.DataScaler()

x_train = ds.fit_scale_data(x_train)
y_train = ds.scale_data(y_train)

x_test = ds.scale_data(x_test)


############### Hiperparametros ##############################
learning_rate=1e-4
number_epochs = 200
batch=300
loss_function='mse'
neuron='relu'
 
precision="mae"

# optimizer = SGD(lr=learning_rate)
# optimizer = RMSprop(lr=learning_rate,rho=0.9)
# optimizer = Adadelta(learning_rate=1.0, rho=0.95)
optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

############### Modelo ############################## 
model = Sequential([
    Dense(17, input_shape=[input_shape], activation=neuron), #Numero de neuronas, tamaño datos entrada, activacion
    Dense(1, activation=neuron)
    
    # Dense(85, input_shape=[input_shape], activation='relu'), #Numero de neuronas, tamaño datos entrada, activacion
    # Dense(17, activation='relu'),
    # Dense(1, activation='relu')
])

model.compile(loss=loss_function, optimizer=optimizer, metrics=[precision])

# lr_schedule = LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
# history = model.fit(x_train, y_train, batch_size=batch, validation_split=0.2 ,epochs=number_epochs, verbose=2, callbacks=[lr_schedule])
# #Mostrar gráfico del learning rate
# lrs = 1e-8 * (10 ** (np.arange(number_epochs) / 30))
# plt.semilogx(lrs, history.history["loss"])
# plt.axis([1e-8, 1e-1, 0, max(history.history["loss"])])
# plt.show()

history = model.fit(x_train, y_train, batch_size=batch, validation_split=0.2 ,epochs=number_epochs, verbose=2) #entrenar la red #validar


# Gráfico con la evolución del coste para cada epoch
Plotter.plot_loss(number_epochs,history)

################# Predicción ################# 
predictions = model.predict(x_test, verbose=0)

### Desescalar los datos ##
x_train = ds.inverse_scale_data(x_train)
y_train = ds.inverse_scale_data(y_train)

x_test = ds.inverse_scale_data(x_test)
predictions = ds.inverse_scale_data(predictions)

#Gráfica Predicción
Plotter.plot_prediction(y_test, predictions,start=100,end=1000)
Plotter.plot_prediction(y_test, predictions,start=600,end=800)

print(f"Overfitting = {Plotter.overfitting(history):.2f}%")
print(f"Precisión (mae) = {Plotter.mae(y_test, predictions):.2f}")
# print(f"Precisión (mape) = {DataSetLoader.mape(y_test, predictions):.3f}")

#Mostrar tiempo de ejecución
print(f"\nTiempo de ejecución = {(time.time() - start_time):.2f} seconds")



