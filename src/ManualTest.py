import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Activation, Lambda, Dropout
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.regularizers import l2
from keras.callbacks.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

import time
import DataSetLoader as dsl, DataScaler, Plotter, ModelTester as mt

keras.backend.clear_session()

results_path = 'C:/Users/PABLO/Desktop/Results/Pruebas manuales'
test_name = 'P43 lr sch'

path = f"{results_path}/{test_name}"
mt.newFolder(path)

start_time = time.time()
############### Carga de datos ##############################
forecast=6
timesteps=60
measure_frec=15 #1 #15 #30
standarize=False

sensors_data = dsl.get_sensors_data(standarize, measure_frec)
x_series, y_series = dsl.window_data(sensors_data,measure_frec,timesteps,forecast)
# x_series = dsl.flatten_windowed_data(x_series)

(x_train, y_train),(x_test, y_test) = dsl.generateTrainTestSet(x_series,y_series,split=9/10)

# input_shape=[x_train.shape[1]]
input_shape=[x_train.shape[1],x_train.shape[2]]

tiempo_carga_datos = (time.time() - start_time)
print(f"\nDatos cargados: {(time.time() - start_time):2f} seconds")

############### Escalado de datos ##############################
ds = DataScaler.DataScaler(feature_range=(0, 1))

x_train = ds.fit_scale_data(x_train)
y_train = ds.scale_data(y_train)

x_test = ds.scale_data(x_test)

############### Hiperparametros ##############################
learning_rate=5e-4
number_epochs = 100
batch=200
loss_function='mse'
neuron='tanh'
 
precision="mae"

# optimizer = SGD(learning_rate=learning_rate)
# optimizer = RMSprop(learning_rate=learning_rate,rho=0.9)
# optimizer = Adadelta(rho=0.95)
optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)

############### Modelo ############################## 
# model = Sequential([
#     Dense(85, input_shape=input_shape, activation=neuron), #Numero de neuronas, tamaño datos entrada, activacion
#     Dense(17, activation=neuron),
#     Dense(5, activation=neuron),
#     Dense(1, activation=neuron)
# ])

model = Sequential([
    LSTM(60, activation=neuron, input_shape=input_shape,return_sequences=False),
    # Dropout(0.3),
    Dense(1, activation='relu')
])

model_arq='60L -> 1D'

model.compile(loss=loss_function, optimizer=optimizer, metrics=[precision])
history = model.fit(x_train, y_train, validation_split=0.05 ,batch_size=batch, epochs=number_epochs, verbose=2) #entrenar la red #validar


#### Early stopping
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
# history = model.fit(x_train, y_train, validation_split=0.05 ,batch_size=batch, epochs=number_epochs, verbose=2, callbacks=[es]) #entrenar la red #validar


#### Lr schedule
# lr_schedule = LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
# history = model.fit(x_train, y_train, batch_size=batch, validation_split=0.2 ,epochs=number_epochs, verbose=2, callbacks=[lr_schedule])
# #Mostrar gráfico del learning rate
# Plotter.plot_lr_schedule(history,path)


# Gráfico con la evolución del coste para cada epoch
Plotter.plot_loss(history,path)

################# Predicción ################# 
predictions = model.predict(x_test, verbose=0)

### Desescalar los datos ##
x_train = ds.inverse_scale_data(x_train)
y_train = ds.inverse_scale_data(y_train)

x_test = ds.inverse_scale_data(x_test)
predictions = ds.inverse_scale_data(predictions)

#Gráfica Predicción
Plotter.plot_prediction(y_test, predictions,path,start=100,end=1000)
Plotter.plot_prediction(y_test, predictions,path,start=600,end=800)
Plotter.plot_prediction(y_test, predictions,path,start=400,end=600)

# Saving results in .txt file
fich_res = open(f"{path}/info_{test_name}.txt","w+")
fich_res.write(f"learning_rate={learning_rate}\n, number_epochs = {number_epochs}\n,batch={batch}\n,neuron={neuron}\n,model={model_arq}\n")
fich_res.write(f"\nPrecision (mae) = {Plotter.mae(y_test, predictions):.2f}\n")
fich_res.write(f"Overfitting = {Plotter.overfitting(history):.2f}%\n")
fich_res.close()
print(f"Overfitting = {Plotter.overfitting(history):.2f}%")
print(f"Precisión (mae) = {Plotter.mae(y_test, predictions):.2f}")

np.savetxt(f"{path}/{test_name}_pred.csv", predictions, delimiter=";")
np.savetxt(f"{path}/{test_name}_loss.csv", history.history['loss'], delimiter=";")
np.savetxt(f"{path}/y_test.csv", y_test, delimiter=";")
#Mostrar tiempo de ejecución
print(f"\nTiempo de ejecución = {(time.time() - start_time):.2f} seconds")



