import time
import Plotter, DataSetLoader

import numpy as np
from numpy import genfromtxt
start_time = time.time()
############### Carga de datos ##############################
dsl = DataSetLoader.DataSetLoader()

# dsl.generateCSVs(False,5,60)

# x_series = genfromtxt('C:/Users/PABLO/Desktop/Datos_prueba/X_flat_stand_01m_ts5_pred60.csv', delimiter=';')
# y_series = genfromtxt('C:/Users/PABLO/Desktop/Datos_prueba/y_stand_01m_ts5_pred60.csv', delimiter=';')

x_series = genfromtxt('C:/Users/PABLO/Desktop/Datos_prueba/X_flat_filled_01m_ts5_pred60.csv', delimiter=';')
y_series = genfromtxt('C:/Users/PABLO/Desktop/Datos_prueba/y_filled_01m_ts5_pred60.csv', delimiter=';')
y_series = np.expand_dims(y_series, axis=1)

(x_train, y_train),(x_test, y_test) = dsl.generateTrainTestSet(x_series,y_series,split=9/10)

################# Predicción ################# 

# predictions= genfromtxt('Predictions/naive_prediction.csv', delimiter=';')
predictions = genfromtxt('Predictions/moving_average_prediction.csv', delimiter=';')
predictions = dsl.generateTestSet(predictions)
predictions = np.expand_dims(predictions, axis=1)


#Gráfica Predicción
Plotter.plot_prediction(y_test, predictions,start=100,end=1000)
Plotter.plot_prediction(y_test, predictions,start=600,end=800)

print(f"Precisión (mae) = {Plotter.mae(y_test, predictions):.3f}")
# print(f"Precisión (mape) = {DataSetLoader.mape(y_test, predictions):.3f}")

#Mostrar tiempo de ejecución
print(f"\nTiempo de ejecución = {(time.time() - start_time):.2f} seconds")

