

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

import tensorflow as tf

def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(actual - predicted))

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

y_series = genfromtxt('Datos_prueba/y_train_1min_dh6_100000.csv', delimiter=';')

total_time = y_series.shape[0]
split_time = int(total_time*(9/10))

time = np.arange(total_time, dtype="float32")
time_train = time[:split_time]
time_valid = time[split_time:]

y_test = y_series[split_time:]

predictions = genfromtxt('Predictions/predictions.csv', delimiter=';')
print(predictions.shape)
print(y_test.shape)

plt.figure(figsize=(10, 6))

plot_series(time_valid, y_test)
plot_series(time_valid, predictions)
plt.legend(['Valor real', 'Predicción'], loc='upper left')

#print(mae(y_test, predictions))
plt.show()
