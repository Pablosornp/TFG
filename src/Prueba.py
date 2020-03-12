import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler

x_series = genfromtxt('Datos_prueba/x_train_1min_dh6_100000.csv', delimiter=';')
y_series = genfromtxt('Datos_prueba/y_train_1min_dh6_100000.csv', delimiter=';')

print(x_series)
print(y_series)

scaler1 = MinMaxScaler(feature_range=(0,1))
scaler2 = MinMaxScaler(feature_range=(0,1))
print(scaler1.fit(x_series))
x_series_norm=scaler1.transform(x_series)
x_series_back=scaler1.inverse_transform(x_series_norm)
print(scaler2.fit(y_series.reshape(-1,1)))
y_series_norm=scaler2.transform(y_series.reshape(-1,1))
y_series_back=scaler2.inverse_transform(y_series_norm)
print(x_series_norm)
print(y_series_norm)
print(x_series_back)
print(y_series_back)
