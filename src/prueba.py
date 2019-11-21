import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from numpy import genfromtxt

X_train = genfromtxt('Datos_prueba/x_train_1min_dh6_600.csv', delimiter=';')
Y_train = genfromtxt('Datos_prueba/y_train_1min_dh6_600.csv', delimiter=';')

print(X_train)

model = Sequential([
    Dense(9, input_shape=(9,), activation='tanh'), #Numero de neuronas, tamaño datos entrada, activacion
    Dense(16, activation='tanh'),
    Dense(1, activation='relu')
])

model.summary()

sgd = SGD(lr=.001)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])

history = model.fit(X_train, Y_train, batch_size=1, validation_split=0.2 ,epochs=100, shuffle=False, verbose=2) #entrenar la red #validar
