import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
def plot_loss(number_epochs,history,path):
    plt.figure(figsize=(10, 6))
    epochs=np.arange(number_epochs)
    fig, axs = plt.subplots(2)
    axs[0].plot(epochs,history.history['loss'],epochs,history.history['val_loss'])
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'validation'], loc='upper right')
    axs[1].plot(epochs[100:],history.history['loss'][100:],epochs[100:],history.history['val_loss'][100:])
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'validation'], loc='upper right')
    plt.savefig(f"{path}/_loss.png")
    plt.show()
    
def plot_comparative_loss(number_epochs,history,path):
    rms = genfromtxt('Loss/rms.csv', delimiter=';')
    adadelta = genfromtxt('Loss/adadelta.csv', delimiter=';')
    plt.figure(figsize=(10, 6))
    epochs=np.arange(number_epochs)
    fig, axs = plt.subplots(2)
    axs[0].plot(epochs,history.history['loss'])
    axs[0].plot(epochs,rms)
    axs[0].plot(epochs,adadelta)
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['Value 1', 'Value 2', 'Value 3'], loc='upper right')
    axs[1].plot(epochs[100:],history.history['loss'][100:])
    axs[1].plot(epochs[100:],rms[100:])
    axs[1].plot(epochs[100:],adadelta[100:])
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['Value 1', 'Value 2', 'Value 3'], loc='upper right')
    plt.savefig(f"{path}/comp_loss.png")
    plt.show()
    
def plot_precision(number_epochs,history,path,start=0,end=None):
    plt.figure(figsize=(10, 6))
    epochs=np.arange(number_epochs)
    plt.plot(epochs[start:end],history.history['mae'],'r')
    plt.plot(epochs[start:end],history.history['val_mae'],'b')
    plt.xlabel("Time (min)")
    plt.ylabel("MAE (W^2/m)")   
    plt.legend(['Precisión entrenamiento', 'Precisión validación'], loc='upper left')
    plt.savefig(f"{path}/prec.png")
    plt.show()
    
def plot_prediction(y_test,prediction,path,start=0,end=None):
    time_valid = np.arange(y_test.shape[0],dtype="float32")
    plt.figure(figsize=(10, 6))
    plt.plot(time_valid[start:end],y_test[start:end])
    plt.plot(time_valid[start:end],prediction[start:end])
    plt.xlabel("Time (min)")
    plt.ylabel("Irradiance(W^2/m)")   
    plt.legend(['Valor real', 'Predicción'], loc='upper left')
    plt.grid(True)
    # plt.annotate('Something', xy=(0.05, 0.65), xycoords='axes fraction')
    plt.savefig(f"{path}/pred_{start}_{end}.png")
    plt.show()
    
def plot_comparative_prediction(y_test,prediction,path,start=0,end=None,color='tab:orange'):
    last = genfromtxt('Predictions/last_pred.csv', delimiter=';')
    time_valid = np.arange(y_test.shape[0],dtype="float32")
    plt.figure(figsize=(10, 6))
    plt.plot(time_valid[start:end],y_test[start:end])
    plt.plot(time_valid[start:end],last[start:end],'g')
    plt.plot(time_valid[start:end],prediction[start:end],color)
    plt.xlabel("Time (min)")
    plt.ylabel("Irradiance(W^2/m)")   
    plt.legend(['Valor real', 'Predicción anterior', 'Predicción'], loc='upper left')
    # plt.legend(['Valor real', 'tanh', 'relu'], loc='upper left')
    plt.grid(True)
    plt.savefig(f"{path}/comp_{start}_{end}.png")
    plt.show()
    
def plot_lr_schedule(number_epochs,history,path):
    #Mostrar gráfico del learning rate
    lrs = 1e-8 * (10 ** (np.arange(number_epochs) / 30))
    plt.semilogx(lrs, history.history["loss"])
    plt.xlabel("Tasa de aprendizaje")
    plt.ylabel("Coste")
    plt.axis([1e-8, 1e-1, 0, max(history.history["loss"])])
    plt.savefig(f"{path}/lr_schedule.png")
    plt.show()
    
def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(actual - predicted))

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(actual - predicted))

def mape(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Percenatage Error """
    epsilon = 0.001
    return np.abs((actual - predicted)/(actual+epsilon)).mean()*100  

def average(lst): 
    return sum(lst) / len(lst) 

def overfitting(history):
    avr_loss=average(history.history['loss'])
    avr_val_loss=average(history.history['val_loss'])
    overfitting = (avr_val_loss-avr_loss)/avr_loss
    overfitting_perc = overfitting * 100
    return overfitting_perc
