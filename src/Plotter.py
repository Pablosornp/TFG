import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
def plot_loss(history,path):
    plt.figure(figsize=(10, 6))
    number_epochs=int(len(history.history['loss']))
    epochs=np.arange(number_epochs)
    number_epochs = int(number_epochs)
    fig, axs = plt.subplots(2)
    axs[0].plot(epochs,history.history['loss'],epochs,history.history['val_loss'])
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'validation'], loc='upper right')
    axs[1].plot(epochs[int(number_epochs/2):],history.history['loss'][int(number_epochs/2):],epochs[int(number_epochs/2):],history.history['val_loss'][int(number_epochs/2):])
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'validation'], loc='upper right')
    plt.savefig(f"{path}/_loss.png")
    plt.show()
    

def plot_precision(history,measure,path,start=0,end=None):
    plt.figure(figsize=(10, 3))
    number_epochs=int(len(history.history['loss']))
    epochs=np.arange(number_epochs)
    plt.plot(epochs[start:end],history.history['mae'],'r')
    plt.plot(epochs[start:end],history.history['val_mae'],'b')
    plt.xlabel(f"Time ({measure}min)")
    plt.ylabel("MAE (W^2/m)")   
    plt.legend(['Precisión entrenamiento', 'Precisión validación'], loc='upper left')
    plt.savefig(f"{path}/prec.png")
    plt.show()
    
def plot_prediction(y_test,prediction,measure,path,start=0,end=None):
    time_valid = np.arange(y_test.shape[0],dtype="float32")
    plt.figure(figsize=(10, 6))
    plt.plot(time_valid[start:end],y_test[start:end])
    plt.plot(time_valid[start:end],prediction[start:end])
    plt.xlabel(f"Time ({measure}min)")
    plt.ylabel("Irradiance(W^2/m)")   
    plt.legend(['Valor real', 'Predicción'], loc='upper left')
    plt.grid(True)
    # plt.annotate('Something', xy=(0.05, 0.65), xycoords='axes fraction')
    plt.savefig(f"{path}/pred_{start}_{end}.png")
    plt.show()
    
def plot_comparative_prediction(y_test,prediction,measure,path,start=0,end=None,color='tab:orange'):
    last = genfromtxt('Predictions/last_pred.csv', delimiter=';')
    time_valid = np.arange(y_test.shape[0],dtype="float32")
    plt.figure(figsize=(10, 6))
    plt.plot(time_valid[start:end],y_test[start:end])
    plt.plot(time_valid[start:end],last[start:end],'g')
    plt.plot(time_valid[start:end],prediction[start:end],color)
    plt.xlabel(f"Time ({measure}min)")
    plt.ylabel("Irradiance(W^2/m)")   
    plt.legend(['Valor real', 'Predicción anterior', 'Predicción'], loc='upper left')
    # plt.legend(['Valor real', 'tanh', 'relu'], loc='upper left')
    plt.grid(True)
    plt.savefig(f"{path}/comp_{start}_{end}.png")
    plt.show()
    
def plot_lr_schedule(history,path):
    #Mostrar gráfico del learning rate
    number_epochs=int(len(history.history['loss']))
    lrs = 1e-8 * (10 ** (np.arange(number_epochs) / 20))
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

def manual_pred_plot(forecast,timesteps,measure,nombre_prueba,start,end):
    path_results='C:/Users/PABLO/Desktop/Results'
    path_carpeta_pruebas=f'{path_results}/Pruebas {forecast}h {timesteps}ts {measure}min'
    
    prueba1=nombre_prueba
    path_prueba1=f'{path_carpeta_pruebas}/{prueba1}'
    prediction = np.genfromtxt(f'{path_prueba1}/{prueba1}_pred.csv', delimiter=';')
    y_test = np.genfromtxt(f'{path_carpeta_pruebas}/y_test.csv', delimiter=';')
    plot_prediction(y_test,prediction,measure,path_prueba1,start=start,end=end)

def plot_comparative_loss(forecast,timesteps,measure,nombre_prueba1,nombre_prueba2):
    path_results='C:/Users/PABLO/Desktop/Results'
    path_carpeta_pruebas=f'{path_results}/Pruebas {forecast}h {timesteps}ts {measure}min'
    
    path_prueba1=f'{path_carpeta_pruebas}/{nombre_prueba1}'
    prueba1_loss = np.genfromtxt(f'{path_prueba1}/{nombre_prueba1}_loss.csv', delimiter=';')
    
    path_prueba2=f'{path_carpeta_pruebas}/{nombre_prueba2}'
    prueba2_loss = np.genfromtxt(f'{path_prueba2}/{nombre_prueba2}_loss.csv', delimiter=';')

    plt.figure(figsize=(10, 6))
    epochs = np.arange(prueba1_loss.shape[0],dtype="float32")
    number_epochs = int(prueba1_loss.shape[0])
    fig, axs = plt.subplots(2)
    names = ['RED 18 (20 LSTM)', 'RED 19 (60 LSTM)']
    axs[0].plot(epochs,prueba1_loss,color='g')
    axs[0].plot(epochs,prueba2_loss,color='y')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(names, loc='upper right')
    axs[1].plot(epochs[int(number_epochs/2):],prueba1_loss[int(number_epochs/2):],color='g')
    axs[1].plot(epochs[int(number_epochs/2):],prueba2_loss[int(number_epochs/2):],color='y')
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(names, loc='upper right')
    plt.savefig(f"{path_prueba1}/comp_loss.png")
    plt.savefig(f"{path_prueba2}/comp_loss.png")
    plt.show()
    

if __name__ == '__main__':
    manual_pred_plot(forecast=6,timesteps=60,measure=15,nombre_prueba='P36',start=180,end=250)
    # plot_comparative_loss(forecast=1,timesteps=15,measure=1,nombre_prueba1="P42",nombre_prueba2="42_R60")
