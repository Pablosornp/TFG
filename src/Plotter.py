import matplotlib.pyplot as plt
import numpy as np

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
def plot_loss(number_epochs,history):
    plt.figure(figsize=(10, 6))
    epochs=np.arange(number_epochs)
    fig, axs = plt.subplots(2)
    axs[0].plot(epochs,history.history['loss'],epochs,history.history['val_loss'])
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'validation'], loc='upper left')
    axs[1].plot(epochs[10:],history.history['loss'][10:],epochs[10:],history.history['val_loss'][10:])
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'validation'], loc='upper left')
    plt.savefig('C:/Users/PABLO/Desktop/Plots/_loss.png')
    plt.show()
    
def plot_prediction(y_test,prediction,start=0,end=None):
    time_valid = np.arange(y_test.shape[0],dtype="float32")
    plt.figure(figsize=(10, 6))
    plt.plot(time_valid[start:end],y_test[start:end])
    plt.plot(time_valid[start:end],prediction[start:end])
    plt.xlabel("Irradiance")
    plt.ylabel("Time (min)")
    plt.legend(['Valor real', 'Predicción'], loc='upper left')
    plt.grid(True)
    plt.savefig(f"C:/Users/PABLO/Desktop/Plots/pred_{start}_{end}.png")
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
