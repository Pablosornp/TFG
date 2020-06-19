import h5py
import tables
import matplotlib.pyplot as plt
import numpy as np
import time

base_path = 'Datos_prueba'
stand_file = 'Datos_prueba/stand_1m.h5'
filled_file = 'Datos_prueba/filled_1m.h5'
sensors = {'ap01':0, 'ap03':1, 'ap04':2, 'ap05':3, 'ap06':4, 'ap07':5, 'dh01':6, 'dh02':7, 'dh03':8, 'dh04':9, 'dh05':10, 'dh06':11, 'dh07':12, 'dh08':13, 'dh09':14, 'dh10':15, 'dh11':16}
    
def get_sensors_data(standarized=False, measure_frec=1):
    if standarized:
        path_file = f"{base_path}/stand_"
    else:
        path_file = f"{base_path}/filled_"
    path_file = f"{path_file}{measure_frec}m.h5"
    
    radiation = h5py.File(path_file, 'r')
    features=[]
    for year in radiation:
        for month in radiation[year]:
            for day in radiation[year][month]:
                for measure in radiation[year][month][day]:
                    measures=[]
                    for sensor in sensors:
                        measures.append(measure[sensors[sensor]+1])
                    features.append(measures)

    input_data = np.asarray(features,'float32')
    # print("Sensores cargados:", input_data.shape)
    return input_data
    
    
def window_data(input_data, measure_frec=1, timesteps=5, forecast=1, sensorToPredict='dh06'):
    X=[]
    y=[]
    steps_ahead = int((forecast*60)/measure_frec)
    for i in range(timesteps-1,len(input_data)-steps_ahead):     
        t=[]
        for j in range(-(timesteps-1),1):
            t.append(input_data[i+j,:].T)

        X.append(t)
        y.append(input_data[i+ steps_ahead,sensors[sensorToPredict]])
    X = np.asarray(X,'float32')
    y = np.asarray(y,'float32')
    y = np.expand_dims(y, axis=1)
    print("Datos preparados:", X.shape, y.shape)
    return X,y
    

def flatten_windowed_data(windowed_data):
    flattened_data = []
    for window in windowed_data:
        flattened_data.append(window.T.flatten())
    flattened_data=np.asarray(flattened_data, 'float32')
    print("Flattened data:", flattened_data.shape)
    return flattened_data

def naive_forecast(input_data,measure_frec=1,timesteps=5,forecast=1,sensorToPredict='dh06'):
    prediction=[]
    steps_ahead = int((forecast*60)/measure_frec)
    for i in range(timesteps-1,len(input_data)-steps_ahead):
        prediction.append(input_data[i,sensors[sensorToPredict]])
    prediction = np.asarray(prediction,'float32')
    return prediction
    
def moving_average_forecast(input_data,measure_frec=1,timesteps=5,forecast=1,sensorToPredict='dh06'):
    prediction = []
    steps_ahead = int((forecast*60)/measure_frec)
    for i in range(timesteps-1,len(input_data)-steps_ahead):
         prediction.append(input_data[i-(timesteps-1):i,sensors[sensorToPredict]].mean())
    prediction = np.asarray(prediction,'float32')
    return prediction
                
def generateTrainTestSet(x_series,y_series,split=9/10):
    total_time = x_series.shape[0]
    split_time = int(total_time*(9/10))      
    x_train = x_series[:split_time]
    y_train = y_series[:split_time]  
    x_test = x_series[split_time:]
    y_test = y_series[split_time:]
    return (x_train, y_train),(x_test, y_test)

def generateTestSet(y_series,split=9/10):
    total_time = y_series.shape[0]
    split_time = int(total_time*(9/10))      
    y_test = y_series[split_time:]
    return y_test
    
def toCSV(data, name):
    np.savetxt("C:/Users/PABLO/Desktop/Datos_prueba/"+name+".csv", data, delimiter=";")
    
def plot_sensor_data(sensor='dh06', start=0, end=None):
    plt.figure(figsize=(10, 5))
    series = get_sensors_data()
    time_ = np.arange(series.shape[0], dtype="float32")
    plt.plot(time_[start:end], series[start:end,sensors[sensor]])
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)   
    plt.show()
    

