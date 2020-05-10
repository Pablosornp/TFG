import h5py
import tables
import matplotlib.pyplot as plt
import numpy as np
import time


class DataSetLoader:
    def __init__(self):
        self.stand_file = 'Datos_prueba/stand_01m.h5'
        self.filled_file = 'Datos_prueba/filled_01m.h5'
        self.sensors = {'ap01':1, 'ap03':2, 'ap04':3, 'ap05':4, 'ap06':5, 'ap07':6, 'dh01':7, 'dh02':8, 'dh03':9, 'dh04':10, 'dh05':11, 'dh06':12, 'dh07':13, 'dh08':14, 'dh09':15, 'dh10':16, 'dh11':17}
    
    def get_sensors_data(self,all_sensors=False, standarized=False):
        if standarized:
            radiation = h5py.File(self.stand_file, 'r')
        else:
            radiation = h5py.File(self.filled_file, 'r')
        features=[]
        if all_sensors is False:   
            for year in radiation:
                for month in radiation[year]:
                    for day in radiation[year][month]:
                        for measure in radiation[year][month][day]:
                            features.append([measure[self.sensors['dh06']],measure[self.sensors['ap07']]])
        else:
            for year in radiation:
                for month in radiation[year]:
                    for day in radiation[year][month]:
                        for measure in radiation[year][month][day]:
                            measures=[]
                            for sensor in self.sensors:
                                measures.append(measure[self.sensors[sensor]])
                            features.append(measures)
    
        input_data = np.asarray(features,'float32')
        print("Sensores cargados:", input_data.shape)
        return input_data
    
    
    def window_data(self,input_data,window_size=5,lookforward=1):
        X=[]
        y=[]
        for i in range(window_size-1,len(input_data)-lookforward):
            t=[]
            for j in range(-(window_size-1),1):
                t.append(input_data[i+j,:].T)
    
            X.append(t)
            y.append(input_data[i+ lookforward,11])
        X = np.asarray(X,'float32')
        y = np.asarray(y,'float32')
        y = np.expand_dims(y, axis=1)
        print("Datos preparados:", X.shape, y.shape)
        return X,y
    
    def naive_forecast(self,input_data,window_size=5,lookforward=1):
        y_pred=[]
        for i in range(window_size-1,len(input_data)-lookforward):
            y_pred.append(input_data[i,11])
        y_pred = np.asarray(y_pred,'float32')
        return y_pred
    
    def moving_average_forecast(self, input_data, window_size=5,lookforward=1):
        y_pred = []
        for i in range(window_size-1,len(input_data)-lookforward):
             y_pred.append(input_data[i-(window_size-1):i,11].mean())
        y_pred = np.asarray(y_pred,'float32')
        return y_pred
        
    
    def flatten_windowed_data(self,windowed_data):
        flattened_data = []
        for window in windowed_data:
            flattened_data.append(window.T.flatten())
        flattened_data=np.asarray(flattened_data, 'float32')
        print("Flattened data:", flattened_data.shape)
        return flattened_data
    
    def generateTrainTestSet(self,x_series,y_series,split=9/10):
        total_time = x_series.shape[0]
        split_time = int(total_time*(9/10))      
        x_train = x_series[:split_time]
        y_train = y_series[:split_time]  
        x_test = x_series[split_time:]
        y_test = y_series[split_time:]
        return (x_train, y_train),(x_test, y_test)

    def generateTestSet(self,y_series,split=9/10):
        total_time = y_series.shape[0]
        split_time = int(total_time*(9/10))      
        y_test = y_series[split_time:]
        return y_test
    
    def toCSV(self, data, name):
        np.savetxt("C:/Users/PABLO/Desktop/Datos_prueba/"+name+".csv", data, delimiter=";")

def generateCSVs(standarized=True,window_size=5,lookforward=60):
    dsl = DataSetLoader()
    print("Generating CSV with raw data")
    input_data = dsl.get_sensors_data(True,standarized)
    if standarized:
        name="stand_01m"
    else:
        name="filled_01m"    
    dsl.toCSV(input_data, name)
    
    print("Generating CSV with labels")
    X,y = dsl.window_data(input_data,window_size,lookforward)
    name1="y_"+name+"_ts"+str(window_size)+"_pred"+str(lookforward)
    dsl.toCSV(y, name1)
    
    print("Generating CSV with naive forecast")
    y_pred = dsl.naive_forecast(input_data,window_size,lookforward)
    np.savetxt("Predictions/naive_prediction.csv", y_pred, delimiter=";")
    
    print("Generating CSV with moving average forecast")
    y_pred = dsl.moving_average_forecast(input_data,window_size,lookforward)
    np.savetxt("Predictions/moving_average_prediction.csv", y_pred, delimiter=";")
    
    print("Generating CSV flattened features")
    flattened_data = dsl.flatten_windowed_data(X)
    name2="X_flat_"+name+"_ts"+str(window_size)+"_pred"+str(lookforward)
    dsl.toCSV(flattened_data, name2)
    
    
def plot_series(format="-", start=0, end=None):
    dsl = DataSetLoader()
    series = dsl.get_sensors_data()
    time = np.arange(series.shape[0], dtype="float32")
    plt.plot(time[start:end], series[start:end,0], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)   
    
#Funcion Main
def main():
    start_time = time.time()
    dsl = DataSetLoader()
    input_data = dsl.get_sensors_data(True,False)
    x_series, y_series = dsl.window_data(input_data,window_size=5,lookforward=1)

    
    # generateCSVs(False,5,60)
    # plot_series(start=0,end=900*2)
    
    print(f"\nTiempo de ejecución = {(time.time() - start_time)} seconds")
    
if __name__ == '__main__':
    main()
