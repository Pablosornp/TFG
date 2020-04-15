import h5py
import tables

import numpy as np


import time

class DataSetLoader:
    def __init__(self):
        self.stand_file = 'Datos_prueba/stand_01m.h5'
        self.filled_file = 'Datos_prueba/filled_01m.h5'
    
    
    def get_sensors_data(self,all_sensors=False, standarized=False):
        sensors = {'ap01':1, 'ap03':2, 'ap04':3, 'ap05':4, 'ap06':5, 'ap07':6, 'dh01':7, 'dh02':8, 'dh03':9, 'dh04':10, 'dh05':11, 'dh06':12, 'dh07':13, 'dh08':14, 'dh09':15, 'dh10':16, 'dh11':17}
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
                            features.append([measure[sensors['dh06']],measure[sensors['ap07']]])
        else:
            for year in radiation:
                for month in radiation[year]:
                    for day in radiation[year][month]:
                        for measure in radiation[year][month][day]:
                            measures=[]
                            for sensor in sensors:
                                measures.append(measure[sensors[sensor]])
                            features.append(measures)
    
        input_data = np.asarray(features,'float64')
        print("Sensores cargados:", input_data.shape)
        return input_data
    
    
    def window_data(self,input_data,timesteps=5,lookback=1):
        X=[]
        y=[]
        for i in range(timesteps-1,len(input_data)-lookback):
            t=[]
            for j in range(-(timesteps-1),1):
                t.append(input_data[i+j,:].T)
    
            X.append(t)
            y.append(input_data[i+ lookback,0])
        X = np.asarray(X,'float64')
        y = np.asarray(y,'float64')
        print("Datos preparados", X.shape, y.shape)
        return X,y
    
    def flatten_windowed_data(self,windowed_data):
        flattened_data = []
        for window in windowed_data:
            flattened_data.append(window.T.flatten())
        flattened_data=np.asarray(flattened_data, 'float64')
        print("Datos aplanados", flattened_data.shape)
        return flattened_data
    
#Funcion Main
def main():
    dsl = DataSetLoader()
    start_time = time.time()
    input_data = dsl.get_sensors_data()
    X,y = dsl.window_data(input_data)
    flat = dsl.flatten_windowed_data(X)
    print(f"\nTiempo de ejecución = {(time.time() - start_time)} seconds")
    
if __name__ == '__main__':
    main()
