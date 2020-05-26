import os
import errno
import keras
import numpy as np
import time
import DataSetLoader as dsl, DataScaler, ModelManager as mm, Plotter as pl
import pandas as pd

class ModelTester:
    def __init__(self,sheetName):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.dataFormat = None #(forecast, timesteps, standarized, flattened_data)
        self.model = None
        self.currentModel = 0
        
        dfs = pd.read_excel('Datos_prueba/Pruebas.xlsx', sheet_name=None)
        self.table = dfs[sheetName]
        self.sheetName = sheetName
        self.resultsPath = f'C:/Users/PABLO/Desktop/Results/{sheetName}'
        self.ds = None
        
    def setData(self):
        dfs = pd.read_excel('Datos_prueba/Pruebas.xlsx', sheet_name=None)
        self.table = dfs[self.sheetName]
        loadedData = False
        #Setting dataset format
        datasetFormat = self.readDataFormat()
        if(datasetFormat != self.dataFormat):
            loadedData = True
            self.dataFormat = datasetFormat
            self.loadData() 
            print("The data has been loaded")
        else:
            print("No need to load")
        return loadedData
    
    def setModel(self):
        #Get model
        modelType = self.table['Model type'][self.currentModel]
        layers = self.table['Layers'][self.currentModel]
        units = int(layers.split(' ')[0])
        activation = self.table['Activation'][self.currentModel]
        out_activation = self.table['Output activation'][self.currentModel]
        flattened = self.dataFormat[3]
        if flattened is True:
            input_shape=[self.x_train.shape[1]]
        else:
            input_shape=[self.x_train.shape[1],self.x_train.shape[2]]
        dropout = self.table['Regularization'][self.currentModel]
        self.model = mm.getModel(modelType,units,activation,out_activation,input_shape,dropout)
        
        #Compile
        learning_rate = self.table['Learning rate'][self.currentModel] 
        optimizer_name = self.table['Optimizer'][self.currentModel] 
        optimizer = mm.getOptimizer(optimizer_name,learning_rate)
        loss_function = self.table['Loss'][self.currentModel]
        self.model.compile(loss=loss_function, optimizer=optimizer, metrics=['mae'])

 
    def trainModel(self):
        print("Training model...")
        batch = int(self.table['Batch size'][self.currentModel])
        number_epochs = int(self.table['Epochs'][self.currentModel])
        history = self.model.fit(self.x_train, self.y_train, validation_split=0.05 ,batch_size=batch, epochs=number_epochs, verbose=0) #entrenar la red #validar
        print("Training complete")
        return history
    
    def predict(self):
        predictions = self.model.predict(self.x_test, verbose=0)
        predictions = self.ds.inverse_scale_data(predictions)
        np.savetxt("C:/Users/PABLO/Google Drive/UPM/fi UPM/4 - Cuarto/Trabajo de Fin de Grado/Cabahla/src/Predictions/last_pred.csv", predictions, delimiter=";")
        return predictions
    
    def savePlotsAndResults(self,history,predictions,time):
        modelID = self.table['ID'][self.currentModel]
        newFolder(self.resultsPath)
        path=self.resultsPath+f"/{modelID}"
        newFolder(path)
        number_epochs= self.table['Epochs'][self.currentModel]
        # Plots
        pl.plot_loss(number_epochs,history,path)
        pl.plot_precision(number_epochs,history,path)
        pl.plot_prediction(self.y_test, predictions,path,start=100,end=1000)
        pl.plot_prediction(self.y_test, predictions,path,start=600,end=800)
        pl.plot_prediction(self.y_test, predictions,path,start=400,end=600)
        # Results
        np.savetxt(f"{path}/{modelID}_pred.csv", predictions, delimiter=";")
        np.savetxt(f"{path}/{modelID}_loss.csv", history.history['loss'], delimiter=";")
        overfitting = pl.overfitting(history)
        mae = pl.mae(self.y_test, predictions)
        self.table['MAE'][self.currentModel] = mae
        self.table['Overfitting'][self.currentModel] = overfitting
        self.table['Time'][self.currentModel] = time
        # Saving model info into txt file
        fich_res = open(f"{path}/info_{modelID}.txt","w+")
        fich_res.write(str(self.table.iloc[self.currentModel]))
        fich_res.close()
        # Saving into excel file
        self.table.to_excel("Datos_prueba/Pruebas.xlsx",sheet_name=self.sheetName, index=False)
        print("Results are ready")
        
    def saveBaselineResults(self):
        modelID = self.table['ID'][self.currentModel]
        newFolder(self.resultsPath)
        path=self.resultsPath+f"/{modelID}"
        newFolder(path)
        predictions = np.genfromtxt('Predictions/moving_average_prediction.csv', delimiter=';')
        predictions = dsl.generateTestSet(predictions)
        predictions = np.expand_dims(predictions, axis=1)
        # Plots
        pl.plot_prediction(self.y_test, predictions,path,start=100,end=1000)
        pl.plot_prediction(self.y_test, predictions,path,start=600,end=800)
        pl.plot_prediction(self.y_test, predictions,path,start=400,end=600)
        # Results
        mae = pl.mae(self.y_test, predictions)
        self.table['MAE'][self.currentModel] = mae
        # Saving into txt file
        fich_res = open(f"{path}/info_{modelID}.txt","w+")
        fich_res.write(str(self.table.iloc[self.currentModel]))
        fich_res.close()
        # Saving into excel file
        self.table.to_excel("Datos_prueba/Pruebas.xlsx",sheet_name=self.sheetName, index=False)
        
    def isBaseline(self):
        return self.table['Architecture'][self.currentModel] in ('Naive foracast','Weighted av. forecast')
    
    def readDataFormat(self):
        forecast = int(self.table['Forecast'][self.currentModel])
        timestep = int(self.table['Timestep'][self.currentModel])
        standarized = self.table['Standarized'][self.currentModel]
        if standarized == 'Yes':
            standarized = True
        else:
            standarized = False
        architecture = self.table['Architecture'][self.currentModel]
        if architecture == 'Fully Connected':
            flattened_data=True
        else: 
            flattened_data=False
        return (forecast, timestep, standarized, flattened_data)
    
    def loadData(self):
        (forecast, timesteps, standarized, flattened_data) = self.dataFormat     
        #Loading dataset based on the format
        sensors_data = dsl.get_sensors_data(standarized)
        moving_avg_pred = dsl.moving_average_forecast(sensors_data,timesteps,forecast)
        np.savetxt("Predictions/moving_average_prediction.csv", moving_avg_pred, delimiter=";")
   
        x_series,y_series = dsl.window_data(sensors_data,timesteps,forecast)
        if flattened_data:
            x_series = dsl.flatten_windowed_data(x_series)
        (self.x_train, self.y_train),(self.x_test, self.y_test) = dsl.generateTrainTestSet(x_series,y_series,split=9/10)

        
    def scaleData(self):
        #Scaling data
        self.ds = DataScaler.DataScaler(feature_range=(0, 1))
        self.x_train = self.ds.fit_scale_data(self.x_train)
        self.y_train = self.ds.scale_data(self.y_train)
        self.x_test = self.ds.scale_data(self.x_test)
        
    def inverseScaleData(self):
        self.x_train = self.ds.inverse_scale_data(self.x_train)
        self.y_train = self.ds.inverse_scale_data(self.y_train)
        self.x_test = self.ds.inverse_scale_data(self.x_test)
        
    def nextModel(self):
        self.currentModel = self.currentModel + 1
 
def newFolder(path):
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('La carpeta '+path+' ya existe')
        
if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    
    global_start_time = time.time()
    mt = ModelTester("Pruebas 3h 30ts")
    number_of_models = mt.table['ID'].shape[0]
    while(mt.currentModel != number_of_models): 
        print(f"################ {mt.currentModel}: PRUEBA {mt.table['ID'][mt.currentModel]} ###############")
        start_time = time.time()
        keras.backend.clear_session()
        loadedData = mt.setData()
        mt.scaleData()
        if mt.isBaseline():
            mt.saveBaselineResults()
        else:
            mt.setModel()
            history=mt.trainModel()
            predictions = mt.predict()
            finish_time = time.time() - start_time
            mt.savePlotsAndResults(history,predictions,finish_time)
            
        mt.inverseScaleData()
        mt.nextModel()
    np.savetxt(f"{mt.resultsPath}/y_test.csv", mt.y_test, delimiter=";")
    print(f"\nTiempo total de ejecución = {(time.time() - global_start_time):.2f} seconds")

