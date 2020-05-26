import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataScaler:
    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range)
        
    def fit_scale_data(self,data):
        data_prov = data.reshape([-1,1])
        data_prov_scaled = self.scaler.fit_transform(data_prov)
        data_scaled = data_prov_scaled.reshape(data.shape)
        return data_scaled
    
    def scale_data(self,data):
        data_prov = data.reshape([-1,1])
        data_prov_scaled = self.scaler.transform(data_prov)
        data_scaled = data_prov_scaled.reshape(data.shape)
        return data_scaled

    def inverse_scale_data(self,data):
        data_prov = data.reshape([-1,1])
        data_prov_scaled = self.scaler.inverse_transform(data_prov)
        data = data_prov_scaled.reshape(data.shape)
        return data

   
    
if __name__ == '__main__':
    ds = DataScaler((-1,1))
    a = np.array(np.mat('1 2; 3 4'))   
    b = ds.fit_scale_data(a)
    c = ds.inverse_scale_data(b)