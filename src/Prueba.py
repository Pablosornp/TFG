# importing pandas as pd 
import pandas as pd 
   
# open hdf5 file for reading
radiation = pd.HDFStore('Datos_prueba/filled_01m.h5', mode='r')

print(radiation.keys())
