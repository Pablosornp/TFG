import h5py
from keras.utils.io_utils import HDF5Matrix

#X_data = HDF5Matrix('Datos_prueba/filled_carray.h5','2010/10/20');
f = h5py.File('Datos_prueba/filled_carray.h5', 'r')
print(list(f.keys()))
dset = f['2010']
