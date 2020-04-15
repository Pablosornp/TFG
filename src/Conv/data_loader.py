import  pandas  as  pd
import  numpy  as  np

path="nachog\\matrices_def"

def  load(filename,sep):
    dataset  =  pd.read_csv(filename,sep=sep)
    values = list(dataset.columns.values)
    X  =  np.array(dataset,dtype='float32') print("Data loaded: "+str(X.shape[0])+" rows")
    return X

def  load_station(filename,sep,station_name,n_cols):
    cols=[station_name+str(i) for  i  in  range(0,n_cols)]
    dataset  =  pd.read_csv(filename,sep=sep)
#	print(dataset)
    values = list(dataset.columns.values)
    dataset  =  dataset[cols]
    X  =  np.array(dataset,dtype='float32')
    print("Station "+station_name+" loaded: "+str(X.shape[0])+" rows")
    return X

def  load_training(pred):
    X_train	= load_station(path+"\\X_train_"+pred+".csv",',','dh6_rel_ns',9) Y_train  =  load(path+"\\Y_train_"+pred+".csv",',')
    split  =  int(Y_train.shape[0]*0.17)
    X_tr,X_va  =  X_train[split:,:],X_train[:split,:] Y_tr,Y_va  =  Y_train[split:,:],Y_train[:split,:]
    X_te=load_station(path+"\\X_test_"+pred+".csv",',','dh6_rel_ns',9)
    Y_te  =  load(path+"\\Y_test_"+pred+".csv",',')
    return ((X_tr,Y_tr),(X_va,Y_va),(X_te,Y_te))


def  load_toy():
    x_tr=load_station(path+"\\..\\X_ejemplo_4muestras.csv",';','dh6_rel_ns',4)
    y_tr = load(path+"\\..\\Y_ejemplo_4muestras.csv",';')
    return (x_tr, y_tr)
