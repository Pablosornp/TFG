import  h5py

import  numpy  as  np 
import  tables 
import keras

filename="C:\\Users\\dan_1\\pyworkspace\\cabahlanet\\data\\stand_carray.h5"

carray  =  h5py.File(filename,'r')
datos  =  []
media_datos = 0.7848976
std_datos  =  0.3395678


def  init_loader():
    global media_datos
    global  std_datos
    i=0
    print("Cargando  datos")
    for year in carray:
        datos.append([])
        j=0
        for mon in carray[year]:
            datos[i].append([])
            k=0
            for  day  in  carray[year][mon]:
                datos[i][j].append(carray[year][mon][day])
                k+=1
            j+=1
        i+=1
    print("Carga  terminada")

def normalizar(vector):
    normavec    =    0.5    *    (np.tanh(0.01    *    ((vector    -    media_datos)    / std_datos))  +  1)
    return normavec


def denormalizar(vector):
    denormavec  =  media_datos+(std_datos*(np.arctanh((vector  /  0.5)-1))/ 0.01)
    return denormavec



def  vector2mat(vector):
    normavec = normalizar(vector)
    mat  =  np.zeros((4,4,1))
    mat[0][0][0]  =  normavec[7]
    mat[0][1][0]  =  normavec[8]
    mat[0][2][0]  =  normavec[11]
    mat[0][3][0]  =  normavec[6]
    mat[1][0][0]  =  normavec[13]
    mat[1][1][0]  =  normavec[10]
    mat[1][2][0]  =  normavec[1]
    mat[1][3][0]  =  normavec[3]
    mat[2][0][0]  =  normavec[15]
    mat[2][1][0]  =  normavec[12]
    mat[2][2][0]  =  normavec[16]
    mat[2][3][0]  =  normavec[9]
    mat[3][0][0]  =  normavec[14]
    mat[3][1][0]  =  normavec[17]
    mat[3][2][0]  =  normavec[4]
    mat[3][3][0]  =  normavec[5]
    return  mat


class  DataGenerator(keras.utils.Sequence):
    def __init__(self,idList,batch_size=10,dim=(4,4,1),shuffle=True):
        self.batch_size=batch_size
        self.dim=dim
        self.indexes = idList
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return  int(np.floor(len(self.indexes)/self.batch_size))


    def __getitem__(self,  index):
        pre_batch=self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim))
        for  i,element  in  enumerate(pre_batch):
            X[i,]=vector2mat(datos[element[0]][element[1]][element[2]][element[3]])
            y[i,]=vector2mat(datos[element[0]][element[1]][element[2]][element[3]+1])
            return  (X,  y)

    def  on_epoch_end(self):
        if  self.shuffle:
            np.random.shuffle(self.indexes)


def  load_dataset(group=None,val=None,test=None):
    init_loader()
    dias = []
    dias_val  =  []
    dias_test = []
    idList = []
    valList = []
    testList = []
    print("Creando  listas  de  indices")


    def  get_all(name):
        if len(name)>7:
            if group==None  or  (group!=None  and  group  in  name):
                if  val  !=  None  and  val  in  name:
                    dias_val.append(name)   
            elif test != None and test in name:
                dias_test.append(name)
            else:
                dias.append(name)


    carray.visit(get_all)
    for  f  in  dias:
        y,m,d=parseIndex(f)
        for  i  in  range(0,54000):
            idList.append((y,m,d,i))
    for  f  in  dias_val:
         y,m,d=parseIndex(f)
         for  i  in  range(0,54000):
             valList.append((y,m,d,i))
    for  f  in  dias_test:
        y,m,d=parseIndex(f)
        for  i  in  range(0,54000):
            testList.append((y,m,d,i))
    print("Listas terminadas")
    return idList,valList,testList

def load_testdata(size,first=20000):
    oct111 = datos[1][9][0]
    print(oct111)
    X  = []
    y  = []
    for  i  in  range(first,first+size):
        X.append(vector2mat(oct111[i]))
        y.append(oct111[i+1][9])
    return (np.asarray(X),y)


def parseIndex(string):
    indices  =  string.split("/")
    mes2010 = {"03":0, "04":1, "05":2, "06":3, "07":4, "08":5, "09":6,"10":7,  "11":8,  "12":9}
    mes2011 = {"01":0, "02":1, "03":2, "04":3, "05":4, "06":5, "07":6,"08":7,  "09":8,  "10":9}
    if  indices[0]  ==  '2010':
        year = 0
        mon  =  mes2010.get(indices[1],"Error  de  mes")
        if mon == 0:
            day = int(indices[2])-18
        else:
            day = int(indices[2])-1
    elif  indices[0]  ==  '2011':
        year = 1
        mon  =  mes2011.get(indices[1],"Error  de  mes")
        day = int(indices[2])-1
    return year, mon, day
