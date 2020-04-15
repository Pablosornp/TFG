import  keras
from  keras  import  backend  as  K

from  keras.datasets  import  mnist
from  keras.layers  import  Dense,  Flatten
from  keras.layers  import  Conv2D,  MaxPooling2D
from  keras.models  import  Sequential

import  tensorflow  as  tf
import  numpy  as  np
import  matplotlib.pylab  as  plt
import  time

###START###
start_t  =  time.time()
from  data_loader  import  load_training
training_data,  validation_data,  test_data  =  load_training("10s")
loading_t  =  time.time()

#  Python  optimisation  variables
learning_rate  =  0.1
epochs = 200
batch_size  =  500
reduccion_lr = 60 #learning_rate/epochs coste_elegido  =  "rmse"  #  'rmse',  'mae'
optimizador  =  "adagrad"  #  'gradient_descent',  'adagrad'



model  =  Sequential()
model.add(Dense(7,input_shape=(9,),activation='tanh',name="Hidden1"))
model.add(Dense(4,input_shape=(7,),activation='tanh',name="Hidden2"))
model.add(Dense(1,input_shape=(4,),activation='tanh',name="Output"))

def  rmse(y_true,  y_pred):
    return  K.sqrt(K.mean(K.square(y_pred  -  y_true)))

def  mean_absolute_error(y_true,y_pred):
    return K.mean(K.abs(y_pred - y_true))

adagrad = keras.optimizers.Adagrad(lr=learning_rate)
sgd  =  keras.optimizers.SGD(lr=learning_rate)
rmsprop  =  keras.optimizers.RMSprop(lr=learning_rate)

model.compile(loss='mse', optimizer='adadelta',metrics=['mae','mse',rmse])


print("Transformando")
reference   =   training_data[1]   #   Usamos   los   Y   de   entrenamiento   como referencia  para  la  normalización  tanh,  ya  que  es  el  dataset  mas  extenso que  no  tiene  datos  repetidos
media_ref  =  np.mean(reference,axis=0)
std_ref  =  np.std(reference,axis=0)

training_data_I	=	0.5	*	(np.tanh(0.01	*	((training_data[0]	- np.mean(reference,axis=0))  /  np.std(reference,axis=0)))  +  1)
validation_data_I	=	0.5	*	(np.tanh(0.01	*	((validation_data[0]	- np.mean(reference,axis=0))  /  np.std(reference,axis=0)))  +  1)
test_data_I	=	0.5	*	(np.tanh(0.01	*	((test_data[0]	- np.mean(reference,axis=0))  /  np.std(reference,axis=0)))  +  1)
training_data_O	=	0.5	*	(np.tanh(0.01	*	((training_data[1]	- np.mean(reference,axis=0))  /  np.std(reference,axis=0)))  +  1)
validation_data_O	=	0.5	*	(np.tanh(0.01	*	((validation_data[1]	- np.mean(reference,axis=0))  /  np.std(reference,axis=0)))  +  1)
test_data_O	=	0.5	*	(np.tanh(0.01	*	((test_data[1]	- np.mean(reference,axis=0))  /  np.std(reference,axis=0)))  +  1)


class  AccuracyHistory(keras.callbacks.Callback):
    def  on_train_begin(self,  logs={}):
        self.acc = []

    def  on_epoch_end(self,  batch,  logs={}):
        self.acc.append(logs.get(mean_absolute_error))

history  =  AccuracyHistory()

lr_sched = keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5,patience=r educcion_lr/4,verbose=1,mode='min',min_lr=0.001,min_delta=0.0000001)
early_stop	= keras.callbacks.EarlyStopping(monitor='loss',patience=reduccion_lr,ver bose=1,mode='min',min_delta=0.0000001)


print("Entrenamiento")
snn	=	model.fit(x=training_data_I,	y=training_data_O, batch_size=batch_size,	epochs=epochs,		verbose=1, validation_data=(validation_data_I,   validation_data_O),   shuffle=True, callbacks=[history,lr_sched,early_stop])
finish_t  =  time.time()
score	=	model.evaluate(x=test_data_I,	y=test_data_O, batch_size=batch_size,  verbose=1)
output  =  model.predict(x=test_data_I,  batch_size=batch_size,  verbose=1) testing_t  =  time.time()
output_denormalized	= np.mean(test_data[1],axis=0)+(np.std(test_data[1],axis=0)*(np.arctanh( (output  /  0.5)-1))  /  0.01)


plt.figure(0)
plt.plot(snn.history['mean_squared_error'])
plt.plot(snn.history['mean_absolute_error'])
#plt.plot(snn.history['mean_absolute_percentage_error'])
#plt.plot(snn.history['cosine_proximity'])
plt.plot(snn.history['rmse']) plt.plot(snn.history['loss'],'r')
#plt.xticks(np.arange(0,  11,  2.0))
#plt.rcParams['figure.figsize']  =  (8,  6)
plt.xlabel("Num  of  Epochs")
plt.ylabel("Accuracy") plt.title("Training  Cost")
plt.legend(['mse','mae','rmse','cost'])

plt.figure(1)
plt.plot(snn.history['loss'],'r')
plt.plot(snn.history['val_loss'],'g')
#plt.xticks(np.arange(0,  11,  2.0))
#plt.rcParams['figure.figsize']  =  (8,  6)
plt.xlabel("Num  of  Epochs")
plt.ylabel("Loss")
plt.title("Training  Loss  vs  Validation  Loss")
plt.legend(['train','validation'])



plt.figure(2)
plt.plot(test_data_O[:400,:],label="Correct")
plt.plot(output[:400,:],label="Calculated")
plt.xlabel('Record')
plt.ylabel('Value')
plt.title("Normalized  Output")
#plt.ylim(top=1)
plt.legend(loc=2)

plt.figure(3)
plt.plot(test_data[1][:400,:],label="Correct")
plt.plot(output_denormalized[:400,:],label="Calculated")
plt.xlabel('Record')
plt.ylabel('Value')
plt.title("Denormalized  Output")
#plt.ylim(top=1)
#plt.legend(loc=2)

plt.figure(4)
plt.scatter(test_data[1][:400,:],output_denormalized[:400,:])
plt.xlabel('Correct')
plt.ylabel('Calculated')
plt.title("Denormalized  Output  Scatter")
#plt.ylim(top=1)
#plt.legend(loc=2)
plt.show()
print("Test  results")

for  i  in  range(0,len(score)):
    print(model.metrics_names[i]+":  ","{:.10f}".format(score[i]))


print()
print("Datasets  loading  time:  "+str(loading_t  -  start_t)+"  seconds")
print("Training  time:  "+str(finish_t  -  loading_t)+"  seconds")
print("Testing	time:  "+str(testing_t  -  finish_t)+"  seconds")
