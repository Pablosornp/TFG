import  keras
from  keras  import  backend  as  K
from  keras.datasets  import  mnist
from  keras.layers  import  Dense,  Flatten,  Reshape
from  keras.layers  import  Conv2D,  MaxPooling2D
from  keras.models  import  Sequential



import  numpy  as  np

import  matplotlib.pylab  as  plt
import  time
from	h5_loader	import DataGenerator,load_dataset,load_testdata,denormalizar



###START###
start_t  =  time.time()

loading_t  =  time.time()


#  Python  optimisation  variables
learning_rate  =  1
epochs = 30
batch_size  =  200
steps  =  500
reduccion_lr = 12 #learning_rate/epochs
coste_elegido  =  "rmse"  #  'rmse',  'mae'
optimizador  =  "adagrad"  #  'gradient_descent',  'adagrad'
input_shape  =  (4,4,1)




model  =  Sequential()
model.add(Conv2D(1,  kernel_size=(2,2),  strides=(1,  1),activation='relu',input_shape=input_shape,data_format='channels_last',name="Conv1"))
#model.add(MaxPooling2D(pool_size=(1,	1),	strides=(1, 1),data_format='channels_last',name="Pool1"))
model.add(Conv2D(1,	kernel_size=(2,	2), activation='relu',data_format='channels_last',name="Conv2"))
#model.add(Conv2D(1,	kernel_size=(2,	2), activation='relu',data_format='channels_last',name="Conv3"))
#model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_last',name="Pool2"))
model.add(Flatten(name="Flat",data_format='channels_last')) model.add(Dense(32,  activation='tanh',name="Dense1")) model.add(Dense(16,  activation='tanh',name="Dense2")) model.add(Reshape((4,4,1)))
#model.add(Dense(num_classes,  activation='softmax'))


def  rmse(y_true,  y_pred):
    return  K.sqrt(K.mean(K.square(y_pred  -  y_true)))

def  mean_absolute_error(y_true,y_pred):
    return K.mean(K.abs(y_pred - y_true))


adagrad = keras.optimizers.Adagrad(lr=learning_rate)
sgd  =  keras.optimizers.SGD(lr=learning_rate)
rmsprop  =  keras.optimizers.RMSprop(lr=learning_rate)

model.compile(loss='mse', optimizer='adadelta',metrics=['mae','mse',rmse])


print("Transformando")


trainList,valList,testList	=
load_dataset(val='2011/09',test='2011/10')
training_gen  =  DataGenerator(trainList,  batch_size=batch_size)
validation_gen  =  DataGenerator(valList,batch_size=batch_size)
test_gen  =  DataGenerator(testList,batch_size=batch_size,shuffle=False)


class  AccuracyHistory(keras.callbacks.Callback):
    def  on_train_begin(self,  logs={}):
        self.acc = []

    def  on_epoch_end(self,  batch,  logs={}):
        self.acc.append(logs.get(mean_absolute_error))

history  =  AccuracyHistory()
lr_sched	= keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5,patience=r educcion_lr/3,verbose=1,mode='min',min_lr=0.001,min_delta=0.01)
early_stop	= keras.callbacks.EarlyStopping(monitor='loss',patience=reduccion_lr,ver bose=1,mode='min',min_delta=0.01)

print("Entrenamiento")

start_t  =  time.time()
snn  =  model.fit_generator(training_gen,epochs=epochs,steps_per_epoch=steps, verbose=1, validation_data=validation_gen,
validation_steps=steps,
#	shuffle=True,
callbacks=[history
#	,lr_sched
#	,early_stop
]
)
finish_t  =  time.time()


score=model.evaluate_generator(test_gen, verbose=1)
test_x,test_y=load_testdata(batch_size)

output  =  model.predict(test_x,  batch_size=batch_size,  verbose=1)
testing_t  =  time.time()

output_vector  =  [denormalizar(i)[2][3][0]  for  i  in  output]

data_acc_x,data_acc_y  =  load_testdata(50000,0)
output_acc  =  model.predict(data_acc_x,verbose=1)
denorm_acc  =  [denormalizar(i)[2][3][0]  for  i  in  output_acc]
accuracy	=	np.mean(np.abs(np.asarray(data_acc_y) - np.asarray(denorm_acc)))


plt.figure(0)
plt.plot(snn.history['mean_squared_error'])
plt.plot(snn.history['mean_absolute_error'])

plt.plot(snn.history['rmse'])
plt.plot(snn.history['loss'],'r')

plt.xlabel("Num  of  Epochs")
plt.ylabel("Accuracy")
plt.title("Training  Cost")
plt.legend(['mse','mae','rmse','cost'])

#
plt.figure(1)
plt.plot(snn.history['loss'],'r')
plt.plot(snn.history['val_loss'],'g')
plt.ylim(top=1)

plt.xlabel("Num  of  Epochs")
plt.ylabel("Loss")
plt.title("Training  Loss  vs  Validation  Loss")
plt.legend(['train','validation'])


plt.figure(2)
plt.plot(snn.history['mean_absolute_error'])
plt.xlabel('Record')
plt.ylabel('Error')
plt.title("Testing  Mean  Absolute  Error")
plt.legend(loc=2)

plt.figure(3)
plt.plot(test_y,label="Correct")
plt.plot(output_vector,label="Calculated")
plt.xlabel('Record')
plt.ylabel('Value')
plt.title("Output")
#plt.ylim(top=1)
plt.legend(loc=2)

plt.show()
print("{:.2f}".format(finish_t-start_t))

print(score)
print("Acuracy  on  Station  DHHL3:"  +str(accuracy))
