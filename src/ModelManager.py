from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
import sys

dropout_value=0.3

########### Methods ######################################
def getOptimizer(optimizer,learningRate):
    optimizers = {'SGD': SGD(learning_rate=learningRate),
                  'RMSProp': RMSprop(learning_rate=learningRate,rho=0.9), 
                  'Adadelta': Adadelta(rho=0.95),
                  'Adam': Adam(learning_rate=learningRate, beta_1=0.9, beta_2=0.999)
                  }
    return optimizers[optimizer]

def getModel(modelType,units,activation,output_activation,input_shape,regularization):
    this_module = sys.modules[__name__]
    if regularization == 'Dropout':
        regularization = True
    else:
        regularization = False
    return getattr(this_module, modelType)(units, activation,output_activation, input_shape, regularization)

########### Models ######################################
def fc1(units, neuron, output_neuron, input_shape, regularization):
    # 17(D) -> d(0.3) -> 1(D)
    model = Sequential()
    model.add(Dense(units, input_shape=input_shape, activation=neuron))
    if regularization:
        model.add(Dropout(dropout_value))
        print(f"# {units}(D) -> d(0.3) -> 1(D)")
    else:
        print(f"# {units}(D) -> 1(D)")
    model.add(Dense(1, activation=output_neuron))
    
    return model
    
def fc2(units, neuron,output_neuron, input_shape, regularization):
    # 85(D) -> 17(D) -> d(0.3) -> 1(D)
    model = Sequential()
    model.add(Dense(units, input_shape=input_shape, activation=neuron))
    model.add(Dense(17, activation=neuron))
    if regularization:
        model.add(Dropout(dropout_value))
        print(f"# {units}(D) -> 17(D) -> d(0.3) -> 1(D)")
    else:
        print(f"# {units}(D) -> 17(D) -> 1(D)")
    model.add(Dense(1, activation=output_neuron))
    return model

def fc3(units, neuron, output_neuron, input_shape, regularization):
    # 85(D) -> 17(D) -> 5(D) -> d(0.3) -> 1(D)
    model = Sequential()
    model.add(Dense(units, input_shape=input_shape, activation=neuron))
    model.add(Dense(17, activation=neuron))
    model.add(Dense(5, activation=neuron))
    if regularization:
        model.add(Dropout(dropout_value))
        print(f"# {units}(D) -> 17(D) -> 5(D) -> d(0.3) -> 1(D)")
    else:
        print(f"# {units}(D) -> 17(D) -> 5(D) -> 1(D)")
    model.add(Dense(1, activation=output_neuron))
    
    return model

def fc4(units, neuron, output_neuron, input_shape, regularization):
    # 510(D) -> 85(D) -> 17(D) -> 5(D) -> d(0.3) -> 1(D)
    model = Sequential()
    model.add(Dense(units, input_shape=input_shape, activation=neuron))
    model.add(Dense(85, activation=neuron))
    model.add(Dense(17, activation=neuron))
    model.add(Dense(5, activation=neuron))
    if regularization:
        model.add(Dropout(dropout_value))
        print(f"# {units}(D) -> 85(D) -> 17(D) -> 5(D) -> d(0.3) -> 1(D)")
    else:
        print(f"# {units}(D) -> 85(D) -> 17(D) -> 5(D) -> 1(D)")
    model.add(Dense(1, activation=output_neuron))   
    return model

def rnn(units, neuron, output_neuron, input_shape, regularization):
    # 20(L rd(0,3)) -> d(0.3) -> 1(D)
    model = Sequential()
    if regularization:
        model.add(SimpleRNN(units,activation=neuron,input_shape=input_shape, recurrent_dropout=dropout_value))
        model.add(Dropout(dropout_value))
        print(f"# {units}(R rd(0,3)) -> d(0.3) -> 1(D)")
    else:
        model.add(SimpleRNN(units, activation=neuron,input_shape=input_shape))
        print(f"# {units}(R) -> d(0.3) -> 1(D)")
    model.add(Dense(1, activation=output_neuron))
    
    return model

def rnn1(units, neuron,output_neuron, input_shape, regularization):
    # 20(L) -> 20(L rd(0,3)) -> d(0.3) -> 1(D)
    model = Sequential()
    model.add(SimpleRNN(units, activation=neuron,input_shape=input_shape ,return_sequences=True))
    if regularization:
        model.add(SimpleRNN(units, activation=neuron, recurrent_dropout=dropout_value))
        model.add(Dropout(dropout_value))
        print(f"# {units}(R) -> {units}(R rd(0,3)) -> d(0.3) -> 1(D)")
    else:
        model.add(SimpleRNN(units,  activation=neuron))  
        print(f"# {units}(R) -> {units}(R) -> 1(D)")
    model.add(Dense(1, activation=output_neuron))
    
    return model

def lstm(units, neuron,output_neuron, input_shape, regularization):
    # 20(L rd(0,3)) -> d(0.3) -> 1(D)
    model = Sequential()
    if regularization:
        model.add(LSTM(units, activation=neuron,input_shape=input_shape, recurrent_dropout=dropout_value))
        model.add(Dropout(dropout_value))
        print(f"# {units}(L rd(0,3)) -> d(0.3) -> 1(D)")
    else:
        model.add(LSTM(units, activation=neuron,input_shape=input_shape))
        print(f"# {units}(L) -> 1(D)")
    model.add(Dense(1, activation=output_neuron))
    
    return model

def lstm1(units, neuron, output_neuron, input_shape, regularization):
    # 20 (L) -> 20(L rd(0,3)) -> d(0.3) -> 1(D)
    model = Sequential()
    model.add(LSTM(units, activation=neuron,input_shape=input_shape ,return_sequences=True))
    if regularization:
        model.add(LSTM(units, activation=neuron, recurrent_dropout=dropout_value))
        model.add(Dropout(dropout_value))
        print(f"# {units} (L) -> {units}(L rd(0,3)) -> d(0.3) -> 1(D)")
    else:
        model.add(LSTM(units, activation=neuron))
        print(f"# {units} (L) -> {units}(L) -> 1(D)")
    model.add(Dense(1, activation=output_neuron))
    return model

