from keras.models import Sequential, Model
from keras.constraints import maxnorm
from keras.layers import merge, Dense, Dropout, Input, concatenate
from keras.optimizers import Nadam

def create_one_layer(data_width, neurons=25, optimizer='adam', dropout_rate=0.0, weight_constraint=0):
    #baseline Model
    model = Sequential()
    #The first param in Dense is the number of neurons in the first hidden layer
    #model.add(Dense(neurons, input_dim=22300, kernel_initializer='normal', activation='relu',kernel_constraint=maxnorm(weight_constraint) ))
    model.add(Dense(neurons, input_dim=data_width, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_fourSameLayer(neurons=25, optimizer='adam', dropout_rate=0.0, weight_constraint=0):
    #baseline Model
    model = Sequential()
    #The first param in Dense is the number of neurons in the first hidden layer
    model.add(Dense(neurons, input_dim=22300, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_fourDecrLayer(neurons=25, optimizer='adam', dropout_rate=0.0, weight_constraint=0):
    #baseline Model
    model = Sequential()
    n2 = neurons // 2 if neurons // 2 > 0 else 1
    n3 = neurons // 3 if neurons // 3 > 0 else 1
    n4 = neurons // 4 if neurons // 4 > 0 else 1
    #The first param in Dense is the number of neurons in the first hidden layer
    model.add(Dense(neurons, input_dim=22300, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n2, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    if n2 > 1 : model.add(Dropout(dropout_rate))
    model.add(Dense(n3, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    if n3 > 1: model.add(Dropout(dropout_rate))
    model.add(Dense(n4, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    if n4 > 1: model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_binaryDecrease(neurons=25, optimizer='adam'):
    model = Sequential()
    #The first param in Dense is the number of neurons in the first hidden layer
    model.add(Dense(neurons, input_dim=22300, kernel_initializer='normal', activation='relu'))
    while (neurons/2 >=1):
        model.add(Dense(neurons/2, kernel_initializer='normal', activation='relu'))
        neurons/=2
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def create_dualInputSimple(input_ratio, feat_width, perm_width, neurons=32, dropout_rate=0.1):
    '''this simple model performs no additional analysis after concatenation'''
    perm_input = Input(shape=(perm_width,), name='permissions_input')
    x = Dense(neurons, activation='relu')(perm_input)
    feat_input = Input(shape=(feat_width,), name='features_input')
    y = Dense(int(neurons*input_ratio), activation='relu')(feat_input)
    x = concatenate([x, y])
    output = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=[perm_input, feat_input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model

def create_dualInputLarge(input_ratio, feat_width, perm_width, neurons=32, dropout_rate=0.1):
    '''this model performs additional analysis with layers after concatenation'''
    perm_width=int(perm_width)
    perm_input = Input(shape=(perm_width,), name='permissions_input')
    x = Dense(neurons, activation='relu')(perm_input)
    x = Dropout(dropout_rate)(x)
    x = Dense(neurons, activation='relu')(x)
    feat_input = Input(shape=(feat_width,), name='features_input')
    y = Dense(int(neurons*input_ratio), activation='relu')(feat_input)
    x = concatenate([x, y])
    x = Dense(int((neurons+(neurons*input_ratio))/2), activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(int((neurons+(neurons*input_ratio))/2), activation='relu')(x)
    output = Dense(1, activation='sigmoid', name="output")(x)
    model = Model(inputs=[perm_input, feat_input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model

#1000-500-250-100-10-1 Model  -WGH
def create_thousand_layer(data_width, neurons=25, optimizer='adam', dropout_rate=0.1, weight_constraint=0):
    #baseline Model
    model = Sequential()
    #The first param in Dense is the number of neurons in the first hidden layer
    #model.add(Dense(neurons, input_dim=22300, kernel_initializer='normal', activation='relu',kernel_constraint=maxnorm(weight_constraint) ))
    model.add(Dense(1000, input_dim=data_width, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(500, input_dim=data_width, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(250, input_dim=data_width, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, input_dim=data_width, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, input_dim=data_width, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
