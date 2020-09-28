from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense,Dropout
from tensorflow.keras.layers import Activation, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2


def FullyConnectedModel(input_shape=(224,224,3),weight_penalty=0.0001,momentum=.9):
    ''' Creates a simple image regressor model with specified parameters
    Args:        input_shape: shape of the input tensor


    Returns: a Keras Model
    '''
    model = Sequential() 
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid" ,use_bias=False,kernel_regularizer=l2(weight_penalty),input_shape=input_shape))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Activation('relu'))
    model.add(Conv2D(36, (3, 3), strides=(2, 2), padding="valid",use_bias=False,kernel_regularizer=l2(weight_penalty)))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='elu',kernel_initializer='he_normal',bias_regularizer=l2(weight_penalty),kernel_regularizer=l2(weight_penalty)))
    model.add(Dense(50, activation='elu',kernel_initializer='he_normal',bias_regularizer=l2(weight_penalty),kernel_regularizer=l2(weight_penalty)))
    model.add(Dense(10, activation='elu',kernel_initializer='he_normal',bias_regularizer=l2(weight_penalty),kernel_regularizer=l2(weight_penalty)))
    model.add(Dense(2,activation='softsign',kernel_initializer='he_normal')) 
    
    return model
    
    #model.add(Conv2D(48, (5, 5), strides=(2, 2),padding="valid",use_bias=False,kernel_regularizer=l2(weight_penalty)))
    #model.add(BatchNormalization(momentum=momentum))
    #model.add(Activation('elu'))
    #model.add(Conv2D(64, (3, 3), padding="valid",use_bias=False,kernel_regularizer=l2(weight_penalty)))
    #model.add(BatchNormalization(momentum=momentum))
    #model.add(Activation('elu'))
    #model.add(Conv2D(64, (3, 3), padding="valid",use_bias=False,kernel_regularizer=l2(weight_penalty)))
    #model.add(BatchNormalization(momentum=momentum))
    #model.add(Activation('elu'))
    #model.add(Flatten())
    #model.add(Dense(1000, activation='relu',kernel_initializer='he_normal',bias_regularizer=l2(weight_penalty),kernel_regularizer=l2(weight_penalty)))
    #model.add(Dense(500, activation='relu',kernel_initializer='he_normal',bias_regularizer=l2(weight_penalty),kernel_regularizer=l2(weight_penalty)))
    #model.add(Dropout(0.3))
    #model.add(Dense(1000, activation='relu',kernel_initializer='he_normal',bias_regularizer=l2(weight_penalty),kernel_regularizer=l2(weight_penalty)))
    #model.add(BatchNormalization(momentum=momentum))
    #model.add(Dense(250, activation='relu',kernel_initializer='he_normal',bias_regularizer=l2(weight_penalty),kernel_regularizer=l2(weight_penalty)))
    #model.add(Dropout(0.3))
    #model.add(Dense(50, activation='relu',kernel_initializer='he_normal',bias_regularizer=l2(weight_penalty),kernel_regularizer=l2(weight_penalty)))
    #model.add(Dense(2,activation='softsign',kernel_initializer='he_normal')) 
    