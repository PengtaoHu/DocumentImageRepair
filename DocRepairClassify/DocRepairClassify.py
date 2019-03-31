from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler, ModelCheckpoint
import numpy.random as rng
import para
import random
import copy
import time
import numpy as np
import tensorflow as tf
import keras
import generator
import keras.initializers

def W_init(shape,name=None):
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)

def b_init(shape,name=None):
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

input = Input(para.input_shape3)
convnet = Sequential()
convnet.add(Conv2D(64,(5,5),activation='relu',input_shape=para.input_shape3,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(3,3),activation='relu',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(2,2),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(Flatten())
convnet.add(Dense(2048,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))
features=convnet(input)
prediction = Dense(para.n_class,activation='sigmoid',bias_initializer=b_init)(features)
net = Model(input=input,output=prediction)

optimizer = Adam(0.00006)
net.compile(loss="categorical_crossentropy",optimizer=optimizer, metrics=['accuracy'])

net.summary()
callback_list = []
train_name = str(int(time.time()))
callback_list.append(ModelCheckpoint(filepath=para.data_result_path+'/models/checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'))
#callback_list.append(EarlyStopping(monitor='val_custom_metric', patience=para.patience))
callback_list.append(TensorBoard(log_dir=para.data_result_path+'/logs/' + str(train_name),update_freq='epoch'))


train_generator = generator.DataGenerator()
val_generator = generator.ValidationDataGenerator()

net.fit_generator(epochs=para.epochs,
                generator=train_generator,
                validation_data=val_generator,
                validation_steps=para.validation_steps,
                steps_per_epoch=para.steps_per_epoch,
                callbacks=callback_list)

net.save(para.data_result_path+'/models_classify/'+str(0)+'.h5')




