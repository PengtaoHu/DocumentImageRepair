from keras.layers import Input, Conv2D, Dense, Flatten,MaxPooling2D,SeparableConv2D
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
from keras.initializers import RandomNormal

convnet = Sequential()
convnet.add(Conv2D(64,(5,5),activation='relu',input_shape=para.input_shape3, kernel_initializer=RandomNormal(0,1e-2)
                   ,kernel_regularizer=l2(2e-4),bias_initializer=RandomNormal(0.5,1e-2)))
#convnet.add(SeparableConv2D(32,(5,5),depth_multiplier=64,activation='relu',input_shape=para.input_shape3, depthwise_initializer=RandomNormal(0,1e-2),pointwise_initializer=RandomNormal(0,1e-2)
#                   ,kernel_regularizer=l2(2e-4),bias_initializer=RandomNormal(0.5,1e-2)))

convnet.add(MaxPooling2D())

convnet.add(Conv2D(128,(3,3),activation='relu', kernel_initializer=RandomNormal(0,1e-2)
                   ,kernel_regularizer=l2(2e-4),bias_initializer=RandomNormal(0.5,1e-2)))
#convnet.add(SeparableConv2D(64,(3,3),depth_multiplier=4,activation='relu', depthwise_initializer=RandomNormal(0,1e-2),pointwise_initializer=RandomNormal(0,1e-2)
#                   ,kernel_regularizer=l2(2e-4),bias_initializer=RandomNormal(0.5,1e-2)))

convnet.add(MaxPooling2D())

convnet.add(Conv2D(256,(2,2),activation='relu', kernel_initializer=RandomNormal(0,1e-2)
                   ,kernel_regularizer=l2(2e-4),bias_initializer=RandomNormal(0.5,1e-2)))
#convnet.add(SeparableConv2D(256,(2,2),depth_multiplier=4,activation='relu', depthwise_initializer=RandomNormal(0,1e-2),pointwise_initializer=RandomNormal(0,1e-2)
#                   ,kernel_regularizer=l2(2e-4),bias_initializer=RandomNormal(0.5,1e-2)))


convnet.add(Flatten())
convnet.add(Dense(1024,activation="sigmoid", kernel_initializer=RandomNormal(0,1e-2)
                  ,kernel_regularizer=l2(1e-3),bias_initializer=RandomNormal(0.5,1e-2)))
convnet.add(Dense(para.n_class,activation='softmax',bias_initializer=RandomNormal(0.5,1e-2)))

optimizer = Adam(0.00006)

def top_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

convnet.compile(loss="categorical_crossentropy",optimizer=optimizer, metrics=['accuracy',top_accuracy])

convnet.summary()
callback_list = []
train_name = str(int(time.time()))
callback_list.append(ModelCheckpoint(filepath=para.data_result_path+'/models/checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'))
#callback_list.append(EarlyStopping(monitor='val_loss', patience=para.patience))
callback_list.append(TensorBoard(log_dir=para.data_result_path+'/logs/' + str(train_name),update_freq='epoch'))


train_generator = generator.DataGenerator()
val_generator = generator.ValidationDataGenerator()

convnet.fit_generator(epochs=para.epochs,
                generator=train_generator,
                validation_data=val_generator,
                validation_steps=para.validation_steps,
                steps_per_epoch=para.steps_per_epoch,
                callbacks=callback_list)

convnet.save(para.data_result_path+'/models_classify/'+str(0)+'.h5')




