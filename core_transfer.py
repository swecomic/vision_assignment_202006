# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 15:07:57 2020

@author: sweco
"""
from tensorflow.keras.models import load_model
import os 
import numpy as np 
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from plot import plot_history

#################################### 
#### create a data generator
#################################### 
train_datagen = ImageDataGenerator(
                                 horizontal_flip=True
                                 ,rescale=1./255)       
train_target = train_datagen.flow_from_directory('../input/transfer/downsized/test'
                                           , class_mode='categorical'
                                           , target_size=(32, 32)
                                           , batch_size=32)

test_datagen = ImageDataGenerator(rescale=1./255)
test_target = test_datagen.flow_from_directory('../input/transfer/downsized/train'
                                           , class_mode='categorical'
                                           , target_size=(32, 32)
                                           , batch_size=32)

input_shape=train_target.next()[0].shape[1:]
nb_class = train_target.next()[1].shape[1]


#################################### 
#### Load base model
#################################### 
base_model = load_model("../input/basemodel/my_model.h5")

base_model.include_top = False
base_model.trainable = False

base_model.summary()


#################################### 
#### Transfer to Target model - change only output layer
#################################### 

base_model.get_config()                                               

## Change output class
base_model.layers[-1].units = nb_class #100
base_model.layers[-1].trainable = True
target_model = model_from_json(base_model.to_json())
target_model.summary()
target_model.get_config()    

optimizer = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
target_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = target_model.fit(train_target
                    , epochs=650                                  
                    , validation_data=test_target                         
                    , verbose=1                 
                    #, callbacks=[es]
                    )

plot_history(history)

print(max(history.history['accuracy']))
print(max(history.history['val_accuracy']))


#################################### 
#### create a data generator
#################################### 
train_datagen = ImageDataGenerator(
#                                     width_shift_range=0.1
#                                  , height_shift_range=0.1
                                    horizontal_flip=True
                                 ,  brightness_range=[0.2,1.0]
#                                  ,  zoom_range=[0.5,1.0]
#                                  , rotation_range=10
                                 ,rescale=1./255)     
train_target = train_datagen.flow_from_directory('../input/transfer/downsized/test'
                                           , class_mode='categorical'
                                           , target_size=(32, 32)
                                           , batch_size=32)

test_datagen = ImageDataGenerator(rescale=1./255)
test_target = test_datagen.flow_from_directory('../input/transfer/downsized/train'
                                           , class_mode='categorical'
                                           , target_size=(32, 32)
                                           , batch_size=32)

input_shape=train_target.next()[0].shape[1:]
nb_class = train_target.next()[1].shape[1]


#################################### 
#### Load base model
#################################### 
base_model = load_model("../input/basemodel/my_model.h5")

base_model.include_top = False
base_model.trainable = False

base_model.summary()


#################################### 
#### Transfer to Target model - Fine Tuning
#################################### 
base_model.get_config()                                               

len(base_model.layers)

base_model.layers[18]

base_model.layers[-1].units = nb_class #100
for i in range(len(base_model.layers)):
    print(len(base_model.layers)-1-i)
    n_layer=base_model.layers[len(base_model.layers)-1-i]
    n_layer.trainable = True
    
    if(len(base_model.layers)-1-i == 10):
        break   

target_model = model_from_json(base_model.to_json())
target_model.summary()
target_model.get_config()    

optimizer = Adam(learning_rate=0.00003, beta_1=0.9, beta_2=0.999, amsgrad=False)
target_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=50)
history = target_model.fit(train_target
                    , epochs=650                                  
                    , validation_data=test_target                         
                    , verbose=1                 
                    #, callbacks=[es]
                    )

plot_history(history)

print(max(history.history['accuracy']))
print(max(history.history['val_accuracy']))

