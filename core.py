
import pandas as pd
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

from gen_data01 import get_CIFAR10_data
from plot import plot_history

# Get cifar-10 data
x_train, y_train, x_test, y_test = get_CIFAR10_data()


print("Training data:")
print("Number of examples: ", x_train.shape[0])
print("Number of channels:", x_train.shape[3])
print("Image size:", x_train.shape[1], x_train.shape[2])

print("======================")
print("Test data:")
print("Number of examples:", x_test.shape[0])
print("Number of channels:", x_test.shape[3])
print("Image size:", x_test.shape[1], x_test.shape[2])

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[np.argmax(y_train[i])])
plt.show()  # Define Alexnet Model


def AlexnetModel(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=2, input_shape=input_shape, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
   
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
   
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
   
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
   
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
   
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
   
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
   
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
   
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
   
    model.add(Dense(num_classes, activation='softmax'))

    return model


num_classes = y_train.shape[1]
input_shape = x_train.shape[1:]

#################################### 
#### Training ###
#################################### 

model = AlexnetModel(input_shape, num_classes)

optimizer = Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

#data augmentation
datagen = ImageDataGenerator(width_shift_range=0.2
                                 , height_shift_range=0.2
                                 , rotation_range=10
                                 )

history = model.fit(datagen.flow(x_train, y_train, batch_size=32)
                    , epochs=100            
                    , validation_data=(x_test, y_test)                  
                    , verbose=1                 
                    , callbacks=[es]
                    )

train_hist = pd.DataFrame(history.history)

plot_history(history)


#save model
model.save('model_alexnet.h5')

x_train[0]
