import matplotlib.pyplot as plt
from glob import glob
import cv2
import random
import os
#%matplotlib inline

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from IPython.display import SVG, Image
import tensorflow as tf
#print("tensorflow version:",tf.__version__)

images=glob("train/**/**")
for i in range(9):
    image=random.choice(images)
    plt.figure(figsize=(12,12))
    plt.subplot(331+i)
    plt.imshow(cv2.imread(image));plt.axis('off')

img_size=48
batch_size=64
datagen_train=ImageDataGenerator()
traingenerator=datagen_train.flow_from_directory("train/",target_size=(img_size,img_size),
                                                 color_mode="grayscale",
                                                 batch_size=batch_size,
                                                 class_mode="categorical",
                                                 shuffle=True)
datagen_validation=ImageDataGenerator()
trainvalidation=datagen_train.flow_from_directory("test/",target_size=(img_size,img_size),
                                                 color_mode="grayscale",
                                                 batch_size=batch_size,
                                                 class_mode="categorical",
                                                 shuffle=True)

def convolution(input_tensor, filters, kernel_size):
    x=Conv2D(filters=filters,kernel_size=kernel_size, padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Dropout(0.25)(x)
    return x

def dense_f(input_tensor, nodes):
    x=Dense(nodes)(input_tensor)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=Dropout(0.25)(x)
    return x

def model_fer(input_shape):
    inputs=Input(input_shape)
    conv1=convolution(inputs,32,(3,3))
    conv2=convolution(inputs,64,(5,5))
    conv3=convolution(inputs,128,(3,3))
    flatten=Flatten()(conv3)
    dense1=dense_f(flatten,256)
    output=Dense(7,activation="softmax")(dense1)
    model=Model(inputs=[inputs],outputs=[output])
    model.compile(loss=['categorical_crossentropy'], optimizer='adam',metrics=['accuracy'])

    return model

model=model_fer((48,48,1))
model.summary()

epochs = 15
steps_per_epoch = traingenerator.n//traingenerator.batch_size
validation_steps = trainvalidation.n//trainvalidation.batch_size

checkpoint = ModelCheckpoint("model_weights.h5", monitor  ='val_accuracy', save_weights_only = True, mode = 'max', verbose =1)
callbacks = [checkpoint]

history = model.fit(
x = traingenerator,
steps_per_epoch = steps_per_epoch,
epochs = epochs,
validation_data = trainvalidation,
validation_steps = validation_steps,
callbacks = callbacks)

model.evaluate(trainvalidation)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.legend(['Train','Validation'],loc = 'upper left')
plt.subplots_adjust(top=1.0,bottom=0.0,right =0.95,left=0.0,hspace=0.25,wspace=0.35)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.legend(['Train','Validation'],loc = 'upper left')
plt.subplots_adjust(top=1.0,bottom=0.0,right =0.95,left=0.0,hspace=0.25,wspace=0.35)

model_json = model.to_json()
with open("model_a.json","w") as json_file:
    json_file.write(model_json)
