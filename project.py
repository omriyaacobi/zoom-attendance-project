# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:17:30 2021

@author: ori1j
"""
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import  Dense, Flatten, Conv2D,MaxPool2D
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
def AproachData():
    path=input('enter location of data file: ')
    path+='/data'
    if(os.path.exists(path)):
        return path
    else:
        while(os.path.exists(path)==False):
            path=input('file not found. enter the correct path to the data file: ')
            path+='/data'
        return path     

def ProcessImage(imagepath):
    imgpath=input('enter location of the image, including the name of the image')
    
    while(True):
        try:
            im=Image.open(imgpath)
            return imgpath
        except:
            imgpath=input('image path not valid. enter the correct imagae path')
        
    
def PlotIMages(image_arr):
    fig,axes= plt.subplots(1,10,figsize=(20,20))
    axes=axes.flatten()
    for img, ax in zip(image_arr,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()    

def TrainAndSaveModel(trainbatches,validbatches):
    model=Sequential([
        Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same',input_shape=(224,224,3)),
        MaxPool2D(pool_size=(2,2),strides=2),
        Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same'),
        MaxPool2D(pool_size=(2,2),strides=2),
        Flatten(),
        Dense(units=2,activation='sigmoid')
        ])
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(x=trainbatches,
              steps_per_epoch=len(trainbatches),
              validation_data=validbatches,
              validation_steps=len(validbatches),
              epochs=10,
              verbose=2
              )
    model.save('C:/data/model.h5')

def Test(model,testbatches):
    score=model.evaluate(x=testbatches,verbose=0)
    print('Test loss: ',score[0])
    print('Test acuracy: ',score[1])
    


datapath=AproachData()
train_path=datapath+'/train'
validation_path = datapath+'/validation'
test_path = datapath+'/test'
imagepath=datapath+'/one'

ProcessImage(imagepath)


trainbatches= ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=train_path, target_size=(224,224),classes=['bad','good'],batch_size=10)
validbatches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=validation_path, target_size=(224,224), classes=['bad', 'good'], batch_size=10)
testbatches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['bad', 'good'], batch_size=10, shuffle=False)
imagebatches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=imagepath, target_size=(224,224),classes=['bad','good'],batch_size=10)

imgs,labels=next(trainbatches)

PlotIMages(imgs)
print(labels)  

#TrainAndSaveModel(trainbatches, validbatches)



new_model = load_model('C:/data/model.h5')

#Test(new_model,testbatches)







prediction=new_model.predict(
    x=imagebatches
    ,batch_size=10
    ,verbose=0)
if(np.argmax(prediction,axis=-1)[0]==1):
    print('person attending zoom class')
else:
    print('person not attending zoom class')
    






