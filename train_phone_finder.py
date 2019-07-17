#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, \
    UpSampling2D, Dropout, Cropping2D
from keras.layers import concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import *
import cv2
import warnings
warnings.filterwarnings("ignore")

img_rows=256
img_cols=256
EPOCHS = 100 
BATCH_SIZE = 16

folder_path = sys.argv[1]

class Unet(object):

    def __init__(self):
        pass

    def get_unet(self):
        
        inputs = Input((img_rows, img_cols, 3))
        
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
        #merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
        #merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
        #merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
        #merge9 = concatenate([conv1, up9],axis=3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(input = inputs, output = conv10)
        opt = Adam(lr = 1e-4) 
        model.compile(optimizer = opt , loss = 'binary_crossentropy', metrics = ['accuracy'])

        return model

    def train(self):

        print ('loading data')
        (imgs_train, mask_train) = get_data(img_rows, img_cols, folder_path)
        imgs_train = np.array(imgs_train)
        mask_train = np.array(mask_train)
       
        print ('loading data done')
        model = self.get_unet()
        print ('got unet')

        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',
                verbose=1, save_best_only=True)

        print ('Augmentation...')
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, \
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,\
    horizontal_flip=True, fill_mode="nearest")

        print ('Fitting model...')
        model.fit(imgs_train, mask_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.25,
            shuffle=True,
            callbacks=[model_checkpoint],
            )

        #(trainX, valX, trainY, valY) = train_test_split(imgs_train,mask_train,test_size=0.25, random_state=10)

#         H = model.fit_generator(aug.flow(trainX, trainY, batch_size=4), \
# # # #H = parallel_model.fit(aug.flow(trainX, trainY, batch_size=BS), \
#      validation_data=(valX, valY), \
#      steps_per_epoch=len(trainX) //4, epochs=200, verbose=1)

        


if __name__ == '__main__':
    Unet = Unet()
    Unet.train()


            