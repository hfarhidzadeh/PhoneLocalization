import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

import train_phone_finder


image_path = sys.argv[1]
#image_path = r"E:\Kaggle\find_phone_task\80.jpg"

model = load_model('unet.hdf5')

#if not os.path.exists(main_folder+'//find_phone_test_images'):
#    os.makedirs(main_folder+'//find_phone_test_images')        

imgs_test = img_to_array(load_img(image_path))/255
#imgs_test = cv2.resize(imgs_test, (train_phone_finder.img_rows, train_phone_finder.img_cols))
imgs_test = cv2.resize(imgs_test, (256, 256)) 
imgs_test = np.expand_dims(imgs_test, axis=0)

print ('predict test data')
imgs_mask_test = model.predict(imgs_test, batch_size=1,
        verbose=1)
#np.save('imgs_mask_test.npy', imgs_mask_test) 


ref_image = cv2.imread(image_path,0)
org_height, org_width = ref_image.shape[:2]

#print ('array to image')
#imgs = np.load('imgs_mask_test.npy') * 255
imgs = imgs_mask_test * 255
imgs = np.squeeze(imgs, axis=3)
imgs = np.squeeze(imgs, axis=0)


imgs = cv2.resize(imgs, (org_width, org_height)) 
cv2.imwrite("result.jpg", imgs)

#gray_image = cv2.cvtColor(img_to_array(load_img("result.jpg")), cv2.COLOR_BGR2GRAY)

gray_image = img_to_array(load_img("result.jpg"))
shape = gray_image.shape

ind = np.where(gray_image == 128.) 

box = [np.amin(ind[0]), np.amax(ind[0]), np.amin(ind[1]), np.amax(ind[1])] 

center_x = (box[0] + box[1])/(2 * shape[0])
center_y = (box[2] + box[3])/(2 * shape[1])

print( round(center_y, 4), round(center_x, 4))