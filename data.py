import cv2
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import sys



def create_mask(points, shape):
    
    mask = np.zeros((shape[0],shape[1]), np.uint8)
    
    center_x = int(shape[0] * float(points[0][2]))
    center_y = int(shape[1] * float(points[0][1]))

    mask[center_x - 20 :center_x + 20, center_y - 20 :center_y + 20] = 255
    return mask


   
def get_data(WIDTH, HEIGHT, path):
    with open(path + '//labels.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
        labels_point = [x.strip() for x in content] 
    
    
    data = []
    mask = []
    tuple_list = []
    for line in labels_point:
        a = line.split( )
        tuple_list.append((a[0], a[1],a[2]))

    for file in os.listdir(path):
        if ".jpg" in file:
            img = load_img(os.path.join(path, file)) 
            arr = img_to_array(img) / 255.0
            arr = cv2.resize(arr, (HEIGHT,WIDTH)) 
            data.append(arr)
            
            t =  [item for item in tuple_list if item[0] == file]
            corr_mask = create_mask(t, arr.shape) 
            arr = img_to_array(corr_mask) / 255.0
            arr = cv2.resize(arr, (HEIGHT,WIDTH)) 
            arr = np.expand_dims(arr, axis=2)
            mask.append(arr)
    
    return data, mask

#get_data(128, 128, r"E:\Kaggle\find_phone_task\find_phone" )
