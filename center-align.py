# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 18:44:43 2019

@author: 726094
"""

import numpy as np
import cv2
import os


def add_padding(img, pad_l, pad_t, pad_r, pad_b):
    
    height, width = img.shape
    #Adding padding to the left side.
    pad_left = np.zeros((height, pad_l), dtype = np.int)
    img = np.concatenate((pad_left, img), axis = 1)

    #Adding padding to the top.
    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis = 0)

    #Adding padding to the right.
    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis = 1)

    #Adding padding to the bottom
    pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
    img = np.concatenate((img, pad_bottom), axis = 0)

    return img


images = []
folder='./result/characters/'
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename),0)
    th, a = cv2.threshold(img, 127, 255,cv2.THRESH_OTSU)
    if a is not None:
        a=cv2.resize(a,(100,80))
        # create blank image - y, x
        col_sum = np.where(np.sum(a, axis = 0)>0)
        row_sum = np.where(np.sum(a, axis = 1)>0)
        y1, y2 = row_sum[0][0], row_sum[0][-1]
        x1, x2 = col_sum[0][0], col_sum[0][-1]
        
        cropped_image = a[y1:y2, x1:x2]        
        cropped_image=cv2.resize(cropped_image,(20,20))
        padded_image = add_padding(cropped_image, 4, 4, 4, 4)
        cv2.imwrite('./result/resized_images/'+filename,padded_image)

print("Images resized and saved into designated folder")
       



