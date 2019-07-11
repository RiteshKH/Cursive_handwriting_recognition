# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 13:52:42 2018

@author: 726094
"""

from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model
import cv2
import os
import numpy

from keras import backend as K
K.set_image_dim_ordering('th')

#Load model  
model=load_model('./gzip/Mnist1L_5Conv.h5')
print(model.summary())

import string
letter_count = dict(zip(string.ascii_lowercase, range(1,27)))
print('Letter_count: ',letter_count.items())


x=[]
res=[]
fname=[]
folder='./result/resized_images/'
dirFiles=os.listdir(folder)
dirFiles = sorted(dirFiles,key=lambda x: int(os.path.splitext(x)[0]))
for filename in dirFiles:
    imt = cv2.imread(os.path.join(folder,filename))
    imt = cv2.blur(imt,(6,6))
    gray = cv2.cvtColor(imt,cv2.COLOR_BGR2GRAY)
    ret, imt = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    if imt is not None:
        imt = imt.reshape((-1, 28, 28))
#        plt.imshow(imt)
#        plt.show()
        imt=imt/255
        x.append(imt)
        fname.append(filename)
       
x=np.array(x);    
predictions = model.predict(x)
classes = np.argmax(predictions, axis=1)    

for i in range(len(classes)):
    imt = cv2.imread(os.path.join(folder,dirFiles[i]))
    plt.imshow(imt)
    plt.show()
    print([k for k,v in letter_count.items() if v == classes[i]])
#print(filename,classes)








