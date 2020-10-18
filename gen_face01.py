# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:09:23 2020

@author: sweco
"""
# demonstrate face detection on 5 Celebrity Faces Dataset
from glob import glob
from PIL import Image
from numpy import asarray
from matplotlib import pyplot

 
data_list = glob('face\\downsized\\train\\*\\*.jpg')


def get_label_from_path(path):
    return (path.split('\\')[-2])


path = data_list[0]

get_label_from_path(path)

image = np.array(Image.open(path))


def read_image(path):
    image = np.array(Image.open(path))
    # Channel 1을 살려주기 위해 reshape 해줌
    #return image.reshape(image.shape[0], image.shape[1], 1)
    return image

read_image(path)

image.shape