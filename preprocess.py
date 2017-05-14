# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:51:07 2017

@author: n.aoi
"""

import os
import cv2
import pandas as pd
import pickle
import numpy as np
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt

def extract_contours(image):
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gaussian filter
    im_gray_smooth = cv2.GaussianBlur(im_gray, (11, 11), 0)

    # apply threshold
    ret, th1 = cv2.threshold(im_gray_smooth, 250, 255, cv2.THRESH_BINARY_INV)

    # extract contour
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def extract_clothes_color(image):
    contours = extract_contours(image)

    mask = np.zeros((image.shape[0], image.shape[1])).astype('uint8')
    cv2.fillPoly(mask, contours, 255)
    # masked_img = cv2.bitwise_and(image, image, mask=mask)

    bool_y, bool_x = np.where(mask==255)
    clothes_color = image[bool_y, bool_x].flatten()

    return clothes_color

def make_hist(data):
    print('calculate histograms ...')

    ch = 3
    bins = 255
    hist = []
    for d in data:
        ## make color histogram for each channels (RGB)
        hist.append([np.histogram(d[c::ch], bins, range=(0, 255), normed=True)[0] for c in range(ch)])
        '''
        print f
        n_b, bins_b = np.histogram(d[0::ch], bins, range=(0, 255), normed=True)
        n_g, bins_g = np.histogram(d[1::ch], bins, range=(0, 255), normed=True)
        n_r, bins_r = np.histogram(d[2::ch], bins, range=(0, 255), normed=True)

        plt.step(bins_b[:-1], n_b, label='blue')
        plt.step(bins_g[:-1], n_g, label='green')
        plt.step(bins_r[:-1], n_r, label='red')
        plt.legend()
        plt.show()
        '''

    hist_array = np.array(hist).reshape(len(hist), ch * bins)

    return hist_array