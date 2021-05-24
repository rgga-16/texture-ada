'''
Code borrowed from:
https://github.com/MarshalLeeeeee/Tamura-In-Python/blob/master/tamura-numpy.py
'''

import math 
import kaolin
import torch 
from defaults import DEFAULTS as D
import torch.nn.functional as F
import time,sys
from PIL import Image, ImageOps
from helpers import image_utils
import os 

import args as args_

import numpy as np
import matplotlib.pyplot as plt

from distutils.dir_util import copy_tree


def compute_coarseness(img,kmax=5):
    w=img.shape[0]
    h=img.shape[1]
    average = np.zeros([kmax,w,h])
    horizontal = np.zeros([kmax,w,h])
    vertical = np.zeros([kmax,w,h])
    sbest = np.zeros([w,h])
    for k in range(kmax):
        window = np.power(2,k)
        for wi in range(w)[window:(w-window)]:
            for hi in range(h)[window:(h-window)]:
                average[k][wi][hi] = torch.sum(img[wi-window:wi+window,hi-window:hi+window]) / np.power(2,2*k)

        for wi in range(w)[window:(w-window)]:
            for hi in range(h)[window:(h-window)]:
                horizontal[k][wi][hi] = np.abs((average[k][wi+window][hi] - average[k][wi-window][hi]))
                vertical[k][wi][hi] = np.abs((average[k][wi][hi+window] - average[k][wi][hi-window]))
    
    for wi in range(w):
        for hi in range(h):
            h_max = np.max(horizontal[:,wi,hi])
            h_max_id = np.argmax(horizontal[:,wi,hi])
            v_max = np.max(vertical[:,wi,hi])
            v_max_id = np.argmax(vertical[:,wi,hi])
            max_id = h_max_id if h_max > v_max else v_max_id
            sbest[wi][hi] = np.power(2,max_id)
    fcrs = np.mean(sbest)
    print()
    return fcrs 

def compute_contrast(img,n=0.25):
    img_flat = torch.flatten(img)
    mean = torch.mean(img_flat)
    v = torch.var(img_flat)

    m4 = torch.mean(torch.pow(img - mean,4))
    std = torch.sqrt(v)
    alpha4 = m4/torch.pow(v,2)
    fcon = v / torch.pow(alpha4,n)
    print()
    return fcon.item()

def compute_directionality(img):
    w=img.shape[0]
    h=img.shape[1]

    deltah = torch.zeros([w,h])
    deltav = torch.zeros([w,h])
    theta = torch.zeros([w,h])
    h_window = torch.tensor([
        [-1,0,1],
        [-1,0,1],
        [-1,0,1],
    ],device=D.DEVICE())

    v_window = torch.tensor([
        [1,1,1],
        [0,0,0],
        [-1,-1,-1],
    ],device=D.DEVICE())

    # Get deltah
    for hi in range(h)[1:h-1]:
        for wi in range(w)[1:w-1]:
            deltah[hi][wi] = torch.sum(torch.mul(img[hi-1:hi+2, wi-1:wi+2],h_window))
    for wi in range(w)[1:w-1]:
        deltah[0][wi] = img[0][wi+1] - img[0][wi]
        deltah[h-1][wi] = img[h-1][wi+1] - img[h-1][wi]
    for hi in range(h):
        deltah[hi][0] = img[hi][1] - img[hi][0]
        deltah[hi][w-1] = img[hi][w-1] - img[hi][w-2]
    
    # get deltav
    for hi in range(h)[1:h-1]:
        for wi in range(w)[1:w-1]:
            deltav[hi][wi] = torch.sum(torch.mul(img[hi-1:hi+2, wi-1:wi+2], v_window))
    for wi in range(w):
        deltav[0][wi] = img[1][wi] - img[0][wi]
        deltav[h-1][wi] = img[h-1][wi] - img[h-2][wi]
    for hi in range(h)[1:h-1]:
        deltav[hi][0] = img[hi+1][0] - img[hi][0]
        deltav[hi][w-1] = img[hi+1][w-1] - img[hi][w-1]
    
    # get deltag
    deltag = (torch.abs(deltah) + torch.abs(deltav)) / 2

    # get theta
    theta = (torch.arctan(deltav/deltah)) + (np.pi/2)
    print()

    

style_dir= './inputs/style_images/filipino_designer_furniture_textures/grayscale'


for f in os.listdir(style_dir):
    input_path = os.path.join(style_dir,f)
    if os.path.isdir(input_path):
        continue
    if (os.path.splitext(input_path)[1] not in ['.png']):
        continue 

    img_tensor = image_utils.image_to_tensor(image_utils.load_image(input_path,mode='L'),phase='default',normalize=False)
    img_tensor.squeeze_()
    # coarseness = compute_coarseness(img_tensor)
    # contrast = compute_contrast(img_tensor)
    compute_directionality(img_tensor)
    # print(f'{f} coarseness: {coarseness:.3f}')

