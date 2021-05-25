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

from skimage.feature import greycomatrix

from shutil import copyfile


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
    deltag = (torch.abs(deltah) + torch.abs(deltav)) / 2.0
    deltag_vec = torch.flatten(deltag)

    # get theta
    for hi in range(h):
        for wi in range(w):
            if (deltah[hi][wi] == 0 and deltav[hi][wi] == 0):
                theta[hi][wi] = 0
            elif(deltah[hi][wi] == 0):
                theta[hi][wi] = np.pi
            else:
                theta[hi][wi] = (np.arctan(deltav[hi][wi]/deltah[hi][wi])) + (np.pi/2.0)
    theta_vec = torch.flatten(theta)

    n = 16
    hd = torch.zeros(n)
    # t=12 
    t = torch.mean(deltag_vec).item()

    counti=0
    for k in range(n):
        countk = 0 
        for dg in range(deltag_vec.shape[0]):
            if (deltag_vec[dg] >= t) and (theta_vec[dg] >= (((2*k-1) * np.pi) /(2*n))) and (theta_vec[dg]  < (((2*k+1)* np.pi)/(2*n))):
                countk+=1
                counti+=1
        hd[k]=countk
    hd = hd / counti
    hd_max_index = np.argmax(hd)
    fdir = 0
    for ni in range(n):
        fdir += np.power((ni - hd_max_index), 2) * hd[ni]
    # r = 1/n
    # fdir = 1 - r*n

    return fdir


def compute_linelikeness(img):
    w=img.shape[0]
    h=img.shape[1]
    img_ = np.uint8(img.detach().cpu().numpy()*225)
    coocurrence_matrix = greycomatrix(img_,distances=[1],levels=w,angles=[0])
    print()




    return 

def compute_roughness(fcrs,fcon):
    return fcrs+fcon


style_dir= './inputs/style_images/filipino_designer_furniture_textures/grayscale'

categories = [
    'coarseness',
    'contrast',
    'directionality',
    # 'linelikeliness',
    # 'regularity',
    'roughness'
]




coarses = {}
contrasts = {}
directions = {}
roughs = {}

for f in os.listdir(style_dir):
    input_path = os.path.join(style_dir,f)
    if os.path.isdir(input_path):
        continue
    if (os.path.splitext(input_path)[1] not in ['.png']):
        continue 

    img_tensor = image_utils.image_to_tensor(image_utils.load_image(input_path,mode='L'),phase='default',normalize=False)
    img_tensor.squeeze_()
    linelikeness = compute_linelikeness(img_tensor)
    coarseness = compute_coarseness(img_tensor)
    contrast = compute_contrast(img_tensor)
    directionality = compute_directionality(img_tensor)

    roughness = compute_roughness(coarseness,contrast)

    coarses[f]=coarseness
    contrasts[f]=contrast
    directions[f]=directionality
    roughs[f]=roughness
    


mean_coarse = np.mean(list(coarses.values()))
mean_contrast = np.mean(list(contrasts.values()))
mean_direction = np.mean(list(directions.values()))
mean_rough = np.mean(list(roughs.values()))

for cat in categories:
    high_dir = os.path.join(style_dir,'categories',cat,'high')
    low_dir = os.path.join(style_dir,'categories',cat,'low')

    try:
        os.makedirs(high_dir,exist_ok=True)
        os.makedirs(low_dir,exist_ok=True)
    except FileExistsError:
        pass

for f in os.listdir(style_dir):
    input_path = os.path.join(style_dir,f)
    if os.path.isdir(input_path):
        continue
    if (os.path.splitext(input_path)[1] not in ['.png']):
        continue 
    
    if coarses[f] > mean_coarse:
        copyfile(input_path, os.path.join(style_dir,'categories/coarseness/high',f)) 
    else: 
        copyfile(input_path, os.path.join(style_dir,'categories/coarseness/low',f)) 
    
    if contrasts[f] > mean_contrast:
        copyfile(input_path, os.path.join(style_dir,'categories/contrast/high',f)) 
    else: 
        copyfile(input_path, os.path.join(style_dir,'categories/contrast/low',f)) 
    
    if directions[f] > mean_direction:
        copyfile(input_path, os.path.join(style_dir,'categories/directionality/high',f)) 
    else: 
        copyfile(input_path, os.path.join(style_dir,'categories/directionality/low',f)) 
    
    if roughs[f] > mean_rough:
        copyfile(input_path, os.path.join(style_dir,'categories/roughness/high',f)) 
    else: 
        copyfile(input_path, os.path.join(style_dir,'categories/roughness/low',f)) 
