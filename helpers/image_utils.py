

import matplotlib.pyplot as plt
from torchvision import transforms
import torch

from defaults import DEFAULTS as D

from PIL import Image
import numpy as np

import os 


# Preprocesses image and converts it to a Tensor
def image_to_tensor(image,phase:str,image_size=D.IMSIZE.get(),device=D.DEVICE(),normalize=True):
    transformer = {
        'train': transforms.Compose([
                    transforms.RandomResizedCrop((image_size,image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ]),
        'val': transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.CenterCrop(round(0.875*image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
        'test': transforms.Compose([
                   transforms.Resize((image_size,image_size)),
                    transforms.ToTensor(),
                ]),
    }
    tensor = transformer[phase](image)

    if normalize:
        norm = transforms.Normalize(D.NORM_MEAN.get(),D.NORM_STD.get())
        temp = tensor[:3,:,:]
        tensor[:3,:,:] = norm(temp)

    return tensor.to(device)

# Converts tensor to an image
def tensor_to_image(tensor,image_size=D.IMSIZE.get(),denorm=True):

    postprocessor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size,image_size)),
    ])
    
    tensor_ = tensor.detach().cpu().numpy().squeeze(0)
    tensor_ = np.transpose(tensor_,(1,2,0))

    if denorm:
        temp = tensor_[...,:3]
        tensor_[...,:3] = temp * np.array(D.NORM_STD.get()) + np.array(D.NORM_MEAN.get())

    image = postprocessor(np.uint8(tensor_*225))
    return image


# Show image
def show_images(images,show=True,save_path=None):
    f, axs = plt.subplots(1,len(images),figsize=(15,15))
    for img, ax in zip(images,axs):
        ax.imshow(img)
        ax.axis('off')
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)

# Loads a single image
def load_image(filename,mode="RGBA"):
    img = Image.open(filename).convert(mode)
    return img


def save_gif(images:list,filename='output.gif'):
    images[0].save(filename,save_all=True,append_images=images[1:],loop=0,optimize=False,duration=75)




