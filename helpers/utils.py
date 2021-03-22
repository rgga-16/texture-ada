
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

from defaults import DEFAULTS as D

from PIL import Image
import numpy as np

import os 
from args import args

def normalize_vertices(vertices):
        """
        Normalize mesh vertices into a unit cube centered at zero.
        """
        vertices = vertices - vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2
        return vertices


# Preprocesses image and converts it to a Tensor
def image_to_tensor(image,image_size=D.IMSIZE.get(),device=D.DEVICE(),normalize=True):

    preprocessor = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ])

    tensor = preprocessor(image)

    if normalize:
        norm = transforms.Normalize(D.NORM_MEAN.get(),D.NORM_STD.get())
        temp = tensor[:3,:,:]
        tensor[:3,:,:] = norm(temp)

    # c,h,w = tensor.shape

    # if(c > 3):
    #     tensor = tensor[:3,:,:]
    # tensor = tensor.unsqueeze(0)
    return tensor.to(device)

# Converts tensor to an image
def tensor_to_image(tensor,image_size=D.IMSIZE.get(),denorm=True):

    postb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size,image_size)),
    ])
    
    tensor_ = tensor.detach().cpu().numpy().squeeze(0)
    tensor_ = np.transpose(tensor_,(1,2,0))

    if denorm:
        temp = tensor_[...,:3]
        tensor_[...,:3] = temp * np.array(D.NORM_STD.get()) + np.array(D.NORM_MEAN.get())

    image = postb(np.uint8(tensor_*225))
    return image


# Show image
def show_image(img):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Loads a single image or multiple images
def load_image(filename,mode="RGBA"):
    img = Image.open(filename).convert(mode)
    return img


def save_gif(images:list,filename='output.gif'):
    images[0].save(filename,save_all=True,append_images=images[1:],loop=0,optimize=False,duration=75)




