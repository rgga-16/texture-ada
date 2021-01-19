
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

from defaults import DEFAULTS as D

from PIL import Image
import numpy as np

def default_normalization(mean = D.NORM_MEAN.get(),std = D.NORM_STD.get()):
    norm = transforms.Normalize(mean,std)

    return norm

def default_preprocessor(image_size):
    preprocessor = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ])
    
    return preprocessor



# Loads image
def load_image(filename,mode="RGBA",size=D.IMSIZE.get()):
    img = Image.open(filename).convert(mode)
    img = img.resize((size,size))

    return img

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
def image_to_tensor(image,image_size=D.IMSIZE.get(),device=D.DEVICE(),preprocessor=None,normalize=True):

    if preprocessor == None:
        preprocessor=default_preprocessor(image_size)

    tensor = preprocessor(image)

    if normalize:
        norm = default_normalization([0.40760392, 0.45795686, 0.48501961],[1,1,1])
        temp = tensor[:3,:,:]
        tensor[:3,:,:] = norm(temp)

    c,h,w = tensor.shape

    if(c > 3):
        print('yes more than 3 channels')
        tensor = tensor[:3,:,:]
    tensor = tensor.unsqueeze(0)
    return tensor.to(device)


# Show image
def show_image(img):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()

def save_gif(images:list,filename='output.gif'):
    images[0].save(filename,save_all=True,append_images=images[1:],loop=0,optimize=False,duration=75)


# Converts tensor to an image
def tensor_to_image(tensor,image_size=D.IMSIZE.get(),denorm=True):

    posta = transforms.Compose([
        default_normalization([-0.40760392, -0.45795686, -0.48501961],[1,1,1]),
    ])

    postb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size,image_size)),
    ])
    
    tensor_ = tensor.cpu().clone()
    tensor_ = tensor_.squeeze(0)

    if denorm:
        temp = tensor_[:3,:,:]
        tensor_[:3,:,:] = posta(temp)

    image = postb(tensor_)

    return image


