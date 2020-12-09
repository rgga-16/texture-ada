
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

from defaults import DEFAULTS as D

from PIL import Image
import numpy as np

def default_normalization():
    mean = D.NORM_MEAN.get()
    std = D.NORM_STD.get()
    norm = transforms.Normalize(mean,std)

    return norm

def default_preprocessor(image_size):
    preprocessor = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        default_normalization(),
    ])
    return preprocessor



# Loads image
def load_image(filename):
    img = Image.open(filename).convert("RGB")

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
def image_to_tensor(image,image_size=D.IMSIZE.get(),device=D.DEVICE(),preprocessor=None):

    if preprocessor == None:
        preprocessor=default_preprocessor(image_size)

    tensor = preprocessor(image)

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
def tensor_to_image(tensor):
    unloader = transforms.Compose([
        transforms.ToPILImage(),
    ])
    
    image = tensor.cpu().clone()

    image = image.squeeze(0)
    image = unloader(image)

    return image
