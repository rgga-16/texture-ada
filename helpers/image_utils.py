

import matplotlib.pyplot as plt
import torchvision 
from torchvision import transforms

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
        'default': transforms.Compose([
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

def white_to_transparency_gradient(img):
    x = np.asarray(img.convert('RGBA')).copy()

    x[:, :, 3] = (255 - x[:, :, :3].mean(axis=2)).astype(np.uint8)

    return Image.fromarray(x)

def white_to_transparency(img):
    x = np.asarray(img.convert('RGBA')).copy()

    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)

def white_to_transparency_threshold(img,threshold=100,dist=5):
    x = np.asarray(img.convert('RGBA')).copy()
    r,g,b,a=np.rollaxis(x,axis=-1)   

    mask=((r>threshold)
        & (g>threshold)
        & (b>threshold)
        & (np.abs(r-g)<dist)
        & (np.abs(r-b)<dist)
        & (np.abs(g-b)<dist)
        )
    x[mask,3]=0

    return Image.fromarray(x,mode='RGBA')

def round_to_white(img_arr,minval=200):
    x = img_arr.copy()

    r1,g1,b1 = minval,minval,minval
    r2,g2,b2 = 255,255,255
    r,g,b = x[:,:,0], x[:,:,1], x[:,:,2]
    mask = (r>=r1) & (g>=g1) & (b>=b1)
    x[:,:,:3][mask] = [r2,g2,b2]

    # x[x>=minval]=255
    print()
    return x

# Converts tensor to an image
def tensor_to_image(tensor,image_size=D.IMSIZE.get(),denorm=True,mode='RGB'):

    postprocessor = transforms.Compose([
        transforms.ToPILImage(mode=mode),
        transforms.Resize((image_size,image_size)),
    ])
    
    # tensor_ = tensor.detach().cpu().numpy().squeeze(0)
    tensor_ = tensor.detach().cpu().numpy()
    tensor_ = np.transpose(tensor_,(1,2,0))

    if denorm:
        temp = tensor_[...,:3]
        tensor_[...,:3] = temp * np.array(D.NORM_STD.get()) + np.array(D.NORM_MEAN.get())

    
    preimage = np.uint8(tensor_*225)
    # preimage = round_to_white(preimage,130)
    image = postprocessor(preimage)
    # image = white_to_transparency(image)
    return image



def show_images(images,save_path=None,normalize=True):
    img_grid = torchvision.utils.make_grid(images,normalize=normalize)
    plt.axis('off')
    plt.imshow(img_grid.cpu().permute(1,2,0))
    plt.show()
    if save_path:
        plt.savefig(save_path)
    
    plt.clf()


# Loads a single image
def load_image(filename,mode="RGBA"):
    img = Image.open(filename).convert(mode)
    return img


def save_gif(images:list,filename='output.gif'):
    images[0].save(filename,save_all=True,append_images=images[1:],loop=0,optimize=False,duration=75)




