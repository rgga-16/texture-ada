
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

from defaults import DEFAULTS as D

from PIL import Image


def default_preprocessor(image_size):
    preprocessor = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ])
    return preprocessor

def default_normalization():
    mean = D.NORM_MEAN.get()
    std = D.NORM_STD.get()
    norm = transforms.Normalize(mean,std)

    return norm

# Loads image
def load_image(filename):
    img = Image.open(filename).convert("RGB")

    return img

# Preprocesses image and converts it to a Tensor
def image_to_tensor(image,image_size=D.IMSIZE.get(),device=D.DEVICE(),preprocessor=None, to_normalize=True):

    if preprocessor == None:
        preprocessor=default_preprocessor(image_size)

    tensor = preprocessor(image)

    # if(to_normalize):
    #     norm = default_normalization()
    #     tensor = norm(tensor)

    tensor = tensor.unsqueeze(0)
    return tensor.to(device)


# Show image
def show_image(img):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()

# Converts tensor to an image
def tensor_to_image(tensor):
    unloader = transforms.Compose([
        # transforms.Normalize([-0.485,-0.456,-0.406],[0.229,0.224,0.225]),
        # denormalization,
        transforms.ToPILImage(),
    ])
    
    image = tensor.cpu().clone()

    image = image.squeeze(0)
    image = unloader(image)

    return image
