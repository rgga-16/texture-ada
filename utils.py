
import matplotlib.pyplot as plt
from torchvision import transforms
import torch


from PIL import Image


def setup_device(use_gpu):
    device = torch.device('cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu')
    return device

def default_preprocessor(image_size):
    preprocessor = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ])
    return preprocessor



# Loads image
def load_image(filename):
    img = Image.open(filename)

    return img

# Preprocesses image and converts it to a Tensor
def image_to_tensor(image,image_size=256,device=setup_device(True),preprocessor=None):

    if preprocessor == None:
        preprocessor=default_preprocessor(image_size)

    tensor = preprocessor(image).unsqueeze(0)
    return tensor.to(device)


# Show image
def show_image(img):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()

# Converts tensor to an image
def tensor_to_image(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()

    image = image.squeeze(0)
    image = unloader(image)

    return image
