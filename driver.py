import torch
import torchvision
from torchsummary import summary
import models


if __name__ == "__main__":
    print("Main Driver")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg16 = models.VGG16().to(device)

    inp = torch.randn(1,3,244,244).to(device)
    output = vgg16(inp)
    
    for layer, feature in output.items():
        print(feature.size())
    