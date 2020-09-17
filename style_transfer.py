import torch
import torchvision

from torchvision import models
from torchvision import transforms

import losses
import utils

def get_optimizer(output_img):
    optim = torch.optim.Adam([output_img.requires_grad_()],lr=1e-2)
    return optim




def run_style_transfer(cnn,normalization_mean, normalization_std,
                    content_img, style_img, output_img, EPOCHS=500,
                    style_weight=1e6,content_weight=1,mask=None):
    
    model, style_losses, content_losses = losses.get_model_and_losses(cnn,normalization_mean,
                                                                    normalization_std,
                                                                    style_img,content_img,mask=mask)

    optimizer = get_optimizer(output_img)

    run = [0] 
    while run[0] <= EPOCHS:

        def closure():
            output_img.data.clamp_(0,1)
            optimizer.zero_grad()

            model(output_img)

            style_loss = 0
            content_loss = 0

            for sl in style_losses:
                style_loss += sl.loss
            
            for cl in content_losses:
                content_loss += cl.loss

            style_loss *= style_weight
            content_loss *= content_weight
            loss = style_loss + content_loss
            loss.backward()

            run[0] += 1
            if(run[0] % 50 == 0):
                print('Iter {} | Total Loss: {:4f} | Style Loss: {:4f} | Content Loss: {:4f}'.format(run[0],loss.item(),content_loss.item(),style_loss.item()))
        
        optimizer.step(closure)
    
    output_img.data.clamp_(0,1)
    return output_img


# content_file = "./data/images/chairs/generic/armchair.jpeg"
# style_file = "./data/images/chairs/cobonpue/chair-2.jpg"

content_file = "./data/images/chairs/generic/armchair.jpeg"
style_file = "./data/images/chairs/cobonpue/chair-2.jpg"

IMSIZE = 256

if __name__ == "__main__":
    print("Main Driver")

    # Load content image (C)
    content_img = utils.load_image(content_file)
    utils.show_image(content_img)

    device = utils.setup_device(use_gpu = True)
    
    # Setup model. Use pretrained VGG-19
    model = models.vgg19(pretrained=True).features.to(device).eval()

    # Preprocess C

    content = utils.image_to_tensor(content_img,IMSIZE,device=device)
    print("Content img dimensions: ", content.size())
    
    # Load style image (S)
    style_img = utils.load_image(style_file)
    # utils.show_image(style_img)

    # Preprocess S
    style = utils.image_to_tensor(style_img,IMSIZE, device=device)
    print("Style img dimensions: ", style.size())

    # setup normalization mean and std
    normalization_mean = torch.tensor([0.485,0.456,0.406]).to(device)
    normalization_std = torch.tensor([0.229,0.224,0.225]).to(device)

    initial = content.clone()

    output = run_style_transfer(model,normalization_mean,normalization_std,
                                content,style,initial,EPOCHS=2000)

    output_img = utils.tensor_to_image(output)
    
    utils.show_image(output_img)
    output_img.save('outputs/stylized_output.png')
