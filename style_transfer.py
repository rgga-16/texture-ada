import torch
from torch.nn import MSELoss
import torchvision

from torchvision import models
from torchvision import transforms

import losses
import models
from models import VGG19


import utils
import dextr.segment as seg
from defaults import DEFAULTS as D

from PIL import Image
from tqdm import tqdm

def get_features(model, tensor, 
                content_layers:dict = D.CONTENT_LAYERS.get(), 
                style_layers:dict = D.STYLE_LAYERS.get()):

    features = {}
    x=tensor

    if isinstance(model,VGG19):
        model = model.features

    for name, layer in model._modules.items():
        x=layer(x)

        if name in style_layers.keys():
            features[style_layers[name]] = losses.gram_matrix(x)
        
        if name in content_layers.keys():
            features[content_layers[name]] = x


    return features

    

def style_transfer_gatys(content, style, output, 
                        model=VGG19(),EPOCHS=D.EPOCHS(),
                        content_layers = D.CONTENT_LAYERS.get(),
                        style_layers = D.STYLE_LAYERS.get(),
                        style_weight=1e6,content_weight=1,
                        c_layer_weights=D.CL_WEIGHTS.get(), 
                        s_layer_weights=D.SL_WEIGHTS.get()):

    optimizer = torch.optim.Adam([output.requires_grad_()],lr=1e-2)
    content_feats = get_features(model,content)
    style_feats = get_features(model,style)

    mse_loss = MSELoss()

    run = [0]
    while run[0] < EPOCHS:
        def closure():
            output.data.clamp_(0,1)
            optimizer.zero_grad()

            content_loss = 0
            style_loss = 0

            output_feats = get_features(model,output)

            for c in content_layers.values():
                diff = mse_loss(output_feats[c],content_feats[c])
                content_loss += c_layer_weights[c] * diff

            for s in style_layers.values():
                c1,c2 = output_feats[s].shape
                diff_s = mse_loss(output_feats[s],style_feats[s]) / (c1**2)
                style_loss += s_layer_weights[s] * diff_s
            
            content_loss = content_loss * content_weight
            style_loss = style_loss * style_weight
            
            total_loss = content_loss + style_loss

            total_loss.backward(retain_graph=True)
            
            if(run[0] % 50 == 0):
                print('Epoch {} | Style Loss: {} | Content Loss: {} | Total Loss: {}'.format(run[0],style_loss.item(), 
                                                                            content_loss.item(), 
                                                                            total_loss.item()))
            run[0]+=1
        optimizer.step(closure)
    output.data.clamp_(0,1)
    return output

def get_mean_std(feat):
    eps=1e-5

    N,C,W,H = feat.shape

    variance = feat.view(N,C,-1).var(dim=2) + eps
    std = variance.sqrt().view(N,C,1,1)
    mean = feat.view(N,C,-1).mean(dim=2).view(N,C,1,1)
    return mean,std

def style_transfer_adain(content,style,vgg=models.vgg_normalized(),alpha=1.0):

    vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
    vgg = torch.nn.Sequential(*list(vgg.children())[:31])
    vgg.eval()
    vgg.to(D.DEVICE())


    decoder = models.decoder().eval()
    decoder.load_state_dict(torch.load('models/decoder.pth'))
    decoder.to(D.DEVICE())

    with torch.no_grad():
        content_feats = vgg(content)
        style_feats = vgg(style)

        content_mean,content_std = get_mean_std(content_feats)
        style_mean, style_std = get_mean_std(style_feats)

        size = content_feats.size()

        adain_output = style_std.expand(size) * ((content_feats - content_mean.expand(size)) / content_std.expand(size)) + style_mean.expand(size)
        adain_output = adain_output * alpha + content_feats * (1-alpha)
        output = decoder(adain_output)
        
        return output









if __name__ == '__main__':

    init = torch.randn(1,3,D.IMSIZE.get(),
                        D.IMSIZE.get(),requires_grad=True,
                        device=D.DEVICE()).detach()

    
    style = utils.image_to_tensor(utils.load_image(D.STYLE_PATH())).detach()

    init_clone = init.clone().detach()

    # content_path = './data/images/others/6.jpg'
    # content = utils.image_to_tensor(utils.load_image(content_path))
    
    
    output = style_transfer_adain(init,style)
    utils.tensor_to_image(output).save('output.png')
    





