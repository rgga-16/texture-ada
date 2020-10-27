

import torch
import torch.nn as nn
from torchvision import models

from models import VGG19
import utils
from defaults import DEFAULTS as D 


import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

if __name__ == '__main__':


    content = utils.image_to_tensor(utils.load_image(D.STYLE_PATH())).detach()

    # mask = utils.image_to_tensor(utils.load_image(D.MASK_PATH())).detach()
    # content=content*mask
    # Create style feature extractor model
    model = VGG19()

    set_ = 1
    relu_inst = 1
    conv_inst = 1
    layer_num = 0
    
    visualize_layers = {}

    for layer in model.modules():
        if isinstance(layer,nn.ReLU):
            name='relu{}_{}'.format(set_,relu_inst)
            visualize_layers[str(layer_num)]=name
            layer_num+=1
            relu_inst+=1
            
        elif isinstance(layer,nn.Conv2d):
            name='conv{}_{}'.format(set_,conv_inst)
            visualize_layers[str(layer_num)]=name
            conv_inst+=1
            layer_num+=1
            
        elif isinstance(layer,nn.MaxPool2d):
            name='maxpool_{}'.format(set_)
            visualize_layers[str(layer_num)]=name
            relu_inst=1
            conv_inst=1
            set_+=1

        

    content_feats = model(content,visualize_layers)

    mse_loss = nn.MSELoss()

    loss=0

    feature_maps = {}
    num=0
    epochs = 50
    for key, val in visualize_layers.items():
        img = torch.zeros(1,3,D.IMSIZE.get(),D.IMSIZE.get(),requires_grad=True,device=D.DEVICE())
        optimizer = torch.optim.Adam([img],lr=0.1)
        progress = tqdm(range(epochs))
        for i in progress:
            img.data.clamp_(0,1)
            optimizer.zero_grad()
            output_feats = model(img,{key:val})
            loss = mse_loss(output_feats[val],content_feats[val])
            
            loss.backward(retain_graph=True)
            optimizer.step()
   
            progress.set_description("Image {} | Loss: {}".format(num+1,loss.item()))
        img.data.clamp_(0,1) 
        feature_maps[val]=img
        num+=1

    axes = []
    fig=plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title('Feature Map Visualizations')
    subplot = 0
    for k,v in feature_maps.items():
        axes.append(fig.add_subplot(8,5,subplot+1))

        for n,l in visualize_layers.items():
            if l == k:
                layer_num = n
                break

        subplot_title = ("Layer {}: {}".format(layer_num if layer_num is not None else '?', k))
        axes[-1].set_title(subplot_title,size=6)
        axes[-1].axis('off')
        display_img = np.asarray(utils.tensor_to_image(v))
        plt.imshow(display_img)
        subplot+=1

    fig.tight_layout()
    # fig.suptitle('Feature Map Visualizations')
    plt.show()
    plt.savefig('Feature Map Visualizations.png')
   


