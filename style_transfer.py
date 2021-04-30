import torch
from torch.nn import MSELoss
import torchvision

from torchvision import models
from torchvision import transforms

import losses
import models
from models.texture_transfer_models import VGG19
import args as args_



import helpers.image_utils as utils
from defaults import DEFAULTS as D

from PIL import Image
from tqdm import tqdm
import numpy as np 

def filter_k_feature_maps(raw_feature_maps,k):
    feature_maps = raw_feature_maps[0]
    feature_maps_total_num = feature_maps.shape[0]

    activation_list = []

    # Add max activation value for each feature map into a list
    for i in range(feature_maps_total_num):
        feat_map = feature_maps[i, :, :]
        activation_val = torch.max(feat_map)
        # activation_val = torch.mean(feat_map)
        activation_list.append(activation_val.item())

    # Get k indices of k highest activation values in the list
    ids = np.argpartition(np.array(activation_list),-k)[-k:]
    
    for i in range(feature_maps_total_num):
        if i not in ids:
            feature_maps[i, :, :] = 0
        else:
            max_map = feature_maps[i, :, :].clone().detach()
            activation = np.array(activation_list)[i]
            max_map = torch.where(max_map == activation.item(),max_map,torch.zeros(max_map.shape).to(D.DEVICE()))
            feature_maps[i,:,:]=max_map

    thing = feature_maps.view(feature_maps.shape[0],-1)
    return feature_maps.unsqueeze_(0)


def get_features(model, tensor, is_style=False,
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
            # if is_style:
            #     _,c,_,_ = x.shape
            #     k = round(0.05 * c) 
            #     x = filter_k_feature_maps(x,c)
            # features[style_layers[name]] = losses.covariance_matrix(x)
            # losses.covariance_matrix(x)
            # losses.weighted_style_rep(x)
        
        if name in content_layers.keys():
            features[content_layers[name]] = x


    return features

    

def style_transfer_gatys(content, style,
                        model=VGG19(),EPOCHS=D.EPOCHS(),
                        content_layers = D.CONTENT_LAYERS.get(),
                        style_layers = D.STYLE_LAYERS.get(),
                        style_weight=1e6,content_weight=1,
                        c_layer_weights=D.CL_WEIGHTS.get(), 
                        s_layer_weights=D.SL_WEIGHTS.get()):

    output = content.clone().detach()
    output.requires_grad_(True)

    optimizer = torch.optim.Adam([output],lr=1e-2)
    content_feats = get_features(model,content,is_style=False)

                                
    style_feats = get_features(model,style,is_style=True)

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
                print('Epoch {} | Style Loss: {:.4f} | Content Loss: {:.4f} | Total Loss: {:.4f}'.format(run[0],style_loss.item(), 
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


if __name__ == '__main__':
    import os
    args = args_.parse_arguments()
    # style_file = 'chair-5_masked.png'
    # content_file = 'uv_map_right_foot.png'

    style_path = args.style
    style_name = os.path.splitext(os.path.basename(style_path))[0]
    style_image = utils.load_image(style_path,mode='RGB')

    for imsize in [128,256,512,768]:

        style = utils.image_to_tensor(style_image,image_size=imsize,normalize=True).detach()

        init = torch.full([1,3,D.IMSIZE.get(),
                            D.IMSIZE.get()],1.0,requires_grad=True,
                            device=D.DEVICE())
        
        # content_path = os.path.join('inputs/uv_maps',content_file)
        # content_image = utils.load_image(content_path,mode='RGB')
        # content = utils.image_to_tensor(content_image,normalize=True)

        output = style_transfer_gatys(init,style,EPOCHS=2000,content_weight=0)
        # output_dir = 'inputs/texture_maps/gatys/uv_maps'
        output_dir = './'
        
        utils.tensor_to_image(output,denorm=True).save(os.path.join(output_dir,'{}_{}.png'.format(style_name,imsize)))






