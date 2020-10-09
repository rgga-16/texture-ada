import torch
from torch.nn import MSELoss
import torchvision

from torchvision import models
from torchvision import transforms

import losses
import utils
import dextr.segment as seg

from PIL import Image

def get_optimizer(output_img):
    optim = torch.optim.Adam([output_img.requires_grad_()],lr=1e-2)
    return optim

content_layers_default = ['relu4_2']
style_layers_default = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']

content_layer_weights_default = {
    layer: 1.0 for layer in content_layers_default
}
style_layer_weights_default = {
    layer: 0.2 for layer in style_layers_default
}

vgg19_style_layers = {
    '3': 'relu1_2',   # Style layers
    '8': 'relu2_2',
    '17' : 'relu3_3',
    '26' : 'relu4_3',
    '35' : 'relu5_3',
}

vgg19_content_layers = {
    '22' : 'relu4_2', # Content layers
    #'31' : 'relu5_2'
}


def style_transfer_gatys2(model,content, style, output, EPOCHS=500,
                        content_layers = content_layers_default,
                        style_layers = style_layers_default,
                        style_weight=1e6,content_weight=1,
                        c_layer_weights=content_layer_weights_default, 
                        s_layer_weights=style_layer_weights_default,
                        mask=None):

    # Am i not supposed to optimize the texture??
    optimizer = get_optimizer(output)
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

            for c in content_layers:
                diff = mse_loss(output_feats[c],content_feats[c])
                content_loss += c_layer_weights[c] * diff

            for s in style_layers:
                c1,c2 = output_feats[s].shape
                # diff_s = mse_loss(output_feats[s],style_feats[s]) / (c1**2)
                diff_s = mse_loss(output_feats[s],style_feats[s])
                style_loss += s_layer_weights[s] * diff_s
            
            total_loss = content_loss * content_weight + style_loss * style_weight

            total_loss.backward(retain_graph=True)
            
            run[0]+=1

            if(run[0] % 50 == True):
                print('Epoch {} | Style Loss: {} | Content Loss: {} | Total Loss: {}'.format(run[0],style_loss.item(), 
                                                                            content_loss.item(), 
                                                                            total_loss.item()))
        optimizer.step(closure)
    output.data.clamp_(0,1)
    return output

def get_features(model, tensor, 
                content_layers = content_layers_default, 
                style_layers = style_layers_default):

    features = {}
    x=tensor

    for name, layer in model._modules.items():
        x=layer(x)

        if name in vgg19_style_layers:
            features[vgg19_style_layers[name]] = losses.gram_matrix(x)
        
        if name in vgg19_content_layers:
            features[vgg19_content_layers[name]] = x


    return features


def style_transfer_gatys(cnn,content_img, style_img, output_img, 
                    normalization_mean= torch.tensor([0.485,0.456,0.406]).to(utils.setup_device()), 
                    normalization_std=torch.tensor([0.229,0.224,0.225]).to(utils.setup_device()),
                    EPOCHS=500,
                    style_weight=1e6,content_weight=1,mask=None,style_layers=style_layers_default):
    

    model, style_losses, content_losses = losses.get_model_and_losses(cnn,normalization_mean,
                                                                    normalization_std,
                                                                    style_img,content_img,mask=mask,style_layers=style_layers)

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

def interactive_style_transfer(model,content_path,style_paths,device,IMSIZE=256,EPOCHS=1000):
    save_path = content_path
    i=0
    for style_path in style_paths:

        _,mask_path = seg.segment_points(save_path,device=device)
        mask_img = utils.load_image(mask_path)
        mask = utils.image_to_tensor(mask_img,image_size=IMSIZE).to(device)

        _,style_mask_path = seg.segment_points(style_path,device=device)
        style_mask_img = utils.load_image(style_mask_path)
        style_mask = utils.image_to_tensor(style_mask_img,image_size=IMSIZE).to(device)

        content_img = utils.load_image(save_path)
        style_img = utils.load_image(style_path)

        content = utils.image_to_tensor(content_img,image_size=IMSIZE).to(device)
        style = utils.image_to_tensor(style_img,image_size=IMSIZE).to(device)

        content_clone = content.clone().detach()

        content = content * mask

        style = style * style_mask
        
        print("Mask shape: {}".format(mask.shape))
        print("content shape: {}".format(content.shape))
        print("style shape: {}".format(style.shape))

        # setup normalization mean and std
        normalization_mean = torch.tensor([0.485,0.456,0.406]).to(device)
        normalization_std = torch.tensor([0.229,0.224,0.225]).to(device)

        initial = content.clone()

        output = style_transfer_gatys(model,normalization_mean,normalization_std,content,style,initial,EPOCHS=EPOCHS)

        save_path = 'outputs/stylized_output_{}.png'.format(i+1)
        i+=1
        final = (output * mask) + (content_clone * (1-mask))
        final_img = utils.tensor_to_image(final)
        final_img.save(save_path)
