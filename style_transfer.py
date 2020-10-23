import torch
from torch.nn import MSELoss
import torchvision

from torchvision import models
from torchvision import transforms

import losses
from vgg import VGG19
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


def style_transfer_gatys(model,content, style, output, EPOCHS=D.EPOCHS(),
                        content_layers = D.CONTENT_LAYERS.get(),
                        style_layers = D.STYLE_LAYERS.get(),
                        style_weight=1e6,content_weight=0,
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



if __name__ == '__main__':

    init = torch.randn(1,3,D.IMSIZE.get(),
                        D.IMSIZE.get(),requires_grad=True,
                        device=D.DEVICE())

    
    style = utils.image_to_tensor(utils.load_image(D.STYLE_PATH()))


    content = init.clone().detach()
    model = VGG19()
    
    output = style_transfer_gatys(model,content,style,init)
    utils.tensor_to_image(output).save('output.png')
    

















# def style_transfer_gatys(cnn,content_img, style_img, output_img, 
#                     normalization_mean= torch.tensor([0.485,0.456,0.406]).to(utils.setup_device()), 
#                     normalization_std=torch.tensor([0.229,0.224,0.225]).to(utils.setup_device()),
#                     EPOCHS=500,style_weight=1e6,content_weight=1,
#                     mask=None,style_layers=D.STYLE_LAYERS.get().values()):
    

#     model, style_losses, content_losses = losses.get_model_and_losses(cnn,normalization_mean,
#                                                                     normalization_std,
#                                                                     style_img,content_img,mask=mask,style_layers=style_layers)

#     optimizer = get_optimizer(output_img)

#     run = [0] 
#     while run[0] <= EPOCHS:

#         def closure():
#             output_img.data.clamp_(0,1)
#             optimizer.zero_grad()

#             model(output_img)

#             style_loss = 0
#             content_loss = 0

#             for sl in style_losses:
#                 style_loss += sl.loss
            
#             for cl in content_losses:
#                 content_loss += cl.loss

#             style_loss *= style_weight
#             content_loss *= content_weight
#             loss = style_loss + content_loss
#             loss.backward()

#             run[0] += 1
#             if(run[0] % 50 == 0):
#                 print('Iter {} | Total Loss: {:4f} | Style Loss: {:4f} | Content Loss: {:4f}'.format(run[0],loss.item(),content_loss.item(),style_loss.item()))
        
#         optimizer.step(closure)
    
#     output_img.data.clamp_(0,1)
#     return output_img

# def interactive_style_transfer(model,content_path,style_paths,device,IMSIZE=256,EPOCHS=1000):
#     save_path = content_path
#     i=0
#     for style_path in style_paths:

#         _,mask_path = seg.segment_points(save_path,device=device)
#         mask_img = utils.load_image(mask_path)
#         mask = utils.image_to_tensor(mask_img,image_size=IMSIZE).to(device)

#         _,style_mask_path = seg.segment_points(style_path,device=device)
#         style_mask_img = utils.load_image(style_mask_path)
#         style_mask = utils.image_to_tensor(style_mask_img,image_size=IMSIZE).to(device)

#         content_img = utils.load_image(save_path)
#         style_img = utils.load_image(style_path)

#         content = utils.image_to_tensor(content_img,image_size=IMSIZE).to(device)
#         style = utils.image_to_tensor(style_img,image_size=IMSIZE).to(device)

#         content_clone = content.clone().detach()

#         content = content * mask

#         style = style * style_mask
        
#         print("Mask shape: {}".format(mask.shape))
#         print("content shape: {}".format(content.shape))
#         print("style shape: {}".format(style.shape))

#         # setup normalization mean and std
#         normalization_mean = torch.tensor([0.485,0.456,0.406]).to(device)
#         normalization_std = torch.tensor([0.229,0.224,0.225]).to(device)

#         initial = content.clone()

#         output = style_transfer_gatys(model,normalization_mean,normalization_std,content,style,initial,EPOCHS=EPOCHS)

#         save_path = 'outputs/stylized_output_{}.png'.format(i+1)
#         i+=1
#         final = (output * mask) + (content_clone * (1-mask))
#         final_img = utils.tensor_to_image(final)
#         final_img.save(save_path)
