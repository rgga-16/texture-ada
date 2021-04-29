import torch
from torch.utils.data import DataLoader
import torchvision
import args as args_

from dataset import UV_Style_Paired_Dataset
from defaults import DEFAULTS as D
from helpers import logger, image_utils 
from models.texture_transfer_models import VGG19, Pyramid2D
import style_transfer as st

import numpy as np
from torch.utils.data import DataLoader
import os, copy, time, datetime ,json, matplotlib.pyplot as plt



def test_texture(generator,texture,gen_path,output_path,mask=None):
    args = args_.parse_arguments()
    generator.eval()
    generator.to(device=D.DEVICE())

    generator.load_state_dict(torch.load(gen_path))
    
    w = args.uv_test_sizes[0]
    inputs = [torch.rand(1,3,sz,sz,device=D.DEVICE()) for sz in [w, w//2,w//4,w//8,w//16,w//32]]
    ################################ Input for AdaIN
    inputs = torch.rand(1,3,w,w,device=D.DEVICE())
    ################################ Input for AdaIN
    style_input = texture.expand(1,-1,-1,-1).clone().detach()
    
    with torch.no_grad():
        output = generator(inputs,style_input)

    output_path_ = '{}_{}.png'.format(output_path,w)
    output_image = image_utils.tensor_to_image(output,image_size=args.output_size)

    img_grid = torchvision.utils.make_grid(torch.cat((style_input,output),dim=0),normalize=True)
    plt.imshow(img_grid.cpu().permute(1,2,0))
    plt.savefig(f'{output_path}_compare.png')
    
    if mask is not None:
        mask = mask.resize(output_image.size) if mask.size != output_image.size else ...
        output_image.putalpha(mask)
    output_image.save(output_path_,'PNG')
    print('Saving image as {}'.format(output_path_))


# def test_texture2(generator,texture,gen_path,output_path):
#     args = args_.parse_arguments()
#     generator.eval()
#     generator.to(device=D.DEVICE())

#     generator.load_state_dict(torch.load(gen_path))
    
#     w = args.uv_test_sizes[0]
#     inputs = [torch.rand(1,3,sz,sz,device=D.DEVICE()) for sz in [w, w//2,w//4,w//8,w//16,w//32]]
#     ################################ Input for AdaIN
#     inputs = torch.rand(1,3,w,w,device=D.DEVICE())
#     ################################ Input for AdaIN
#     style_input = texture.expand(1,-1,-1,-1).clone().detach()
    
#     with torch.no_grad():
#         output = generator(inputs,style_input)

#     output_path_ = '{}_{}.png'.format(output_path,w)
#     output_image = image_utils.tensor_to_image(output,image_size=args.output_size)
#     style_img = image_utils.tensor_to_image(style_input,image_size=args.output_size)
#     output_image.save(output_path_,'PNG')
#     print('Saving image as {}'.format(output_path_))


# def test(generator,input,gen_path,output_path):
#     generator.eval()
#     generator.cuda(device=D.DEVICE())

#     generator.load_state_dict(torch.load(gen_path))
    
#     # uvs = input[:-1]
#     # style=input[-1][:3,...].unsqueeze(0).detach()
#     uvs=input[:-1]
#     style=input[-1]
#     for uv in uvs:
#         _,_,w = uv.shape
#         input_sizes = [w//2,w//4,w//8,w//16,w//32]
#         inputs = uv[:3,...].unsqueeze(0).clone().detach()
#         uv_mask = uv[3,...].expand(1,1,-1,-1).clone().detach()
#         input_style = style[:3,...].unsqueeze(0).clone().detach()

#         with torch.no_grad():
#             output = generator(inputs,input_style)

#         output_path_ = '{}_{}.png'.format(output_path,w)
#         output_image = image_utils.tensor_to_image(output,image_size=args.output_size)
#         mask = image_utils.tensor_to_image(uv_mask,image_size=args.output_size,denorm=False)
#         output_image.putalpha(mask)
#         output_image.save(output_path_,'PNG')
#         print('Saving image as {}'.format(output_path_))

# def test_ulyanov(generator,input,gen_path,output_path):
#     generator.eval()
#     generator.cuda(device=D.DEVICE())

#     generator.load_state_dict(torch.load(gen_path))
    
#     uvs = input
#     for uv in uvs:
#         _,_,w = uv.shape
#         input_sizes = [w//2,w//4,w//8,w//16,w//32]
#         inputs = [uv[:3,...].unsqueeze(0).clone().detach()]
#         inputs.extend([torch.rand(1,3,sz,sz,device=D.DEVICE()) for sz in input_sizes])
#         uv_mask = uv[3,...].expand(1,1,-1,-1).clone().detach()

#         with torch.no_grad():
#             output = generator(inputs)

#         output_path_ = '{}_{}.png'.format(output_path,w)
#         output_image = image_utils.tensor_to_image(output,image_size=args.output_size)
#         mask = image_utils.tensor_to_image(uv_mask,image_size=args.output_size,denorm=False)
#         output_image.putalpha(mask)
#         output_image.save(output_path_,'PNG')

#         colors = ['r','g','b']
#         for i in range(len(colors)):
#             output_1channel = output.clone()[:,i,...]

#             tensor = None 
#             if colors[i]=='r':
#                 r = output_1channel
#                 g = torch.zeros_like(r)
#                 b = torch.zeros_like(r)
#                 tensor = torch.stack([r,g,b],dim=1)
            
#             elif colors[i]=='g':
#                 g = output_1channel
#                 r = torch.zeros_like(g) 
#                 b = torch.zeros_like(g)
#                 tensor = torch.stack([r,g,b],dim=1)
            
#             elif colors[i]=='b':
#                 b = output_1channel
#                 r = torch.zeros_like(b)
#                 g = torch.zeros_like(b) 
#                 tensor = torch.stack([r,g,b],dim=1)

#             output_1channel_image = image_utils.tensor_to_image(tensor,image_size = args.output_size)
#             output_1channel_image.putalpha(mask)
#             output_1channel_image.save('{}_{}.png'.format(output_path_[:-4],colors[i]),'PNG')


#         print('Saving image as {}'.format(output_path_))



# def test_ulyanov_adain(generator,input,gen_path,output_path):
#     generator.eval()
#     generator.cuda(device=D.DEVICE())

#     generator.load_state_dict(torch.load(gen_path))
    
#     # uvs = input
#     uvs=input[:-1]
#     style=input[-1]
#     for uv in uvs:
#         _,_,w = uv.shape
#         input_sizes = [w//2,w//4,w//8,w//16,w//32]
#         # inputs = [uv[:3,...].unsqueeze(0).clone().detach()]
#         inputs = [torch.rand(1,3,w,w,device=D.DEVICE())]
#         inputs.extend([torch.rand(1,3,sz,sz,device=D.DEVICE()) for sz in input_sizes])
#         uv_mask = uv[3,...].expand(1,1,-1,-1).clone().detach()
#         style_input = style[:3,...].unsqueeze(0).clone().detach()
#         with torch.no_grad():
#             output = generator(inputs,style_input)

#         output_path_ = '{}_{}.png'.format(output_path,w)
#         output_image = image_utils.tensor_to_image(output,image_size=args.output_size)
#         mask = image_utils.tensor_to_image(uv_mask,image_size=args.output_size,denorm=False)
#         output_image.putalpha(mask)
#         output_image.save(output_path_,'PNG')
#         print('Saving image as {}'.format(output_path_))
