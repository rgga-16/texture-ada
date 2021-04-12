import torch
from torchvision import transforms
import torch.nn.functional as F

from args import args

from defaults import DEFAULTS as D
from helpers import logger
import style_transfer as st

import numpy as np
import torch



import os, copy, time, datetime ,json


def train(generator,feat_extractor,dataloader):
    lr = args.lr
    iters = args.epochs
    generator.train()
    generator.cuda(device=D.DEVICE())

    checkpoint=5

    optim = torch.optim.Adam(generator.parameters(),lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=checkpoint,gamma=0.1)

    mse_loss = torch.nn.MSELoss()

    
    loss_history=[]
    epoch_chkpts=[]

    lowest_loss = np.inf
    best_model = generator.state_dict()

    style_layers = D.STYLE_LAYERS.get()
    # style_layers = {
    #                                     '1': 'relu1_1',   # Style layers
    #                                     '6': 'relu2_1',
    #                                     '11' : 'relu3_1',
    #                                     '20' : 'relu4_1',
    #                                 }
    s_layer_weights = D.SL_WEIGHTS.get()
    # s_layer_weights = {
    #                         layer: 0.25 for layer in style_layers.values()
    #                     }

    for i in range(iters):
        for _, sample in enumerate(dataloader):
            optim.zero_grad()

            uvs = sample['uvs']
            style = sample['style']
            
            style_feats = st.get_features(feat_extractor,style,is_style=True,style_layers=style_layers)
           

            mse_loss = torch.nn.MSELoss()

            avg_loss=0

            for uv in uvs:
                # Setup inputs 
                _,_,_,w = uv.shape
                # input_sizes = [w//2,w//4,w//8,w//16,w//32]
                # inputs = [uv[:,:3,...].clone().detach()]
                # inputs.extend([torch.rand(1,3,sz,sz,device=D.DEVICE()) for sz in input_sizes])
                # inputs.extend([F.interpolate(style,sz,mode='nearest') for sz in input_sizes])
                input_uv = uv[:,:3,...]
                input_style = style[:,:3,...]

                # Get output
                output = generator(input_uv,input_style)
                # output = generator(inputs)

                # Get output FG
                # output_mask = output[:,3,...].unsqueeze(0)

                # Get output features
                output=output[:,:3,...]
                out_feats = st.get_features(feat_extractor,output,is_style=False,style_layers=style_layers)

                # Get style loss
                style_loss=0
                for s in style_layers.values():
                    diff = mse_loss(out_feats[s],style_feats[s])
                    style_loss += s_layer_weights[s] * diff

                # Get uv FG
                # uv_mask = uv[:,3,...].unsqueeze(0)

                # Get FG MSE Loss
                # fg_loss = mse_loss(output_mask,uv_mask)
                # fg_weight = args.foreground_weight
                
                # Get loss
                # loss = (style_loss * style_weight) + (fg_loss * fg_weight)
                loss = (style_loss * args.style_weight)
                avg_loss+=loss
            
            avg_loss/= len(uvs)
            avg_loss.backward()
            optim.step()

            if avg_loss < lowest_loss:
                lowest_loss = avg_loss.item() 
                best_model = copy.deepcopy(generator.state_dict())
                best_iter = i

        if(i%checkpoint==checkpoint-1):
            print('ITER {} | LOSS: {}'.format(i+1,avg_loss.item()))
            loss_history.append(avg_loss)
            epoch_chkpts.append(i)
        
    print("Lowest Loss at epoch {}: {}".format(best_iter,lowest_loss))

    model_file = '{}_iter{}.pth'.format(best_model.__class__.__name__,best_iter)
    # model_file = '{}_iter{}.pth'.format(generator.__class__.__name__,i)
    gen_path = os.path.join(args.output_dir,model_file)
    # torch.save(best_model,gen_path)
    torch.save(generator.state_dict(),gen_path)
    print('Model saved in {}'.format(gen_path))

    losses_file = 'losses.png'
    losses_path = os.path.join(args.output_dir,losses_file)
    logger.log_losses(loss_history,epoch_chkpts,losses_path)
    print('Loss history saved in {}'.format(losses_path))
    return gen_path