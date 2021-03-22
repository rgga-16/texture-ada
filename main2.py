# SEED=0


import torch
from torch.utils.data import DataLoader
# torch.manual_seed(SEED)
from torchvision import transforms

import style_transfer as st

from helpers import logger, utils 
import args

from models import VGG19, ConvAutoencoder,TextureNet, Pyramid2D

from args import args
import os
import time
import datetime

from dataset import UV_Style_Paired_Dataset


from defaults import DEFAULTS as D
from torchsummary import summary


def train(generator,feat_extractor,dataloader):
    lr = args.lr
    iters = args.epochs
    generator.train()
    generator.cuda(device=D.DEVICE())

    optim = torch.optim.Adam(generator.parameters(),lr=lr)

    mse_loss = torch.nn.MSELoss()

    checkpoint=100
    loss_history=[]
    epoch_chkpts=[]
    for i in range(iters):
        for _, sample in enumerate(dataloader):
            optim.zero_grad()

            uvs = sample['uvs']
            style = sample['style']
            
            style_feats = st.get_features(feat_extractor,style)
            style_layers = D.STYLE_LAYERS.get()
            s_layer_weights = D.SL_WEIGHTS.get()

            mse_loss = torch.nn.MSELoss()

            avg_loss=0

            for uv in uvs:
                # Setup inputs 
                _,_,_,w = uv.shape
                input_sizes = [w//2,w//4,w//8,w//16,w//32]
                inputs = [uv[:,:3,...].clone().detach()]
                inputs.extend([torch.rand(1,3,sz,sz,device=D.DEVICE()) for sz in input_sizes])

                # Get output
                output = generator(inputs)

                # Get output FG
                output_mask = output[:,3,...].unsqueeze(0)

                # Get output features
                output=output[:,:3,...]
                out_feats = st.get_features(feat_extractor,output)

                # Get style loss
                style_loss=0
                for s in style_layers.values():
                    diff = mse_loss(out_feats[s],style_feats[s])
                    style_loss += s_layer_weights[s] * diff
                    style_weight=args.style_weight

                # Get uv FG
                uv_mask = uv[:,3,...].unsqueeze(0)

                # Get FG MSE Loss
                fg_loss = mse_loss(output_mask,uv_mask)
                fg_weight = args.foreground_weight
                
                # Get loss
                loss = (style_loss * style_weight) + (fg_loss * fg_weight)
                avg_loss+=loss
            
            avg_loss/= len(uvs)
            avg_loss.backward()
            optim.step()

        if(i%checkpoint==checkpoint-1):
            print('ITER {} | LOSS: {}'.format(i+1,avg_loss.item()))
            loss_history.append(avg_loss)
            epoch_chkpts.append(i)

    today = datetime.datetime.today().strftime('%y-%m-%d %H-%M')
    model_file = '[{}]{}-{}_iters.pth'.format(today,generator.__class__.__name__,iters)
    gen_path = os.path.join(D.MODEL_DIR.get(),model_file)
    print('Model saved in {}'.format(gen_path))
    torch.save(generator.state_dict(),gen_path)

    # losses_file = '[{}]-losses.png'.format(today)
    losses_file = 'losses.png'.format(today)
    losses_path = os.path.join(args.output_dir,'outputs',losses_file)
    logger.log_losses(loss_history,epoch_chkpts,losses_path)
    print('Loss history saved in {}'.format(losses_path))
    return gen_path

def test(generator,input,gen_path,output_dir):
    generator.eval()
    generator.cuda(device=D.DEVICE())

    generator.load_state_dict(torch.load(gen_path))
    
    uvs = input['uvs']
    for uv in uvs:
        _,_,w = uv.shape
        input_sizes = [w//2,w//4,w//8,w//16,w//32]
        inputs = [uv[:3,...].unsqueeze(0).detach()]
        inputs.extend([torch.rand(1,3,sz,sz,device=D.DEVICE()) for sz in input_sizes])

        y = generator(inputs)
        # y = y.clamp(0,1)
        _,_,_,h = y.shape

        output_file = '{}.png'.format(w)
        utils.tensor_to_image(y,image_size=args.imsize).save(os.path.join(output_dir,output_file))
        print('Saving image as {}'.format(output_dir))
 

def main():
    print("Starting texture transfer..")
    print("="*10)
    device = D.DEVICE()

    imsize = args.imsize

    start=time.time()
    

    # Setup generator model 
    net = Pyramid2D().to(device)
            
    # Setup feature extraction model 
    feat_extractor = VGG19()
    for param in feat_extractor.parameters():
        param.requires_grad = False
    
    # Initialize dataset for training
    dataset = UV_Style_Paired_Dataset(
        uv_dir=args.content_dir,
        style_dir=args.style_dir,
        uv_sizes=[128,256],
        style_size=imsize,
    )

    dataloader = DataLoader(dataset,num_workers=0,)

    gen_path=train(generator=net,feat_extractor=feat_extractor,dataloader=dataloader)

    date = datetime.datetime.today().strftime('%m-%d-%y %H-%M-%S')
    output_folder = os.path.join(args.output_dir,"[{}]".format(date))
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass
    
    test_inputs = dataset.__getitem__(0)

    test(net,test_inputs,gen_path,output_folder)
    
    # record losses and configurations
    time_elapsed = time.time() - start 
    
    log_file = '[{}]_log.txt'.format(date)
    
    logger.log_args(os.path.join(output_folder,log_file),
                    Time_Elapsed=time_elapsed,
                    Model_Name=net.__class__.__name__,
                    Seed = torch.seed())
    print("="*10)
    print("Transfer completed. Outputs saved in {}".format(output_folder))



if __name__ == "__main__":
    
    main()
    






   
