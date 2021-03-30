


import torch
from torch.utils.data import DataLoader

from args import args
from dataset import UV_Style_Paired_Dataset
from defaults import DEFAULTS as D
from helpers import logger, utils 
from models import VGG19, ConvAutoencoder,TextureNet, Pyramid2D
import style_transfer as st

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


import os, copy, time, datetime ,json


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

    lowest_loss = np.inf
    best_model = generator.state_dict()

    for i in range(iters):
        for _, sample in enumerate(dataloader):
            optim.zero_grad()

            uvs = sample['uvs']
            style = sample['style']
            
            style_feats = st.get_features(feat_extractor,style,is_style=True)
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
                out_feats = st.get_features(feat_extractor,output,is_style=False)

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

            if avg_loss < lowest_loss:
                lowest_loss = avg_loss.item() 
                best_model = copy.deepcopy(generator.state_dict())
                best_iter = i

        if(i%checkpoint==checkpoint-1):
            print('ITER {} | LOSS: {}'.format(i+1,avg_loss.item()))
            loss_history.append(avg_loss)
            epoch_chkpts.append(i)

    model_file = '{}_iter{}.pth'.format(generator.__class__.__name__,best_iter)
    gen_path = os.path.join(args.output_dir,model_file)
    torch.save(best_model,gen_path)
    print('Model saved in {}'.format(gen_path))

    losses_file = 'losses.png'
    losses_path = os.path.join(args.output_dir,losses_file)
    logger.log_losses(loss_history,epoch_chkpts,losses_path)
    print('Loss history saved in {}'.format(losses_path))
    return gen_path

def test(generator,input,gen_path,output_path):
    generator.eval()
    generator.cuda(device=D.DEVICE())

    generator.load_state_dict(torch.load(gen_path))
    
    uvs = input
    for uv in uvs:
        _,_,w = uv.shape
        input_sizes = [w//2,w//4,w//8,w//16,w//32]
        inputs = [uv[:3,...].unsqueeze(0).detach()]
        inputs.extend([torch.rand(1,3,sz,sz,device=D.DEVICE()) for sz in input_sizes])

        y = generator(inputs)

        output_path_ = '{}_{}.png'.format(output_path,w)
        utils.tensor_to_image(y,image_size=args.output_size).save(output_path_)
        print('Saving image as {}'.format(output_path_))
 

def main():
    print("Starting texture transfer..")
    print("="*10)
    device = D.DEVICE()

    start=time.time()

    # Setup generator model 
    net = Pyramid2D().to(device)
            
    # Setup feature extraction model 
    feat_extractor = VGG19()
    for param in feat_extractor.parameters():
        param.requires_grad = False
    
    # Create output folder named date today and time (ex. [3-12-21 17-00-04])
    # This will store the model, output images, loss history chart and configurations log
    output_folder = args.output_dir
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass

    data = json.load(open(args.uv_style_pairs))
    uv_style_pairs = data['uv_style_pairs']

    for k,v in uv_style_pairs.items():
        # Setup dataset for training
        dataset = UV_Style_Paired_Dataset(
            uv_dir=args.uv_dir,
            style_dir=args.style_dir,
            uv_sizes=args.uv_train_sizes,
            style_size=args.style_size,
            uv_style_pairs={k:v}
        )

        # Setup dataloader for training
        dataloader = DataLoader(dataset,num_workers=0,)

        # Training. Returns path of the generator weights.
        gen_path=train(generator=net,feat_extractor=feat_extractor,dataloader=dataloader)
        
        test_uv_files = [k]

        for uv_file in test_uv_files:
            test_uvs = []
            for test_size in args.uv_test_sizes:
                uv = utils.image_to_tensor(utils.load_image(os.path.join(args.uv_dir,uv_file)),image_size=test_size)
                test_uvs.append(uv)
            output_path = os.path.join(output_folder,uv_file)

            test(net,test_uvs,gen_path,output_path)
        
    # record losses and configurations
    time_elapsed = time.time() - start 
        
    log_file = 'configs.txt'
    
    logger.log_args(os.path.join(output_folder,log_file),
                    Time_Elapsed='{:.2f}s'.format(time_elapsed),
                    Model_Name=net.__class__.__name__,
                    Seed = torch.seed())
    print("="*10)
    print("Transfer completed. Outputs saved in {}".format(output_folder))



if __name__ == "__main__":
    
    main()
    






   
