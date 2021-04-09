import torch
from torch.utils.data import DataLoader

from args import args
from dataset import UV_Style_Paired_Dataset
from defaults import DEFAULTS as D
from helpers import logger, utils 
from models.texture_transfer_models import VGG19, Pyramid2D
from models.feedforward import FeedForwardNetwork
import style_transfer as st
from trainer import train
from tester import test

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


import os, copy, time, datetime ,json


def main():
    print("Starting texture transfer..")
    print("="*10)
    device = D.DEVICE()

    start=time.time()

    # Setup generator model 
    # net = Pyramid2D().to(device)
    net = FeedForwardNetwork(in_channels=3,out_channels=3).to(device)
            
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
        
        # uv map as test input only (uses rand input tensors)
        # test_uv_files = [k]
        # for uv_file in test_uv_files:
        #     test_uvs = []
        #     for test_size in args.uv_test_sizes:
        #         uv = utils.image_to_tensor(utils.load_image(os.path.join(args.uv_dir,uv_file)),image_size=test_size)
        #         test_uvs.append(uv)
        #     output_path = os.path.join(output_folder,uv_file)

        #     test(net,test_uvs,gen_path,output_path)
        
        # uv map and style as test input (replace rand inplut tensors with small style imges)
        # test_= []
        # style = utils.image_to_tensor(utils.load_image(os.path.join(args.style_dir,v)),image_size=args.style_size)
        # for test_size in args.uv_test_sizes:
        #     uv = utils.image_to_tensor(utils.load_image(os.path.join(args.uv_dir,k)),image_size=test_size)
        #     test_.append(uv)
        # test_.append(style)

        test_= []
        for test_size in args.uv_test_sizes:
            uv = utils.image_to_tensor(utils.load_image(os.path.join(args.uv_dir,k)),image_size=test_size)
            test_.append(uv)

        output_path = os.path.join(output_folder,k)

        test(net,test_,gen_path,output_path)
        
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
    






   
