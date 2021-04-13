import torch
from torch.utils.data import DataLoader

from args import args
from dataset import UV_Style_Paired_Dataset
from defaults import DEFAULTS as D
from helpers import logger, utils 
from models.texture_transfer_models import VGG19, Pyramid2D
from models.adain import FeedForwardNetwork_AdaIN, Network_AdaIN
import style_transfer as st
from trainer import train
from tester import test

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


import os, copy, time, datetime ,json,itertools


def main():
    print("Starting texture transfer..")
    print("="*10)
    device = D.DEVICE()

    start=time.time()

    # Setup generator model 
    # net = Pyramid2D().to(device)
    net = Network_AdaIN().to(device)
            
    # Setup feature extraction model 
    feat_extractor = VGG19()
    for param in feat_extractor.parameters():
        param.requires_grad = False
    
    # Create output folder
    # This will store the model, output images, loss history chart and configurations log
    output_folder = args.output_dir
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass

    data = json.load(open(args.uv_style_pairs))
    uv_style_pairs = data['uv_style_pairs']

    uv_style_trainpairs = dict(itertools.islice(uv_style_pairs.items(),6))
    uv_style_testpairs = dict(itertools.islice(uv_style_pairs.items(),4))
    
    # Setup dataset for training
    dataset = UV_Style_Paired_Dataset(
        uv_dir=args.uv_dir,
        style_dir=args.style_dir,
        uv_sizes=args.uv_train_sizes,
        style_size=args.style_size,
        uv_style_pairs=uv_style_trainpairs
    )

    # Setup dataloader for training
    dataloader = DataLoader(dataset,num_workers=0,)

    # Training. Returns path of the generator weights.
    gen_path=train(generator=net,feat_extractor=feat_extractor,dataloader=dataloader)
    
    test_files = uv_style_testpairs

    for uv_file,style_file in test_files.items():
        test_ = []
        style = utils.image_to_tensor(utils.load_image(os.path.join(args.style_dir,style_file)),image_size=args.style_size)
        for test_size in args.uv_test_sizes:
            uv = utils.image_to_tensor(utils.load_image(os.path.join(args.uv_dir,uv_file)),image_size=test_size)
            test_.append(uv)
        test_.append(style)
        output_path = os.path.join(output_folder,uv_file)

        test(net,test_,gen_path,output_path)
    
    # record time elapsed and configurations
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
    






   
