import torch
from torch._C import Value
from torch.utils.data import DataLoader

from args import args
from dataset import UV_Style_Paired_Dataset
from defaults import DEFAULTS as D
from helpers import logger, image_utils 
from models.texture_transfer_models import Pyramid2D, VGG19, Pyramid2D_custom, Pyramid2D_adain
from models.adain import FeedForwardNetwork_AdaIN, Network_AdaIN
import style_transfer as st
from trainer import train_ulyanov, train_ulyanov_adain
from tester import test_ulyanov, test_ulyanov_adain

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
    net = Pyramid2D_adain(ch_in=3, ch_step=64,ch_out=3).to(device)
            
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

    uv_dir = None 
    if args.uv_dir is not None:
        uv_dir = args.uv_dir
    elif 'uv_dir' in data and data['uv_dir'] is not None:
        uv_dir = data['uv_dir']
    else: 
        raise ValueError('UV maps directory was not specified in terminal or in UV-Style pairs json file.')

    style_dir = None
    if args.style_dir is not None:
        style_dir = args.style_dir
    elif 'style_dir' in data and data['style_dir'] is not None:
        style_dir = data['style_dir']
    else: 
        raise ValueError('Style images directory was not specified in terminal or in UV-Style pairs json file..')
    
    # Setup dataset for training
    dataset = UV_Style_Paired_Dataset(
        uv_dir=uv_dir,
        style_dir=style_dir,
        uv_sizes=args.uv_train_sizes,
        style_size=args.style_size,
        uv_style_pairs=uv_style_trainpairs
    )

    # train_size = int(0.75 * dataset.__len__())
    # valid_size = dataset.__len__() - train_size 

    # Setup dataloader for training
    dataloader = DataLoader(dataset,num_workers=0,)

    # Training. Returns path of the generator weights.
    gen_path=train_ulyanov_adain(generator=net,feat_extractor=feat_extractor,dataloader=dataloader)
    
    test_files = uv_style_testpairs.items()

    for uv_file,style_file in test_files:
        test_uvs = []
        
        for test_size in args.uv_test_sizes:
            uv = image_utils.image_to_tensor(image_utils.load_image(os.path.join(uv_dir,uv_file)),image_size=test_size)
            test_uvs.append(uv)

        style = image_utils.image_to_tensor(image_utils.load_image(os.path.join(style_dir,style_file)),image_size=args.style_size)
        test_uvs.append(style)
        output_path = os.path.join(output_folder,uv_file)

        test_ulyanov_adain(net,test_uvs,gen_path,output_path)

    
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
    






   
