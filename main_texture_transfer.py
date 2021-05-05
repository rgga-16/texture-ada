
import torch
from torch.utils.data import DataLoader,random_split

import args as args_

from seeder import SEED, init_fn
from dataset import UV_Style_Paired_Dataset, Describable_Textures_Dataset as DTD, Styles_Dataset
from defaults import DEFAULTS as D
from helpers import logger, image_utils 
from models.texture_transfer_models import FeedForward,TextureNet,AdaIN_Autoencoder,ProposedModel
import style_transfer as st
from trainer import train_texture
from tester import test_texture
from models.structure_transfer_net import Pointnet_Autoencoder2
import numpy as np
import multiprocessing


import os, copy, time, datetime ,json,itertools

def main():
    print("Starting texture transfer..\n")
    
    device = D.DEVICE()
    
    start=time.time()
    args = args_.parse_arguments()

    # Setup generator model 
    model = ProposedModel()

    data = json.load(open(args.uv_style_pairs))
    uv_style_pairs = data['uv_style_pairs']

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
    

    n_workers = multiprocessing.cpu_count()//2 if args.multiprocess else 0

    # Setup dataset for training
    # Filipino furniture
    ####################
    fil_dataset = Styles_Dataset(style_dir='./inputs/style_images/filipino_designer_furniture_textures',
                                style_size=args.style_size,
                                set='test')
    fil_dataloader = DataLoader(fil_dataset,batch_size=8,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)
    fil_testloader = DataLoader(fil_dataset,batch_size=1,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)
    # fil_dataset = Styles_Dataset(style_dir=style_dir,style_size=args.style_size,
    #                                 style_files=uv_style_pairs.values())
    # train_size, val_size, test_size = round(0.60 * fil_dataset.__len__()),round(0.20 * fil_dataset.__len__()),round(0.20 * fil_dataset.__len__())
    # train_set, val_set, test_set = random_split(fil_dataset,[train_size,val_size,test_size],
    #                                                         generator = torch.Generator().manual_seed(SEED))
    # train_set.dataset.set='train'; val_set.dataset.set='val'; test_set.dataset.set='test'
    ####################
    # DTD
    ####################
    train_set = DTD('train')
    val_set = DTD('val')
    test_set = DTD('test')
    ####################

    # Create output folder
    # This will store the model, output images, loss history chart and configurations log
    output_folder = args.output_dir
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass

    # Setup dataloader for training
    
    train_loader = DataLoader(train_set,batch_size=8,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)
    val_loader = DataLoader(val_set,batch_size=8,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)
    test_loader = DataLoader(test_set,batch_size=1,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)

    # Training. Returns path of the generator weights.
    # gen_path=train_texture(generator=net,feat_extractor=feat_extractor,train_loader=train_loader,val_loader=val_loader)
    gen_path=train_texture(model,train_loader=fil_dataloader,val_loader=fil_dataloader)

    # Test on DTD Test Set
    #######################################
    # for i, texture in enumerate(test_loader):
    #     output_path = os.path.join(output_folder,'{}.png'.format(i))
    #     test_texture(model,texture,gen_path,output_path)
    
    # Test on Furniture pairngs
    #######################################
    # test_files = uv_style_pairs.items()
    # for uv_file,style_file in test_files:
    #     uv = image_utils.load_image(os.path.join(uv_dir,uv_file),mode='L')
    #     texture = image_utils.image_to_tensor(image_utils.load_image(os.path.join(style_dir,style_file),mode='RGB'),phase='test',image_size=args.style_size)
    #     test_texture(model,texture,gen_path,os.path.join(output_folder,uv_file),mask=uv)
    #######################################

    # Test on all Filipino Designer Furniture textures
    #######################################
    for j,texture in enumerate(fil_testloader):
        test_texture(model,texture,gen_path,os.path.join(output_folder,str(j)))
    #######################################
    
    
    # INSERT RENDERING MODULE HERE
    #######################################

    #######################################
    
    # record time elapsed and configurations
    time_elapsed = time.time() - start 
    log_file = 'configs.txt'
    
    logger.log_args(os.path.join(output_folder,log_file),
                    Time_Elapsed='{:.2f}s'.format(time_elapsed),
                    Model_Name=model.__class__.__name__,
                    Seed = torch.seed())
    print("="*10)
    print("Transfer completed. Outputs saved in {}".format(output_folder))



if __name__ == "__main__":
    
    main()
    






   
