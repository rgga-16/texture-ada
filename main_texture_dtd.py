
import torch
from torch.utils.data import DataLoader,random_split

import args as args_

from seeder import SEED, init_fn
from dataset import UV_Style_Paired_Dataset, Describable_Textures_Dataset as DTD, Styles_Dataset
from defaults import DEFAULTS as D
from helpers import logger, image_utils 
from models.texture_transfer_models import FeedForward,TextureNet,AdaIN_Autoencoder,ProposedModel
from models.networks.texturenet import Pyramid2D_adain2
import style_transfer as st
from trainer import train_texture
from tester import predict_texture,evaluate_texture
import numpy as np
import multiprocessing


import os, copy, time, datetime ,json,itertools

def main():
    print("Starting texture transfer..\n")
    
    device = D.DEVICE()
    
    
    args = args_.parse_arguments()

    

    n_workers = multiprocessing.cpu_count()//2 if args.multiprocess else 0

    # Setup datasets

    # DTD
    ####################
    train_set = DTD('train')
    val_set = DTD('val')
    test_set = DTD('test')
    ####################

    train_loader = DataLoader(train_set,batch_size=32,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)
    val_loader = DataLoader(val_set,batch_size=32,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)
    test_loader = DataLoader(test_set,batch_size=32,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)
    
    
    # Filipino furniture
    ####################
    fil_dataset = Styles_Dataset(style_dir='./inputs/style_images/filipino_designer_furniture_textures',
                                style_size=round(args.style_size),
                                set='test')
    fil_dataloader = DataLoader(fil_dataset,batch_size=1,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)
  

    models = [ProposedModel()]
    for model in models:

        # Create output folder
        # This will store the model, output images, loss history chart and configurations log
        output_folder = os.path.join(args.output_dir, f'{model.__class__.__name__}')
        try:
            os.makedirs(output_folder)
        except FileExistsError:
            pass

        
        start=time.time()
        # Training. Returns path of the generator weights.
        gen_path=train_texture(model,train_loader=train_loader,val_loader=val_loader)
        # record time elapsed and configurations
        time_elapsed = time.time() - start 

        model.net.load_state_dict(torch.load(gen_path))

        # Test on DTD Test Set
        ######################################
        avg_tloss_dtd,avg_twdists_dtd=evaluate_texture(model,test_loader)

        # Test on all Filipino Designer Furniture textures
        #######################################
        tlosses=0
        twdists=0
        for j,texture in enumerate(fil_dataloader):
            filename = fil_dataset.style_files[j]
            filename = os.path.splitext(os.path.basename(filename))[0]
            test_loss,test_wdist = predict_texture(model,texture,os.path.join(output_folder,f'FDF_{filename}.png'))
            tlosses+=test_loss
            twdists+=test_wdist
        avg_tloss = tlosses / fil_dataloader.dataset.__len__()
        avg_twdists = twdists / fil_dataloader.dataset.__len__()
        #######################################

       
        log_file = 'configs.txt'
        
        logger.log_args(os.path.join(output_folder,log_file),
                        Train_Time='{:.2f}s'.format(time_elapsed),
                        Model_Name=model.__class__.__name__,
                        Seed = torch.seed(),
                        Average_CovMatrix_TestLoss_DTD = avg_tloss_dtd,
                        Average_CovMatrix_TestLoss_FDF = avg_tloss,
                        Average_WassDist_DTD=avg_twdists_dtd,
                        Average_WassDist_FDF=avg_twdists)
        print("="*10)
        print("Transfer completed. Outputs saved in {}".format(output_folder))

    
    
    

   
    
    
    # INSERT RENDERING MODULE HERE
    #######################################

    #######################################
    
   



if __name__ == "__main__":
    
    main()
    






   
