
import torch
from torch.utils.data import DataLoader,random_split

import args as args_

import seeder
from seeder import SEED, init_fn
from dataset import UV_Style_Paired_Dataset, Describable_Textures_Dataset as DTD, Styles_Dataset
from defaults import DEFAULTS as D
from helpers import logger, image_utils 
from models.texture_transfer_models import FeedForward,TextureNet,AdaIN_Autoencoder,ProposedModel
from models.networks.texturenet import Pyramid2D_adain2
import style_transfer as st
from trainer import train_texture
from tester import predict_texture, evaluate_texture
import numpy as np
import multiprocessing


import os, copy, time, datetime ,json,itertools

def main():
    print("Starting texture transfer..\n")
    
    device = D.DEVICE()

    args = args_.parse_arguments()

    n_workers = multiprocessing.cpu_count()//2 if args.multiprocess else 0

    # Setup dataset for training
    # Filipino furniture
    ####################
    bs=2
    fil_dataset = Styles_Dataset(style_dir=args.style_dir,
                                style_size=round(args.style_size),
                                set='default')
    fil_trainloader = DataLoader(fil_dataset,batch_size=bs,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)                            
    fil_testloader = DataLoader(fil_dataset,batch_size=bs,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)
    fil_dataloader = DataLoader(fil_dataset,batch_size=1,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)

    models = [ProposedModel()]
    for model in models:
        print(f'Running using model {model.__class__.__name__}')

        # Create output folder
        # This will store the model, output images, loss history chart and configurations log
        output_folder = os.path.join(args.output_dir, f'{model.__class__.__name__}')
        try:
            os.makedirs(output_folder)
        except FileExistsError:
            pass

        # Training. Returns path of the generator weights.
        start=time.time()
        gen_path=train_texture(model,train_loader=fil_trainloader,val_loader=fil_trainloader)
        time_elapsed = time.time() - start 

        model.net.load_state_dict(torch.load(gen_path))

        start_test=time.time()
        avg_tloss_fdf,avg_twdists_fdf=evaluate_texture(model,fil_testloader)
        time_elapsed_test = time.time()-start_test
        avg_testtime= time_elapsed_test/fil_dataloader.dataset.__len__()

        # Test on all Filipino Designer Furniture textures
        #######################################    

        # for j,texture in enumerate(fil_dataloader):
        for j in range(0,fil_dataset.__len__()):
            texture = fil_dataset.__getitem__(j)
            filename = os.path.splitext(os.path.basename(fil_dataset.style_files[j]))[0]
            predict_texture(model,texture,os.path.join(output_folder,f'FDF_{filename}.png'))

        #######################################

        # record time elapsed and configurations
        
        log_file = 'configs.txt'
        
        logger.log_args(os.path.join(output_folder,log_file),
                        Train_Time='{:.2f}s'.format(time_elapsed),
                        Avg_Test_Time=f'{avg_testtime:.2f}s',
                        Model_Name=model.__class__.__name__,
                        Seed = SEED,
                        Average_CovMatrix_TestLoss = avg_tloss_fdf,
                        Average_WassDist=avg_twdists_fdf,
                        batch_size=bs)
        print("="*10)
        print("Transfer completed. Outputs saved in {}".format(output_folder))

    
    # INSERT RENDERING MODULE HERE
    #######################################

    #######################################
    
   



if __name__ == "__main__":
    
    main()
    






   
