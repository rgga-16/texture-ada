import torch
from torch.utils.data import DataLoader
import torchvision
import args as args_

from dataset import UV_Style_Paired_Dataset
from defaults import DEFAULTS as D
from helpers import logger, image_utils 
from models.base_model import BaseModel
import style_transfer as st

import numpy as np
from torch.utils.data import DataLoader
import os, copy, time, datetime ,json, matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader,random_split

import args as args_

from seeder import SEED, init_fn, set_seed
from dataset import Describable_Textures_Dataset as DTD, Styles_Dataset
from defaults import DEFAULTS as D
from helpers import logger, image_utils 
from models.texture_transfer_models import FeedForward,TextureNet,AdaIN_Autoencoder,ProposedModel
import style_transfer as st
from tester import predict_texture, evaluate_texture,predict_texture_batch
import numpy as np
import multiprocessing

import argparse


import os, copy, time, datetime ,json,itertools





def main():
    args = args_.parse_arguments()

    n_workers = multiprocessing.cpu_count()//2 if args.multiprocess else 0

    bs=2
    fil_dataset = Styles_Dataset(style_dir='./inputs/style_images/filipino_designer_furniture_textures',
                                style_size=round(args.style_size),
                                set='default')
    fil_dataloader = DataLoader(fil_dataset,batch_size=bs,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)

    models = [ProposedModel()]

    
    # seeds = [32,64,128,256,512]
    seeds=[32]
    for model in models:
        seed_losses =[]
        seed_wdists=[]
        seed_test_times=[]
        for seed in seeds:
            print(f'Running using model {model.__class__.__name__}')

            set_seed(seed)

            output_folder = os.path.join(args.output_dir, f'seed_{seed}',f'{model.__class__.__name__}')
            try:
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
            except FileExistsError:
                pass

            gen_path=args.model_path
            model.net.load_state_dict(torch.load(gen_path))

            start_test=time.time()
            avg_loss,avg_wdist=predict_texture_batch(model,fil_dataloader,output_folder)
            time_elapsed_test = time.time()-start_test
            avg_testtime= time_elapsed_test/fil_dataloader.dataset.__len__()

            seed_losses.append(avg_loss)
            seed_wdists.append(avg_wdist)
            seed_test_times.append(avg_testtime)

            # record time elapsed and configurations
            log_file = 'configs.txt'
            
            logger.log_args(os.path.join(output_folder,log_file),
                            Sample_Test_Time=f'{avg_testtime:.2f}s',
                            Model_Name=model.__class__.__name__,
                            Seed = SEED,
                            Average_CovMatrix_TestLoss = avg_loss,
                            Average_WassDist=avg_wdist,
                            batch_size=bs)
            print("="*10)
            print("Transfer completed. Outputs saved in {}".format(output_folder))
        # record time elapsed and configurations
        log_file = f'log_{model.__class__.__name__}.txt'
        logger.log_args(os.path.join(args.output_dir,log_file),
                        Overall_Sample_TestTimes=f'{np.mean(np.array(seed_test_times)):.2f}s',
                        Model_Name=model.__class__.__name__,
                        Seed = seeds,
                        Overall_CovMatrix_Loss = np.mean(np.array(seed_losses)),
                        Overall_WassDist=np.mean(np.array(seed_wdists)),
                        batch_size=bs)


if __name__ == "__main__":
    
    main()
    






   
