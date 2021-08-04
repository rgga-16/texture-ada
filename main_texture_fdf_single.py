
import torch
from torch.utils.data import DataLoader,random_split

import args as args_

from helpers.seeder import SEED, init_fn
from data.dataset import Describable_Textures_Dataset as DTD, Styles_Dataset
from defaults import DEFAULTS as D
from helpers import logger, image_utils 
from models.texture_transfer_models import FeedForward,TextureNet,AdaIN_Autoencoder,ProposedModel
from models.networks.texturenet import Pyramid2D_adain2
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
    fil_dataset = Styles_Dataset(style_dir='./inputs/style_images/filipino_designer_furniture_textures',
                                style_size=round(args.style_size),
                                set='default',lower_size=17)

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
        
        avg_train_time=0
        avg_test_time=0
        tlosses = {}
        twdists= {}
        for i in range(0,fil_dataset.__len__()):
            model_ = model.__class__()
            texture_file = fil_dataset.style_files[i]

            single_dataset = Styles_Dataset(style_dir='./inputs/style_images/filipino_designer_furniture_textures',
                                            style_size=round(args.style_size),set='default',
                                            style_files=[os.path.basename(texture_file)])

            single_loader = DataLoader(single_dataset,batch_size=1,worker_init_fn=init_fn,shuffle=True,num_workers=n_workers)

            # Training. Returns path of the generator weights.
            start_train=time.time()
            gen_path=train_texture(model_,train_loader=single_loader,val_loader=single_loader)
            avg_train_time += time.time() - start_train 

            model_.net.load_state_dict(torch.load(gen_path))

            start_test=time.time()
            tloss,twdist=evaluate_texture(model_,single_loader)
            avg_test_time += time.time()-start_test

            tlosses[os.path.basename(texture_file)]=tloss 
            twdists[os.path.basename(texture_file)]=twdist 

            # Generate final texture
            #######################################    
            for j,texture in enumerate(single_loader):
                filename = single_dataset.style_files[j]
                filename = os.path.splitext(os.path.basename(filename))[0]
                predict_texture(model_,texture,os.path.join(output_folder,f'FDF_{filename}.png'))
            #######################################
        avg_train_time/=fil_dataset.__len__()
        avg_test_time/=fil_dataset.__len__()

        tlosses_l = np.array(list(tlosses.values()))
        twdists_l = np.array(list(twdists.values()))
        avg_tloss = np.mean(tlosses_l)
        avg_twdists = np.mean(twdists_l)

        log_file = 'configs.txt'
            
        logger.log_args(os.path.join(output_folder,log_file),
                        Train_Time='{:.2f}s'.format(avg_train_time),
                        Avg_Test_Time=f'{avg_test_time:.2f}s',
                        Model_Name=model_.net.__class__.__name__,
                        Seed = SEED,
                        Test_Losses = tlosses,
                        WassDists = twdists,
                        Average_CovMatrix_TestLoss = avg_tloss,
                        Average_WassDist=avg_twdists,
                        batch_size=1)
        print("="*10)
        print("Transfer completed. Outputs saved in {}".format(output_folder))




if __name__ == "__main__":
    
    main()
    






   
