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

def evaluate_texture(model:BaseModel,test_loader):
    args = args_.parse_arguments()

    assert isinstance(model,BaseModel)
    model.eval()

    w = args.uv_test_sizes[0]
    running_dist=0.0
    running_loss=0.0
    for i,texture in enumerate(test_loader):
        model.set_input(texture)
        with torch.no_grad():
            model.forward()
            loss,wdist=model.get_losses()
            running_dist+=wdist*texture.shape[0]
            running_loss+=loss*texture.shape[0]
    
    eval_wdist = running_dist/test_loader.dataset.__len__()
    eval_loss = running_loss/test_loader.dataset.__len__()
    return eval_loss,eval_wdist

def predict_texture_batch(model:BaseModel,test_loader,output_dir):
    args = args_.parse_arguments()

    assert isinstance(model,BaseModel)

    model.eval()
    
    avg_loss,avg_wdist=0.0,0.0
    for i,texture in enumerate(test_loader):
        model.set_input(texture)
        with torch.no_grad():
            model.set_input(texture)
            output = model.forward()
            loss,wdist = model.get_losses()
        avg_loss+=loss*texture.shape[0]
        avg_wdist+=wdist*texture.shape[0]

        if texture.shape[0] > 8:
            image_utils.show_images(torch.cat((texture,output),dim=0),save_path=os.path.join(output_dir,f'output_batch_{i+1}.png'))

        for o in range(len(output)): 
            save_path = os.path.join(output_dir,f'output_{(i+1)}_{(o+1)}.png')
            output_image = image_utils.tensor_to_image(output[o],image_size=args.output_size)
            output_image.save(save_path,'PNG')
            print('Saving image as {}'.format(save_path))
    
    avg_loss/=test_loader.dataset.__len__()
    avg_wdist/=test_loader.dataset.__len__()
    
    return avg_loss,avg_wdist

def predict_texture(model:BaseModel,texture,output_path):
    args = args_.parse_arguments()

    assert isinstance(model,BaseModel)

    model.eval()
    
    w = args.uv_test_sizes[0]
    texture = texture.expand(1,-1,-1,-1).clone().detach()
    
    with torch.no_grad():
        model.set_input(texture)
        output = model.forward()
        loss,wdist = model.get_losses()

    output_image = image_utils.tensor_to_image(output.squeeze(0),image_size=args.output_size)

    output_image.save(output_path,'PNG')
    print('Saving image as {}'.format(output_path))
    return loss,wdist




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
    






   
