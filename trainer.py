import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, copy

import torchvision

import args as args_
from defaults import DEFAULTS as D
from helpers import logger
import style_transfer as st

from helpers import image_utils

def train_texture(model,train_loader,val_loader,output_folder=None):
    args = args_.parse_arguments()
    lr,epochs = args.lr,args.epochs
    start_epoch=0
    best_model_wts = copy.deepcopy(model.net.state_dict())
    best_val_loss = np.inf 
    train_loss_history=[]
    val_loss_history = []
    wdist_history = []
    epoch_chkpts=[]

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',patience=10,verbose=True)
    
    if args.checkpoint_path is not None: 
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint)
        start_epoch = checkpoint['epoch']+1
        best_model_wts=checkpoint['best_net_weights']
        best_val_loss=checkpoint['best_val_loss']
        train_loss_history=checkpoint['train_loss_history'] 
        val_loss_history=checkpoint['val_loss_history'] 
        wdist_history=checkpoint['wdist_history'] 
        epoch_chkpts=checkpoint['epoch_chkpts']
        print(f'Checkpoint found. Resuming training from epoch {start_epoch+1}..\n')

    
    writer = SummaryWriter(f'runs/{os.path.dirname(args.output_dir)}')
    for epoch in range(start_epoch,epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('='*10)
        for phase in ['Train','Val']:
            if phase.casefold()=='train':
                model.train()
                dataloader = train_loader
                loss_history = train_loss_history
            else:
                model.eval()
                dataloader = val_loader
                loss_history = val_loss_history
            batch_chkpt = 1 if len(dataloader) <= args.num_batch_chkpts else len(dataloader)//args.num_batch_chkpts

            running_loss, batch_running_loss = 0.0, 0.0
            running_dist,batch_running_dist=0.0,0.0
            for i,texture in enumerate(dataloader):
                model.set_input(texture)
                # Get output
                with torch.set_grad_enabled(phase.casefold()=='train'):
                    output = model.forward()
                    if phase.casefold()=='train':
                        loss = model.get_losses()
                    else:
                        loss,wdist = model.get_losses()
                        running_dist += wdist * texture.shape[0]
                        batch_running_dist += wdist 
                
                if phase.casefold()=='train':
                    model.optimize_parameters()
                
                running_loss += loss * texture.shape[0]
                batch_running_loss += loss
                if(i%batch_chkpt==batch_chkpt-1):
                    batch_str = f'[{phase} Batch {i+1}/{len(dataloader)}] {phase} Loss: {batch_running_loss/batch_chkpt:.3f}'
                    writer.add_scalar(f'{phase} loss', batch_running_loss/batch_chkpt, epoch*len(dataloader)+i)
                    batch_running_loss=0.0
                    if phase.casefold()=='val':
                        batch_str = f'{batch_str} Wasserstein Dist: {batch_running_dist/batch_chkpt:.5f}'
                        writer.add_scalar(f'Wasserstein Distance', batch_running_dist/batch_chkpt, epoch*len(dataloader)+i)
                        batch_running_dist=0.0
                    print(batch_str)
            
            epoch_loss = running_loss / dataloader.dataset.__len__()
            epoch_str = f'{phase} Loss: {epoch_loss:.4f}'
            if phase.casefold()=='val':
                epoch_wdist = running_dist / dataloader.dataset.__len__()
                epoch_str = f'{epoch_str} Wass. Dist: {epoch_wdist:.5f}'
                wdist_history.append(epoch_wdist)

            print(epoch_str)
            loss_history.append(epoch_loss)

            if phase.casefold()=='val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.net.state_dict()) 
                print(f'Found best net params at Epoch {epoch+1}')         
        epoch_chkpts.append(epoch)


        state_dict = model.get_state_dict()
        state_dict['epoch']=epoch 
        state_dict['best_net_weights']=best_model_wts
        state_dict['best_val_loss']=best_val_loss
        state_dict['train_loss_history'] = train_loss_history
        state_dict['val_loss_history'] = val_loss_history
        state_dict['wdist_history'] = wdist_history
        state_dict['epoch_chkpts']=epoch_chkpts
        if output_folder is not None:
            checkpoint_path = os.path.join(output_folder,f'{model.__class__.__name__}_chkpt.pt')
        else:
            checkpoint_path = os.path.join(args.output_dir,f'{model.__class__.__name__}_chkpt.pt')
        torch.save(state_dict,checkpoint_path)
        print(f'Checkpoint saved in {checkpoint_path}')
        
        
        print('='*10)
        print('')        

    
    
    if output_folder is not None:
        gen_path = os.path.join(output_folder,f'{model.net.__class__.__name__}_final.pth')
    else: 
        gen_path = os.path.join(args.output_dir,f'{model.net.__class__.__name__}_final.pth')
    torch.save(best_model_wts,gen_path)
    print('Final model saved in {}\n'.format(gen_path))

    losses_file = f'losses_{model.net.__class__.__name__}.png'
    
    if output_folder is not None:
        losses_path = os.path.join(output_folder,losses_file)
    else: 
        losses_path = os.path.join(args.output_dir,losses_file)
    logger.log_losses(train_loss_history,val_loss_history,epoch_chkpts,losses_path)
    print('Loss history saved in {}'.format(losses_path))
    writer.close()
    return gen_path

def train_structure(model,train_loader,val_loader):
    args = args_.parse_arguments()
    lr,epochs = args.lr,args.epochs
    start_epoch=0
    best_model_wts = copy.deepcopy(model.net.state_dict())
    best_val_loss = np.inf 

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',patience=10,verbose=True)
    
    if args.checkpoint_path is not None: 
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint)
        start_epoch = checkpoint['epoch']+1
        best_model_wts=checkpoint['best_net_weights']
        best_val_loss=checkpoint['best_val_loss']
        print(f'Checkpoint found. Resuming training from epoch {start_epoch+1}..\n')

    train_loss_history=[]
    val_loss_history = []
    wdist_history = []
    epoch_chkpts=[]
    writer = SummaryWriter(f'runs/{os.path.dirname(args.output_dir)}')
    for epoch in range(start_epoch,epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('='*10)
        for phase in ['Train','Val']:
            if phase.casefold()=='train':
                model.train()
                dataloader = train_loader
                loss_history = train_loss_history
            else:
                model.eval()
                dataloader = val_loader
                loss_history = val_loss_history
            batch_chkpt = 1 if len(dataloader) <= args.num_batch_chkpts else len(dataloader)//args.num_batch_chkpts

            running_loss, batch_running_loss = 0.0, 0.0
            running_dist,batch_running_dist=0.0,0.0
            for i,texture in enumerate(dataloader):
                model.set_input(texture)
                # Get output
                with torch.set_grad_enabled(phase.casefold()=='train'):
                    output = model.forward()
                    if phase.casefold()=='train':
                        loss = model.get_losses()
                    else:
                        loss,wdist = model.get_losses()
                        running_dist += wdist * texture.shape[0]
                        batch_running_dist += wdist 
                
                if phase.casefold()=='train':
                    model.optimize_parameters()
                
                running_loss += loss * texture.shape[0]
                batch_running_loss += loss
                if(i%batch_chkpt==batch_chkpt-1):
                    batch_str = f'[{phase} Batch {i+1}/{len(dataloader)}] {phase} Loss: {batch_running_loss/batch_chkpt:.3f}'
                    writer.add_scalar(f'{phase} loss', batch_running_loss/batch_chkpt, epoch*len(dataloader)+i)
                    batch_running_loss=0.0
                    if phase.casefold()=='val':
                        batch_str = f'{batch_str} Wasserstein Dist: {batch_running_dist/batch_chkpt:.5f}'
                        writer.add_scalar(f'Wasserstein Distance', batch_running_dist/batch_chkpt, epoch*len(dataloader)+i)
                        batch_running_dist=0.0
                    print(batch_str)
            
            epoch_loss = running_loss / dataloader.dataset.__len__()
            epoch_str = f'{phase} Loss: {epoch_loss:.4f}'
            if phase.casefold()=='val':
                epoch_wdist = running_dist / dataloader.dataset.__len__()
                epoch_str = f'{epoch_str} Wass. Dist: {epoch_wdist:.5f}'
                wdist_history.append(epoch_wdist)

            print(epoch_str)
            loss_history.append(epoch_loss)

            if phase.casefold()=='val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.net.state_dict()) 
                print(f'Found best net params at Epoch {epoch+1}')         
        
        state_dict = model.get_state_dict()
        state_dict['epoch']=epoch 
        state_dict['best_net_weights']=best_model_wts
        state_dict['best_val_loss']=best_val_loss
        checkpoint_path = os.path.join(args.output_dir,f'{model.net.__class__.__name__}_chkpt.pt')
        torch.save(state_dict,checkpoint_path)
        print(f'Checkpoint saved in {checkpoint_path}')
        
        epoch_chkpts.append(epoch)
        print('='*10)
        print('')        

    gen_path = os.path.join(args.output_dir,f'{model.net.__class__.__name__}_final.pth')
    torch.save(best_model_wts,gen_path)
    print('Final model saved in {}\n'.format(gen_path))

    losses_file = f'losses_{model.net.__class__.__name__}.png'
    losses_path = os.path.join(args.output_dir,losses_file)
    logger.log_losses(train_loss_history,val_loss_history,epoch_chkpts,losses_path)
    print('Loss history saved in {}'.format(losses_path))
    writer.close()
    return gen_path
    