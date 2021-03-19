# SEED=0


import torch
# torch.manual_seed(SEED)
from torchvision import transforms

import style_transfer as st

from helpers import logger, utils 
import args

from models import VGG19, ConvAutoencoder,TextureNet, Pyramid2D

from args import args
import os
import time
import datetime


from defaults import DEFAULTS as D
from torchsummary import summary


def train(generator,input,style,content,feat_extractor,lr=args.lr):
    
    epochs = args.epochs
    generator.train()
    generator.cuda(device=D.DEVICE())

    content_mask = content[:,3,...].unsqueeze(0)
    content =content[:,:3,...]

    optim = torch.optim.Adam(generator.parameters(),lr=lr)

    style_feats = st.get_features(feat_extractor,style)
    style_layers = D.STYLE_LAYERS.get()
    s_layer_weights = D.SL_WEIGHTS.get()


    # content_feats = st.get_features(feat_extractor,content)
    # content_layers = D.CONTENT_LAYERS.get()
    # c_layer_weights = D.CL_WEIGHTS.get()

    mse_loss = torch.nn.MSELoss()

    checkpoint=100
    loss_history=[]
    epoch_chkpts=[]
    for i in range(epochs):
        # sizes = [imsize/1,imsize/2,imsize/4,imsize/8,imsize/16,imsize/32]
        # samples = [torch.rand(1,3,int(sz),int(sz),device=D.DEVICE()) for sz in sizes]
        # samples = [transforms.Resize((int(size),int(size)))(style) for size in sizes ]
        # samples = input.clone().detach()
        samples=input

        optim.zero_grad()
        style_loss=0
        output = generator(samples)
        # output = output.clamp(0,1)

        output_mask = output[:,3,...].unsqueeze(0).clamp(0,1)
        # output_mask = torch.round(output_mask)
        output=output[:,:3,...]

        style_loss=0
        content_loss=0
        
        out_feats = st.get_features(feat_extractor,output)

        for s in style_layers.values():
            diff = mse_loss(out_feats[s],style_feats[s])
            style_loss += s_layer_weights[s] * diff
        style_weight=args.style_weight

        # for c in content_layers.values():
        #     c_diff = mse_loss(out_feats[c], content_feats[c])
        #     content_loss += c_layer_weights[c] * c_diff
        # content_weight=args.content_weight

        fg_loss = mse_loss(output_mask,content_mask)
        fg_weight = args.foreground_weight

        # loss = (content_loss*content_weight) + (style_loss * style_weight)
        loss = (style_loss * style_weight) + (fg_loss * fg_weight)
        loss.backward()
        
        optim.step()

        if(i%checkpoint==checkpoint-1):
            print('ITER {} | LOSS: {}'.format(i+1,loss.item()))
            loss_history.append(loss)
            epoch_chkpts.append(i)

    today = datetime.datetime.today().strftime('%y-%m-%d %H-%M')
    model_file = '[{}]{}-{}_iters.pth'.format(today,generator.__class__.__name__,epochs)
    gen_path = os.path.join(D.MODEL_DIR.get(),model_file)
    print('Model saved in {}'.format(gen_path))
    torch.save(generator.state_dict(),gen_path)

    # losses_file = '[{}]-losses.png'.format(today)
    losses_file = 'losses.png'.format(today)
    losses_path = os.path.join(args.output_dir,'outputs',losses_file)
    logger.log_losses(loss_history,epoch_chkpts,losses_path)
    print('Loss history saved in {}'.format(losses_path))
    # vis.display_losses(loss_history,epoch_chkpts,title='Training Loss History')


    return gen_path

def test(generator,input,gen_path,output_path):
    generator.eval()
    generator.cuda(device=D.DEVICE())

    generator.load_state_dict(torch.load(gen_path))
    # sizes = [imsize/1,imsize/2,imsize/4,imsize/8,imsize/16,imsize/32]
    # samples = [torch.rand(1,3,int(sz),int(sz),device=D.DEVICE()) for sz in sizes]
    # samples = [transforms.Resize((int(size),int(size)))(style) for size in sizes ]
    samples=input

    y = generator(samples)
    # y = y.clamp(0,1)
    _,_,_,h = y.shape

    utils.tensor_to_image(y,image_size=args.imsize).save(output_path)
    print('Saving image as {}'.format(output_path))
 

def main():
    print("Starting texture transfer..")
    print("="*10)
    device = D.DEVICE()
    imsize = args.imsize
    # Get pairings between UV maps and style images

    # armchair sofa
    # uv_map_style_pairings = {
    #     'uv_map_backseat.png':'chair-2_tiled.png',
    #     'uv_map_left_arm.png':'chair-3_tiled.png',
    #     'uv_map_right_arm.png':'chair-3_tiled.png',
    #     'uv_map_left_foot.png':'chair-4_masked.png',
    #     'uv_map_right_foot.png':'chair-4_masked.png',
    #     'uv_map_base.png':'chair-1_masked.png',
    #     'uv_map_seat.png':'chair-6_masked.png'
    # }

    # office chair
    # uv_map_style_pairings = {
    #     # 'uv_map_backseat.png':'chair-2_masked.png',
    #     'uv_map_arms.png':'chair-3_masked.png',
    #     'uv_map_feet.png':'chair-3_masked.png',
    #     'uv_map_seat.png':'chair-1_masked.png',
    # }

    # lounge sofa
    uv_map_style_pairings = {
        'left_arm_uv.png':['chair-3_tiled.png',256],
        # 'right_arm_uv.png':['chair-3_tiled.png',256],
        # 'left_backseat_uv.png':['cobonpue-17_tiled.png',512],
        # 'mid_backseat_uv.png':['chair-2_tiled.png',256],
        # 'right_backseat_uv.png':['cobonpue-17_tiled.png',512],
        # 'left_base_uv.png':['cobonpue-80_tiled.png',512],
        # 'right_base_uv.png':['cobonpue-80_tiled.png',512],
        # 'left_seat_uv.png':['cobonpue-99_tiled.png',512],
        # 'mid_seat_uv.png':['chair-2_tiled.png',256],
        # 'right_seat_uv.png':['cobonpue-99_tiled.png',512],
    }

    # Retrieve style images and UV maps
    style_files = list(uv_map_style_pairings.values())
    uv_map_files = list(uv_map_style_pairings.keys())

    assert len(uv_map_files) == len(style_files)

    # sizes = [imsize//2,imsize//4,imsize//8,imsize//16,imsize//32]

    start=time.time()
    date = datetime.datetime.today().strftime('%m-%d-%y %H-%M-%S')

    output_folder = os.path.join(args.output_dir,"[{}]".format(date))
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass
    
    for uv_size in [128,256,512,768]:
        sizes = [uv_size//2,uv_size//4,uv_size//8,uv_size//16,uv_size//32]
        for uvf,sf_ in zip(uv_map_files,style_files):
            sf=sf_[0]
            style_size=sf_[1]
            print("Transferring {} ==> {} ...".format(sf,uvf))
            style_img = utils.load_image(os.path.join(args.style_dir,sf))
            # Convert to tensor 
            style = utils.image_to_tensor(style_img,image_size=imsize,normalize=True).detach()
            style = style[:,:3,...]

            uv_img =utils.load_image(os.path.join(args.content_dir,uvf))
            uv = utils.image_to_tensor(uv_img,image_size=uv_size,normalize=True).detach()

            # Setup inputs 
            inputs = [uv[:,:3,...].clone().detach()]
            # inputs.extend([torch.rand(1,3,sz,sz,device=D.DEVICE()) for sz in sizes])
            for sz in sizes:
                inputs.append(utils.image_to_tensor(uv_img,image_size=sz,normalize=True).detach()[:,:3,...])

            # Setup generator model 
            net = Pyramid2D().to(device)
            # net = Pyramid2D_small().to(device)

            # Setup feature extraction model 
            feat_extractor = VGG19()
            for param in feat_extractor.parameters():
                param.requires_grad = False

            # train model 
            gen_path = train(net,inputs,style,uv,feat_extractor)
            
            output_filename = '{}_{}_{}.png'.format(uvf[:-4],sf[:-4],uv_size)
            output_path =os.path.join(output_folder,output_filename)

            # test model to output texture 
            test(net,inputs,gen_path,output_path)

    # record losses and configurations
    time_elapsed = time.time() - start 
    
    log_file = '[{}]_log.txt'.format(date)
    
    logger.log_args(os.path.join(output_folder,log_file),
                    Time_Elapsed=time_elapsed,
                    Model_Name=Pyramid2D().__class__.__name__,
                    Seed = torch.seed())
    print("="*10)
    print("Transfer completed. Outputs saved in {}".format(output_folder))



if __name__ == "__main__":
    
    main()
    






   
