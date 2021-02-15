import torch
from torchvision import transforms

import style_transfer as st

import helpers.utils as utils
import args

from models import Pyramid2D_2, VGG19, ConvAutoencoder,TextureNet, Pyramid2D

from args import parse_arguments
import os
import time
import datetime

# from dextr.segment import segment_points

from defaults import DEFAULTS as D
from torchsummary import summary


def train(args,generator,input,style,content,texture_patch,feat_extractor,lr=0.001):
    
    epochs = args.epochs
    generator.train()
    generator.cuda(device=D.DEVICE())

    optim = torch.optim.Adam(generator.parameters(),lr=lr)

    style_feats = st.get_features(feat_extractor,style)
    style_layers = D.STYLE_LAYERS.get()
    s_layer_weights = D.SL_WEIGHTS.get()

    # style_feats = st.get_features(feat_extractor,texture_patch)
    # style_layers = D.STYLE_LAYERS.get()
    # s_layer_weights = D.SL_WEIGHTS.get()

    content_feats = st.get_features(feat_extractor,content)
    content_layers = D.CONTENT_LAYERS.get()
    c_layer_weights = D.CL_WEIGHTS.get()

    mse_loss = torch.nn.MSELoss()

    checkpoint=100
    for i in range(epochs):
        # sizes = [imsize/1,imsize/2,imsize/4,imsize/8,imsize/16,imsize/32]
        # samples = [torch.rand(1,3,int(sz),int(sz),device=D.DEVICE()) for sz in sizes]
        # samples = [transforms.Resize((int(size),int(size)))(style) for size in sizes ]
        samples = input.clone().detach()

        optim.zero_grad()
        style_loss=0
        output = generator(samples)
        output = output.clamp(0,1)

        style_loss=0
        content_loss=0
        
        out_feats = st.get_features(feat_extractor,output)

        for s in style_layers.values():
            diff = mse_loss(out_feats[s],style_feats[s])
            style_loss += s_layer_weights[s] * diff
        style_weight=1e6

        for c in content_layers.values():
            c_diff = mse_loss(out_feats[c], content_feats[c])
            content_loss += c_layer_weights[c] * c_diff
        content_weight=1e3


        loss = (content_loss*content_weight) + (style_loss * style_weight)
        loss.backward()
        optim.step()

        if(i%checkpoint==checkpoint-1):
            print('ITER {} | LOSS: {}'.format(i+1,loss.item()))

    _,style_filename = os.path.split(args.style)
    today = datetime.datetime.today().strftime('%y-%m-%d %H-%M')
    model_file = '[{}]{}-{}-{}_iters.pth'.format(today,generator.__class__.__name__,style_filename[:-4],epochs)
    gen_path = os.path.join(D.MODEL_DIR.get(),model_file)
    print('Model saved in {}'.format(gen_path))
    torch.save(generator.state_dict(),gen_path)

    return gen_path

def test(args,generator,input,gen_path):
    generator.eval()
    imsize=args.imsize
    generator.cuda(device=D.DEVICE())

    _,style_filename = os.path.split(args.style)
    generator.load_state_dict(torch.load(gen_path))
    # sizes = [imsize/1,imsize/2,imsize/4,imsize/8,imsize/16,imsize/32]
    # samples = [torch.rand(1,3,int(sz),int(sz),device=D.DEVICE()) for sz in sizes]
    # samples = [transforms.Resize((int(size),int(size)))(style) for size in sizes ]
    samples=input.clone().detach()

    y = generator(samples)
    y = y.clamp(0,1)
    b,c,w,h = y.shape

    date = time.time

    output_filename = 'output_{}.png'.format(style_filename[:-4])
    output_path =os.path.join(args.output,'output_images',output_filename)
    utils.tensor_to_image(y,image_size=h).save(output_path)
    print('Image saved in {}'.format(output_path))

    

def main():
    print("Main Driver")
    args = parse_arguments()

    device = D.DEVICE()
    # net = ConvAutoencoder().to(device)
    # net = TextureNet().to(device)
    # net = DenseNet(small_inputs=False)

    imsize=args.imsize
    style_img = utils.load_image(args.style)
    style = utils.image_to_tensor(style_img,image_size=imsize,normalize=True).detach()

    texture_img = utils.load_image(args.texture)
    texture = utils.image_to_tensor(texture_img,image_size=imsize,normalize=True).detach()

    content_img =utils.load_image(args.content)
    content = utils.image_to_tensor(content_img,image_size=imsize,normalize=True).detach()

    input = utils.image_to_tensor(content_img,image_size=imsize//32,normalize=True).detach()

    # net = Pyramid2D().to(device)
    net = Pyramid2D_2().to(device)

    feat_extractor = VGG19()
    for param in feat_extractor.parameters():
        param.requires_grad = False

    gen_path = train(args,net,input,style,content,texture,feat_extractor)
    # gen_path = './models/[21-02-08 07-22]Pyramid2D-chair-1_masked-2500_epochs.pth'

    test(args,net,input,gen_path)

    



if __name__ == "__main__":
    main()
    






   
