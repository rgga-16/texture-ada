import torch

import style_transfer as st
import utils
import args
from densenet import DenseNet
from models import VGG19

from args import parse_arguments
import os

# from dextr.segment import segment_points

from defaults import DEFAULTS as D

torch.autograd.set_detect_anomaly(True)

def train(args):
    style = utils.image_to_tensor(utils.load_image(args.style),image_size=args.imsize).detach()

    epochs = args.epochs
    lr=0.1
    generator = DenseNet(small_inputs=False)
    feat_extractor = VGG19()

    generator.train()
    generator.cuda(device=D.DEVICE())

    optim = torch.optim.Adam(generator.parameters(),lr=lr)

    style_feats = st.get_features(feat_extractor,style)
    style_layers = D.STYLE_LAYERS.get()
    s_layer_weights = D.SL_WEIGHTS.get()

    mse_loss = torch.nn.MSELoss()

    checkpoint=100
    print('Training for {} epochs'.format(epochs))
    for i in range(epochs):
        x = torch.rand(1,3,args.imsize,args.imsize,device=D.DEVICE()).detach()
        optim.zero_grad()
        loss=0
        output = generator(x)
        
        out_feats = st.get_features(feat_extractor,output)

        for s in style_layers.values():
            diff = mse_loss(out_feats[s],style_feats[s])
            loss += s_layer_weights[s] * diff

        loss.backward()
        optim.step()

        if(i%checkpoint==checkpoint-1):
            print('EPOCH {} | LOSS: {}'.format(i+1,loss.item()))
    
    _,style_filename = os.path.split(args.style)
    model_path = './models/{}.pth'.format(style_filename[:-4])
    torch.save(generator.state_dict(),model_path)

def test(args):
    generator = DenseNet(small_inputs=False)
    feat_extractor = VGG19()

    generator.eval()
    generator.cuda(device=D.DEVICE())

    _,style_filename = os.path.split(args.style)
    model_path = './models/{}.pth'.format(style_filename[:-4])
    generator.load_state_dict(torch.load(model_path))
    x = torch.rand(1,3,args.imsize,args.imsize,device=D.DEVICE()).detach()

    y = generator(x)

    output_filename = 'output_{}.png'.format(style_filename[:-4])
    utils.tensor_to_image(y).save(output_filename)
    print('Image saved in {}'.format(output_filename))

    

def main():
    print("Main Driver")
    args = parse_arguments()
    train(args)
    test(args)
    



if __name__ == "__main__":
    main()
    






   
