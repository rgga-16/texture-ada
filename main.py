import torch


import style_transfer as st
import utils
import args

from models import VGG19, ConvAutoencoder,TextureNet, Pyramid2D

from args import parse_arguments
import os
import datetime

# from dextr.segment import segment_points

from defaults import DEFAULTS as D
from torchsummary import summary

torch.autograd.set_detect_anomaly(True)

def train(args,generator,feat_extractor,lr=0.001):
    style_img = utils.load_image(args.style)
    imsize=args.imsize
    style = utils.image_to_tensor(style_img,image_size=imsize).detach()

    epochs = args.epochs
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
        sizes = [imsize/1,imsize/2,imsize/4,imsize/8,imsize/16,imsize/32]
        samples = [torch.rand(1,3,int(sz),int(sz),device=D.DEVICE()) for sz in sizes]

        optim.zero_grad()
        loss=0
        output = generator(samples)
        output = output.clamp(0,1)
        
        out_feats = st.get_features(feat_extractor,output)

        for s in style_layers.values():
            diff = mse_loss(out_feats[s],style_feats[s])
            loss += s_layer_weights[s] * diff
        style_weight=1e6
        loss = loss * style_weight

        loss.backward()
        optim.step()

        if(i%checkpoint==checkpoint-1):
            print('EPOCH {} | LOSS: {}'.format(i+1,loss.item()))

    _,style_filename = os.path.split(args.style)
    today = datetime.datetime.today().strftime('%y-%m-%d %H-%M')
    model_file = '[{}]{}-{}-{}_epochs.pth'.format(today,generator.__class__.__name__,style_filename[:-4],epochs)
    gen_path = os.path.join(D.MODEL_DIR.get(),model_file)
    print('Model saved in {}'.format(gen_path))
    torch.save(generator.state_dict(),gen_path)

    return gen_path

def test(args,generator,gen_path):
    generator.eval()
    imsize=args.imsize
    generator.cuda(device=D.DEVICE())

    _,style_filename = os.path.split(args.style)
    generator.load_state_dict(torch.load(gen_path))
    # x = torch.rand(1,3,64,64,device=D.DEVICE()).detach()
    sizes = [imsize/1,imsize/2,imsize/4,imsize/8,imsize/16,imsize/32]
    samples = [torch.rand(1,3,int(sz),int(sz),device=D.DEVICE()) for sz in sizes]

    y = generator(samples)
    y = y.clamp(0,1)
    b,c,w,h = y.shape

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
    net = Pyramid2D().to(device)

    feat_extractor = VGG19()
    for param in feat_extractor.parameters():
        param.requires_grad = False

    gen_path = train(args,net,feat_extractor)

    test(args,net,gen_path)

    # Algorithm
    



if __name__ == "__main__":
    main()
    






   
