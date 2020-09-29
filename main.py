import torch
from torchvision import models


import argparse
import os


import style_transfer as st
import utils


IMSIZE=256

datapath = './data'
datatype='images'
furniture='chairs'

generic='generic/chair-1.jpg'

style='selected_styles'


def create_arg_parser():

    default_style_path = os.path.join(datapath,datatype,style)
    default_content_path = os.path.join(datapath,datatype,furniture,generic)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--content_path',type=str,help='Path to content image',default=default_content_path)
    parser.add_argument('-s','--style_path',type=str,help='Path to style image or directory (if using 2+ style images)', default=default_style_path)
    parser.add_argument('-imsize', '--image_size', type=int,default=256)
    parser.add_argument('-o','--output_path', type=str,help='Path of output image')

    return parser


if __name__ == "__main__":
    print("Main Driver")

    parser = create_arg_parser()
    args = parser.parse_args()

    style_paths = [os.path.join(args.style_path,fil) for fil in os.listdir(args.style_path)]
    content_path = args.content_path

    device = utils.setup_device(use_gpu = True)
    model = models.vgg19(pretrained=True).features.to(device).eval()
    st.interactive_style_transfer(model,content_path,style_paths,device)

    # save_path = content_path
    # i=0
    # for style_path in style_paths:

    #     _,mask_path = seg.segment_points(save_path,device=device)
    #     mask_img = utils.load_image(mask_path)
    #     mask = utils.image_to_tensor(mask_img,image_size=IMSIZE).to(device)

    #     _,style_mask_path = seg.segment_points(style_path,device=device)
    #     style_mask_img = utils.load_image(style_mask_path)
    #     style_mask = utils.image_to_tensor(style_mask_img,image_size=IMSIZE).to(device)

    #     print("Mask shape: {}".format(mask.shape))
    #     print("Style mask shape: {}".format(style_mask.shape))

    #     content_img = utils.load_image(save_path)
    #     style_img = utils.load_image(style_path)

    #     content = utils.image_to_tensor(content_img,image_size=IMSIZE).to(device)
    #     style = utils.image_to_tensor(style_img,image_size=IMSIZE).to(device)

    #     content_clone = content.clone().detach()

    #     content = content * mask

    #     style = style * style_mask
        
    #     print("Mask shape: {}".format(mask.shape))
    #     print("content shape: {}".format(content.shape))
    #     print("style shape: {}".format(style.shape))

    #     # setup normalization mean and std
    #     normalization_mean = torch.tensor([0.485,0.456,0.406]).to(device)
    #     normalization_std = torch.tensor([0.229,0.224,0.225]).to(device)

    #     initial = content.clone()

    #     output = st.style_transfer_gatys(model,normalization_mean,normalization_std,content,style,initial,EPOCHS=1000)

    #     save_path = 'outputs/stylized_output_{}.png'.format(i+1)
    #     i+=1
    #     final = (output * mask) + (content_clone * (1-mask))
    #     final_img = utils.tensor_to_image(final)
    #     final_img.save(save_path)


