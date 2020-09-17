
import utils
import style_transfer as st

import blender

from PIL import Image
import imageio
import cv2

import numpy as np
import torch

import dextr.segment as seg
import style_transfer as st

import argparse

import torch.nn.functional as F

from torchvision import models



if __name__ == "__main__":
    print("Main Driver")

    device = utils.setup_device(use_gpu = True)

    # Setup model. Use pretrained VGG-19
    model = models.vgg19(pretrained=True).features.to(device).eval()

    # print(model)

    content_path = './data/images/chairs/generic/armchair.jpeg'
    style_path = './data/images/chairs/cobonpue/chair-2.jpg'
    mask_path = './data/images/masks/segmented_seat.png'

    # ## Get mask by segmenting the content image via user input
    # mask,_ = seg.segment_points(content_path,device=device)
    # # Convert mask from Numpy array to Torch tensor
    # mask = torch.from_numpy(mask)
    # w,h,c = mask.shape
    # # Make mask have dimensions b,c,w,h
    # mask = mask.view(c,w,h).unsqueeze(0).to(device)

    ## Or get mask by loading from path
    mask = utils.image_to_tensor(utils.load_image(mask_path)).to(device)

    # Add a 3rd dimension to mask
    # mask = mask[:,:,None]/255
    print("Mask shape: {}".format(mask.shape))

    probe = torch.zeros((3,) + mask.shape[2:]).unsqueeze(0).to(device)
    print("Probe shape: {}".format(probe.shape))

    for i in range(0, 3):
        print(probe[0,i].shape)
        print(mask[0,0].shape)
        probe[0,i,:,:]=mask[0,0,:,:]

    content = utils.image_to_tensor(utils.load_image(content_path)).to(device)
    style = utils.image_to_tensor(utils.load_image(style_path)).to(device)
    content = torch.matmul(content,probe)
    f = utils.tensor_to_image(content)
    f.save('attemtped cropped image.png')
    # print("Mask shape: {}".format(mask.shape))
    # print("content shape: {}".format(content.shape))
    # print("style shape: {}".format(style.shape))

    # # setup normalization mean and std
    # normalization_mean = torch.tensor([0.485,0.456,0.406]).to(device)
    # normalization_std = torch.tensor([0.229,0.224,0.225]).to(device)

    # initial = content.clone()

    # output = st.run_style_transfer(model,normalization_mean,normalization_std,content,style,initial,EPOCHS=2000)
    # output_img = utils.tensor_to_image(output)
    # save_path = 'outputs/stylized_output.png'
    # output_img.save(save_path)
    # print('Saved at {}'.format(save_path))






















    # content = np.array(utils.load_image(content_path),dtype=np.uint8)
    # stylized = np.array(utils.load_image(stylized_path),dtype=np.uint8)
    # mask = np.array(utils.load_image(mask_path),dtype=np.uint8)
    

    # # blended_img = blender.blend_images(content,stylized,mask)
    # # imageio.imwrite('blended_img.png',blended_img)

    # cropped_img = blender.crop_with_mask(content,mask)
    # imageio.imwrite('cropped_img.png',cropped_img)



    




