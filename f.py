

import torch
import torch.nn as nn
from torchvision import models

from vgg import VGG19
import utils
from defaults import DEFAULTS as D 


import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

img = torch.zeros(1,3,D.IMSIZE.get(),D.IMSIZE.get(),requires_grad=True,device=D.DEVICE())

utils.tensor_to_image(img).save('init.png')

content = utils.image_to_tensor(utils.load_image(D.STYLE_PATH())).detach()

optimizer = torch.optim.Adam([img],lr=0.1)

# Create style feature extractor model
model = VGG19()

layers = {'17' : 'relu3_3'}

_,content_feats = model(content,layers)

mse_loss = nn.MSELoss()

loss=0

progress = tqdm(range(D.EPOCHS()))

for i in progress:
    
    img.data.clamp_(0,1)
    for key,val in layers.items():
        optimizer.zero_grad()
        _,output_feats = model(img,{key:val})
        loss = mse_loss(output_feats[val],content_feats[val])
        
        loss.backward(retain_graph=True)
        optimizer.step()

        progress.set_description("Loss: {}".format(loss.item()))
img.data.clamp_(0,1)    
final = np.asarray(utils.tensor_to_image(img))
plt.imshow(final)
plt.show()


