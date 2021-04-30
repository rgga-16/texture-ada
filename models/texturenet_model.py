from models.networks.texturenet import Pyramid2D,Pyramid2D_custom
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base_model import BaseModel
from defaults import DEFAULTS as D
import style_transfer as st

class TextureNet(BaseModel):
    def __init__(self,version:str='vanilla',ch_in=3, ch_step=8, ch_out=3,n_samples=6) -> None:
        super().__init__()

        assert version.casefold() in ['custom','vanilla']

        if version.casefold() =='vanilla':
            net = Pyramid2D(ch_in,ch_step,ch_out)
        elif version.casefold() =='custom':
            net = Pyramid2D_custom(ch_in,ch_step,ch_out,n_samples)

        self.net = net.to(self.device)
        self.lr = D.LR()
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)
        self.criterion_loss = nn.MSELoss().to(self.device)
    
    def train(self):
        self.net.train()
    
    def eval(self):
        self.net.eval()
    
    def set_input(self, style):
        self.batch_size = style.shape[0]
        
        s = D.IMSIZE.get()
        content = [torch.rand(self.batch_size,3,sz,sz,device=self.device,requires_grad=False).detach() for sz in [s, s//2,s//4,s//8,s//16,s//32]]

        self.style = style.clone().detach().to(self.device)
        self.content = content

    def forward(self):
        self.output = self.net(self.content)
        return self.output
    
    def get_losses(self):
        style_feats = st.get_features(self.style)
        output_feats = st.get_features(self.output)
        style_loss=0
        for s in D.STYLE_LAYERS.get().values():
            diff = self.criterion_loss(output_feats[s],style_feats[s])
            style_loss += D.SL_WEIGHTS.get()[s] * diff
        self.loss = style_loss
        return self.loss.item()

    def optimize_parameters(self):
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()