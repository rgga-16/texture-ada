from models.networks.adain import Network_AdaIN
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base_model import BaseModel
from defaults import DEFAULTS as D
import style_transfer as st


class AdaIN_Autoencoder(BaseModel):
    def __init__(self) -> None:
        

        net = Network_AdaIN().to(self.device)
        self.lr = D.LR()
        optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)
        criterion_loss = nn.MSELoss().to(self.device)
        super().__init__(net,optimizer,criterion_loss)
    
    def train(self):
        self.net.train()
        # self.net.decoder.train()
        # self.net.encoder.eval()
    
    def eval(self):
        self.net.eval()
        # self.net.encoder.eval()
        # self.net.decoder.eval()
    
    def set_input(self, style, content=None):
        
        if content is None: 
            content = torch.rand(style.shape[0],3,D.IMSIZE.get(),D.IMSIZE.get())
        
        assert style.shape == content.shape 
        self.batch_size = style.shape[0]
        self.style = style.clone().detach().to(self.device)
        self.content = content.clone().detach().to(self.device)

    def forward(self):
        self.output = self.net(self.content,self.style)
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