from models.networks.texturenet import Pyramid2D_adain,Pyramid2D_adain2
from models.networks.adain import Network_AdaIN
from models.networks.feedforward import FeedForwardNetwork
from models.networks.texturenet import Pyramid2D,Pyramid2D_custom
import torch
import torch.nn as nn
from torch.nn import functional as F
from args import parse_arguments

from models.base_model import BaseModel
from defaults import DEFAULTS as D
import style_transfer as st
import ops 

class ProposedModel(BaseModel):
    def __init__(self) -> None:
        
        net = Pyramid2D_adain2(3,64,3)
        self.lr = D.LR()
        optimizer = torch.optim.Adam(net.parameters(),lr=self.lr)
        criterion_loss = nn.MSELoss()

        super().__init__(net,optimizer,criterion_loss)
    
    def train(self):
        self.net.train()
        self.net.encoder.eval()
    
    def eval(self):
        self.net.eval()
        self.net.encoder.eval()
    
    def set_input(self, style):
        self.batch_size = style.shape[0]
        args = parse_arguments()
        
        if self.net.training:
            s = args.uv_train_sizes[0]
        else:
            s = args.uv_test_sizes[0]
        content = [torch.rand(self.batch_size,3,sz,sz,device=self.device).detach() for sz in [s, s//2,s//4,s//8,s//16,s//32]]

        self.style = style.clone().detach().to(self.device)
        self.content = content

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

        if not self.net.training: 
            style_means,style_covs = st.get_means_and_covs(self.style)
            output_means,output_covs = st.get_means_and_covs(self.output)

            wass_dist = 0
            for s in D.STYLE_LAYERS.get().values():
                wdist = ops.gaussian_wasserstein_distance(style_means[s],style_covs[s],output_means[s],output_covs[s]).real
                wass_dist += D.SL_WEIGHTS.get()[s] * wdist 
            self.wasserstein_distance = wass_dist
            
            return self.loss.item(),self.wasserstein_distance
        return self.loss.item()

    def optimize_parameters(self):
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


class FeedForward(BaseModel):
    def __init__(self,ch_in=3, ch_out=3,n_resblocks=5) -> None:
        

        net = FeedForwardNetwork(ch_in,ch_out,n_resblocks)
        self.lr = D.LR()
        optimizer = torch.optim.Adam(net.parameters(),lr=self.lr)
        criterion_loss = nn.MSELoss()
        super().__init__(net,optimizer,criterion_loss)
        self.net = self.net.to(self.device)
        self.criterion_loss = self.criterion_loss.to(self.device)
    
    def train(self):
        self.net.train()
    
    def eval(self):
        self.net.eval()
    
    def set_input(self, style,content=None):
        self.batch_size = style.shape[0]

        if content is None: 
            content = torch.rand(style.shape[0],3,D.IMSIZE.get(),D.IMSIZE.get())

        self.style = style.clone().detach().to(self.device)
        self.content = content.clone().detach().to(self.device)

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
        
        if not self.net.training: 
            style_means,style_covs = st.get_means_and_covs(self.style)
            output_means,output_covs = st.get_means_and_covs(self.output)

            wass_dist = 0
            for s in D.STYLE_LAYERS.get().values():
                wdist = ops.gaussian_wasserstein_distance(style_means[s],style_covs[s],output_means[s],output_covs[s]).real
                wass_dist += D.SL_WEIGHTS.get()[s] * wdist 
            self.wasserstein_distance = wass_dist
            
            return self.loss.item(),self.wasserstein_distance
        
        
        return self.loss.item()

    def optimize_parameters(self):
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

class AdaIN_Autoencoder(BaseModel):
    def __init__(self) -> None:
        

        net = Network_AdaIN()
        self.lr = D.LR()
        optimizer = torch.optim.Adam(net.parameters(),lr=self.lr)
        criterion_loss = nn.MSELoss()
        super().__init__(net,optimizer,criterion_loss)
        self.net = self.net.to(self.device)
        self.criterion_loss = self.criterion_loss.to(self.device)
    
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

        if not self.net.training: 
            style_means,style_covs = st.get_means_and_covs(self.style)
            output_means,output_covs = st.get_means_and_covs(self.output)

            wass_dist = 0
            for s in D.STYLE_LAYERS.get().values():
                wdist = ops.gaussian_wasserstein_distance(style_means[s],style_covs[s],output_means[s],output_covs[s]).real
                wass_dist += D.SL_WEIGHTS.get()[s] * wdist 
            self.wasserstein_distance = wass_dist
            
            return self.loss.item(),self.wasserstein_distance
        return self.loss.item()

    def optimize_parameters(self):
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

class TextureNet(BaseModel):
    def __init__(self,version:str='vanilla',ch_in=3, ch_step=8, ch_out=3,n_samples=6) -> None:
        

        assert version.casefold() in ['custom','vanilla']

        if version.casefold() =='vanilla':
            net = Pyramid2D(ch_in,ch_step,ch_out)
        elif version.casefold() =='custom':
            net = Pyramid2D_custom(ch_in,ch_step,ch_out,n_samples)

        
        self.lr = D.LR()
        optimizer = torch.optim.Adam(net.parameters(),lr=self.lr)
        criterion_loss = nn.MSELoss()
        super().__init__(net,optimizer,criterion_loss)
        self.criterion_loss = self.criterion_loss.to(self.device)
        self.net = self.net.to(self.device)
    
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
        if not self.net.training: 
            style_means,style_covs = st.get_means_and_covs(self.style)
            output_means,output_covs = st.get_means_and_covs(self.output)

            wass_dist = 0
            for s in D.STYLE_LAYERS.get().values():
                wdist = ops.gaussian_wasserstein_distance(style_means[s],style_covs[s],output_means[s],output_covs[s]).real
                wass_dist += D.SL_WEIGHTS.get()[s] * wdist 
            self.wasserstein_distance = wass_dist
            
            return self.loss.item(),self.wasserstein_distance
        return self.loss.item()

    def optimize_parameters(self):
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()