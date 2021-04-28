import struct
import torch
from torch import nn 
from torch.nn import functional as F
import torchvision
from defaults import DEFAULTS as D
from losses import adaptive_instance_normalization,adain_pointcloud

from kaolin.metrics.pointcloud import chamfer_distance
# from chamferdist import ChamferDistance
# from external_libs.emd import emd_module as emd
if D.DEVICE().type=='cuda':
    from external_libs.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from external_libs.ChamferDistancePytorch import chamfer_python, fscore


def pointcloud_autoencoder_loss(predicted_pointcloud,real_pointcloud,is_eval=False):
    # earth_movers_distance_loss = emd.emdModule()
    # emd_loss,_ = earth_movers_distance_loss(predicted_pointcloud,real_pointcloud,0.05,3000)
    # emd_loss = torch.sqrt(torch.sum(emd_loss) / float(batch_size))

    if (D.DEVICE().type=='cuda'):
        # Link: https://github.com/ThibaultGROUEIX/ChamferDistancePytorch 
        dist_forward,dist_backward,_,_ = dist_chamfer_3D.chamfer_3DDist()(predicted_pointcloud,real_pointcloud)
    else:
        dist_forward,dist_backward,_,_ = chamfer_python.distChamfer(predicted_pointcloud,real_pointcloud)
    # # Averaging is doen to get the mean chamfer_loss in the batch of pointclouds.
    chamfer_loss = (dist_forward.mean(dim=-1)+dist_backward.mean(dim=-1)).mean()
    if is_eval:
        f_score,precision,recall = fscore.fscore(dist_forward,dist_backward)
        return chamfer_loss,f_score,precision,recall

    return chamfer_loss

class Pointnet_Autoencoder(nn.Module):
    def __init__(self,n_points,point_dim=3):
        super(Pointnet_Autoencoder,self).__init__()
        self.n_points=n_points

        self.conv1 = nn.Conv1d(point_dim,32,kernel_size=1,stride=2)
        self.bn1 = nn.BatchNorm1d(32,affine=True)
        self.conv2 = nn.Conv1d(32,64,kernel_size=1,stride=2)
        self.bn2 = nn.BatchNorm1d(64,affine=True)
        self.conv3 = nn.Conv1d(64,128,kernel_size=1,stride=2)
        self.bn3 = nn.BatchNorm1d(128,affine=True)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1,stride=2)
        self.bn4 = nn.BatchNorm1d(256,affine=True)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=1,stride=2)
        self.bn5 = nn.BatchNorm1d(512,affine=True)

        self.deconv5 = nn.ConvTranspose1d(512,256,kernel_size=2,stride=2)
        self.dbn5 = nn.BatchNorm1d(256)
        self.deconv4 = nn.ConvTranspose1d(256,128,kernel_size=2,stride=2)
        self.dbn4 = nn.BatchNorm1d(128)
        self.deconv3 = nn.ConvTranspose1d(128,64,kernel_size=2,stride=2)
        self.dbn3 = nn.BatchNorm1d(64)
        self.deconv2 = nn.ConvTranspose1d(64,32,kernel_size=2,stride=2)
        self.dbn2 = nn.BatchNorm1d(32)
        self.deconv1 = nn.ConvTranspose1d(32,point_dim,kernel_size=2,stride=2)
        

    def forward(self,pointcloud,image_feats):
        # Change pointcloud to shape (batch_size,3,n_points)
        pointcloud = pointcloud.permute(0,2,1)
        assert pointcloud.dim()==3

        x = F.relu(self.bn1(self.conv1(pointcloud))) 
        pfeat_relu1_2 = F.relu(self.bn2(self.conv2(x))) 
        pfeat_relu1_2 = adain_pointcloud(pfeat_relu1_2,image_feats['relu1_2'])

        pfeat_relu2_2 = F.relu(self.bn3(self.conv3(pfeat_relu1_2))) 
        pfeat_relu2_2 = adain_pointcloud(pfeat_relu2_2,image_feats['relu2_2'])

        pfeat_relu3_4 = F.relu(self.bn4(self.conv4(pfeat_relu2_2))) 
        pfeat_relu3_4 = adain_pointcloud(pfeat_relu3_4,image_feats['relu3_4'])

        pfeat_relu4_4 = F.relu(self.bn5(self.conv5(pfeat_relu3_4))) 
        pfeat_relu4_4 = adain_pointcloud(pfeat_relu4_4,image_feats['relu4_4'])

        x = F.relu(self.dbn5(self.deconv5(pfeat_relu4_4)))
        x = F.relu(self.dbn4(self.deconv4(x)))
        x = F.relu(self.dbn3(self.deconv3(x)))
        x = F.relu(self.dbn2(self.deconv2(x)))
        output =self.deconv1(x)


        output = output.permute(0,2,1)
        return output

class Pointnet_UpconvAutoencoder(nn.Module):
    def __init__(self,n_points,point_dim=3):
        super(Pointnet_UpconvAutoencoder,self).__init__()
        self.n_points=n_points

        # Encoder Portion
        self.conv1 = nn.Conv1d(point_dim,64,kernel_size=point_dim,stride=1)
        self.bn1 = nn.InstanceNorm1d(64,affine=True)
        self.conv2 = nn.Conv1d(64,64,kernel_size=1,stride=1)
        self.bn2 = nn.InstanceNorm1d(64,affine=True)
        self.conv3 = nn.Conv1d(64,64,kernel_size=1,stride=1)
        self.bn3 = nn.InstanceNorm1d(64,affine=True)
        self.conv4 = nn.Conv1d(64,128,kernel_size=1,stride=1)
        self.bn4 = nn.InstanceNorm1d(128,affine=True)
        self.conv5 = nn.Conv1d(128,1024,kernel_size=(point_dim),stride=1)
        self.bn5 = nn.InstanceNorm1d(1024,affine=True)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        # end Encoder Portion

        self.fc = nn.Linear(1024,1024)
        self.bnfc = nn.BatchNorm1d(1024)

        #Start Decoder Portion
        self.deconv1=nn.ConvTranspose2d(512,512,kernel_size=(2,2),stride=(2,2))
        self.dbn1 = nn.InstanceNorm2d(512)
        self.deconv2=nn.ConvTranspose2d(512,256,kernel_size=(3,3),stride=(1,1))
        self.dbn2 = nn.InstanceNorm2d(256)
        self.deconv3=nn.ConvTranspose2d(256,256,kernel_size=(4,5),stride=(2,3))
        self.dbn3 = nn.InstanceNorm2d(256)
        self.deconv4=nn.ConvTranspose2d(256,128,kernel_size=(5,7),stride=(3,3))
        self.dbn4 = nn.InstanceNorm2d(128)
        self.deconv5=nn.ConvTranspose2d(128,3,kernel_size=(1,1),stride=(1,1))


    def forward(self,pointcloud):
        # Change pointcloud to shape (batch_size,3,n_points)
        pointcloud = pointcloud.permute(0,2,1)
        assert pointcloud.dim()==3
        bs = pointcloud.shape[0]
        point_dim = pointcloud.shape[1]

        net = F.relu(self.bn1(self.conv1(pointcloud)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = F.relu(self.bn3(self.conv3(net)))
        net = F.relu(self.bn4(self.conv4(net)))
        net = F.relu(self.bn5(self.conv5(net)))

        global_feat = self.maxpool(net)
        global_feat = torch.reshape(global_feat,(bs,-1))

        net = F.relu(self.bnfc(self.fc(global_feat)))

        net = torch.reshape(global_feat,(bs,-1))
        net = torch.reshape(net,(bs,-1,1,2))

        net = F.relu(self.dbn1(self.deconv1(net)))
        net = F.relu(self.dbn2(self.deconv2(net)))
        net = F.relu(self.dbn3(self.deconv3(net)))
        net = F.relu(self.dbn4(self.deconv4(net)))
        output = self.deconv5(net)

        output=torch.reshape(output,(bs,point_dim,-1))
        output = output.permute(0,2,1)
        return output


# Transformer network. Encodes pointcloud into features
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_transform = Tnet(k=3) # Input transform Tnet
        self.feature_transform = Tnet(k=64) # Feature transform Tnet
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
    def forward(self, input):
        # Assume n_points=3000

        #Get 3x3 matrix for input 
        matrix3x3 = self.input_transform(input) #input(1,3,3000) => matrix3x3 (1,3,3)

        # Matrix multiply input pointcloud with 3x3 matrix to make it transformation invariant
        yes = torch.transpose(input,1,2) #input(1,3,3000) => yes(1,3000,3)
        xb = torch.bmm(yes, matrix3x3).transpose(1,2) #yes(1,3000,3) mm (1,3,3) => xb(1,3000,3) => xb(1,3,3000)
        
        #Transform input pointcloud to features
        xb = F.relu(self.bn1(self.conv1(xb))) #xb(1,3,3000)=>xb(1,64,3000)

        #Get 64x64 matrix for features
        matrix64x64 = self.feature_transform(xb) #xb(1,64,3000)=>matrix64x64(1,64,64)
        
        # Matrix multiply pointcloud features with 64x64 matrix to make it transformation invariant
        #xb(1,64,3000)=>(1,3000,64) mm (1,64,64)=> xb(1,3000,64) => xb(1,64,3000)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb))) #xb(1,64,3000) => xb(1,128,3000)
        xb = self.bn3(self.conv3(xb)) #xb(1,128,3000) => xb(1,1024,3000)

        # Add maxpool to make the pointcloud permutation invariant
        # Maxpool to get highest point per feature map
        xb = nn.MaxPool1d(xb.size(-1))(xb) #xb(1,1024,3000) => xb(1,1024,1)

        # Output is a global feature vector that aggregates the features
        output = nn.Flatten(1)(xb) #xb(1,1024,1) => output(1,1024)
        return output, matrix3x3, matrix64x64


# Tnet outputs a 3x3 matrix which makes the pointcloud invariant to transformations (i.e. rotation, moving)
class Tnet(nn.Module):
    def __init__(self,k=3):
        super(Tnet,self).__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # self.bn1 = nn.InstanceNorm1d(64)
        # self.bn2 = nn.InstanceNorm1d(128)
        # self.bn3 = nn.InstanceNorm1d(1024)
        # self.bn4 = nn.InstanceNorm1d(512)
        # self.bn5 = nn.InstanceNorm1d(256)
       
    def forward(self, input):
        # input.shape == (bs,3,n)
        # n= number of points in the pointcloud
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input))) #xb = (bs,3,n) => (bs,64,n)
        xb = F.relu(self.bn2(self.conv2(xb))) #xb =  (bs,64,n) => (bs,128,n)
        xb = F.relu(self.bn3(self.conv3(xb))) #xb = (bs,128,n) => (bs,1024,n)

        pool = nn.MaxPool1d(xb.size(-1))(xb) #pool = (bs,1024,1)
        flat = nn.Flatten(1)(pool) #flat = (1,1024)
        # xb = F.relu(self.fc1(flat)) #xb = (1,512)
        # xb = F.relu(self.fc2(xb)) #xb = (1,256)

        xb = F.relu(self.bn4(self.fc1(flat))) #xb = (1,512)
        xb = F.relu(self.bn5(self.fc2(xb))) #xb = (1,256)
        
        # Bias of the 3x3 matrix. initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1) #init=(1,3,3)
        if xb.is_cuda:
            init=init.cuda()
        # add identity to the output
        xb = self.fc3(xb) #xb = (1,9)
        matrix = xb.view(-1,self.k,self.k) + init #matrix = (1,3,3)
        return matrix

class PointNet_Classifier(nn.Module):
    def __init__(self,n_classes=10):
        super(PointNet_Classifier,self).__init__()
        self.transform = Transformer()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        # Assume n_points=3000
        # Get global feature vector from input
        xb, matrix3x3, matrix64x64 = self.transform(input) #input(1,3,3000)=>xb(1,1024)
        xb = F.relu(self.fc1(xb)) #xb(1,1024) => xb(1,512)
        xb = F.relu(self.dropout(self.fc2(xb))) #xb(1,512) => xb(1,256)
        # xb = F.relu(self.bn1(self.fc1(xb))) #xb(1,1024) => xb(1,512)
        # xb = F.relu(self.bn2(self.dropout(self.fc2(xb)))) #xb(1,512) => xb(1,256)
        output = self.fc3(xb) #xb(1,256) => output(1,10)
        # logsoftmax puts values in the range [-inf,0)
        return self.logsoftmax(output), matrix3x3, matrix64x64

    
def regularization_loss(m3x3, m64x64,batch_size,device=D.DEVICE(), alpha=1e-4):
     # Make identity matrix (batch_size,3,3) for m3x3 matrices. Used for regularization loss for matrix.
    identity_3x3 = torch.eye(3,requires_grad=True,device=device).repeat(batch_size,1,1)
    # Make identity matrix (batch_size,64,64) for m64x64 matrices. Used for regularization loss for matrix.
    identity_64x64 = torch.eye(64,requires_grad=True,device=device).repeat(batch_size,1,1)
    # Get regularization loss for 3x3 matrix batch
    reg_loss3x3 = identity_3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    # Get regularization loss for 64x64 matrix batch
    reg_loss64x64 = identity_64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    regularization_loss = alpha* (torch.norm(reg_loss3x3) + torch.norm(reg_loss64x64)) / float(batch_size)
    return regularization_loss

def pointnet_classifier_losses(output_labels,real_labels,m3x3,m64x64):
    criterion = nn.NLLLoss() #Classification loss between output labels and real labels
    reg_loss = regularization_loss(m3x3,m64x64,output_labels.shape[0])
    classification_loss = criterion(output_labels, real_labels)
    return classification_loss + reg_loss