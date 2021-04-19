import struct
import torch
from torch import nn 
from torch.nn import functional as F
import torchvision
from defaults import DEFAULTS as D

from kaolin.metrics.pointcloud import chamfer_distance


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

def pointcloud_autoencoder_losses(predicted_pointcloud,real_pointcloud,m3x3,m64x64,batch_size):
    reg_loss = regularization_loss(m3x3,m64x64,batch_size)
    chamfer_loss = chamfer_distance(predicted_pointcloud,real_pointcloud)
    chamfer_loss = torch.sum(chamfer_loss) / float(batch_size)
    return chamfer_loss



class Pointnet_Autoencoder(nn.Module):
    def __init__(self,n_points):
        super(Pointnet_Autoencoder,self).__init__()
        self.encoder = Transformer()

        # Decoder
        #################################
        self.fc1 = nn.Linear(1024,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,n_points*3)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(n_points*3)
        #################################

    def forward(self,pointcloud):
        pointcloud = pointcloud.permute(0,2,1)
        assert pointcloud.dim()==3

        batch_size = pointcloud.shape[0]
        n_points = pointcloud.shape[2]
        features,matrix3x3,matrix64x64 = self.encoder(pointcloud)

        # x=F.relu(self.fc1(features))
        # x=F.relu(self.fc2(x))

        # x = F.relu(self.fc3(x))

        x=F.relu(self.bn1(self.fc1(features)))
        x=F.relu(self.bn2(self.fc2(x)))

        x = F.relu(self.bn3(self.fc3(x)))

        output =torch.reshape(x,(batch_size,n_points,3))
        return output, matrix3x3, matrix64x64


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