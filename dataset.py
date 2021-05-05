import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import os,h5py, scipy.io , itertools
from helpers import image_utils
from defaults import DEFAULTS as D
import args as args_


class Describable_Textures_Dataset(Dataset):

    def __init__(self,set,root='./data/dtd',imdb_path='./data/dtd/imdb/imdb.mat',remove_classes=[],only_class=[],lower_size=None) -> None:
        assert set.lower() in ['train','val','test']
        self.set = set.lower()
        self.root = root
        mat = scipy.io.loadmat(imdb_path)

        self.classes = list(itertools.chain(*mat['meta']['classes'][0][0][0]))
        
        if remove_classes:
            assert all(x in self.classes for x in remove_classes)

        image_files = list(itertools.chain(*mat['images']['name'][0][0][0]))
        ids = list(itertools.chain(*mat['images']['id'][0][0]))
        sets = list(itertools.chain(*mat['images']['set'][0][0]))
        class_ = list(itertools.chain(*mat['images']['class'][0][0]))

        self.data = np.asarray([ids,image_files,sets,class_]).T

        if remove_classes:
            self.data = np.asarray([x for x in self.data if os.path.dirname(x[1]) not in remove_classes])
        
        if only_class:
            self.data = np.asarray([x for x in self.data if os.path.dirname(x[1]) in only_class])
        
        if set.lower()=='train':
            self.data = self.data[(self.data[:,2]=='1')] 
        elif set.lower()=='val':
            self.data = self.data[(self.data[:,2]=='2')]
        elif set.lower()=='test':
            self.data = self.data[(self.data[:,2]=='3')]

        if lower_size:
            self.data = self.data[np.random.choice(self.data.shape[0],lower_size,replace=False)]
            print(self.data[:,1])

        self.size = len(self.data)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        args = args_.parse_arguments()

        # Each row in the dataset has [id,image_path,set,class]
        sample = self.data[index]
        # id = sample[0]
        image_path = os.path.join(self.root,'images',sample[1])
        # set=sample[2]
        # class_=sample[3]

        image = image_utils.image_to_tensor(image_utils.load_image(image_path,'RGB'),phase='test',image_size=args.style_size).detach()
        
        return image

class Styles_Dataset(Dataset):

    def __init__(self, style_dir, style_size, style_files=None,set=None,lower_size=None):
        super().__init__()

        self.style_dir = style_dir 
        self.style_size = style_size 
        self.style_files = []
        self.set=set

        # If no particular files are specified, load all files from the style_dir
        if style_files is None:
            for s in os.listdir(self.style_dir):
                self.style_files.append(os.path.join(self.style_dir,s))
        else:
            for s in style_files:
                self.style_files.append(os.path.join(self.style_dir,s))
        
        if lower_size:
            self.style_files = self.style_files[:lower_size]
    
    def __len__(self):
        return len(self.style_files)

    def __getitem__(self, index):
        style_im = self.style_files[index]
        style = image_utils.image_to_tensor(image_utils.load_image(style_im),phase=self.set,image_size=self.style_size).detach()
        style = style[:3,...]
        return style


class UV_Style_Paired_Dataset(Dataset):

    def __init__(self, uv_dir,style_dir, uv_sizes, style_size, uv_style_pairs:dict) -> None:
        super().__init__()

        self.uv_dir = uv_dir
        self.style_dir = style_dir 
        self.uv_sizes = uv_sizes 
        self.style_size = style_size 
        self.uv_files = []
        self.style_files = []
        
        for k,v in uv_style_pairs.items():
            self.uv_files.append(os.path.join(self.uv_dir,k))
            self.style_files.append(os.path.join(self.style_dir,v))

        assert len(self.style_files) == len(self.uv_files)
    
    def __len__(self):
        return len(self.style_files)

    def __getitem__(self, index):
        
        style_im = self.style_files[index]
        style = image_utils.image_to_tensor(image_utils.load_image(style_im),image_size=self.style_size).detach()
        style = style[:3,...]

        uv_map_set = []
        for uv_size in self.uv_sizes:
            uv_map_set.append(image_utils.image_to_tensor(image_utils.load_image(self.uv_files[index]),image_size = uv_size).detach())

        sample = {'style': style, 'uvs':uv_map_set}
        
        return sample



if __name__ =='__main__':
    td = Describable_Textures_Dataset('train',remove_classes=['banded'],lower_size=10)
    yes = td.__getitem__(0)