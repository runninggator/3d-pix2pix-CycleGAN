import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
#from data.image_folder import make_dataset
import pickle
import numpy as np


class MRI2CTDataset(BaseDataset):

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.type = 'train' if opt.isTrain else 'test'
        self.samples = os.listdir(os.path.join(opt.dataroot, f'A/{self.type}'))

        random.shuffle(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # load t2 and ct scans
        t2_scan = np.load(os.path.join(self.root, f'A/{self.type}/', sample))
        ct_scan = np.load(os.path.join(self.root, f'B/{self.type}/', sample))

        # convert to torch tensors with dimension [channel, z, x, y]
        # t2_scan = torch.from_numpy(t1_scan[None, ])
        # ct_scan = torch.from_numpy(ct_scan[None, ])
        t2_scan = torch.as_tensor(t2_scan[None,])
        ct_scan = torch.as_tensor(ct_scan[None,])
        
        return {
            'A' : t2_scan,
            'B' : ct_scan,
            'Name': sample
        }

    def __len__(self):
        return len(self.samples)
        # return len(self.AB_paths)

    def name(self):
        return 'MRI2CTDataset'

if __name__ == '__main__':
    #test
    n = MRI2CTDataset()
    n.initialize("datasets/mri2ct")
    print(len(n))
    print(n[0])
    print(n[0]['A'].size())
