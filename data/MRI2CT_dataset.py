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
        self.samples = os.listdir(os.path.join(opt.dataroot, 'A/train'))

        random.shuffle(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # load t1 and ct scans
        t1_scan = np.load(os.path.join(self.root, 'A/train/', sample))
        ct_scan = np.load(os.path.join(self.root, 'B/train/', sample))

        # convert to torch tensors with dimension [channel, z, x, y]
        # t1_scan = torch.from_numpy(t1_scan[None, ])
        # ct_scan = torch.from_numpy(ct_scan[None, ])
        t1_scan = torch.as_tensor(t1_scan[None,])
        ct_scan = torch.as_tensor(ct_scan[None,])
        
        return {
            'A' : t1_scan,
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
