import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.NiftiDataset import *

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    elif opt.dataset_mode == 'nodule':
        from data.nodule_dataset import NoduleDataset
        dataset = NoduleDataset()
    elif opt.dataset_mode == 'mri2ct':
        from data.MRI2CT_dataset import MRI2CTDataset
        dataset = MRI2CTDataset()
    elif opt.dataset_mode == 'nifti':
        import data.NiftiDataset as NiftiDataset

        min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
        trainTransforms = [
            NiftiDataset.Resample(opt.new_resolution, opt.resample),
            NiftiDataset.Augmentation(),
            NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
            NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
        ]
        dataset = NifitDataSet(opt.dataroot, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=True, train=True)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)

        if opt.dataset_mode == 'nifti':
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset, 
                batch_size=opt.batchSize, 
                shuffle=True, 
                num_workers=opt.workers, 
                pin_memory=True
            )
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads)
            )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
