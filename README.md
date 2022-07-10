# 3D pix2pix/CycleGAN

Added 3D convolutional support to pix2pix/CycleGAN (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) in order to use 3D CT scans as a data source.  Used in experiments in segmenting lung nodules.

## Train example

```
python train.py --dataroot /blue/ruogu.fang/runninggator/MRI2CT/project/data/flirt_reg_AISIM_33/organized/train --input_nc 1 --output_nc 1 --name MRI2CT_7_19_22 --dataset_mode nifti --model pix2pix3d --which_model_netG unet_256 --which_direction AtoB --checkpoints_dir /blue/ruogu.fang/runninggator/MRI2CT/project/data/flirt_reg_AISIM_33/checkpoints/3d_pix2pix/
```