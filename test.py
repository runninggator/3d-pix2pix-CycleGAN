import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from pdb import set_trace as st
import shutil
import numpy as np

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

# Create empty result dir
if os.path.exists(opt.results_dir):
  # Delete output directory if it already exists
  shutil.rmtree(opt.results_dir)

os.mkdir(opt.results_dir)

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    name = data['Name'][0][:-4] # remove ".npy" from string
    print('process image... %s' % name)
    
    # create output dir
    output_dir = os.path.join(opt.results_dir, name)
    os.mkdir(output_dir)

    print('---saving real and fake data')
    # save data
    np.save(os.path.join(output_dir, 'real_A'), visuals['real_A'])
    np.save(os.path.join(output_dir, 'real_B'), visuals['real_B'])
    np.save(os.path.join(output_dir, 'fake_B'), visuals['fake_B'])
