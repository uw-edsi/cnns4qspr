from cnns4qspr.loader import voxelize
from cnns4qspr.visualizer import plot_field, plot_internals
from cnns4qspr.featurizer import featurize, gen_feature_set
from cnns4qspr.trainer import Trainer, CNNTrainer, BigDataset
import argparse
import numpy as np
from silx.io.dictdump import h5todict
import os
from timeit import default_timer as timer
from sys import getsizeof
import sys
import h5py

channels = ['protein', 'ligand', 'all_C', 'all_O', 'all_N', 'acidic', 'basic', 'polar', 'nonpolar',\
                 'charged', 'amphipathic','hydrophobic', 'aromatic', 'acceptor', 'donor',\
                 'ring', 'hyb', 'heavyvalence', 'heterovalence', 'partialcharge']
#channels = ['all_C', 'all_O']


path = '/gscratch/pfaendtner/Chowdhury/cnns4qspr/test_codes/feature_30/validation/'
#path = '/gscratch/pfaendtner/Chowdhury/cnns4qspr/test_codes/small_set'

#file_name = 'features_all.npz'

def convert_h5_data(file_path, channels_considered):

    file_names = os.listdir(file_path)
    data_for_cnn = []
    target = []
    num_files =  next(os.walk(file_path))[2]
    print(len(num_files))
    #num_npy = int(len(num_files) / 5)
    i = 0
    cur_dir = os.getcwd()
    
    for ix, file in enumerate(file_names):
        input_file = os.path.join(file_path, file)
        vox = h5todict(input_file)
        data_structure = []
        for channel in channels_considered:
            data_structure.append(vox[channel])
        data_for_cnn.append(data_structure)
        target.append(vox['affinity'])
        if (ix > 0 and ix % 500 ==0) or ix == len(num_files) - 1 :
            print(i)
            target = np.array(target).squeeze()
            data_for_cnn = np.expand_dims(np.array(data_for_cnn), 5)
            data_for_cnn[:,0,0,0,0,0] = target
            file_to_save = os.path.join(cur_dir, f'data_30/validation/feature{i}.npy')
            #np.save(f'data/feature{i}.npy', data_for_cnn)
            np.save(file_to_save, data_for_cnn)
            data_for_cnn = []
            target = []
            i = i + 1
        #print(ix)
        #if ix > 50:
        #    break

    size = str(round(getsizeof(np.array(data_for_cnn)) / 1024 / 1024,2))
    print(size, flush=True)
    sys.stdout.write('\r'+'structures loaded...\n')
    #sys.stdout.write(("Value is %s" % size)
    sys.stdout.flush()
    #return np.array(data_for_cnn), np.array(target)


if __name__ = "__main__":
    start = timer()
    convert_h5_data(path, channels)
    end = timer()
    print(end - start)
