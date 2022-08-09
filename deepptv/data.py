#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
# import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset



class FluidflowDataset(Dataset):
    def __init__(self, npoints=2048, root='data_preprocessing/data_processed_maxcut_35_both_mask_20k_2k', partition='train'):
        self.npoints = npoints
        self.partition = partition
        self.root = root
        if self.partition=='train':
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
        self.cache = {}
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######
        self.datapath.sort()
        print(self.partition, ': ',len(self.datapath))
        #print(self.partition, ': ',self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, color1, color2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1'].astype('float32')
                pos2 = data['pos2'].astype('float32')
                color1 = data['intensity1'].astype('float32')
                color2 = data['intensity2'].astype('float32')
                flow = data['flow'].astype('float32')
                #mask1 = data['valid_mask1']

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, color1, color2, flow)

        if self.partition == 'train':
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            color1 = color1[sample_idx1, :]
            color2 = color2[sample_idx2, :]
            flow = flow[sample_idx1, :]
            #mask1 = mask1[sample_idx1]
        else:
            pos1 = pos1[:self.npoints, :]
            pos2 = pos2[:self.npoints, :]
            color1 = color1[:self.npoints, :]
            color2 = color2[:self.npoints, :]
            flow = flow[:self.npoints, :]
            #mask1 = mask1[:self.npoints]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        return pos1, pos2, color1, color2, flow

    def __len__(self):
        return len(self.datapath)


class FluidflowDataset3D(Dataset):
    def __init__(self, npoints=2048, root='data_preprocessing/data_processed_maxcut_35_both_mask_20k_2k', partition='train'):
        self.npoints = npoints
        self.partition = partition
        self.root = root
        if self.partition=='train':
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
        self.cache = {}
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######
        self.datapath.sort()
        print(self.partition, ': ',len(self.datapath))
        # print(self.partition, ': ',self.datapath)
        # datalist=np.array(self.datapath)
        # np.save('test_result/eval_allflow/datalist.npy', datalist)

    def __getitem__(self, index):
        if index in self.cache:
            # pos1, pos2, color1, color2, flow = self.cache[index]
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1'].astype('float32')
                pos2 = data['pos2'].astype('float32')
                # color1 = data['intensity1'].astype('float32')
                # color2 = data['intensity2'].astype('float32')
                flow = data['flow'].astype('float32')
                #mask1 = data['valid_mask1']
            
            # if 's025' in fn:
            #     pos1 = pos1 * 8.
            #     pos2 = pos2 * 8.
            #     flow = flow * 8.
            # elif 's050' in fn:
            #     pos1 = pos1 * 4.
            #     pos2 = pos2 * 4.
            #     flow = flow * 4.
            # elif 's100' in fn:
            #     pos1 = pos1 * 2.
            #     pos2 = pos2 * 2.
            #     flow = flow * 2.
            # elif 'beltrami' in fn :
            #     pos1 = (pos1 + 1.) * np.pi
            #     pos2 = (pos2 + 1.) * np.pi
            #     flow = flow * np.pi

            if len(self.cache) < self.cache_size:
                # self.cache[index] = (pos1, pos2, color1, color2, flow)
                self.cache[index] = (pos1, pos2, flow)

        if self.partition == 'train':
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            # color1 = color1[sample_idx1, :]
            # color2 = color2[sample_idx2, :]
            flow = flow[sample_idx1, :]
            #mask1 = mask1[sample_idx1]
        else:
            pos1 = pos1[:self.npoints, :]
            pos2 = pos2[:self.npoints, :]
            # color1 = color1[:self.npoints, :]
            # color2 = color2[:self.npoints, :]
            flow = flow[:self.npoints, :]
            #mask1 = mask1[:self.npoints]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        return pos1, pos2, flow # color1, color2, 

    def __len__(self):
        return len(self.datapath)


class FluidflowDataset3D_eval(Dataset):
    def __init__(self, npoints=2048, root='data_preprocessing/data_processed_maxcut_35_both_mask_20k_2k', partition='train'):
        self.npoints = npoints
        self.partition = partition
        self.root = root
        if self.partition=='train':
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
        self.cache = {}
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######
        self.datapath.sort()
        print(self.partition, ': ',len(self.datapath))
        #print(self.partition, ': ',self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            # pos1, pos2, color1, color2, flow = self.cache[index]
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1'].astype('float32')
                pos2 = data['pos2'].astype('float32')
                # color1 = data['intensity1'].astype('float32')
                # color2 = data['intensity2'].astype('float32')
                flow = data['flow'].astype('float32')
                #mask1 = data['valid_mask1']
            
            # if 's025' in fn:
            #     pos1 = pos1 * 8.
            #     pos2 = pos2 * 8.
            #     flow = flow * 8.
            # elif 's050' in fn:
            #     pos1 = pos1 * 4.
            #     pos2 = pos2 * 4.
            #     flow = flow * 4.
            # elif 's100' in fn:
            #     pos1 = pos1 * 2.
            #     pos2 = pos2 * 2.
            #     flow = flow * 2.
            # elif 'beltrami' in fn :
            #     pos1 = (pos1 + 1.) * np.pi
            #     pos2 = (pos2 + 1.) * np.pi
            #     flow = flow * np.pi

            if len(self.cache) < self.cache_size:
                # self.cache[index] = (pos1, pos2, color1, color2, flow)
                self.cache[index] = (pos1, pos2, flow)

        if self.partition == 'train':
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            # color1 = color1[sample_idx1, :]
            # color2 = color2[sample_idx2, :]
            flow = flow[sample_idx1, :]
            #mask1 = mask1[sample_idx1]
        else:
            pos1 = pos1[:self.npoints, :]
            pos2 = pos2[:self.npoints, :]
            # color1 = color1[:self.npoints, :]
            # color2 = color2[:self.npoints, :]
            flow = flow[:self.npoints, :]
            #mask1 = mask1[:self.npoints]

        # pos1_center = np.mean(pos1, 0)
        # pos1 -= pos1_center
        # pos2 -= pos1_center

        return pos1, pos2, flow # color1, color2, 

    def __len__(self):
        return len(self.datapath)

# if __name__ == '__main__':
#     train = ModelNet40(1024)
#     test = ModelNet40(1024, 'test')
#     for data in train:
#         print(data[0].shape)
#         break
