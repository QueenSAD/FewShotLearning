from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label




class FewShotModelNet40Cls(data.Dataset):
    def __init__(self,num_points=1024, transforms=None, state="train",ways=5,shots=5,query_num=1,epoch=10000):
        super().__init__()

        self.transforms = transforms
        self.state = state
        self.ways=ways
        self.shots=shots
        self.query_num=query_num
        self.__size = epoch
        self.data=[]
        self.pt_idxs = np.arange(0, num_points)
        # if self.state=="train":
        #     select = range(0,20)
        # if self.state=="val":
        #     select = range(20,30)
        # if self.state=="test":
        #     select = range(30,40)
        if self.state=="train":
            select = [0,3,4,5,6,7,8,9,11,12,13,15,17,18,19,20,21,22,23,26]
        if self.state=="val":
            select = [27,28,29,30,31,33,34,36,38,39]
        if self.state=="test":
            select = [1,2,10,14,16,24,25,32,35,37]
       # merge_ply_data.h5 can download from http://modelnet.cs.princeton.edu/
        with h5py.File('./data/merge_ply_data.h5') as f:
            for i in select:
                self.data.append(f[str(i)][:])
                
    def __getitem__(self, index):
        supportPoints = torch.FloatTensor()
        queryPoint = torch.FloatTensor()
        queryBelongs = torch.LongTensor(self.ways * self.query_num, 1)

        selected_classes = np.random.choice(len(self.data), self.ways, False)

        for i in range(self.ways):
            points_id = np.random.choice(len(self.data[selected_classes[i]]),
                                     self.shots+self.query_num, False)
            for j in range(self.shots):
                spoints = self.transforms(self.data[selected_classes[i]][points_id[j]])
                np.random.shuffle(self.pt_idxs)
                spoints = spoints[self.pt_idxs]
                spoints=spoints.unsqueeze(0)
                supportPoints = torch.cat((supportPoints, spoints), 0)

            for j in range(self.query_num):
                qpoints = self.transforms(self.data[selected_classes[i]][points_id[self.shots+j]])
                np.random.shuffle(self.pt_idxs)
                qpoints = qpoints[self.pt_idxs]
                qpoints=qpoints.unsqueeze(0)
                queryPoint = torch.cat((queryPoint, qpoints), 0)
                queryBelongs[i * self.query_num + j, 0] = i

        return supportPoints, queryPoint,queryBelongs

    def __len__(self):
        return self.__size


