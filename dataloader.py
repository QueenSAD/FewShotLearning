import os
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import numpy as np
import random

np.random.seed(9102)
filenameToPILImage = lambda x: Image.open(x).convert("RGB")
filenameToPILImageRGB = lambda x: Image.open(x).convert("RGB")

# omniglot
class OmnigloatDataSet(data.Dataset):
    def __init__(self,
                 root='./datasets/omniglot/',
                 state="train",
                 ways=5,
                 shots=5,
                 query_num=1,
                 epoch=100,shapesize=28):
        '''
        get the path of images and divid the dataset
        '''
        self.ways = ways
        self.state = state
        self.__size = epoch
        self.shots = shots
        self.query_num = query_num
        if self.state=="test":
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
        else:
            imgs = []
            classes = [os.path.join(root, path) for path in os.listdir(root)]
            for path in classes:
                subimgs = [os.path.join(path, img) for img in os.listdir(path)]
                imgs.append(subimgs)
            
        self.transforms = T.Compose([
                filenameToPILImage,
                T.Resize((shapesize, shapesize)),
                T.ToTensor(),
            ])
            

        self.data = imgs

    def __getitem__(self, index):
        '''
        return the supportset and queryset
        '''
        supportImages = torch.FloatTensor()
        queryImages = torch.FloatTensor()
        queryBelongs = torch.LongTensor(self.ways * self.query_num, 1)

        selected_classes = np.random.choice(len(self.data), self.ways, False)

        for i in range(self.ways):
            files = np.random.choice(self.data[selected_classes[i]],
                                     self.shots+self.query_num, False)
            # Sampling support set 
            for j in range(self.shots):
                image = self.transforms(files[j])
                image = image.unsqueeze(0)
                supportImages = torch.cat((supportImages, image), 0)

            # Sampling query set 
            for j in range(self.query_num):
                image = self.transforms(files[self.shots+j])
                image = image.unsqueeze(0)
                queryImages = torch.cat((queryImages, image), 0)
                queryBelongs[i * self.query_num + j, 0] = i

        return supportImages, queryImages, queryBelongs

    def __len__(self):
        return self.__size

#fc100
class fc100(data.Dataset):
    def __init__(self,
                 root='./datasets/mini-imagenet/',
                 state="train",
                 ways=5,
                 shots=5,
                 query_num=5,
                 epoch=100,shapesize=84):
        '''
        get the path of images and divid the dataset
        '''
        self.ways = ways
        self.state = state
        self.__size = epoch
        self.shots = shots
        self.query_num = query_num
        if self.state=="test":
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
        else:
            imgs = []
            classes = [os.path.join(root, path) for path in os.listdir(root)]
            for path in classes:
                subimgs = [os.path.join(path, img) for img in os.listdir(path)]
                imgs.append(subimgs)
        if self.state=="train":
            self.transforms = T.Compose([
                            filenameToPILImage,
                            T.RandomCrop(32, padding=4),
                            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(), T.Normalize(mean=np.array([125.3, 123.0, 113.9]) / 255.0,
                                         std=np.array([63.0, 62.1, 66.7]) / 255.0)])
        else:
            self.transforms = T.Compose([
                    filenameToPILImage, 
                             T.ToTensor(),  T.Normalize(mean=np.array([125.3, 123.0, 113.9]) / 255.0,
                                         std=np.array([63.0, 62.1, 66.7]) / 255.0)])
        self.data = imgs

    def __getitem__(self, index):
        '''
        return the supportset and queryset
        '''
        supportImages = torch.FloatTensor()
        queryImages = torch.FloatTensor()
        queryBelongs = torch.LongTensor(self.ways * self.query_num, 1)

        selected_classes = np.random.choice(len(self.data), self.ways, False)

        for i in range(self.ways):
            files = np.random.choice(self.data[selected_classes[i]],
                                     self.shots+self.query_num, False)
            # Sampling support set 
            for j in range(self.shots):
                image = self.transforms(files[j])
                image = image.unsqueeze(0)
                supportImages = torch.cat((supportImages, image), 0)
            # Sampling query set 
            for j in range(self.query_num):
                image = self.transforms(files[self.shots+j])
                image = image.unsqueeze(0)
                queryImages = torch.cat((queryImages, image), 0)
                queryBelongs[i * self.query_num + j, 0] = i

        return supportImages, queryImages, queryBelongs

    def __len__(self):
        return self.__size

# miniimagenet
class MiniImageDataSet(data.Dataset):
    def __init__(self,
                 root='./datasets/mini-imagenet/',
                 state="train",
                 ways=5,
                 shots=5,
                 query_num=5,
                 epoch=100,shapesize=224):
        '''
        get the path of images and divid the dataset
        '''
        self.ways = ways
        self.state = state
        self.__size = epoch
        self.shots = shots
        self.query_num = query_num
        if self.state=="test":
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
        else:
            imgs = []
            classes = [os.path.join(root, path) for path in os.listdir(root)]
            for path in classes:
                subimgs = [os.path.join(path, img) for img in os.listdir(path)]
                imgs.append(subimgs)
        if self.state=="train":
            self.transforms = T.Compose([
                            T.RandomCrop(224, padding=4),
                            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                            T.RandomHorizontalFlip(),
                            filenameToPILImage,T.Resize((shapesize,shapesize)), 
                            ImageNetPolicy(), 
                            T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
        else:
            self.transforms = T.Compose([
                    filenameToPILImage,T.Resize((shapesize,shapesize)), 
                             T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
        self.data = imgs

    def __getitem__(self, index):
        '''
        return the supportset and queryset
        '''
        supportImages = torch.FloatTensor()
        queryImages = torch.FloatTensor()
        queryBelongs = torch.LongTensor(self.ways * self.query_num, 1)

        selected_classes = np.random.choice(len(self.data), self.ways, False)

        for i in range(self.ways):
            files = np.random.choice(self.data[selected_classes[i]],
                                     self.shots+self.query_num, False)
            # Sampling support set 
            for j in range(self.shots):
                image = self.transforms(files[j])
                image = image.unsqueeze(0)
                supportImages = torch.cat((supportImages, image), 0)
            # Sampling query set 
            for j in range(self.query_num):
                image = self.transforms(files[self.shots+j])
                image = image.unsqueeze(0)
                queryImages = torch.cat((queryImages, image), 0)
                queryBelongs[i * self.query_num + j, 0] = i
        return supportImages, queryImages, queryBelongs
    def __len__(self):
        return self.__size