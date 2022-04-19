#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tfs
import os
from PIL import Image
import matplotlib.pyplot as plt
from Augmentations import Augment


# In[2]:


class ImageNette(Dataset):
    def __init__(self, train = True):
        super().__init__()
        if os.path.exists('imagenette2-160'):
            print('ImageNette exists. Loading data now...')
        else:
            if os.path.exists('tiny-imagenet-200.zip'):
                print('Zip File exists unzipping data now...')
                get_ipython().system('tar -xvzf imagenette2-160.tgz')
            else:
                print('Data not in folder. Downloading data and unzipping now...')
                get_ipython().system('wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz')
                get_ipython().system('tar -xvzf imagenette2-160.tgz')
                
        self.train = train
        self.labels = None
        self.images = None

        self.classes = 10
        self.resize1 = tfs.Resize(int(124 * (256 / 224)), interpolation = tfs.InterpolationMode.BICUBIC)
        
        if train: self.path = "imagenette2-160/train/"
        else:     self.path = "imagenette2-160/val/"
        
        self.return_labels()
        self.get_all_images()
        
        print('Data Loaded')
        return
        
        
        
    def return_labels(self):

        #map the keys to the words with a number for each for easy classification
        _labels = {}
            
        for i, l in enumerate(sorted(os.listdir(self.path))):
            if l == '.DS_Store': continue
            if self.train:
                _labels[l] = [l, i - 1]
            else:
                _labels[l] = [l, i]

        self.labels = _labels
        return


    
    
    def get_all_images(self):

        #each nested list will be path, class label, class label name
        images = []
        labels = []

        for i, l in enumerate(sorted(os.listdir(self.path))):
            if l == '.DS_Store': continue

            l = l + '/'

            for _i, _l in enumerate(sorted(os.listdir(self.path + l))):
                image = self.path + l + _l
                images.append(image)
                labels.append(self.labels[l[:-1]][1])

        self.labels = labels
        self.images = images
        return

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns two augmented images according to the configurations of the augmentations,
        the label and the label name in normal english
        """
        
        image = Image.open(self.images[idx])
        image = tfs.ToTensor()(image)
        
        #resizes and takes a center crop according to BYOL paper
        image = self.resize1(image)
        image = tfs.CenterCrop(124)(image)
        if image.shape[0] == 1:
            image = torch.stack((image,) * 3, axis = 1)[0]
        
        label = self.labels[idx]
        
        
        return image, label


# In[3]:


class TinyImageNetTrain(Dataset):
    def __init__(self):
        if os.path.exists('tiny-imagenet-200'):
            print('Tiny Image Net exists. Loading data now...')
        else:
            if os.path.exists('tiny-imagenet-200.zip'):
                print('Zip File exists unzipping data now...')
                get_ipython().system('unzip tiny-imagenet-200')
            else:
                print('Data not in folder. Downloading data and unzipping now...')
                get_ipython().system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
                get_ipython().system('unzip tiny-imagenet-200')
                
        self.path = "tiny-imagenet-200/train/"
        self.labels = None
        self.images = None
        self.classes = 200
        
        self.return_labels()
        self.get_all_images()
        
        print('Data Loaded')
        return
        
        
        
    def return_labels(self):

        #open the labels and get the words needed
        labels = {}
        with open("tiny-imagenet-200/words.txt") as file:
            for line in file:
                line = line.split()
                key = line[0]
                value = " ".join(line[1:])
                labels[key] = value
        file.close()

        #map the keys to the words with a number for each for easy classification
        _labels = {}
        for i, l in enumerate(sorted(os.listdir(self.path))):
            if l == '.DS_Store': continue
            _labels[l] = [labels[l], i - 1]

        self.labels = _labels
        return


    
    
    def get_all_images(self):

        #each nested list will be path, class label, class label name
        images = []


        for i, l in enumerate(sorted(os.listdir(self.path))):
            if l == '.DS_Store': continue

            for _i, _l in enumerate(sorted(os.listdir(self.path + l + "/images"))):
                image = [self.path + l + "/images/" + _l, self.labels[l][1], self.labels[l][0]]
                images.append(image)
                
        
        self.images = images
        return
        
        
        
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns images and the label
        """
        
        image = Image.open(self.images[idx][0])
        image = tfs.ToTensor()(image)
        if image.shape[0] == 1:
            image = torch.stack((image,) * 3, axis = 1)[0]
        #no need to resize as all are 64x64 piccs
        
        return image1, image2, self.images[idx][1]


# In[4]:


class TinyImageNetVal(Dataset):
    def __init__(self):
        if os.path.exists('tiny-imagenet-200'):
            print('Tiny Image Net exists. Loading data now...')
        else:
            if os.path.exists('tiny-imagenet-200.zip'):
                print('Zip File exists unzipping data now...')
                get_ipython().system('unzip tiny-imagenet-200')
            else:
                print('Data not in folder. Downloading data and unzipping now...')
                get_ipython().system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
                get_ipython().system('unzip tiny-imagenet-200')
                
        self.path = "tiny-imagenet-200/val/images/"
        self.labels = None
        self.images = None
        self.classes = 200
        
        self.return_labels()
        self.get_all_images()
        
        print('Data Loaded')
        return
        
        
        
    def return_labels(self):
        
        #open the labels and get the words needed
        labels = {}
        with open("tiny-imagenet-200/words.txt") as file:
            for line in file:
                line = line.split()
                key = line[0]
                value = " ".join(line[1:])
                labels[key] = value
        file.close()

        #map the keys to the words with a number for each for easy classification
        _labels = {}
        for i, l in enumerate(sorted(os.listdir("tiny-imagenet-200/train/"))):
            if l == '.DS_Store': continue
            _labels[l] = [labels[l], i - 1]

        self.labels = _labels

        #open the labels and get the words needed
        new_labels = {}
        
        with open("tiny-imagenet-200/val/val_annotations.txt") as file:
            for line in file:
                line = line.split()
                key = line[0]
                value = line[1]
                new_labels[key] = self.labels[value][-1]
        file.close()
        self.labels = new_labels
        return


    def get_all_images(self):

        #each nested list will be path, class label, class label name
        images = []


        for i, l in enumerate(sorted(os.listdir(self.path))):
            if l == '.DS_Store': continue
            image = [self.path + l, self.labels[l]]
            images.append(image)
                
        
        self.images = images
        return
        
        
        
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns two augmented images according to the configurations of the augmentations,
        the label and the label name in normal english
        """
        
        image = Image.open(self.images[idx][0])
        image = tfs.ToTensor()(image)
        
        return image, self.images[idx][1]


# In[10]:


class CIFAR_10(Dataset):
    def __init__(self, train = True):
        if train:
            self.data =  torchvision.datasets.CIFAR10(root='./CIFAR-10', train=True,
                                        download=True)
        else:
            self.data = torchvision.datasets.CIFAR10(root='./CIFAR-10', train=False,
                                        download=True)
        self.classes = 10
        
        
    def __len__(self):
        return self.data.__len__()
        
    def __getitem__(self, idx):
        image, label = self.data.__getitem__(idx)
        image = tfs.ToTensor()(image)
        return image, label
        


# In[11]:



class CIFAR_100(Dataset):
    def __init__(self, train = True):
        if train:
            self.data =  torchvision.datasets.CIFAR100(root='./CIFAR-100', train=True,
                                        download=True)
        else:
            self.data = torchvision.datasets.CIFAR100(root='./CIFAR-100', train=False,
                                        download=True)
            
        self.classes = 100
    def __len__(self):
        return self.data.__len__()
        
    def __getitem__(self, idx):
        image, label = self.data.__getitem__(idx)
        image = tfs.ToTensor()(image)
        return image, label


# In[ ]:




