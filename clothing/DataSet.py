import torch
import torch.utils.data as data
import numpy as np
import random
import os
from PIL import Image, PILLOW_VERSION, ImageEnhance
import numbers
from torchvision.transforms.functional import _get_inverse_affine_matrix
import math
from sklearn.model_selection import train_test_split
from collections import defaultdict

class Simple_Clothing_Dataset1(data.Dataset):
    def __init__(self, dataset_dir,
                 transform_pre=None,require_number=None,rand_state=888,require_index=False,remove_unknown=False):
        super(Simple_Clothing_Dataset1, self).__init__()
        self.root = os.path.abspath(dataset_dir)
        listfiles=os.listdir(self.root)
        self.trainlist = [os.path.join(dataset_dir, x) for x in listfiles if "trainset" in x]
        self.soft_labels = [os.path.join(dataset_dir, x) for x in listfiles if "softlabel" in x]
        self.aimlist = [os.path.join(dataset_dir, x) for x in listfiles if "aimset" in x]
        self.soft_labels.sort()
        self.trainlist.sort()
        self.aimlist.sort()
        if require_number is not None:
            repeat_times=int(require_number/len(self.trainlist))
            self.trainlist=self.trainlist*repeat_times
            self.aimlist=self.aimlist*repeat_times
            self.soft_labels=self.soft_labels*repeat_times
            print("Extended to %d examples"%len(self.trainlist))
        self.transform_now=transform_pre
        self.require_index = require_index
        self.recover_label=False
        if remove_unknown:
            tmp_trainlist=[]
            tmp_softlist=[]
            tmp_aimlist=[]
            for k in range(len(self.trainlist)):
                aim_path = self.soft_labels[k]
                target=np.load(aim_path)
                if target!=-1:
                    tmp_trainlist.append(self.trainlist[k])
                    tmp_softlist.append(self.soft_labels[k])
                    tmp_aimlist.append(self.aimlist[k])
            self.soft_labels=tmp_softlist
            self.aimlist=tmp_aimlist
            self.trainlist=tmp_trainlist
            print("After remove unknown noisy label, we have %d labels"%len(self.trainlist))
    def normalise(self,x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),):
        x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
        x -= mean * 255
        x *= 1.0 / (255 * std)
        return x
    def Rebuild_Label(self,soft_label):
        self.soft_labels=soft_label
        self.recover_label=True
    def __getitem__(self, index):
        train_path = self.trainlist[index]  # we change the saving to disk files to directly load all to memory
        soft_path = self.soft_labels[index]
        aim_path = self.aimlist[index]

        img1 = np.load(train_path)
        #img1=Image.fromarray(img1.astype(np.uint8))
        if self.recover_label:
            target=soft_path
        else:
            target = np.load(soft_path)  # soft target
        real_target = np.load(aim_path)
        #img1= self.transform_now(img1)

        #img1 = img1.transpose((1, 2, 0))  # change 3*32*32 to 32*32*3
        #img1 = self.normalise(img1)  # normalize#channel last format
        try:
            img1 = self.normalise(img1)#normalize#channel last format
        except Exception as e:
            print('##We have errors for processin data %s'%e)
            #I don't know why we have thisoperands could not be broadcast together with shapes (310,227) (3,) (310,227)
            if index+1< len(self.aimlist):
                return self.__getitem__(index+1)
            elif index-1>0:
                return self.__getitem__(index - 1)
            else:
                randnumber=np.random.randint(len(self.aimlist))
                return self.__getitem__(randnumber)
        img1 = self.transform_now(img1)
        img1 = img1.transpose((2, 0, 1))
        img1 = torch.from_numpy(img1).float()
        if self.require_index:
            return img1, index,target, real_target
        else:
            return img1,target,real_target
    def __len__(self):
        return len(self.aimlist)