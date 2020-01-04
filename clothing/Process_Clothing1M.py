"""
Please notice I found a strange example in the transformation, please remove such indexes in the trainset dir manually
393191



"""



import os
import sys
from ops.os_operation import mkdir
from PIL import Image
from numpy import array
import  numpy as np
#from ops.Transform_ops import Resize

class Process_Clothing1M(object):
    """

    Clothing1M
    Args:
        root (string): Root directory of dataset where directory
            `
    """

    def __init__(self, save_path):
        #self.resize_func=Resize((256, 256))
        self.root = save_path
        self.final_path=os.path.join(self.root,'Clothing1M')
        mkdir(self.final_path)
        self.train_path = os.path.join(self.final_path, 'trainset')
        self.val_path=os.path.join(self.final_path,'validationset')
        self.test_path = os.path.join(self.final_path, 'testset')
        self.train_clean_path=os.path.join(self.final_path, 'trainset_clean')
        mkdir(self.train_path)
        mkdir(self.test_path)
        mkdir(self.val_path)
        mkdir(self.train_clean_path)
        if os.path.getsize(self.train_path) > 100000 and \
            os.path.getsize(self.val_path) > 100000 and os.path.getsize(self.test_path) > 100000 \
                and os.path.getsize(self.train_clean_path) > 100000:
            return
        self.noisy_labels = {}
        noisy_label_path = os.path.join(self.root, 'noisy_label_kv.txt')
        with open(noisy_label_path, 'r') as f:
            lines = f.read().splitlines()
        for item in lines:
            entry = item.split()
            img_path = os.path.join(self.root, entry[0])
            self.noisy_labels[img_path] = int(entry[1])
        self.clean_labels = {}
        clean_label_path = os.path.join(self.root, 'clean_label_kv.txt')
        with open(clean_label_path, 'r') as f:
            lines = f.read().splitlines()
        for item in lines:
            entry = item.split()
            img_path = os.path.join(self.root, entry[0])
            self.clean_labels[img_path] = int(entry[1])
        if os.path.getsize(self.train_path) < 10000:
            self.Process_Dataset(self.train_path,'Train')
        if os.path.getsize(self.val_path)<10000:
            self.Process_Dataset(self.val_path,'Val')
        if os.path.getsize(self.test_path)<10000:
            self.Process_Dataset(self.test_path,'Test')
        if os.path.getsize(self.train_clean_path)<10000:
            self.Process_Dataset(self.train_clean_path, 'Trainclean')
    def Process_Dataset(self,save_path,data_type):
        if data_type=='Train':
            key_list_path=os.path.join(self.root,'noisy_train_key_list.txt')
            self.train_imgs = []

            with open(key_list_path, 'r') as f:
                lines = f.read().splitlines()
            for file_name in lines:
                img_path = os.path.join(self.root,file_name)
                self.train_imgs.append(img_path)
            for img_idx in range(len(self.train_imgs)):
                train_example_path=self.train_imgs[img_idx]
                train_softlabel=self.noisy_labels[train_example_path]

                #we will not use this in training,
                # but just give two labels for dataset class to make the dataset class can work for all 3 different conditions
                if train_example_path in self.clean_labels:
                    train_reallabel=self.clean_labels[train_example_path]
                else:
                    train_reallabel=-1
                #prepare to save this data
                img = Image.open(train_example_path)
                arr = array(img)
                #arr=self.resize_func(arr)#give up because of limit of disk .
                tmp_train_path=os.path.join(save_path,'trainset'+str(img_idx)+'.npy')
                tmp_soft_path=os.path.join(save_path,'softlabel'+str(img_idx)+'.npy')
                tmp_aim_path=os.path.join(save_path,'aimset'+str(img_idx)+'.npy')
                np.save(tmp_train_path,arr)
                np.save(tmp_soft_path,np.array(train_softlabel,dtype=int))
                np.save(tmp_aim_path,np.array(train_reallabel,dtype=int))
        elif data_type=='Val':
            key_list_path = os.path.join(self.root, 'clean_val_key_list.txt')
            self.val_imgs = []

            with open(key_list_path, 'r') as f:
                lines = f.read().splitlines()
            for file_name in lines:
                img_path = os.path.join(self.root, file_name)
                self.val_imgs.append(img_path)
            for img_idx in range(len(self.val_imgs)):
                train_example_path=self.val_imgs[img_idx]
                train_reallabel=self.clean_labels[train_example_path]
                try:
                    train_softlabel=self.noisy_labels[train_example_path]
                except:
                    train_softlabel=-1
                #prepare to save this data
                img = Image.open(train_example_path)
                arr = array(img)
                #arr = self.resize_func(arr)
                tmp_train_path=os.path.join(save_path,'trainset'+str(img_idx)+'.npy')
                tmp_soft_path=os.path.join(save_path,'softlabel'+str(img_idx)+'.npy')
                tmp_aim_path=os.path.join(save_path,'aimset'+str(img_idx)+'.npy')
                np.save(tmp_train_path,arr)
                np.save(tmp_soft_path,np.array(train_softlabel,dtype=int))
                np.save(tmp_aim_path,np.array(train_reallabel,dtype=int))
        elif data_type=='Test':
            key_list_path = os.path.join(self.root, 'clean_test_key_list.txt')
            self.test_imgs = []

            with open(key_list_path, 'r') as f:
                lines = f.read().splitlines()
            for file_name in lines:
                img_path = os.path.join(self.root, file_name)
                self.test_imgs.append(img_path)
            for img_idx in range(len(self.test_imgs)):
                train_example_path=self.test_imgs[img_idx]
                train_reallabel=self.clean_labels[train_example_path]
                try:
                    train_softlabel=self.noisy_labels[train_example_path]
                except:
                    train_softlabel=-1
                #prepare to save this data
                img = Image.open(train_example_path)
                arr = array(img)#channel last here!!!
                #arr = self.resize_func(arr)
                tmp_train_path=os.path.join(save_path,'trainset'+str(img_idx)+'.npy')
                tmp_soft_path=os.path.join(save_path,'softlabel'+str(img_idx)+'.npy')
                tmp_aim_path=os.path.join(save_path,'aimset'+str(img_idx)+'.npy')
                np.save(tmp_train_path,arr)
                np.save(tmp_soft_path,np.array(train_softlabel,dtype=int))
                np.save(tmp_aim_path,np.array(train_reallabel,dtype=int))
        elif data_type=='Trainclean':
            key_list_path = os.path.join(self.root, 'clean_train_key_list.txt')
            self.trainclean_imgs = []

            with open(key_list_path, 'r') as f:
                lines = f.read().splitlines()
            for file_name in lines:
                img_path = os.path.join(self.root, file_name)
                self.trainclean_imgs.append(img_path)
            for img_idx in range(len(self.trainclean_imgs)):
                train_example_path = self.trainclean_imgs[img_idx]
                train_reallabel = self.clean_labels[train_example_path]
                try:
                    train_softlabel = self.noisy_labels[train_example_path]
                except:
                    train_softlabel = -1
                # prepare to save this data
                img = Image.open(train_example_path)
                arr = array(img)  # channel last here!!!
                # arr = self.resize_func(arr)
                tmp_train_path = os.path.join(save_path, 'trainset' + str(img_idx) + '.npy')
                tmp_soft_path = os.path.join(save_path, 'softlabel' + str(img_idx) + '.npy')
                tmp_aim_path = os.path.join(save_path, 'aimset' + str(img_idx) + '.npy')
                np.save(tmp_train_path, arr)
                np.save(tmp_soft_path, np.array(train_softlabel, dtype=int))
                np.save(tmp_aim_path, np.array(train_reallabel, dtype=int))


