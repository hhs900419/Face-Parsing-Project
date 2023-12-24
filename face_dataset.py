import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import cv2
from utils import *

class CelebAMask_HQ_Dataset(Dataset):
    def __init__(self, 
                root_dir,
                sample_indices, 
                mode,
                tr_transform=None, 
                augmentation=None,
                preprocessing=None
                ):

        assert mode in ('train', "val", "test")

        self.root_dir = root_dir
        self.mode = mode

        self.tr_transform = tr_transform
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.image_dir = os.path.join(root_dir, 'CelebA-HQ-img')  # Path to image folder
        self.mask_dir = os.path.join(root_dir, 'mask')    # Path to mask folder
        self.sample_indices = sample_indices

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        self.train_dataset = []
        self.test_dataset = []
        self.list_preprocess()
        
    def list_preprocess(self):
        for i in range(len([name for name in os.listdir(self.image_dir) if osp.isfile(osp.join(self.image_dir, name))])):
            img_path = osp.join(self.image_dir, str(i)+'.jpg')
            label_path = osp.join(self.mask_dir, str(i)+'.png')

            if self.mode != "test":
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])

    def __getitem__(self, idx):
        idx = self.sample_indices[idx]
        
        if self.mode != "test":
            img_pth, mask_pth = self.train_dataset[idx]
        else:
            img_pth, mask_pth = self.test_dataset[idx]
        
        

        # read img, mask
        image = Image.open(img_pth).convert('RGB')
        image = image.resize((512, 512), Image.BILINEAR)
        # mask = Image.open(mask_pth).convert('P')
        mask = Image.open(mask_pth).convert('L')
        
        # data augmentation
        # if self.mode == 'train':
        #     image, mask = self.tr_transform(image, mask)
        
        ## convert to numpy to fit the required dtype of albumentation
        image = np.array(image)
        mask = np.array(mask)

        # apply augmentations
        if self.mode == 'train':
            if self.augmentation:   # Albumentation
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            elif self.tr_transform:
                image, mask = self.tr_transform(image, mask)
            
        # apply preprocessing
        if self.preprocessing:
            mask = one_hot_encode(mask, 19)
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            mask = reverse_one_hot(mask)
        else:
            image = self.to_tensor(image)
            mask = torch.from_numpy(np.array(mask)).long()
        

        
        return image, mask


    def __len__(self):
        return len(self.sample_indices)
    
    
class Synth_CelebAMask_HQ_Dataset(Dataset):
    def __init__(self, 
                root_dir,
                mode,
                augmentation=None,
                tr_transform=None,
                preprocessing=None,
                split_ratio=0.8
                ):
        assert mode in ('train', "val", "test")

        self.root_dir = root_dir
        self.mode = mode

        self.tr_transform = tr_transform
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.image_dir = os.path.join(root_dir, 'synth-img')  # Path to image folder
        self.mask_dir = os.path.join(root_dir, 'masks')    # Path to mask folder
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.split_ratio = split_ratio
        
        self.whole_dataset = []
        self.train_dataset = []
        self.test_dataset = []
        
        self.list_preprocess()
        print(self.whole_dataset)
        print(len(self.whole_dataset))
        ### split synthesis data with 8:2 ratio by defualt
        split_index = int(len(self.whole_dataset) * self.split_ratio)
        self.train_dataset = self.whole_dataset[:split_index]
        self.test_dataset = self.whole_dataset[split_index:]
        print(len(self.train_dataset))
        print(len(self.test_dataset))
        
        
    def list_preprocess(self):
        img_files = os.listdir(self.image_dir)
        mask_files = os.listdir(self.mask_dir)
        img_files.sort()
        mask_files.sort()
        for img_name in img_files:
            img_path = osp.join(self.image_dir, img_name.split('.')[0] + ".jpg")
            mask_path = osp.join(self.mask_dir, img_name.split('.')[0] + ".png")
            self.whole_dataset.append([img_path, mask_path])
                
            
    def __getitem__(self, idx):
        if self.mode != "test":
            img_pth, mask_pth = self.train_dataset[idx]
        else:
            img_pth, mask_pth = self.test_dataset[idx]
            
        # read img, mask
        image = Image.open(img_pth).convert('RGB')
        image = image.resize((512, 512), Image.BILINEAR)
        mask = Image.open(mask_pth).convert('L')
        
        ## convert to numpy to fit the required dtype of albumentation
        image = np.array(image)
        mask = np.array(mask)

        # apply augmentations
        if self.mode == 'train':
            if self.augmentation:   # Albumentation
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            elif self.tr_transform:
                image, mask = self.tr_transform(image, mask)
            
        # apply preprocessing
        if self.preprocessing:
            mask = one_hot_encode(mask, 19)
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            mask = reverse_one_hot(mask)
        else:
            image = self.to_tensor(image)
            mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

    
    def __len__(self):
        if self.mode != "test":
            return len(self.train_dataset)
        else:
            return len(self.test_dataset)
    
if __name__ == "__main__":
    ROOT_DIR = "/home/hsu/HD/CV/Synth-CelebAMask-HQ"
    trainset = Synth_CelebAMask_HQ_Dataset(root_dir=ROOT_DIR, 
                            mode='train', 
                            tr_transform=None)