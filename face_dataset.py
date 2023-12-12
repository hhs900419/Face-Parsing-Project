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

        if self.mode == 'train':
            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            
        # apply preprocessing
        mask = one_hot_encode(mask, 19)
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        mask = reverse_one_hot(mask)

        # image = self.to_tensor(image)
        # mask = torch.from_numpy(np.array(mask)).long()
        

        
        return image, mask


    def __len__(self):
        return len(self.sample_indices)