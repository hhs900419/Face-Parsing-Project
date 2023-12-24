import os.path as osp
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from configs import *
from tqdm import tqdm


def list_preprocess(image_dir, mask_dir):
    train_dataset = []
    for i in range(len([name for name in os.listdir(image_dir) if osp.isfile(osp.join(image_dir, name))])):
        img_path = osp.join(image_dir, str(i)+'.jpg')
        label_path = osp.join(mask_dir, str(i)+'.png')
        train_dataset.append([img_path, label_path])
    return train_dataset

def synthesize_imgs(img1, img1_mask, img2, img2_mask, least_shift=50, max_shift=350, ratio=0.15):
    # get the bg mask
    img1_one_hot_mask = one_hot_encode(np.array(img1_mask), 19)    # (h,w) -> (19, h, w)
    img2_one_hot_mask = one_hot_encode(np.array(img2_mask), 19)    # (h,w) -> (19, h, w)
    img1_bg_mask = img1_one_hot_mask[0]
    img2_bg_mask = img2_one_hot_mask[0] 
    img2 = np.array(img2)
    img1 = np.array(img1)
    cnt = 0
    while True:
        # Specify the number of pixels to shift
        tx = np.random.randint(least_shift, max_shift+1)  # Shift along the x-axis (horizontal)
        ty = np.random.randint(least_shift, max_shift+1)  # Shift along the y-axis (vertical)
        if random.choice([True, False]):
            tx = -tx
        if random.choice([True, False]):
            ty = -ty

        # Create the translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        # Apply the translation using cv2.warpAffine
        translated_img2 = cv2.warpAffine(img2, translation_matrix, (img2.shape[1], img2.shape[0]))
        translated_img2_bg_mask = cv2.warpAffine(img2_bg_mask, translation_matrix, (img2.shape[1], img2.shape[0]))

        # fill the edge of the translated mask with 1s
        if tx < 0:
            x_start = img2.shape[1] - abs(tx)
            x_end = img2.shape[1]
        else:
            x_start = 0
            x_end = ty

        if ty < 0:
            y_start = img2.shape[0] - abs(ty)
            y_end = img2.shape[0]
        else:
            y_start = 0
            y_end = ty

        translated_img2_bg_mask[:, x_start:x_end] = 1
        translated_img2_bg_mask[y_start:y_end, :] = 1
        
        inverted_trans_img2_bg_mask = cv2.bitwise_not(translated_img2_bg_mask)
        intersection_mask = cv2.bitwise_and(img1_bg_mask, inverted_trans_img2_bg_mask)
        synth_area = np.sum(intersection_mask)
        total_area = img1.shape[0] * img1.shape[1]
        if synth_area > total_area*ratio:
            break
        else:
            cnt += 1
        if cnt > 100:
            return None, None, None
    
    indices = np.where(intersection_mask == 1)
    synth_img = img1.copy()
    synth_img[indices] = translated_img2[indices]
    synth_mask = img1_mask
    synth_bg_mask = img1_bg_mask
    # print(tx, ty)
    return synth_img, synth_mask, synth_bg_mask

if __name__ == "__main__":
    configs = Configs()
    ROOT_DIR = configs.root_dir
    image_dir = os.path.join(ROOT_DIR, 'CelebA-HQ-img')
    new_dataset_path = "/home/hsu/HD/CV/Synth-CelebAMask-HQ"
    synth_image_dir = os.path.join(new_dataset_path, 'synth-img')
    synth_mask_dir = os.path.join(new_dataset_path, 'masks')
    nums = 2400
    
    if not os.path.exists(synth_image_dir):
        os.makedirs(synth_image_dir)
    if not os.path.exists(synth_mask_dir):
        os.makedirs(synth_mask_dir)

    #### get indices of training data (use these data to synthesize images)
    train_indices = set()
    indices_file_pth = os.path.join(ROOT_DIR, 'train.txt')
    with open(indices_file_pth, 'r') as file:
        train_indices = set(map(int, file.read().splitlines()))
        
    sample_indices = list(range(len(os.listdir(image_dir))))
    test_indices = [idx for idx in sample_indices if idx not in train_indices]
    train_indices = list(train_indices)
    print(len(train_indices))
    print(len(test_indices))

    image_dir = os.path.join(ROOT_DIR, 'CelebA-HQ-img')  # Path to image folder
    mask_dir = os.path.join(ROOT_DIR, 'mask')    # Path to mask folder
    train_dataset = list_preprocess(image_dir, mask_dir)
    # print(len(train_dataset))
    
    while len(os.listdir(synth_image_dir)) < nums:
        #### random select 2 training data and load images and masks
        while True:
            idx_1 = random.randint(0, len(train_indices)-1)
            idx_2 = random.randint(0, len(train_indices)-1)
            idx_1 = train_indices[idx_1]
            idx_2 = train_indices[idx_2]
            if idx_1 != idx_2:
                break

        img1_pt, img1_mask_pt = train_dataset[idx_1]
        img2_pt, img2_mask_pt = train_dataset[idx_2]

        img1 = Image.open(img1_pt).convert('RGB')
        img1 = img1.resize((512, 512), Image.BILINEAR)
        img1_mask = Image.open(img1_mask_pt).convert('L')
        img2 = Image.open(img2_pt).convert('RGB')
        img2 = img2.resize((512, 512), Image.BILINEAR)
        img2_mask = Image.open(img2_mask_pt).convert('L')
        
        synth_img, synth_mask, synth_bg_mask = synthesize_imgs(img1, img1_mask, img2, img2_mask, least_shift=50, max_shift=350, ratio=0.1)
        
        if synth_img is None:
            continue
        
        synth_img = cv2.cvtColor(synth_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{synth_image_dir}/synth_{idx_1}.jpg", synth_img)
        synth_mask.save(f"{synth_mask_dir}/synth_{idx_1}.png")
        print(len(os.listdir(synth_image_dir)))
    print(len(os.listdir("/home/hsu/HD/CV/Synth-CelebAMask-HQ/masks")))