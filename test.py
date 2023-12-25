import os.path as osp
import os
import cv2
import numpy as np
from augmentation import *
from face_dataset import *
from models.unet import *
from models.attention_unet import *
from criterion import *
from tester import *
from configs import *
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.backends import cudnn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
import segmentation_models_pytorch as smp
import gc
from utils import *


def test_fn():
    configs = Configs()
    SEED = configs.seed
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.cuda.manual_seed(SEED)

    ### Train/Val/Test Split ###
    """
    create train/val/test index list (only test is used in this script)
    """
    ROOT_DIR = configs.root_dir
    image_dir = os.path.join(ROOT_DIR, 'CelebA-HQ-img')

    train_indices = set()
    indices_file_pth = os.path.join(ROOT_DIR, 'train.txt')
    with open(indices_file_pth, 'r') as file:
        train_indices = set(map(int, file.read().splitlines()))
        
    sample_indices = list(range(len(os.listdir(image_dir))))
    test_indices = [idx for idx in sample_indices if idx not in train_indices]
    # Split indices into training and validation sets
    train_indices = list(train_indices)
    if configs.debug:
        train_indices = train_indices[:100]         
    VAL_SIZE = configs.val_size
    train_indices, valid_indices = train_test_split(train_indices, test_size=VAL_SIZE, random_state=SEED)
    print(len(test_indices))


    ### test dataloader ###
    BATCH_SIZE = configs.batch_size
    N_WORKERS = configs.n_workers

    testset = CelebAMask_HQ_Dataset(root_dir=ROOT_DIR,
                                sample_indices=test_indices,
                                mode='test')

    test_loader = DataLoader(testset,
                        batch_size = BATCH_SIZE,
                        shuffle = False,
                        num_workers = N_WORKERS, 
                        pin_memory = True,
                        drop_last = True)
    
    ####################### for Debugging
    if configs.debug:
        validset = CelebAMask_HQ_Dataset(root_dir=ROOT_DIR, 
                                        sample_indices=valid_indices, 
                                        mode = 'val')
        valid_loader = DataLoader(validset,
                            batch_size = BATCH_SIZE,
                            shuffle = False,
                            num_workers = N_WORKERS, 
                            pin_memory = True,
                            drop_last = True)
    #################################
    

    ### load model weight ###
    DEVICE = configs.device
    SAVEPATH = configs.model_path
    OUTPUT_DIR = f'{configs.cmp_result_dir}/vis_{get_current_timestamp()}'
    OUTPUT_DIR_UNSEEN = f'{configs.cmp_result_dir}/unseen_vis_{get_current_timestamp()}'
    MODEL_WEIGHT = configs.load_model_weight
    
    if configs.debug:
        MODEL_WEIGHT = 'model_debug.pth'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ## make sure the setting is same as the model in train.py
    # ENCODER = 'efficientnet-b3'
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=19, 
    )
    
    # model = model.to(DEVICE)
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(SAVEPATH , MODEL_WEIGHT)))
    

    ## testing
    criterion = DiceLoss()
    
    Tester(model=model, 
       testloader=test_loader, 
       criterion=criterion, 
       device=DEVICE).run()
    
    
    

    ### visualize and generate csv file
    cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
                         (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                         (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                         (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                         (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                        dtype=np.uint8)
    # some preprocessing
    to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    image_dir = os.path.join(ROOT_DIR, 'CelebA-HQ-img') 
    mask_dir = os.path.join(ROOT_DIR, 'mask')    

    test_dataset =[]
    for i in range(len([name for name in os.listdir(image_dir) if osp.isfile(osp.join(image_dir, name))])):
        img_path = osp.join(image_dir, str(i)+'.jpg')
        label_path = osp.join(mask_dir, str(i)+'.png')
        test_dataset.append([img_path, label_path])

    # inference again in file order
    # test_dir = f"result/test_result_{get_current_timestamp()}"
    # for i in tqdm(range(0, len(test_indices))):
    #     ### The below operation is simmilar to the __getitem__() function
    #     idx = test_indices[i]
    #     if configs.debug:
    #         idx = valid_indices[i]
    #         idx = train_indices[i]
    #     img_pth, mask_pth = test_dataset[idx]
    #     image = Image.open(img_pth).convert('RGB')
    #     image = image.resize((512, 512), Image.BILINEAR)
    #     mask = Image.open(mask_pth).convert('L')

    #     image = to_tensor(image).unsqueeze(0)
    #     gt_mask = torch.from_numpy(np.array(mask)).long()

    #     ### predict with model
    #     # pred_mask = model(image.to(DEVICE))     # predict
    #     pred_mask = model(image.cuda())     # predict
    #     pred_mask = pred_mask.data.max(1)[1].cpu().numpy()  # Matrix index  (1,19,h,w) => (1,h,w)
        
    #     image = image.squeeze(0).permute(1,2,0)     # (1,3,h,w) -> (h,w,3)
    #     pred_mask = pred_mask.squeeze(0)            # (1,h,w) -> (h,w)


    #     classes = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
    #             'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    #     one_hot_mask = one_hot_encode(pred_mask, 19)    # (h,w) -> (19, h, w)
    #     # print(one_hot_mask.shape)
    #     # print(one_hot_mask)
        
       
    #     TEST_ID_DIR = f'{test_dir}/Test-image-{idx}'
    #     if not os.path.exists(TEST_ID_DIR):
    #         os.makedirs(TEST_ID_DIR)

    #     dict_path = {}    
    #     # save seperated predict masks 
    #     for j in range(19):
    #         if j == 0:
    #             mask = one_hot_mask[j,:,:] * 0
    #         else:
    #             mask = one_hot_mask[j,:,:] * 255
    #         cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
    #         dict_path[classes[j]] = f"{TEST_ID_DIR}/{classes[j]}.png"


    #     # generate color mask image compared with GT(60 samples)
    #     if i % 100 == 0:
    #         color_gt_mask = cmap[gt_mask]
    #         color_pr_mask = cmap[pred_mask]
    #         plt.figure(figsize=(13, 6))
    #         image = Image.open(img_pth).convert('RGB')      # we want the image without normalization for plotting
    #         image = image.resize((512, 512), Image.BILINEAR)
    #         img_list = [image, color_pr_mask, color_gt_mask]
    #         for n in range(3):
    #             plt.subplot(1, 3, n+1)
    #             plt.imshow(img_list[n])
    #         plt.savefig(f"{OUTPUT_DIR}/result_{idx}.jpg")

    #     ### Reorder the 19 class order since we use a different order during preprocessing
    #     labels_celeb = ['background','skin','nose',
    #     'eye_g','l_eye','r_eye','l_brow',
    #     'r_brow','l_ear','r_ear','mouth',
    #     'u_lip','l_lip','hair','hat',
    #     'ear_r','neck_l','neck','cloth']

    #     right_order_mask_path = {}
    #     for lab in labels_celeb:
    #         right_order_mask_path[lab] = dict_path[lab]
        
    #     # Csv file for submission
    #     mask2csv(mask_paths=right_order_mask_path, image_id=i)
    #     # break

    #### unseen dataset eval #######
    unseen_dir = "/home/hsu/HD/CV/unseen_10samples"
    # unseen_dir = "/home/hsu/HD/CV/unseen"
    image_list = []
    mask_list = []

    # Iterate through the files in the directory
    for filename in os.listdir(unseen_dir):
        if filename.endswith('.jpg'):
            # Append to the image list if it's a JPG file
            image_list.append(os.path.join(unseen_dir, filename))
        elif filename.endswith('.png'):
            # Append to the mask list if it's a PNG file
            mask_list.append(os.path.join(unseen_dir, filename))
    mask_list = sorted(mask_list)
    image_list = sorted(image_list)

    # for image in image_list:
    #     print(image)

    # for mask in mask_list:
    #     print(mask)

    test_dir = f"result/unseen_10_test_result_{get_current_timestamp()}"
    for i in tqdm(range(0, len(image_list))):
        ### The below operation is simmilar to the __getitem__() function
        img_pth = image_list[i]
        mask_pth = mask_list[i]
        image = Image.open(img_pth).convert('RGB')
        image = image.resize((512, 512), Image.BILINEAR)
        mask = Image.open(mask_pth).convert('L')

        image = to_tensor(image).unsqueeze(0)
        gt_mask = torch.from_numpy(np.array(mask)).long()
        print(image.shape)
        ### predict with model
        # pred_mask = model(image.to(DEVICE))     # predict
        pred_mask = model(image.cuda())     # predict
        pred_mask = pred_mask.data.max(1)[1].cpu().numpy()  # Matrix index  (1,19,h,w) => (1,h,w)
        
        image = image.squeeze(0).permute(1,2,0)     # (1,3,h,w) -> (h,w,3)
        pred_mask = pred_mask.squeeze(0)            # (1,h,w) -> (h,w)


        classes = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        one_hot_mask = one_hot_encode(pred_mask, 19)    # (h,w) -> (19, h, w)
        # print(one_hot_mask.shape)
        # print(one_hot_mask)
        
       
        TEST_ID_DIR = f'{test_dir}/Test-image-{i}'
        if not os.path.exists(TEST_ID_DIR):
            os.makedirs(TEST_ID_DIR)

        dict_path = {}    
        # save seperated predict masks 
        for j in range(19):
            if j == 0:
                mask = one_hot_mask[j,:,:] * 0
            else:
                mask = one_hot_mask[j,:,:] * 255
            cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
            dict_path[classes[j]] = f"{TEST_ID_DIR}/{classes[j]}.png"


        # generate color mask image compared with GT(60 samples)
        color_gt_mask = cmap[gt_mask]
        color_pr_mask = cmap[pred_mask]
        plt.figure(figsize=(13, 6))
        image = Image.open(img_pth).convert('RGB')      # we want the image without normalization for plotting
        image = image.resize((512, 512), Image.BILINEAR)
        img_list = [image, color_pr_mask, color_gt_mask]
        for n in range(3):
            plt.subplot(1, 3, n+1)
            plt.imshow(img_list[n])
        plt.savefig(f"{test_dir}/result_{i}.jpg")

        ### Reorder the 19 class order since we use a different order during preprocessing
        labels_celeb = ['background','skin','nose',
        'eye_g','l_eye','r_eye','l_brow',
        'r_brow','l_ear','r_ear','mouth',
        'u_lip','l_lip','hair','hat',
        'ear_r','neck_l','neck','cloth']

        right_order_mask_path = {}
        for lab in labels_celeb:
            right_order_mask_path[lab] = dict_path[lab]
        
        # Csv file for submission
        # mask2csv(mask_paths=right_order_mask_path, image_id=i)
        # break
            
    #### unseen 2000 ####
    test_dir = f"result/unseen_test_result_{get_current_timestamp()}"
    unseen_dir = "/home/hsu/HD/CV/unseen"
    image_list = []

    # Iterate through the files in the directory
    for filename in os.listdir(unseen_dir):
        if filename.endswith('.jpg'):
            image_list.append(os.path.join(unseen_dir, filename))
    plist = []
    for file in image_list:
        name = file.split('_')[0]
        plist.append(int(name.split('/')[-1]))
    print(len(plist))
    combined_list = list(zip(image_list, plist))
    sorted_combined_list = sorted(combined_list, key=lambda x: x[1])
    image_list, _ = zip(*sorted_combined_list)
    print(image_list[:10])

    # for image in image_list:
    #     print(image)

    for i in tqdm(range(0, len(image_list))):
        ### The below operation is simmilar to the __getitem__() function
        img_pth = image_list[i]
        image = Image.open(img_pth).convert('RGB')
        image = image.resize((512, 512), Image.BILINEAR)
        image = to_tensor(image).unsqueeze(0)
        # mask = Image.open(mask_pth).convert('L')
        # gt_mask = torch.from_numpy(np.array(mask)).long()
        # print(image.shape)
        ### predict with model
        pred_mask = model(image.cuda())     # predict
        pred_mask = pred_mask.data.max(1)[1].cpu().numpy()  # Matrix index  (1,19,h,w) => (1,h,w)
        
        image = image.squeeze(0).permute(1,2,0)     # (1,3,h,w) -> (h,w,3)
        pred_mask = pred_mask.squeeze(0)            # (1,h,w) -> (h,w)

        classes = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        one_hot_mask = one_hot_encode(pred_mask, 19)    # (h,w) -> (19, h, w)
       
        TEST_ID_DIR = f'{test_dir}/Test-image-{i}'
        if not os.path.exists(TEST_ID_DIR):
            os.makedirs(TEST_ID_DIR)

        dict_path = {}    
        # save seperated predict masks 
        for j in range(19):
            mask = one_hot_mask[j,:,:] * 255
            cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
            dict_path[classes[j]] = f"{TEST_ID_DIR}/{classes[j]}.png"


        # generate color mask image compared with GT(60 samples)
        # if i % 100 == 0:
        #     color_pr_mask = cmap[pred_mask]
        #     plt.figure(figsize=(13, 6))
        #     image = Image.open(img_pth).convert('RGB')      # we want the image without normalization for plotting
        #     image = image.resize((512, 512), Image.BILINEAR)
        #     img_list = [image, color_pr_mask]
        #     for n in range(2):
        #         plt.subplot(1, 2, n+1)
        #         plt.imshow(img_list[n])
        #     plt.savefig(f"{OUTPUT_DIR_UNSEEN}/result_{i}.jpg")

        ### Reorder the 19 class order since we use a different order during preprocessing
        labels_celeb = ['background','skin','nose',
        'eye_g','l_eye','r_eye','l_brow',
        'r_brow','l_ear','r_ear','mouth',
        'u_lip','l_lip','hair','hat',
        'ear_r','neck_l','neck','cloth']

        right_order_mask_path = {}
        for lab in labels_celeb:
            right_order_mask_path[lab] = dict_path[lab]
        
        # Csv file for submission
        mask2csv(mask_paths=right_order_mask_path, image_id=i)
    

if __name__ == "__main__":
    test_fn()


















