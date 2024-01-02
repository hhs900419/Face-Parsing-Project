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
from retinaface import RetinaFace


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
    test_indices = test_indices[:10]


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
    exp_setting = "crop"
    
    if configs.debug:
        MODEL_WEIGHT = 'model_debug.pth'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ## make sure the setting is same as the model in train.py
    # ENCODER = 'efficientnet-b3'
    ENCODER = 'resnet50'
    # ENCODER = 'resnet101'
    # ENCODER = 'resnext101_32x8d'
    ENCODER_WEIGHTS = 'imagenet'
    # model = smp.FPN(
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
#############################################################################
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
        
##################################################################################################
    def scale_box(box,scale):
            dx = box[2]-box[0]
            dy = box[3]-box[1]
            newbox = [int(box[0]-dx*(scale/2)),int(box[1]-dy*(scale/2)),int(box[2]+dx*(scale/2)),int(box[3]+dy*(scale/2))]
            return newbox
    def bbox_crop(img, bbox ,rate_w):
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        center_x = (box[2]+box[0]) // 2
        center_y = (box[3]+box[1]) // 2
        img_size = img.shape[0]
        new_width = width * (1 + rate_w)
        new_height =  img_size 

        tl_corner_x = center_x - new_width // 2
        br_corner_x = center_x + new_width // 2
        if br_corner_x > img_size:
            br_corner_x = img_size
        if tl_corner_x < 0:
            tl_corner_x = 0
        print(tl_corner_x)
        print(br_corner_x)

        new_img = img.copy()
        new_img[:, :int(tl_corner_x), :] = 0
        new_img[:, int(br_corner_x):, :] = 0

        return new_img
    #### 10 unseen dataset eval #######
    # unseen_dir = "/home/hsu/HD/CV/unseen_10samples"
    # # unseen_dir = "/home/hsu/HD/CV/unseen"
    # image_list = []
    # mask_list = []

    # # Iterate through the files in the directory
    # for filename in os.listdir(unseen_dir):
    #     if filename.endswith('.jpg'):
    #         # Append to the image list if it's a JPG file
    #         image_list.append(os.path.join(unseen_dir, filename))
    #     elif filename.endswith('.png'):
    #         # Append to the mask list if it's a PNG file
    #         mask_list.append(os.path.join(unseen_dir, filename))
    # mask_list = sorted(mask_list)
    # image_list = sorted(image_list)

    # # for image in image_list:
    # #     print(image)

    # # for mask in mask_list:
    # #     print(mask)
    # test_dir = f"result/unseen_10_{exp_setting}_test_result_{get_current_timestamp()}"
    # metrics = SegMetric(n_classes=11)
    # for i in tqdm(range(0, len(image_list))):
    #     ### The below operation is simmilar to the __getitem__() function
    #     img_pth = image_list[i]
    #     mask_pth = mask_list[i]
    #     image = Image.open(img_pth).convert('RGB')
    #     image = image.resize((512, 512), Image.BILINEAR)
    #     mask = Image.open(mask_pth).convert('L')
    #     mask = mask.resize((512, 512), Image.BILINEAR)

    #     image = np.array(image)
    #     resp = RetinaFace.detect_faces(image)
    #     box = resp['face_1']['facial_area']
    #     image = bbox_crop(image, box, 0.8)

    #     image = Image.fromarray(image)

    #     image = to_tensor(image).unsqueeze(0)
    #     gt_mask = torch.from_numpy(np.array(mask)).long()
    #     # print(image.shape)
        
    #     ### predict with model
    #     pred_mask = model(image.cuda())     # predict
        
    #     copy_gt = gt_mask.clone().unsqueeze(0)
    #     rearr_pred_mask = pred_mask.clone()
    #     rearr_pred_mask = rearr_pred_mask[:, [0, 1, 2, 3, 4, 5, 10, 12, 11, 13, 17, 8, 7, 9, 14, 15, 16, 10, 18], :, :]
    #     rearr_pred_mask = rearr_pred_mask[:, :11, :, :]
    #     rearr_pred_mask = rearr_pred_mask.data.max(1)[1].cpu().numpy()  # Matrix index  (1,19,h,w) => (1,h,w)

    #     pred_mask = pred_mask.data.max(1)[1].cpu().numpy()  # Matrix index  (1,19,h,w) => (1,h,w)
    #     gt_mask = gt_mask.cpu().numpy()
    #     copy_gt = copy_gt.cpu().numpy()
    #     metrics.update(copy_gt, rearr_pred_mask)

    #     image = image.squeeze(0).permute(1,2,0)     # (1,3,h,w) -> (h,w,3)
    #     pred_mask = pred_mask.squeeze(0)            # (1,h,w) -> (h,w)


    #     classes = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
    #             'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    #     one_hot_mask = one_hot_encode(pred_mask, 19)    # (h,w) -> (19, h, w)
    #     # print(one_hot_mask.shape)
    #     # print(one_hot_mask)
       
    #     TEST_ID_DIR = f'{test_dir}/Test-image-{i}'
    #     if not os.path.exists(TEST_ID_DIR):
    #         os.makedirs(TEST_ID_DIR)


        
        
    #     dict_path = {}    
    #     # save seperated predict masks 
    #     for j in range(19):
    #         mask = one_hot_mask[j,:,:] * 255
    #         if j==1:
    #             box = scale_box(box, 0.15)
    #             corner1 = [int(box[0]) ,int(box[1])]
    #             corner2 = [int(box[2]), int(box[3])]
    #             mask[:, :corner1[0]] = 0
    #             mask[:, corner2[0]:] = 0
    #             mask[corner2[1]:, :] = 0
    #             mask[:corner1[1], :] = 0
    #         cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
    #         dict_path[classes[j]] = f"{TEST_ID_DIR}/{classes[j]}.png"


    #     # generate color mask image compared with GT(60 samples)
    #     color_gt_mask = cmap[gt_mask]
    #     color_pr_mask = cmap[pred_mask]
    #     plt.figure(figsize=(13, 6))
    #     image = Image.open(img_pth).convert('RGB')      # we want the image without normalization for plotting
    #     image = image.resize((512, 512), Image.BILINEAR)
    #     img_list = [image, color_pr_mask, color_gt_mask]
    #     for n in range(3):
    #         plt.subplot(1, 3, n+1)
    #         plt.imshow(img_list[n])
    #     plt.savefig(f"{test_dir}/result_{i}.jpg")

    #     ### Reorder the 19 class order since we use a different order during preprocessing
    #     labels_celeb = ['background','skin','nose',
    #     'eye_g','l_eye','r_eye','l_brow',
    #     'r_brow','l_ear','r_ear','mouth',
    #     'u_lip','l_lip','hair','hat',
    #     'ear_r','neck_l','neck','cloth']

    #     right_order_mask_path = {}
    #     for lab in labels_celeb:
    #         right_order_mask_path[lab] = dict_path[lab]
            
    # metric_score = metrics.get_scores()[0]
    # for k, v in metric_score.items():
    #     print(k, v)
        
    #     # Csv file for submission
    #     mask2csv(mask_paths=right_order_mask_path, image_id=i)
    #     break
            
##############################################################################################

    #### unseen 2000 ####
    test_dir = f"result/unseen_{exp_setting}_test_result_{get_current_timestamp()}"
    unseen_dir = "/home/hsu/HD/CV/unseen"
    image_list = []

    # Iterate through the files in the directory
    for filename in os.listdir(unseen_dir):
        if filename.endswith('.jpg'):
            image_list.append(os.path.join(unseen_dir, filename))
    plist = []
    for file in image_list:
        name = file.split('_')[0]
        suffix = int(file.split('_')[1].split(".")[0])
        plist.append(int(name.split('/')[-1]) + suffix)
    # print(len(plist))
    combined_list = list(zip(image_list, plist))
    sorted_combined_list = sorted(combined_list, key=lambda x: x[1])
    image_list, _ = zip(*sorted_combined_list)
    # print(image_list)
    # return

    # print(image_list[:10])

    # for image in image_list:
    #     print(image)

    for i in tqdm(range(0, len(image_list))):
        ### The below operation is simmilar to the __getitem__() function
        img_pth = image_list[i]
        image = Image.open(img_pth).convert('RGB')
        image = image.resize((512, 512), Image.BILINEAR)
        img_np = np.array(image)
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
        sklabel_dict = {'eye_g': 6}
        eyeslabel_dict = {'l_eye': 4, 'r_eye': 5}
        bg_mask = one_hot_mask[0,:,:] * 255
        sk_mask = one_hot_mask[1,:,:] * 255
        resp = RetinaFace.detect_faces(np.array(img_np))

        #determine the center face
        face = "face_1"
        if len(resp) > 1:
            point1 = np.array([resp["face_1"]["landmarks"]["nose"][0],resp["face_1"]["landmarks"]["nose"][1]])
            point2 = np.array([256,256])
            point3 = np.array([resp["face_2"]["landmarks"]["nose"][0],resp["face_2"]["landmarks"]["nose"][1]])
            point4 = np.array([256,256])
            euclidean_distance1 = np.linalg.norm(point2 - point1)
            euclidean_distance2 = np.linalg.norm(point4 - point3)
            if euclidean_distance1 > euclidean_distance2:
                face = "face_2"



        for j in range(2, 19, ):
            if j==17:
                # mask = one_hot_mask[j,:,:] * 255
                box = resp[face]['facial_area']
                # print(box)
                box = scale_box(box, 0.3)
                # print(box)
                corner1 = [int(box[0]) ,int(box[1])]
                corner2 = [int(box[2]), int(box[3])]
                mask = one_hot_mask[j,:,:] * 255
                mask[:, :corner1[0]] = 0
                mask[:, corner2[0]:] = 0
                mask[corner2[1]:, :] = 0
                mask[:corner1[1], :] = 0
                cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
                dict_path[classes[j]] = f"{TEST_ID_DIR}/{classes[j]}.png"
                continue
            elif j in sklabel_dict.values():
                mask = one_hot_mask[j,:,:] * 255
                if np.sum(mask==255)>0:
                    # add eye_g into skin
                    sklabel_mask = one_hot_mask[j, :, :] * 255
                    sklabel_mask_indices = np.where(sklabel_mask == 255)
                    sk_mask[sklabel_mask_indices] = sklabel_mask[sklabel_mask_indices]
                    # face detection and draw eyes
                    l_eye = np.array(resp[face]["landmarks"]["left_eye"])
                    r_eye = np.array(resp[face]["landmarks"]["right_eye"])
                    l_eye = l_eye.astype(np.int64)
                    r_eye = r_eye.astype(np.int64)
                    sk_mask[l_eye[1]-5 : l_eye[1]+5, l_eye[0]-10 : l_eye[0]+10] = 0
                    sk_mask[r_eye[1]-5 : r_eye[1]+5, r_eye[0]-10 : r_eye[0]+10] = 0

                    leye_mask = one_hot_mask[4, :, :] * 255
                    reye_mask = one_hot_mask[5, :, :] * 255
                    leye_mask[l_eye[1]-5 : l_eye[1]+5, l_eye[0]-10 : l_eye[0]+10] = 255
                    reye_mask[r_eye[1]-5 : r_eye[1]+5, r_eye[0]-10 : r_eye[0]+10] = 255

                    cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", sklabel_mask)
                    cv2.imwrite(f"{TEST_ID_DIR}/{classes[4]}.png", leye_mask)
                    cv2.imwrite(f"{TEST_ID_DIR}/{classes[5]}.png", reye_mask)
                else:   
                    cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
            elif j == 4:
                mask = one_hot_mask[j,:,:] * 255
                if np.sum(mask==255)==0:
                    l_eye = np.array(resp[face]["landmarks"]["left_eye"])               
                    l_eye = l_eye.astype(np.int64)
                    leye_mask = one_hot_mask[4, :, :] * 255                
                    leye_mask[l_eye[1]-2 : l_eye[1]+2, l_eye[0]-10 : l_eye[0]+10] = 255               
                    cv2.imwrite(f"{TEST_ID_DIR}/{classes[4]}.png", leye_mask)                
                else:   
                    box = resp[face]['facial_area']
                    # print(box)
                    box = scale_box(box, -0.1)
                    # print(box)
                    corner1 = [int(box[0]) ,int(box[1])]
                    corner2 = [int(box[2]), int(box[3])]
                    mask = one_hot_mask[j,:,:] * 255
                    mask[:, :corner1[0]] = 0
                    mask[:, corner2[0]:] = 0
                    mask[corner2[1]:, :] = 0
                    mask[:corner1[1], :] = 0
                    cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
            elif j == 5:
                mask = one_hot_mask[j,:,:] * 255
                if np.sum(mask==255)==0:                 
                    r_eye = np.array(resp[face]["landmarks"]["right_eye"])                   
                    r_eye = r_eye.astype(np.int64)                  
                    reye_mask = one_hot_mask[5, :, :] * 255                   
                    reye_mask[r_eye[1]-2 : r_eye[1]+2, r_eye[0]-10 : r_eye[0]+10] = 255
                    
                    cv2.imwrite(f"{TEST_ID_DIR}/{classes[5]}.png", reye_mask)
                else:   
                    box = resp[face]['facial_area']
                    # print(box)
                    box = scale_box(box, -0.1)
                    # print(box)
                    corner1 = [int(box[0]) ,int(box[1])]
                    corner2 = [int(box[2]), int(box[3])]
                    mask = one_hot_mask[j,:,:] * 255
                    mask[:, :corner1[0]] = 0
                    mask[:, corner2[0]:] = 0
                    mask[corner2[1]:, :] = 0
                    mask[:corner1[1], :] = 0
                    cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
            else:
                # print(i)
                box = resp[face]['facial_area']
                # print(box)
                box = scale_box(box, -0.1)
                # print(box)
                corner1 = [int(box[0]) ,int(box[1])]
                corner2 = [int(box[2]), int(box[3])]
                mask = one_hot_mask[j,:,:] * 255
                mask[:, :corner1[0]] = 0
                mask[:, corner2[0]:] = 0
                mask[corner2[1]:, :] = 0
                mask[:corner1[1], :] = 0
                cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
            
            # dict_path[classes[j]] = f"{TEST_ID_DIR}/{classes[j]}.png"

            
        cv2.imwrite(f"{TEST_ID_DIR}/{classes[0]}.png", bg_mask)
        cv2.imwrite(f"{TEST_ID_DIR}/{classes[1]}.png", sk_mask)
        # dict_path[classes[0]] = f"{TEST_ID_DIR}/{classes[0]}.png"
        # dict_path[classes[1]] = f"{TEST_ID_DIR}/{classes[1]}.png"


        for j in range(19):   #skin
            mask = one_hot_mask[j,:,:] * 255
            if j==1:
                # resp = RetinaFace.detect_faces(img_np)
                box = resp[face]['facial_area']
                box = scale_box(box, 0.1)
                corner1 = [int(box[0]) ,int(box[1])]
                corner2 = [int(box[2]), int(box[3])]
                mask[:, :corner1[0]] = 0
                mask[:, corner2[0]:] = 0
                mask[corner2[1]:, :] = 0
                mask[:corner1[1], :] = 0
                cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
            dict_path[classes[j]] = f"{TEST_ID_DIR}/{classes[j]}.png"
        # print(dict_path)


        # generate color mask image compared with GT(60 samples)
        if i % 20 == 0:
            color_pr_mask = cmap[pred_mask]
            plt.figure(figsize=(13, 6))
            image = Image.open(img_pth).convert('RGB')      # we want the image without normalization for plotting
            image = image.resize((512, 512), Image.BILINEAR)
            img_list = [image, color_pr_mask]
            for n in range(2):
                plt.subplot(1, 2, n+1)
                plt.imshow(img_list[n])
            plt.savefig(f"visualize/result_{i}.jpg")

        ### Reorder the 19 class order since we use a different order during preprocessing
        labels_celeb = ['background','skin','nose',
        'eye_g','l_eye','r_eye','l_brow',
        'r_brow','l_ear','r_ear','mouth',
        'u_lip','l_lip','hair','hat',
        'ear_r','neck_l','neck','cloth']

        right_order_mask_path = {}
        for lab in labels_celeb:
            right_order_mask_path[lab] = dict_path[lab]
        # print(right_order_mask_path)
        # Csv file for submission
        mask2csv(mask_paths=right_order_mask_path, image_id=i)
    

if __name__ == "__main__":
    test_fn()

# # save seperated predict masks
#         # bglabel_dict = {'eye_g': 6, 'ear_r': 9, 'neck': 14, 'neck_l': 15, 'cloth': 16, 'hat': 18}
#         # sklabel_dict = {'l_ear': 7, 'r_ear': 8}
#         # bg_mask = one_hot_mask[0,:,:] * 255
#         # sk_mask = one_hot_mask[1,:,:] * 255
#         # for j in range(2, 19, ):
#         #     if j in bglabel_dict.values():
#         #         bglabel_mask = one_hot_mask[j, :, :] * 255
#         #         bglabel_mask_indices = np.where(bglabel_mask == 255)
#         #         bg_mask[bglabel_mask_indices] = bglabel_mask[bglabel_mask_indices]
#         #         bglabel_mask = 0
#         #         cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", bglabel_mask)
            
#         #     elif j in sklabel_dict.values():
#         #         sklabel_mask = one_hot_mask[j, :, :] * 255
#         #         sklabel_mask_indices = np.where(sklabel_mask == 255)
#         #         sk_mask[sklabel_mask_indices] = sklabel_mask[sklabel_mask_indices]
#         #         sklabel_mask = 0
#         #         cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", sklabel_mask)
            
#         #     else:
#         #         mask = one_hot_mask[j,:,:] * 255
#         #         cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
            
#         #     dict_path[classes[j]] = f"{TEST_ID_DIR}/{classes[j]}.png"
        
#         # cv2.imwrite(f"{TEST_ID_DIR}/{classes[0]}.png", bg_mask)
#         # cv2.imwrite(f"{TEST_ID_DIR}/{classes[1]}.png", sk_mask)
#         # dict_path[classes[0]] = f"{TEST_ID_DIR}/{classes[0]}.png"
#         # dict_path[classes[1]] = f"{TEST_ID_DIR}/{classes[1]}.png"


#         # resp = RetinaFace.detect_faces(images[index])
#         # box = resp['face_1']['facial_area']
#         # corner1 = [int(box[0]) ,int(box[1])]
#         # corner2 = [int(box[2]), int(box[3])]
#         # mask[:, :corner1[0]] = 0
#         # mask[:, corner2[0]:] = 0
#         # mask[corner2[1]:, :] = 0
#         # mask[:corner1[1], :] = 0

#         for j in range(19):
#             mask = one_hot_mask[j,:,:] * 255

#             # if j==1:
#             #     image = Image.open(img_pth).convert('RGB')
#             #     image = image.resize((512, 512), Image.BILINEAR)
#             #     image = np.array(image)
#             #     resp = RetinaFace.detect_faces(image)
#             #     box = resp['face_1']['facial_area']
#             #     corner1 = [int(box[0]) ,int(box[1])]
#             #     corner2 = [int(box[2]), int(box[3])]
#             #     mask[:, :corner1[0]] = 0
#             #     mask[:, corner2[0]:] = 0
#             #     mask[corner2[1]:, :] = 0
#             #     mask[:corner1[1], :] = 0

#             cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
#             dict_path[classes[j]] = f"{TEST_ID_DIR}/{classes[j]}.png"
















