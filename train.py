import os.path as osp
import os
import cv2
import numpy as np
from augmentation import *
from face_dataset import *
from models.unet import *
from models.attention_unet import *
from models.deeplabv3plus_xception import *
from criterion import *
from trainer import *
from configs import *
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.backends import cudnn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
import gc
import wandb
import segmentation_models_pytorch as smp
from albumentation import *
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def train():
    configs = Configs()
    SEED = configs.seed
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.cuda.manual_seed(SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    
    ### 1. Train/Val/Test Split ### (this section is useless since i use testset as validation set directly)
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
    valid_indices = train_indices
    if configs.debug:
        train_indices = train_indices[:100]         
        train_indices, valid_indices = train_test_split(train_indices, test_size=VAL_SIZE, random_state=SEED)
    print(len(train_indices))
    if configs.debug:
        print(len(valid_indices)) 
    print(len(test_indices))
    
    ### 2. augmentations ### (Can either use Albumentations(albumentation.py) or functions in augmentation.py)
    train_tranform = Compose({
        # RandomCrop(448),
        # RandomHorizontallyFlip(p=0.5),
        # AdjustBrightness(bf=0.1),
        # AdjustContrast(cf=0.1),
        # AdjustHue(hue=0.1),
        # AdjustSaturation(saturation=0.1)
    })
    
    ### 3. initialize wandb project as experiment configurations for better model tracking
    wandb.init(
        project="Face Parsing",
        name=f"experiment_{get_current_timestamp()}", 
        # Track hyperparameters and run metadata
        config={
        "model Architecture": "DLv3+",
        "encoder": "mobv2",
        "freeze encoder": False,
        "augmentation": False,
        "batch size": configs.batch_size,
        "learning_rate": configs.lr,
        "epochs": configs.epochs,
        "criterion": "Cross Entropy",
        "scheduler": "Reduce on Plateau",
        "model weight": configs.model_weight
        }
    )
    
    ### 4. Model Initialization 
    # (use smp library or you can put your model under the 'models/' folder and use it)
    
    ############# SMP library ##########
    # ENCODER = 'efficientnet-b3'
    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = configs.device
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=19, 
    )
    # freeze encoder weight(optional)   empirically, unfreezed weight gives better performance
    # model.encoder.eval()
    # for m in model.encoder.modules():
    #     m.requires_grad_ = False
    
    # model = Unet(3,19)
    
    # model = model.to(DEVICE)
    model = model.cuda()
    wandb.watch(model, log="all", log_freq=10)
    print("Model Initialized !")
    
    ### 5. Create CelebAMask-HQ dataset ### (use 6000 test imgs as validation set)
    trainset = CelebAMask_HQ_Dataset(root_dir=ROOT_DIR, 
                                sample_indices=train_indices,
                                mode='train', 
                                tr_transform=None)
                                # tr_transform=train_tranform)
                                # preprocessing = get_preprocessing(preprocessing_fn))
    validset = CelebAMask_HQ_Dataset(root_dir=ROOT_DIR, 
                                sample_indices=test_indices, 
                                mode = 'val')
                                # preprocessing=get_preprocessing(preprocessing_fn))
    if configs.debug:
        validset = CelebAMask_HQ_Dataset(root_dir=ROOT_DIR, 
                                    sample_indices=valid_indices, 
                                    mode = 'val')
                                    # preprocessing=get_preprocessing(preprocessing_fn))
    
    ### 6. dataloader ###
    BATCH_SIZE = configs.batch_size
    N_WORKERS = configs.n_workers

    train_loader = DataLoader(trainset,
                        batch_size = BATCH_SIZE,
                        shuffle = True,
                        num_workers = N_WORKERS,
                        pin_memory = True,
                        drop_last = True)

    valid_loader = DataLoader(validset,
                        batch_size = BATCH_SIZE,
                        shuffle = False,
                        num_workers = N_WORKERS, 
                        pin_memory = True,
                        drop_last = False)
    print(f"training data: {len(trainset)} and test data: {len(validset)} loaded succesfully ...")
    
    
    gc.collect()
    torch.cuda.empty_cache()    
    
    
    ### 7. hyper params ###
    EPOCHS = configs.epochs
    LR = configs.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2, min_lr=1e-6, verbose=True)  # goal: minimize val_loss/maximize miou
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.2, min_lr=1e-6, verbose=True)  # goal: minimize val_loss/maximize miou
    # tmax = len(train_loader) * EPOCHS
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=5e-6)  # goal: maximize miou
    
    # criterion = DiceLoss()
    criterion = nn.CrossEntropyLoss()
    SAVEPATH = configs.model_path
    SAVENAME = configs.model_weight
    
    ### 8. training ###
    Trainer( model=model, 
        trainloader=train_loader,
        validloader=valid_loader,
        epochs=EPOCHS,
        criterion=criterion, 
        optimizer=optimizer,
        scheduler=scheduler, 
        # scheduler=None, 
        device=DEVICE,
        savepath=SAVEPATH, 
        savename=SAVENAME).run()
    
if __name__ == "__main__":
    train()
    

    
    