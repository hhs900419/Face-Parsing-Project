import torch
import os
from utils import *

class Configs():
    def __init__(self):
        # os.environ['CUDA_VISIBLE_DEVICES']='1'
        
        self.seed = 1187
        self.root_dir = "/home/hsu/HD/CV/CelebAMask-HQ"
        self.val_size = 0.15
        self.batch_size = 6
        self.n_workers = 4
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_gpu_id = "1"    # "0", "0,1"
        self.epochs = 30
        self.lr = 1e-4
        self.model_path = '../model_weight/'
        self.model_weight = f'model_dlv3p_r50_noaug_schlrP_{self.epochs}eps_{get_current_timestamp()}.pth'
        self.load_model_weight = 'model_dlv3p_r50_noaug_schlrP_30eps_2023_12_19_210838.pth'
        # self.model_weight = f'model_debug.pth'
        self.cmp_result_dir = './visualize'
        self.debug = False
        # self.debug = True
        

        
