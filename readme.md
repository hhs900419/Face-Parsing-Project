## Dependencies
- Albumentaions
- Segmentation-model-pytorch
- wandb

## Training
1. Prepare training data :
    -- download [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ)

2. Preprocessing
(to stack all the mask images of a person into one)
```Shell
python prepropess_data.py
```

3. Train
Feel free to change different settings (see [smp doc](https://smp.readthedocs.io/en/latest/) for model initialization)
Remember to set the configs in wandb.init() for better experiment recording
```Shell
python train.py
```

- model with best performance (Unet++ with effb3 encoder pretrained on imagenet)
- loss function: cross entropy
- evaluation metric: F1 score (dice coefficient)
- hyperparamters and global variables are in `configs.py`
- beaware of the arguments in the scheduler, unexpected result are produced if the scheduler is initialized unappropriately 

## Inference
```Shell
python test.py
```
- put the correct path for the dataset in `config.py`
- put the payh of the weight in `config.py` and remember to initialize the correct model in test.py
- there are three parts in `test.py`, if you want to generate the submisssion csv file for CelebAMask-HQ, comment out the code for evaluating unseen dataset, if you want to inference on unseen dataset, comment out the CelebAMask-HQ part
- 60 comparison results are generated for better visualization of the model performance
- a csv file will be generated for Kaggle submission

## Notes
- model weight is too large, can't push to gihub
- [segmentation model pytorch](https://github.com/qubvel/segmentation_models.pytorch/tree/master) library might be useful for building different model architecture and applying pretrained weight.
- [this reference](https://github.com/hukenovs/easyportrait) given by TAs shows lots of performance results by using different model architecture
- [this library](https://github.com/open-mmlab/mmsegmentation) includes more architectures, but seems quite difficilt to use
- [10 samples from Unseen dataset](https://drive.google.com/drive/folders/1jbOs1aBDN3myl6WX47Qy8nUqp9svA8-j)
- [FaceSynthetics](https://github.com/microsoft/FaceSynthetics) dataset (optional)


