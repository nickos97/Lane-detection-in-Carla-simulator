import torch 
import utilities.BlobData as BlobData
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as album
from torchvision.transforms import functional as F
import os
#from dotenv import load_dotenv

TRAIN_IMG_DIR = 'train'
TRAIN_LABEL_DIR = 'train_label'
VAL_IMG_DIR = 'val'
VAL_LABEL_DIR = 'val_label'
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 32
def DataLoaders(batch_size,
                train_transform,
                val_transform,
                train_path=TRAIN_IMG_DIR,
                train_label_path=TRAIN_LABEL_DIR,
                val_path=VAL_IMG_DIR,
                val_label_path=VAL_LABEL_DIR,
                num_workers=0,
                pin_memory=True
                ):
    
    train_data = BlobData.LanesDataset(train_path,train_label_path,transforms=train_transform)
    
    train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    val_data = BlobData.LanesDataset(val_path,val_label_path,transforms=val_transform)
    
    val_loader = DataLoader(val_data,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle = False)
    
    return (train_loader,val_loader)

def test():
    val_transform = album.Compose([album.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),ToTensorV2()])
    _,vdl = DataLoaders(16,None,val_transform)
    for data in vdl:
        x,y=data
        
if __name__=='__main__':
    test()
