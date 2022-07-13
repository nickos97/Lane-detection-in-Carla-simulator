import os 
import cv2
from cv2 import COLOR_BGR2RGB
from cv2 import COLOR_BGR2GRAY
from cv2 import COLOR_GRAY2RGB
import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as album
from dotenv import dotenv_values

DATA_DIR = dotenv_values('.env')

#testing
IMAGES_DIR = DATA_DIR['TRAIN_LABEL']
LABELS_DIR = DATA_DIR['VAL_LABEL']
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 330

class LanesDataset(Dataset):
    def __init__(self,imagePath,maskPath,transforms):
        self.imagePath=imagePath
        self.maskPath=maskPath
        self.transforms=transforms
        self.images = os.listdir(imagePath)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.imagePath,self.images[idx])
        mask_path = os.path.join(self.maskPath,self.images[idx].replace('.png','_label.png'))
        image = cv2.imread(img_path)
        image = np.array(cv2.cvtColor(image,COLOR_BGR2RGB),dtype=np.float32)
            
        mask = cv2.imread(mask_path)
        mask = np.array(cv2.cvtColor(mask,COLOR_BGR2GRAY))
        
        if self.transforms != None:
            augmentations = self.transforms(mask=mask, image=image)
            mask = augmentations["mask"]
            #mask = mask.permute((2,0,1))
            image = augmentations["image"]
            
            
        #mask[mask==2.0] = 1.0 
        return (image,mask)

def test():
    val_transform = album.Compose([album.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),ToTensorV2()])
    train_transform = album.Compose([album.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH)])
    data = LanesDataset(IMAGES_DIR,LABELS_DIR,transforms=train_transform)
    #print(data.__getitem__(2)[1].size())
    image = data.__getitem__(2)[0]
    label = data.__getitem__(2)[1]
    print(image.shape)
    cv2.imshow("",label)
    cv2.waitKey()
    con = np.concatenate((image/255,label/2),axis=1)
    cv2.imshow("",con)
    cv2.waitKey()
   
if __name__=='__main__':
    test()