from training.UNet import UNet2
from utilities.utils import ( DataLoaders)
from albumentations.pytorch import ToTensorV2
import albumentations as album
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dotenv import dotenv_values

DATA_DIR = dotenv_values(".env")

IMAGE_HEIGHT = 210
IMAGE_WIDTH = 420
IMG_PATH = DATA_DIR['IMAGE']
MODEL_PATH = 'models\model_20220313_182658_32_512_19'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=32
num_classes=3

val_transform = album.Compose([album.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),ToTensorV2()])

class evaluate():
    
    def __init__(self):
        
        _,self.val_loader = DataLoaders(BATCH_SIZE,None,val_transform)
        self.model = UNet2(in_channels=3, out_channels=3).to(device=DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        
    def overlap(self,pred_y,y):
        overlap_classes=[]
        for i in range(num_classes):
            overlap_classes.append(torch.sum((pred_y==y)*(pred_y==i)))
        return overlap_classes

    def total_pixels(self,pred_y,y):
        total_pixel_classes=[]
        print(torch.sum(pred_y == 2))
        for i in range(num_classes):
            total_pixel_classes.append(torch.sum(pred_y==i) + torch.sum(y==i))
        return total_pixel_classes
    
    def class_distribution(self,y):
        pixels_per_class = []
        for i in range(num_classes):
            pixels_per_class.append(torch.sum(y==i))
        return pixels_per_class

    def MIoU(self,pred_y,y):
        
        intersections=self.overlap(pred_y,y)
        total_px = self.total_pixels(pred_y,y)
        
        unions=[total_px[i] - intersections[i] for i in range(num_classes)]
    
        no_lanes_iou = intersections[0]/unions[0]
        left_lanes_iou = intersections[1]/unions[1]
        right_lanes_iou = intersections[2]/unions[2]
        
        mean_iou = (no_lanes_iou+left_lanes_iou+right_lanes_iou)/num_classes
        
        return mean_iou

    def dice_coefficient(self,pred_y,y):
            
        intersections=self.overlap(pred_y,y)
        total_px = self.total_pixels(pred_y,y)
        
        no_lanes_dice = 2*intersections[0]/total_px[0]
        left_lanes_dice = 2*intersections[1]/total_px[1]
        right_lanes_dice = 2*intersections[2]/total_px[2]
        
        mean_dice = (no_lanes_dice+left_lanes_dice+right_lanes_dice)/num_classes
        
        return mean_dice

        
    def evaluate(self):
        num_correct = 0
        num_pixels = 0
        sum_iou=0
        sum_dice=0
        backgd = 0
        left = 0
        right = 0
        
        with torch.no_grad():
            self.model.eval()
            for batch_idx,data in enumerate(self.val_loader):
                X,y = data
                if torch.cuda.is_available():
                    X = X.to(device=DEVICE)
                    y=y.unsqueeze(1).to(device=DEVICE)
                out = self.model(X)
                soft = nn.Softmax(dim=1)
                preds = soft(out)
                pred_y = torch.argmax(preds,dim=1,keepdim=True)
            
                num_correct += (pred_y == y).sum()
                 
                sum_iou+=self.MIoU(pred_y,y)
                sum_dice+=self.dice_coefficient(pred_y,y)
                pixels_per_class = self.class_distribution(y)
                backgd += pixels_per_class[0].cpu().numpy()
                left += pixels_per_class[1].cpu().numpy()
                right += pixels_per_class[2].cpu().numpy()
                
                
                
                
                num_pixels += len(X)*IMAGE_HEIGHT*IMAGE_WIDTH
                
            MMiou=sum_iou/len(self.val_loader)
            MMdice=sum_dice/len(self.val_loader)
            
            
            
            labels = ['background','left lane','right lane']
            px_class = [float("{:.2f}".format(100*backgd/num_pixels)),float("{:.2f}".format(100*left/num_pixels)),float("{:.2f}".format(100*right/num_pixels))]
            print(px_class)
            
            print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}%, IoU: {MMiou:.2f}% and dice score: {MMdice:.2f}%")
            print(f"{labels[0]}: {100*backgd/num_pixels:.2f}% {labels[1]}: {100*left/num_pixels:.2f}% {labels[2]}: {100*right/num_pixels:.2f}%")
            
            x = np.arange(len(labels))
            width = 0.35
            fig,ax = plt.subplots()
            rec = ax.bar(x,px_class,width)
            ax.set_ylabel('pixels')
            ax.set_xlabel('classes')
            ax.set_title('pixels per class (%)')
            ax.set_xticks(x,labels)
            
            ax.bar_label(rec)
            
            plt.show()
            

   
eval = evaluate()

eval.evaluate()