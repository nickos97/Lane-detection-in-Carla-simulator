import torch
from torch import cuda, device
from torch import cuda, sigmoid
from tqdm import tqdm
import torch.nn as nn
from training.UNet import UNet2
from utilities.utils import ( DataLoaders)
from albumentations.pytorch import ToTensorV2
import albumentations as album
import numpy as np
from torchvision.utils import save_image
from torchsummary import summary
from datetime import datetime
import time
import cv2
import matplotlib.pyplot as plt

L_R = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=8
EPOCHS = 30
NUM_WORKERS=0
IMAGE_HEIGHT = 210
IMAGE_WIDTH = 420
PIN_MEMORY = True
M_PATH = 'model2.pth'
STARTED = datetime.now()


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

train_transform = album.Compose([album.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),ToTensorV2()])
            
val_transform = album.Compose([album.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),ToTensorV2()])
                               

net = UNet2(in_channels=3, out_channels=3).to(DEVICE)
#summary(net,(3,IMAGE_HEIGHT,IMAGE_WIDTH))

class Train_model():
    def __init__(self):
        self.model = self
        self.criterion =  nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net.parameters(),lr=L_R)
        self.scaler = torch.cuda.amp.GradScaler()
        self.val_loader= None
        self.start_time = None
        self.first_feat,self.last_feat = net.filters() 
        self.train_loss = []
        self.val_loss = []
        self.epochs = []
        
        
    def fit(self):
        best_vloss=np.inf
        train_loader,self.val_loader = DataLoaders(BATCH_SIZE,train_transform,val_transform)
        
        counter = 0
        self.start_time = datetime.now()
        for epoch in range(1,EPOCHS+1):
            counter += 1
            average_loss=0
            net.train()
            for batch_idx,data in enumerate(train_loader):
                
                X,y = data
                
                if torch.cuda.is_available():
                    X = X.to(device=DEVICE)
                    y=y.long().to(device=DEVICE)
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=False):
                    out = net(X)
                    loss = self.criterion(out,y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                average_loss += loss.item()
                
                if batch_idx%50==0:
                    print(f"Epoch: {epoch} \t Batch: {batch_idx} \t Loss: {average_loss/(batch_idx+1)}")
            self.train_loss.append(average_loss/(batch_idx+1))       
            net.eval()
            average_vloss=0
            vloss=0
            for vbatch_idx,vdata in enumerate(self.val_loader):
                
                vX,vy = vdata
                if torch.cuda.is_available():
                    vX=vX.to(device=DEVICE)
                    vy=vy.long().to(device=DEVICE)
                vpreds = net(vX)
                vloss=self.criterion(vpreds,vy)
                average_vloss+=vloss.item()
                
            average_vloss = average_vloss/(vbatch_idx+1)
            self.val_loss.append(average_vloss)
            print(f"Train loss: {average_loss/(batch_idx+1)} \t Validation Loss: {average_vloss}")    
            
            if best_vloss>average_vloss:
                counter = 0
                best_vloss=average_vloss
                model_path = f"models\model_{timestamp}_{self.first_feat}_{self.last_feat}_{epoch}"
                torch.save(net.state_dict(),model_path) 
            self.epochs.append(epoch)  
            if counter>5:
                break
            
        print(f"Training time: {datetime.now()-self.start_time}")
                 
    def predict(self):
        test_loader = self.val_loader
        self.sum=0
        self.samples=0
        num_correct = 0
        num_pixels = 0
        with torch.no_grad():
            net.eval()
            for batch_idx,data in enumerate(test_loader):
                X,y = data
                if torch.cuda.is_available():
                    X = X.to(device=DEVICE)
                    y=y.unsqueeze(1).to(device=DEVICE)
                out = net(X)
                soft = nn.Softmax(dim=1)
                preds = soft(out)
                preds = torch.argmax(preds,dim=1,keepdim=True).float()
                save_image(preds,f"saved_images/pred_{batch_idx}.png")   
                num_correct += (preds == y).sum()
                num_pixels += len(X)*IMAGE_HEIGHT*IMAGE_WIDTH
                
            print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}%")
    def plot(self):
        
        plt.plot(self.epochs,self.train_loss,label="Training Loss")
        plt.plot(self.epochs,self.val_loss,label="Validation Loss")
        plt.title("Model Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
                

def main():     
    model = Train_model()
    model.fit()
    model.predict()
    model.plot()
    
if __name__=="__main__":
    main()