import torch
import torch.nn as nn
from tqdm import tqdm
from math import ceil
from torchvision.transforms import functional as TF
from torchsummary import summary

def DoubleConv(in_channels,out_channels):
    conv = nn.Sequential(
        
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1, bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1, bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    
    return conv

def fit_tensor(tensor,target_tensor):
    
    tensor_height = tensor.size()[2]
    target_height = target_tensor.size()[2]

    delta_height = ceil(abs(tensor_height - target_height)/2)

    tensor_width = tensor.size()[3]
    target_width = target_tensor.size()[3]
    
    delta_width = ceil(abs(tensor_width - target_width)//2)
    if (tensor_height-target_height)%2:
        return tensor[:,:,delta_height-1:tensor_height-delta_height,delta_width:tensor_width-delta_width-1]
    else:
        return tensor[:,:,delta_height:tensor_height-delta_height,delta_width:tensor_width-delta_width]
    

    

class UNet2(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,features_list = [32,64,128,256,512]):
    
        super(UNet2,self).__init__()
        self.conv_downs = nn.ModuleList() #double convolutions in down operation
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2) #max poolings in down operation
        self.conv_ups = nn.ModuleList() #double convolutions in up operation
        self.conv_trans = nn.ModuleList() #transposed convolutions in up operation
        self.final_conv = nn.Conv2d(in_channels = features_list[0], out_channels=out_channels, kernel_size=1,stride=1) #final layer
        self.feats = [features_list[0],features_list[-1]]
        self.features_length = len(features_list)
        
        #Convolution in down operation list
        for features in features_list:
            self.conv_downs.append(DoubleConv(in_channels,features))
            in_channels = features # 3-->64-->128-->256-->512-->1024
        
        #Convolution in up operation list
        for features in reversed(features_list[:self.features_length-1]):
            self.conv_ups.append(DoubleConv(features*2,features)) # 1024-->512-->256-->128-->64
        
        #Convolution Transposed in up operation list
        for i,features in enumerate(reversed(features_list)):
            if i<self.features_length-1:
                self.conv_trans.append(nn.ConvTranspose2d(features,features_list[::-1][i+1],kernel_size=2,stride=2))
    
    def filters(self):
        first_feat = self.feats[0]
        last_feat = self.feats[1]
        return first_feat,last_feat
            
    def forward(self,x):
        
        """
        ENCODER
        """
        connections = []
        pool_output=x
        for super_layer in range(len(self.conv_downs)):
            conv_output = self.conv_downs[super_layer](pool_output)
            #print(f"(super layer {super_layer+1}): double conv output: {conv_output.size()}")
            if super_layer==self.features_length-1:
                break
            connections.append(conv_output)
            pool_output = self.maxpool(conv_output)
            #print(f"(super_layer {super_layer+1}): maxpool ouput: {pool_output.size()}")
           
        """
        DECODER
        """
      
        upconv_output = conv_output
        for i in range(len(self.conv_ups)):
            trans_out = self.conv_trans[i](upconv_output)
            tensor_to_concat = TF.resize(trans_out,size=connections[::-1][i].shape[2:])
            #upconv_output = self.conv_ups[i](torch.cat([trans_out,tensor_to_concat],1))
            upconv_output = self.conv_ups[i](torch.cat([tensor_to_concat,connections[::-1][i]],1))
            
            
        final_layer = self.final_conv(upconv_output)
        
        return final_layer      
        
def test():
    x=torch.randn((1,3,210,420))
    model = UNet2()
    y = torch.randn((1,1,210,420))
    pred = nn.CrossEntropyLoss()(model(x),y.long())

    print(pred)             
       
    
    
if __name__=="__main__":
    test()
else:
    print("Initializing parameters...")