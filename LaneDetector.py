from IPM import IPM
import numpy as np
import cv2
import torch
from UNet import UNet2
from albumentations.pytorch import ToTensorV2
import albumentations as album
from matplotlib import pyplot as plt
from target_point import get_target_point
#from fastseg import MobileV3Small

DEVICE="cuda"
IMG_PATH = 'Town04_Clear_Noon_09_09_2020_14_57_22_frame_863.png'
IMAGE_HEIGHT = 210
IMAGE_WIDTH = 420

loader = album.Compose([album.Resize(IMAGE_HEIGHT,IMAGE_WIDTH),ToTensorV2()])

 
def get_trajectory_from_lane_detector(ld, image):
# get lane boundaries using the lane detector
    
    poly_left, poly_right, _, _,preds1 = ld.get_fit_and_probs(image)
    
    x = np.arange(-2,60,1.0)
    y = -0.5*(poly_left(x)+poly_right(x))
    x += 0.5
    traj = np.stack((x,y)).T
    yl = poly_left(x)
    yr = poly_right(x)
    ylr = poly_right(x) + poly_left(x)
    plt.plot(x, yl, label="yl(x)")
    plt.plot(x,-y, label="ym(x)")
    plt.plot(x, yr, label="yr(x)")
    plt.xlabel("x (m)") 
    plt.ylabel("y (m)")
    plt.legend()
    plt.axis("equal")
    plt.show()
    return traj

class Unet_Detector():
    def __init__(self, cam_geom=IPM(), model_path='models/model_20220313_182658_32_512_19'):
        self.cg = cam_geom
        self.grid = self.cg.precompute_grid()
        #self.cut_v = 0
        if torch.cuda.is_available():
             print("GPU found!!")
             self.model = UNet2(in_channels=3, out_channels=3).to(device=DEVICE)
             self.model.load_state_dict(torch.load(model_path))
       
        else:
            self.model = torch.load(model_path, map_location=torch.device("cpu"))
            self.device = "cpu"
        self.model.eval()
        
    def transform(self,image):
        aug = loader(image=image)
        image = aug["image"]    
        image = image.unsqueeze(0)
        image = image.cuda()
        return image

    def _predict(self,img):
        with torch.no_grad():
            img = self.transform(img)
            out = self.model(img)
            soft = torch.nn.Softmax(dim=1)
            out = soft(out)
            preds1 = torch.argmax(out,dim=1,keepdim=True).float().round()
            preds2=out.cpu().numpy()
            
        return preds2,preds1

    def detect(self, img_array):
        model_output,preds1 = self._predict(img_array)
        background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:] 
        #cv2.imshow("",right/2)
        #cv2.waitKey()
        return background,preds1, left, right

    def fit_poly(self, probs):
    
        probs_flat = np.ravel(probs)
        mask = probs_flat > 0.5
        
        if mask.sum() > 0:
            coeffs = np.polyfit(self.grid[:,0][mask], self.grid[:,1][mask], deg=3, w=probs_flat[mask])
        else:
            coeffs = np.array([0.,0.,0.,0.])
        return np.poly1d(coeffs)

    def get_fit_and_probs(self, img):
        _,preds1, left, right = self.detect(img)
        left_poly = self.fit_poly(left)
        right_poly = self.fit_poly(right)
        return left_poly, right_poly, left, right,preds1
    
def main():
    ld = Unet_Detector()
    image = cv2.imread(IMG_PATH)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = np.array(image,dtype=np.float32)
    traj = get_trajectory_from_lane_detector(ld,image)
    #print(get_trajectory_from_lane_detector(ld,image))
    fig, ax = plt.subplots()
    ax.plot(traj[:,0], traj[:,1], color="g")
    circle = plt.Circle((0, 0), 20, color="k", fill=False)
    ax.add_artist(circle)
    intersec = get_target_point(20, traj)
    print(intersec[0],intersec[1])
    if intersec is not None:
        plt.scatter([intersec[0]], [intersec[1]], color="r")
    plt.axis("equal")

    plt.show()
if __name__=="__main__":
    main()