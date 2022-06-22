from IPM import IPM
import numpy as np
import cv2
import torch
from fastseg import MobileV3Small
import albumentations as album
from albumentations.pytorch import ToTensorV2

IMG_PATH = 'C:\\carla-simulator\\val\\Town04_Clear_Noon_09_09_2020_14_57_22_frame_84_validation_set.png'
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 1024

loader = album.Compose([album.Resize(IMAGE_HEIGHT,IMAGE_WIDTH),ToTensorV2()])

class fastai():
    def __init__(self, cam_geom=IPM(), model_path='models/fastai_model.pth'):
        self.cg = cam_geom
        self.cut_v, self.grid = self.cg.precompute_grid()
        if torch.cuda.is_available():
            print("GPU found!!!")
            self.device = "cuda"
            self.model = torch.load(model_path).to(self.device)
        else:
            self.model = torch.load(model_path, map_location=torch.device("cpu"))
            self.device = "cpu"
        self.model.eval()

    def read_imagefile_to_array(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image   

    def detect_from_file(self, filename):
        img_array = self.read_imagefile_to_array(filename)
        return self.detect(img_array)

    def _predict(self, img):
        with torch.no_grad():
            image_tensor = img.transpose(2,0,1).astype('float32')/255
            x_tensor = torch.from_numpy(image_tensor).to(self.device).unsqueeze(0)
            model_output = torch.softmax(self.model.forward(x_tensor), dim=1).cpu().numpy()
            #preds = torch.argmax(model_output,dim=1,keepdim=True).float().round()
        return model_output

    def detect(self, img_array):
        model_output = self._predict(img_array)
        background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:] 
        return background, left, right

    def fit_poly(self, probs):
        probs_flat = np.ravel(probs[self.cut_v:, :])
        mask = probs_flat > 0.3
        if mask.sum() > 0:
            coeffs = np.polyfit(self.grid[:,0][mask], self.grid[:,1][mask], deg=3, w=probs_flat[mask])
        else:
            coeffs = np.array([0.,0.,0.,0.])
        return np.poly1d(coeffs)

    def __call__(self, image):
        if isinstance(image, str):
            image = self.read_imagefile_to_array(image)
        left_poly, right_poly, _, _ = self.get_fit_and_probs(image)
        return left_poly, right_poly

    def get_fit_and_probs(self, img):
        _, left, right = self.detect(img)
        left_poly = self.fit_poly(left)
        right_poly = self.fit_poly(right)
        return left_poly, right_poly, left, right
    
def main():
    ld = fastai()
    image = cv2.imread(IMG_PATH)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = np.array(image,dtype=np.float32)
    #aug = loader(image=image)
    #image = aug["image"]    
    #image = image.unsqueeze(0)
    #image = image.cuda()
    #print(image.size())
    pred = ld._predict(image)
    pred=pred.squeeze(0).squeeze(0).cpu().numpy()
    cv2.imshow("",np.array(100*pred,dtype=np.uint8))
    cv2.waitKey()
    
if __name__=="__main__":
    main()