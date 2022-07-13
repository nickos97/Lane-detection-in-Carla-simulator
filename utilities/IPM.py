import numpy as np
from dotenv import dotenv_values

FOV = 45
WIDTH = 420
HEIGHT = 210
CAM_HEIGHT = 1.3

def get_intrinsic_matrix(fov,image_width,image_height):
    fov_rad = fov * np.pi/180
    alpha = (image_width / 2.0) / np.tan(fov_rad /2.) #focal length: Au=Av=A
    Cu = image_width / 2.0 
    Cv = image_height /2.0
    #Cu,Cv: optical center of image plane
    return np.array([[alpha,0,Cu],
                     [0,alpha,Cv],
                     [0,0,1.0]])
    

class IPM(object):
    def __init__(self,height=CAM_HEIGHT,yaw=0,pitch=-5,roll=0, image_width = WIDTH,image_height = HEIGHT,fov=FOV):
        self.height = height
        self.pitch_deg = pitch
        self.roll_deg = roll
        self.yaw_deg = yaw
        self.image_width = image_width
        self.image_height = image_height
        self.field_of_view_deg = fov
        
        self.intrinsic_matrix = get_intrinsic_matrix(fov,image_width,image_height)
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
        cy,sy = np.cos(yaw),np.sin(yaw)
        cp,sp = np.cos(pitch),np.sin(pitch)
        cr,sr = np.cos(roll),np.sin(roll)
        
        rotation_road_to_cam = np.array([[cr*cy+sp*sr+sy, cr*sp*sy-cy*sr, -cp*sy],
                                            [cp*sr, cp*cr, sp],
                                            [cr*sy-cy*sp*sr, -cr*cy*sp -sr*sy, cp*cy]])
        self.rotation_cam_to_road = rotation_road_to_cam.T #R^(-1)=R.T
        
        self.translation_cam_to_road = np.array([0,-self.height,0])
        # compute vector nc. Note that R_{rc}^T = R_{cr}
        self.road_normal_camframe = self.rotation_cam_to_road.T @ np.array([0,1,0])


    def camframe_to_roadframe(self,vec_in_cam_frame):
        return self.rotation_cam_to_road @ vec_in_cam_frame + self.translation_cam_to_road

    def uv_to_roadXYZ_camframe(self,u,v):
        uv_hom = np.array([u,v,1])
        Kinv_uv_hom = self.inverse_intrinsic_matrix @ uv_hom
        denominator = self.road_normal_camframe@Kinv_uv_hom
        return self.height*Kinv_uv_hom/denominator
    
    def uv_to_roadXYZ_roadframe(self,u,v):
        r_camframe = self.uv_to_roadXYZ_camframe(u,v)
        X,Y,Z = self.camframe_to_roadframe(r_camframe)
        return np.array([Z,-X,-Y])

    def precompute_grid(self):
        xy = []
        for v in range(self.image_height):
            for u in range(self.image_width):
                X,Y,Z= self.uv_to_roadXYZ_roadframe(u,v)
                xy.append(np.array([X,Y]))
        xy = np.array(xy)
        return xy

    
def main():
    cg=IPM()
    
    print(cg.precompute_grid().shape)
    
if __name__=='__main__':
    main()