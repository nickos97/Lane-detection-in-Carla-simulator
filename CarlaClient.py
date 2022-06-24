import glob
import os
import sys
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
import random
import time 
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import weakref
import torch
from cv2 import COLOR_RGB2GRAY
import queue
from Detection.fastai_detect import fastai
from Detection.LaneDetector import Unet_Detector
from Control_system.Control_system import PurePursuitPlusPID,PurePursuit,PIDController
from datetime import datetime

IM_WIDTH = 420
IM_HEIGHT = 210
FOV = 45
FPS = 20
RECORD = False
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

class SychronousClient():

    def __init__(self):
        
        self.client=None
        self.world=None
        self.camera=None
        self.camera_segmentation=None
        self.vehicle=None
        self.npc_list=[]
        self.autoPilot=False
        self.images=[]
        self.seg_image=None
        self.sync = True
        self.nsync = False
        self.tick = 0
        self.fps = FPS
        self.start_time = time.time()
        self.image_queue = queue.Queue()
        #models
        self.fastai_ld = fastai()
        self.Unet_ld = Unet_Detector()
        self.controller = PurePursuitPlusPID()
        self.PIDcontroller = PIDController()
        self.PurePursuitController = PurePursuit()
        self.record = RECORD
        if self.record:
            self.out = cv2.VideoWriter(f'Captures/route{TIMESTAMP}.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (840,210))


    def process_img(self,image):
    
        i = np.array(image.raw_data,dtype=np.float32)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        return i3
    
    def get_trajectory_from_lane_detector(self,ld, image):
    # get lane boundaries using the lane detector
        image = self.process_img(image)
        
        poly_left, poly_right, _, _,preds1 = ld.get_fit_and_probs(image)

        x = np.arange(0,60,1.0)
        y = -0.5*(poly_left(x)+poly_right(x))
        x += 0.5
        traj = np.stack((x,y)).T
        return traj,preds1
        
    def predict(self,pred,image):
    
        i3 = self.process_img(image)
        
        #pred = self.viz.predict2(i3).squeeze(0).squeeze(0).cpu().numpy() #Unet
        #pred = self.fastai_ld._predict(i3).squeeze(0).squeeze(0).cpu().numpy() #fastai
        pred = pred.squeeze(0).squeeze(0).cpu().numpy()
        i3 = np.array(i3,dtype=np.uint8)
        pred = np.array(100*pred,dtype=np.uint8)
        predrgb = cv2.cvtColor(pred,cv2.COLOR_GRAY2RGB)
        con = np.concatenate((i3,predrgb),axis=1)
        self.images.append(con)
        cv2.imshow("",np.array(con,dtype=np.uint8))
        cv2.waitKey(1)
        
    
    
    def camera_bp(self,filter):

        
        #Returns camera blueprint
        

        bp_camera = self.world.get_blueprint_library().find(filter)
        bp_camera.set_attribute('image_size_x',str(IM_WIDTH))
        bp_camera.set_attribute('image_size_y',str(IM_HEIGHT))
        bp_camera.set_attribute('fov',str(FOV))

        return bp_camera

    def set_synchronous_mode(self,fps):
    
        #Sets synchronous mode
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0/fps
        self.world.apply_settings(settings)

    def spawn_vehicle(self):

        #spawns vehicle
    
        blueprint_library=self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        random_location=random.choice(spawn_points)
        location = self.world.get_map().get_spawn_points()[45]
        self.vehicle=self.world.spawn_actor(vehicle_bp,location)

        self.npc_list.append(self.vehicle)
    
    def save_img(self,img):
        img.save_to_disk('home/nickos/Documents/Thesis/data/%.6d.jpg' % img.frame)
        self.images.append(img)

    

    def setup_camera(self):

        #spawn and attach camera
    
        camera_transform = carla.Transform(carla.Location(x=0.5, z=1.3),carla.Rotation(pitch=-5))
        self.camera = self.world.spawn_actor(self.camera_bp('sensor.camera.rgb'), camera_transform, attach_to=self.vehicle)
        
        self.camera.listen(self.image_queue.put)

        self.npc_list.append(self.camera)

    def destroy_npc(self):
        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.npc_list])
        
    def start_game(self):
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout(10.0)
        try:
            

            self.world = self.client.get_world()
            
            self.set_synchronous_mode(self.fps)
            self.spawn_vehicle()    
            self.setup_camera()
            
            frame = 0
            throttle=0
            error = 0
            steer = 0
            prev_speed = 0
            while True:
                if(self.autoPilot):
                    print("Autopilot set!!!")
                    self.vehicle.set_autopilot(True)
                self.world.tick()
                
                
                if frame%100==-1:
                    
                    print(f"location: {self.vehicle.get_location()}")
                    print(f"velocity: {self.vehicle.get_velocity()}")
                    print(f"angular velocity: {self.vehicle.get_angular_velocity()}")
                    print(f"acceleration: {self.vehicle.get_acceleration()}")
                    print(f"Cross track error: {error}")
                    print(f"steering: {steer}")
                image=self.image_queue.get()
                
                traj,preds1 = self.get_trajectory_from_lane_detector(self.Unet_ld,image)
        
                acc_vec = self.vehicle.get_acceleration()
                ax,ay,az = acc_vec.x,acc_vec.y,acc_vec.z
                acceleration = math.sqrt(ax**2+ay**2+az**2)
                
                speed_vec = self.vehicle.get_velocity()
                x,y,z = speed_vec.x,speed_vec.y,speed_vec.z
                speed = math.sqrt(x**2+y**2+z**2)
                
                computed_acc = (speed - prev_speed)/(1./FPS)
                prev_speed = speed
                
                print(f"actual acceleration: {acceleration}, computed acceleration: {computed_acc}")
                if frame%50==0:
                    print(f"speed:{speed}")
                #throttle,steer = self.controller.get_control(traj, speed, desired_speed=20, dt=1.0/self.fps)
                steer,error = self.PurePursuitController.get_control(traj,speed)
                #print(f"desired speed: {self.PIDcontroller.set_point}")
                if(abs(steer)>0.005):
                    self.PIDcontroller.set_point = 20
                else:
                    self.PIDcontroller.set_point = 20
                throttle = self.PIDcontroller.get_control(speed,dt=1.0/self.fps)
                
                control = carla.VehicleControl(throttle, steer)
                self.vehicle.apply_control(control)
                self.predict(preds1,image)
                #self.visualize(image)
                frame +=1
        finally:
            if self.record:
                for image in self.images:
                    self.out.write(image)
                self.out.release()
            if self.npc_list:
                self.destroy_npc()
                print('Actors destroyed')
            else:
                print('Zero actors to be destroyed!!!')

def main():

    try:
        client=SychronousClient()
        client.start_game()
    finally:
        print("Exit game")

if __name__== '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Cancelled by user, bye!!!s")
    finally:
        print("done")
