# Lane-detection-in-Carla-simulator

# Basic overview

![pipeline](https://user-images.githubusercontent.com/33639372/175613609-3512cd95-133c-412d-b02f-d3af4965b2d4.PNG)


The images are being produced by an rgb camera which is attached to the vehicle simulating the driver's point of view.
After generating and properly pre-processing the image via the Python API offered by the simulation, the lane detection model does
the necessary prediction by classifying each pixel of the image at the right and left lanes.
Based on the curvature of the road and the predefined desired speed, 
the steering wheel angle and the necessary throttle are calculated from the pure pursuit algorithm and
the PID controller respectively. Finally, the appropriate control is applied by the Python API
on the vehicle based on the previous calculations.

# Lisence & copyright

© Nikos Tzanettis, University of Patras 
