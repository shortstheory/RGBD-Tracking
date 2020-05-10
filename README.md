# ESE 650 Final Project
*Arnav Dhamija and Mihir Parmar*

## Abstract
In this project, we revisit the problem of 3D object tracking using an RGB-D camera. We present a novel method of tracking an object's translation and rotation relative to a camera using an RGB-D camera such as the Microsoft Kinect or Intel Realsense D435i. Our approach uses a particle filter with correlation computed using 2D feature descriptors. This approach can handle the cases of estimating the position of objects out of the camera's depth range, behind obstacles, and out of the camera's frame. This approach can be extended for active tracking objects on mobile robots.

## Code Structure
```
src/ - Contains the classes need for the particle filter and DepthTracker
   - DepthTracker.py - Class for running the algorithm
   - ParticleFilter.py - Class for implementing the particle filter
   - Helpers.py - File for storing the helper classes
   - Vision.py - Class with static functions for coordinate transforms and keypoint matching
Dataloader.py - loads the data from the Princeton RGB-D dataset
```

## Example
```
wget https://www.dropbox.com/s/mtbwzn2bb1lu7p2/bear_front.zip # run this command from the folder with the code
unzip bear_front.zip
python demo.py
```