First launch:
Open 3 terminals and launch in this order
1) 
roslaunch hector_quadrotor_gazebo quadrotor_empty_world.launch 
2) 
source catkin_build_ws/devel/setup.bash
cd DepthTracker/src 
python3 bounding_box_node.py 
3)
rosrun DepthTracker DepthTracker.py

roslaunch hector_sensors_description realsense_d435_camera_static_transforms.launch camera_name:=camera
