# ICRA RGMC 2024 code for Picking-in-Clutter

This repository is our codes for [the Picking-in-Clutter sub-track](http://cepbbenchmark.eu/) of [9th Robotic Grasping and Manipulation Competition (RGMC)](https://cse.usf.edu/~yusun/rgmc/2024.html) in [ICRA 2024](https://2024.ieee-icra.org/). We (Team THUDA) achieved 3rd place in the sub-track. We introduced our system in [this paper](https://alumnisssup-my.sharepoint.com/personal/salvatore_davella_santannapisa_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsalvatore%5Fdavella%5Fsantannapisa%5Fit%2FDocuments%2FIROS2024%5Fworkshop%2FPurely%20Vision%2DBased%20Robotic%20Grasping%20in%20Clutter%20with%20Parallel%20Jaw%5Ffinal%2Epdf&parent=%2Fpersonal%2Fsalvatore%5Fdavella%5Fsantannapisa%5Fit%2FDocuments%2FIROS2024%5Fworkshop&ga=1). Different from the paper, this codebase is based on the [DH-AG95](https://en.dh-robotics.com/product/ag) gripper, rather than [Robotiq 2f-140](https://robotiq.com/products/adaptive-grippers#Two-Finger-Gripper).

## Setup

- Our codes are tested on `Ubuntu 20.04` and `ROS Noetic`.
- Make sure you have installed these ROS packages: [`ur_robot_driver`](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver), [`moveit`](https://github.com/moveit/moveit), [`easy_handeye`](https://github.com/IFL-CAMP/easy_handeye), [`realsense2_camera`](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy)
- Please refer to [graspnet-baseline](https://github.com/graspnet/graspnet-baseline) to set up environments for `GraspNet`. 

## Usage

Press the power-on button of the robot arm, open a terminial, and enter:

```bash
# Bring up the robot arm
# You can change the IP address according to your own configuration.
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.0.5 
```

On the robot arm screen, press: Program the robot -> Empty program -> structure -> External control -> the "play" button (a triangle pointing right-hand-side) at the bottom of the screen. After you press the play button, a new line will show up in the terminal: 

```
[ INFO] [1739085973.564729658]: Robot connected to reverse interface. Ready to receive control commands.
```

Open another new terminal, and enter:

```bash
# Bring up the Moveit! program for the robot arm
roslaunch ur5_moveit_config moveit_planning_execution.launch
```

Open a third terminal, and enter:

```bash
# Bring up the camera
roslaunch realsense2_camera rs_camera.launch filters:=pointcloud
```

Open a fourth terminal, and enter:

```bash
# Publish the calibration between the robot arm and the camera
roslaunch easy_handeye ur5_realsense_publish.launch
```

Open a fifth terminal, and enter:

```bash
# Enable the serial port of the gripper
sudo chmod 666 /dev/ttyACM0
# Run the main program
source /home/wshf/rgmc2024new_ws/devel/setup.bash
rosrun rgmc2024 main.py
```
