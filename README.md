# ICRA RGMC 2024 code for Picking-in-Clutter

This repository is our codes for [the Picking-in-Clutter sub-track](http://cepbbenchmark.eu/) of [9th Robotic Grasping and Manipulation Competition (RGMC)](https://cse.usf.edu/~yusun/rgmc/2024.html) in [ICRA 2024](https://2024.ieee-icra.org/). We (Team THUDA) achieved 3rd place in the sub-track. This work is submitted to IEEE Robotics and Automation Practice (RA-P).

## Installation

- Our codes are tested on `Ubuntu 20.04` and `ROS Noetic`.
- Make sure you have installed these ROS packages: [`ur_robot_driver`](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver), [`moveit`](https://github.com/moveit/moveit), [`easy_handeye`](https://github.com/IFL-CAMP/easy_handeye), [`realsense2_camera`](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy), [`robotiq_2f`](https://github.com/KevinGalassi/Robotiq-2f-85)
- Please refer to [graspnet-baseline](https://github.com/graspnet/graspnet-baseline) to set up environments for `GraspNet`. 

Install this package:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/frankWang67/rgmc2024.git
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

Furthermore, please configure a `moveit_config` package for the UR5 equipped with a Robotiq gripper using `moveit_setup_assistant` and `ur5_robotiq_xacro` in this repository.

## Usage

Press the power-on button of the robot arm, open a terminial, and enter:

```bash
roslaunch rgmc2024 pipeline.launch
```

On the robot arm screen, press: Program the robot -> Empty program -> structure -> External control -> the "play" button (a triangle pointing right-hand-side) at the bottom of the screen. After you press the play button, a new line will show up in the terminal: 

```
[ INFO] [1739085973.564729658]: Robot connected to reverse interface. Ready to receive control commands.
```

If you want you visulize in RViz, open a new terminal and enter:

```bash
roslaunch ur5robotiq_moveit_config moveit_rviz.launch
```
