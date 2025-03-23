#!/home/wshf/miniconda3/envs/graspnet/bin/python

import rospy
import numpy as np
import time
import socket
import copy
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from geometry_msgs.msg import Pose, PoseStamped

# from hand_ctrl import hand_ctrl

def interpolate_orientation(start, end, num_points, speed=0.2, blend=0.005):
    path = []
    start_pos, end_pos = start[0:3], end[0:3]
    start_end_rot = Rotation.from_rotvec(np.concatenate((start[3:], end[3:])).reshape(2,3))
    interval = (end_pos - start_pos) / num_points

    slerp = Slerp([0,1], start_end_rot)
        
    orientations = slerp(np.linspace(0, 1, num_points+1))
    orientations = orientations.as_rotvec()

    for i in range(num_points+1):
        pos = start_pos + i*interval
        path.append(np.concatenate((pos, orientations[i],np.array([0.5,speed,blend]))).reshape(9,).tolist())
    
    path[-1][-2] = 0.01
    return path

def move_arm(target_pose, rtde_c, rtde_r):
    # rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    # rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

    acutal_pose = rtde_r.getActualTCPPose()
    path = interpolate_orientation(acutal_pose, target_pose, 50)
    rtde_c.moveL(path)
    # rtde_c.moveJ_IK(target_pose, speed=0.2)

'''
    Below are moveit APIs
'''

def moveit_init():
    group_name = "manipulator"  # 指定机械臂的move_group名称
    group = MoveGroupCommander(group_name)
    # group.set_planner_id("RRTConnectkConfigDefault")
    # # group.set_planner_id("LazyPRMkConfigDefault")
    # # group.set_planner_id("BFMTkConfigDefault")
    # # group.set_planning_time(10) # seconds
    # group.set_max_velocity_scaling_factor(1.0)
    eef_link = group.get_end_effector_link()
    robot = moveit_commander.RobotCommander()
    touch_links = robot.get_link_names(group=group_name)
    
    # 创建PlanningSceneInterface实例
    scene = PlanningSceneInterface()
    scene.clear()

    return group, eef_link, touch_links, scene

def add_an_object(scene, obj_name, obj_pos, obj_size):
    obj_pose = PoseStamped()
    obj_pose.header.frame_id = "base_link"
    obj_pose.pose.position.x = obj_pos[0]
    obj_pose.pose.position.y = obj_pos[1]
    obj_pose.pose.position.z = obj_pos[2]
    scene.add_box(obj_name, obj_pose, size=obj_size)

    return scene

def add_objects(scene):
    '''
        workspace is given as the 2D coordinate of the nearest convex of the box's underside in the world coordinate system
    '''
    scene.clear()
    scene = add_an_object(scene, "box_wall", [0.21, 0.15, 0.05], (0.555, 0.02, 0.10))
    scene = add_an_object(scene, "obstacle", [0.30, 0.00, 0.05], (0.10 , 0.10, 0.10))
    scene = add_an_object(scene, "back_obstacle", [0.50, -0.40, 0.05], (1.00 , 0.60, 0.10))

    return scene

def moveit_arm(group, pose):
    # Update planning scene
    # rospy.sleep(1)  # Wait for scene update
    # group.set_planning_time(10)  # Set planning time
    # group.set_start_state_to_current_state()
    # group.set_goal_position_tolerance(0.05)
    # group.set_goal_orientation_tolerance(0.05)
    
    # 设置目标姿态
    current = group.get_current_pose().pose
    quat_current = np.array([current.orientation.x, current.orientation.y, current.orientation.z, current.orientation.w])
    quat_target = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    if np.dot(quat_current, quat_target) < 0:
        pose.orientation.x = -pose.orientation.x
        pose.orientation.y = -pose.orientation.y
        pose.orientation.z = -pose.orientation.z
        pose.orientation.w = -pose.orientation.w
    group.set_pose_target(pose)
    
    # 检查规划是否成功
    # success = False
    # cnt = 0
    # while (not success) and (cnt < 5):
    #     success, plan, _, _ = group.plan()
    #     cnt += 1
    #     if not success:
    #         rospy.logerr("No plan found to move to pose, retrying...")

    group.set_start_state_to_current_state()
    success, plan, _, _ = group.plan()
    if not success:
        # rospy.logerr("No plan found to move to pose after 5 tries")
        rospy.logerr("No plan found to move to pose")
        return None
    
    if len(plan.joint_trajectory.points) > 100:
        rospy.logerr("Too many points in the plan")
        return None
    
    # 执行规划的轨迹
    rospy.loginfo("Executing plan to move to pose")
    group.execute(plan, wait=True)
    
    return plan


def moveit_arm_Q(group, q):
    group.set_joint_value_target(q)
    
    # 检查规划是否成功
    success = False
    cnt = 0
    while (not success) and (cnt < 5):
        success, plan, _, _ = group.plan()
        cnt += 1
        if not success:
            rospy.logerr("No plan found to move to pose, retrying...")
    if not success:
        rospy.logerr("No plan found to move to pose after 5 tries")
        return None
    
    # 执行规划的轨迹
    rospy.loginfo("Executing plan to move to pose")
    group.execute(plan, wait=True)

    return plan


def moveit_arm_straight(group, target_pose, current_pose=None):
    if current_pose is None:
        current_pose = group.get_current_pose().pose
        print(current_pose)
        
    # waypoints = [current_pose, target_pose]

    waypoints = []
    num_points = 50
    start_pos = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
    end_pos = np.array([target_pose.position.x, target_pose.position.y, target_pose.position.z])
    start_quat = np.array([current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w])
    end_quat = np.array([target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w])
    start_end_rot = Rotation.from_quat(np.concatenate([start_quat, end_quat]).reshape(2,4))
    interval = (end_pos - start_pos) / num_points

    slerp = Slerp([0,1], start_end_rot)
        
    orientations = slerp(np.linspace(0, 1, num_points+1))
    orientations = orientations.as_quat()

    for i in range(num_points+1):
        pos = start_pos + i*interval
        orientation = orientations[i]
        waypoint = Pose()
        waypoint.position.x = pos[0]
        waypoint.position.y = pos[1]
        waypoint.position.z = pos[2]
        waypoint.orientation.x = orientation[0]
        waypoint.orientation.y = orientation[1]
        waypoint.orientation.z = orientation[2]
        waypoint.orientation.w = orientation[3]
        waypoints.append(waypoint)

    (plan, fraction) = group.compute_cartesian_path(waypoints, 0.1, 0.0)
    print("Fraction: ", fraction)
    group.execute(plan, wait=True)

    return plan

def moveit_arm_straight_simplify(group, target_pose, current_pose=None):
    if current_pose is None:
        current_pose = group.get_current_pose().pose
    waypoints = [current_pose, target_pose]
    (plan, fraction) = group.compute_cartesian_path(waypoints, 0.01, 0.0)
    print("Fraction: ", fraction)
    group.execute(plan, wait=True)

def moveit_arm_plan(group, plan):
    group.execute(plan, wait=True)

def moveit_arm_plan_reverse(group, plan):
    rev_plan = copy.deepcopy(plan)
    rev_plan.joint_trajectory.points.reverse()
    duration = rev_plan.joint_trajectory.points[-1].time_from_start.to_sec()
    for point in rev_plan.joint_trajectory.points:
        point.time_from_start = rospy.Duration(duration - point.time_from_start.to_sec())
        point.velocities = [-v for v in point.velocities]
        point.accelerations = [-a for a in point.accelerations]
    group.execute(rev_plan, wait=True)

def safety_recover(ip="192.168.0.5", port=30002):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # port = 29999
    # script = "robotmode\n"
    script = "get robot safety mode()\n"
    try:
        s.connect((ip, port))
        # s.send(script.encode("utf-8"))
        time.sleep(1)
        response = s.recv(4)
        print("Response: ", response.decode("utf-8"))
    finally:
        s.close()

if __name__ == "__main__":
    rospy.init_node('ur5_moveit_demo')
    group, eef_link, touch_links, scene = moveit_init()
    
    # target_pose = Pose()
    # target_pose.position.x = 0.18337249425842814
    # target_pose.position.y = 0.23131284480172665
    # target_pose.position.z = 0.2853755274054901
    # target_pose.orientation.x = -0.07897989095485924
    # target_pose.orientation.y = 0.9739419470620377
    # target_pose.orientation.z = 0.08588756482060517
    # target_pose.orientation.w = -0.19448029922578727

    # moveit_arm(group, target_pose)

    current_q = group.get_current_joint_values()
    print(current_q)