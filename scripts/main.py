#!/home/wshf/miniconda3/envs/graspnet/bin/python

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import torch
# import transforms3d as t3d
from graspnetAPI import Grasp, GraspGroup
import rospy
# import rtde_receive, dashboard_client
import copy
# from geometry_msgs.msg import Pose, PoseStamped

from concurrent.futures import ThreadPoolExecutor

# from robotiq_2f_gripper_msgs.msg import CommandRobotiqGripperFeedback, CommandRobotiqGripperResult, CommandRobotiqGripperAction, CommandRobotiqGripperGoal
# from robotiq_2f_gripper_control.robotiq_2f_gripper_driver import Robotiq2FingerGripperDriver

ROOT_DIR = os.path.dirname('/home/wshf/graspnet-baseline/')
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
# from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
# from data_utils import CameraInfo, create_point_cloud_from_depth_image

from grasp_ctrl import *
from hand_ctrl_robotiq import *
from utils import *
from camera_utils import *
from grasp_select import *

desk_z = 0.76
work_space = [-0.22, -0.12]
box_size = (0.51, 0.32, 0.28)
work_space3d = [work_space[0], work_space[0]+box_size[0], work_space[1], work_space[1]+box_size[1], desk_z-box_size[2], desk_z]
work_space3d_small = work_space3d.copy()
work_space3d_small[0] += 0.05
work_space3d_small[1] -= 0.05
work_space3d_small[2] += 0.02
work_space3d_small[3] -= 0.02

# robot_ip = "192.168.0.5"

place_idx = 0

# empty_positions = []

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='/home/wshf/graspnet-baseline/checkpoint-rs.tar', help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.05, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data():
    # load data
    # cloud_path = os.path.join(data_dir, 'point.npy')
    # color_path = os.path.join(data_dir, 'color.npy')
    # cloud = np.load(cloud_path, allow_pickle=True)
    # color = np.load(color_path, allow_pickle=True)

    cloud, color = point_cloud_shot()
    idx = mask(cloud, work_space3d)
    cloud_masked = cloud[idx]
    color_masked = color[idx]
    # color_idx = mask_color(color_masked, color_mask)
    # cloud_masked = cloud_masked[color_idx]
    # color_masked = color_masked[color_idx]

    num_points = 200
    wall1_xz = np.meshgrid(np.linspace(work_space3d[0], work_space3d[1], num_points), np.linspace(work_space3d[4], work_space3d[5] + 0.02, num_points))
    wall1 = np.concatenate([wall1_xz[0].reshape(-1, 1), np.full((num_points*num_points, 1), work_space3d[2]), wall1_xz[1].reshape(-1, 1)], axis=1)
    wall2_xz = wall1_xz
    wall2 = np.concatenate([wall2_xz[0].reshape(-1, 1), np.full((num_points*num_points, 1), work_space3d[3]), wall2_xz[1].reshape(-1, 1)], axis=1)
    wall3_yz = np.meshgrid(np.linspace(work_space3d[2], work_space3d[3], num_points), np.linspace(work_space3d[4], work_space3d[5] + 0.02, num_points))
    wall3 = np.concatenate([np.full((num_points*num_points, 1), work_space3d[0]), wall3_yz[0].reshape(-1, 1), wall3_yz[1].reshape(-1, 1)], axis=1)
    wall3_small_yz = np.meshgrid(np.linspace((work_space3d[2] + work_space3d[3]) / 2 - 0.06, (work_space3d[2] + work_space3d[3]) / 2 + 0.06, num_points // 2), np.linspace(work_space3d[5] - 0.21, work_space3d[5], num_points // 2))
    wall3_small = np.concatenate([np.full((num_points*num_points//4, 1), work_space3d[0] + 0.024), wall3_small_yz[0].reshape(-1, 1), wall3_small_yz[1].reshape(-1, 1)], axis=1)
    wall4_yz = wall3_yz
    wall4 = np.concatenate([np.full((num_points*num_points, 1), work_space3d[1]), wall4_yz[0].reshape(-1, 1), wall4_yz[1].reshape(-1, 1)], axis=1)
    wall4_small_yz = wall3_small_yz
    wall4_small = np.concatenate([np.full((num_points*num_points//4, 1), work_space3d[1] - 0.024), wall4_small_yz[0].reshape(-1, 1), wall4_small_yz[1].reshape(-1, 1)], axis=1)
    ground_xy = np.meshgrid(np.linspace(work_space3d[0], work_space3d[1], num_points), np.linspace(work_space3d[2], work_space3d[3], num_points))
    ground = np.concatenate([ground_xy[0].reshape(-1, 1), ground_xy[1].reshape(-1, 1), np.full((num_points*num_points, 1), work_space3d[5] + 0.02)], axis=1)
    wall = np.concatenate([wall1, wall2, wall3, wall4, ground, wall3_small, wall4_small], axis=0)
    wall_color = np.zeros_like(wall)
    point_walled = np.concatenate([cloud_masked, wall], axis=0)
    color_walled = np.concatenate([color_masked, wall_color], axis=0)

    point_sampled = cloud_masked
    color_sampled = color_masked

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_walled.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_walled.astype(np.float32))

    cloud_without_wall = o3d.geometry.PointCloud()
    cloud_without_wall.points = o3d.utility.Vector3dVector(point_sampled.astype(np.float32))
    cloud_without_wall.colors = o3d.utility.Vector3dVector(color_sampled.astype(np.float32))

    # voxel_size = 0.005
    # cloud = cloud.voxel_down_sample(voxel_size)

    end_points = dict()
    point_sampled = torch.from_numpy(point_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    point_sampled = point_sampled.to(device)
    end_points['point_clouds'] = point_sampled
    end_points['cloud_colors'] = color_sampled
    return end_points, cloud, cloud_without_wall, wall
    # return end_points, cloud, mask_pt

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    gg.nms()
    gg.sort_by_score()
    return gg

def get_grasp(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg_cnt = len(gg_array)
    gg = GraspGroup(gg_array)
    
    gg.sort_by_score()
    sample_range = min(10, gg_cnt)
    # sample_idx = np.random.randint(0, sample_range)
    sample_idx = 0
    grasp = gg[sample_idx]

    return grasp

def collision_detection(gg, cloud, get_GraspGroup=True):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.0, collision_thresh=cfgs.collision_thresh)
    if get_GraspGroup:
        gg = gg[~collision_mask]
        return gg
    else:
        return collision_mask

def first_feasible_grasp(gg, cloud, cloud_without_wall, wall_pts, trys):
    gg.depths[gg.depths > 0.03] = 0.03
    gg.heights = np.ones(len(gg)) * 20e-3
    # gg = collision_detection(gg, cloud)
    # vis_grasps(gg, cloud)
    # gg = collision_detection(gg, np.array(cloud.points))
    # vis_grasps(gg, cloud)
    
    # box_pos = [
    #     (-0.0837931444127552, -0.5158425246947902), 
    #     (-0.0837931444127552+0.56, -0.5158425246947902), 
    #     (-0.0837931444127552+0.56, -0.5158425246947902+0.39), 
    #     (-0.0837931444127552, -0.5158425246947902+0.39)
    # ]
    # # box_pos = mask_pts
    # box_height = 0.28
    # # box_height = 5
    # arm_length = 0.19
    # print(len(gg))
    # gg = box_detection(gg, box_pos, box_height, arm_length)
    # # vis_grasps(gg, cloud)
    # print(len(gg))
    gg.nms().sort_by_score()
    # return gg[0]

    for i in range(len(gg)):
        # to_quit = input("Quit? (y/n): ")
        # if to_quit == 'y':
        #     exit(0)

        g = gg[i]
        g.width += 0.02
        # g_0 = copy.deepcopy(g)
        # direction = g.rotation_matrix[:, 0]
        # g.translation += 0.02 * direction
        # vis_grasp(g, cloud)

        # for empty_g in empty_positions:
        #     if np.linalg.norm(g.translation - empty_g) < 0.01:
        #         continue

        obstacle1 = [ work_space3d[0], work_space3d[0] + 0.03, \
                     (work_space3d[2] + work_space3d[3]) / 2 - 0.06, (work_space3d[2] + work_space3d[3]) / 2 + 0.06, \
                      work_space3d[4], work_space3d[5]]
        obstacle2 = [ work_space3d[1] - 0.03, work_space3d[1], \
                     (work_space3d[2] + work_space3d[3]) / 2 - 0.06, (work_space3d[2] + work_space3d[3]) / 2 + 0.06, \
                      work_space3d[4], work_space3d[5]]
        if in_workspace(g.translation, obstacle1) or in_workspace(g.translation, obstacle2):
            continue

        if not in_workspace(g.translation, work_space3d):
            # print("Not in workspace 1")
            continue

        if trys < 2:
            # print(f"Grasp {i} move_grasp_to_center")
            g = move_grasp_to_center(g, cloud, cloud_without_wall, wall_pts)
            if g is None:
                # print("Move to center failed.")
                continue
            # vis_grasp(g, cloud)

        # print(f"Grasp {i} adjust_grasp")
        g = adjust_grasp(g, cloud, wall_pts)
        if g is None:
            # print("Normal or angle adjustment failed.")
            continue

        if not in_workspace(g.translation, work_space3d_small):
            # print("Not in workspace 2")
            continue

        # x_axis = g.rotation_matrix[:, 0]
        # theta = np.arccos(np.dot(x_axis, np.array([0, 0, 1])))
        # if theta > np.pi / 2:
        #     continue

        # vertex1 = g.translation + g.depth/2 * g.rotation_matrix[:, 0] + g.width/2 * g.rotation_matrix[:, 1]
        # vertex2 = g.translation + g.depth/2 * g.rotation_matrix[:, 0] - g.width/2 * g.rotation_matrix[:, 1]
        # too_low_1 = vertex1[2] > desk_z
        # too_low_2 = vertex2[2] > desk_z
        # if too_low_1 or too_low_2:
        #     continue

        return g
    
    # raise Exception("No feasible grasp found.")
    return None

def move_to_ready_pose(group, ready, start):
    if start:
        wait_Q = [-1.4126904646502894, -0.7241709868060511, -1.3692906538592737, -2.5246546904193323, 1.565276861190796, -1.2765572706805628]
        moveit_arm_Q(group, wait_Q)
        start = False
        return start
    ready1_Q = [-1.5708311239825647, -1.0874975363360804, -1.898360554371969, -0.6553075949298304, 1.6215910911560059, -1.4271395842181605]
    ready2_Q = [-1.5707948843585413, -1.1816704908954065, -1.5608304182635706, -1.7979419867144983, 1.6216390132904053, -1.4271395842181605]
    if not ready:
        moveit_arm_Q(group, ready1_Q)
        moveit_arm_Q(group, ready2_Q)
    return start

def vis_grasps(gg, cloud):
    # gg.nms()
    # gg.sort_by_score()
    # gg = gg[:10]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def vis_grasp(grasp, cloud):
    gripper = grasp.to_open3d_geometry()
    print(gripper)
    o3d.visualization.draw_geometries([cloud, gripper])
    
def get_place_Q():
    global place_idx
    place_Qs = [[-1.5376175085650843, -1.6826584974872034, -1.7544377485858362, -0.3596141974078577, 1.576860785484314, -1.4896062056170862],
                [-1.603865925465719, -1.76933461824526, -1.5779779593097132, -1.0756223837481897, 1.6310282945632935, -1.4588821570025843],
                [-1.764393154774801, -1.8328073660479944, -1.4954283873187464, -1.1074593702899378, 1.6759133338928223, -1.6134985128985804]]
    choice = place_idx
    place_idx = (place_idx + 1) % 3
    return place_Qs[choice]

def demo_grasp(tf_buffer, group):
    net = get_net()
    
    gripper = RobotiqGripper()

    ready = False
    start = True
    trys = 0
    # mask, mask_pts = get_mask()

    place_Q = get_place_Q()
    ready1_Q = [-1.5708311239825647, -1.0874975363360804, -1.898360554371969, -0.6553075949298304, 1.6215910911560059, -1.4271395842181605]
    ready2_Q = [-1.5707948843585413, -1.1816704908954065, -1.5608304182635706, -1.7979419867144983, 1.6216390132904053, -1.4271395842181605]
    wait_Q = [-1.4126904646502894, -0.7241709868060511, -1.3692906538592737, -2.5246546904193323, 1.565276861190796, -1.2765572706805628]
    
    while True:
        end_points, cloud, cloud_without_wall, wall_pts = get_and_process_data()
        # end_points, cloud, cloud_without_wall = get_and_process_datav2(mask=mask)
        gg = get_grasps(net, end_points)
        
        # with ThreadPoolExecutor() as executor:
        #     grasp_future = executor.submit(first_feasible_grasp, gg, cloud, cloud_without_wall, wall_pts, 0)
        #     execute_future = executor.submit(move_to_ready_pose, group, ready, start)
        #     grasp = grasp_future.result()
        #     start = execute_future.result()
        grasp = first_feasible_grasp(gg, cloud, cloud_without_wall, wall_pts, trys)
        print("moveit_to_ready_pose in line 322 start")
        # start = move_to_ready_pose(group, ready, start)
        print("moveit_to_ready_pose in line 322 end")
        
        # grasp = first_feasible_grasp(gg, cloud, cloud_without_wall, wall_pts)
        
        if grasp is None:
            print("No feasible grasp found.")
            moveit_arm_Q(group, wait_Q)
            ready = True
            trys += 1
            if trys >= 5:
                print(f"No feasible grasp found after {trys} tries.")
                return
            continue
        print("Feasible grasp found.")

        vis_grasp(grasp, cloud_without_wall)
        go_on = input("Continue? (y/n): ")
        if go_on == 'n':
            continue
        elif go_on == 'q':
            return

        width_default = grasp.width
        # width_default = width_default if width_default <= 100 else 100
        width = 0.0
        rotation = grasp.rotation_matrix
        translation = grasp.translation
        # translation[2] += 0.01
        direction = rotation[:, 0]
        translation += 0.02 * direction
        translation_prev = translation - 0.10 * direction
        grasp_pose = moveit_target_pose_from_graspnet(tf_buffer, translation, rotation)
        print(grasp_pose)
        grasp_pose_prev = moveit_target_pose_from_graspnet(tf_buffer, translation_prev, rotation)

        gripper.set_pos(pos=width_default)
        # if not ready:
        #     moveit_arm_Q(group, ready1_Q)
        #     moveit_arm_Q(group, ready2_Q)
        #     moveit_arm_Q(group, ready3_Q)
        
        # plan_prev = moveit_arm(group, grasp_pose_prev)
        # if plan_prev is None:
        #     moveit_arm_Q(group, wait_Q)
        #     ready = True
        #     trys += 1
        #     if trys >= 5:
        #         print(f"No feasible grasp found after {trys} tries.")
        #         return
        #     continue

        moveit_arm_Q(group, ready2_Q)
        moveit_arm(group, grasp_pose_prev)
        
        trys = 0
        moveit_arm_straight(group, grasp_pose, grasp_pose_prev)
        
        # q_prev = group.get_current_joint_values()
        
        pause = rospy.get_param('/pause_node', False)
        print(f"protective stop = {pause}")
        # stopped = pause
        cnt = 0
        rate = rospy.Rate(0.5)
        while pause:
            if cnt > 5:
                break
            group.stop()
            pause = rospy.get_param('/pause_node', True)
            rate.sleep()
            cnt += 1
        # if stopped:
        #     position = group.get_current_pose().pose.position
        #     pos = np.array([position.x, position.y, position.z])
        #     pos -= 0.005 * direction
        #     pose = copy.deepcopy(grasp_pose)
        #     pose.position.x = pos[0]
        #     pose.position.y = pos[1]
        #     pose.position.z = pos[2]
        #     print("Moving back a little.")
        #     moveit_arm_straight(group, pose)
        gripper.set_pos(pos=width)
        time.sleep(0.5)
        
        pause = rospy.get_param('/pause_node', False)
        print(f"protective stop = {pause}")
        # stopped = pause
        rate = rospy.Rate(0.5)
        cnt = 0
        while pause:
            if cnt > 5:
                break
            group.stop()
            pause = rospy.get_param('/pause_node', True)
            rate.sleep()
            cnt += 1
        
        moveit_arm_straight(group, grasp_pose_prev, grasp_pose)
        
        pause = rospy.get_param('/pause_node', False)
        print(f"protective stop = {pause}")
        # stopped = pause
        rate = rospy.Rate(0.5)
        cnt = 0
        while pause:
            if cnt > 5:
                break
            group.stop()
            pause = rospy.get_param('/pause_node', True)
            rate.sleep()
            cnt += 1
        time.sleep(0.5)
        
        pre_place_Q = [-1.4204400221454065, -0.8840416113482874, -1.5934847036944788, -1.6350296179400843, 1.5654805898666382, -1.2844165007220667]
        moveit_arm_Q(group, pre_place_Q)
        
        pause = rospy.get_param('/pause_node', False)
        print(f"protective stop = {pause}")
        # stopped = pause
        rate = rospy.Rate(0.5)
        cnt = 0
        while pause:
            if cnt > 5:
                break
            group.stop()
            pause = rospy.get_param('/pause_node', True)
            rate.sleep()
            cnt += 1
        
        pos = gripper.get_pos()
        print(f"Gripper pos: {pos}")
        
        if pos < 0.00186:
            ready = True
            gripper.set_pos(pos=width_default)
            moveit_arm_Q(group, wait_Q)
            continue
        moveit_arm_Q(group, ready1_Q)
        moveit_arm_Q(group, place_Q)
        ready = False

        gripper.set_pos(pos=width_default)
        moveit_arm_Q(group, ready1_Q)
        moveit_arm_Q(group, wait_Q)

def demo_view():
    net = get_net()
    # mask, mask_pts = get_mask()
    end_points, cloud, cloud_without_wall, wall_pts = get_and_process_data()
    # end_points, mask_pts, cloud, cloud_without_wall = get_and_process_datav2(mask)


    gg = get_grasps(net, end_points)
    # if cfgs.collision_thresh > 0:
    #     gg = collision_detection(gg, np.array(cloud.points))
    # vis_grasps(gg, cloud)
        
    gg.sort_by_score()
    vis_grasps(gg[:20], cloud)
    # g = gg[0]
    # vis_grasp(g, cloud)

    # --------------------------------------------------------------------------------
    # g = first_feasible_grasp(gg, cloud, cloud_without_wall, wall_pts, trys=0)
    # vis_grasp(g, cloud)
    # vis_grasp(g_0, cloud)
    # --------------------------------------------------------------------------------

    # points = np.array(cloud.points)
    # points -= g.translation
    # points = np.dot(points, g.rotation_matrix)

    # grasp_mask = (points[:, 0] > -g.depth) & (points[:, 0] < g.depth/2) & \
    #              (points[:, 1] > -g.width/3) & (points[:, 1] < g.width/3) & \
    #              (points[:, 2] > -g.height/2) & (points[:, 2] < g.height/2)
    # points_grasp_masked = np.array(cloud.points)[grasp_mask]
    # colors_grasp_masked = np.array(cloud.colors)[grasp_mask]
    # pcd_grasp_masked = o3d.geometry.PointCloud()
    # pcd_grasp_masked.points = o3d.utility.Vector3dVector(points_grasp_masked)
    # pcd_grasp_masked.colors = o3d.utility.Vector3dVector(colors_grasp_masked)
    # vis_grasp(g, pcd_grasp_masked)

    # g, pcd = move_grasp_to_center(g, cloud_without_wall)
    # vis_grasp(g, pcd)

    # rot = Rotation.from_euler('y', 30, degrees=True).as_matrix()
    # grot = copy.deepcopy(g)
    # grot.rotation_matrix = np.dot(rot, grot.rotation_matrix.T).T
    # grot.score = 0.1
    # gg = GraspGroup()
    # gg.add(g)
    # gg.add(grot)
    # vis_grasps(gg, cloud)

if __name__=='__main__':
    rospy.init_node('graspnet_demo', anonymous=True)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    group, eef_link, touch_links, scene = moveit_init()

    # scene = add_objects(scene)

    demo_grasp(tf_buffer, group)
    # pointing up to the ceiling
    quit_Q = [-1.5708430449115198, -0.7167013327227991, -1.678051773701803, -2.1457460562335413, 1.6216870546340942, -1.4270919005023401]
    moveit_arm_Q(group, quit_Q)
    
    # demo_view()
