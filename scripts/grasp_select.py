import numpy as np
import open3d as o3d
import copy
import random
from scipy.spatial.transform import Rotation
from graspnetAPI import Grasp

from utils import mask

desk_z = 0.85

def vis_grasp(grasp, cloud):
    gripper = grasp.to_open3d_geometry()
    print(gripper)
    o3d.visualization.draw_geometries([cloud, gripper])

def box_detection_pcd(g: Grasp, wall_pts):
# def box_detection_pcd(g: Grasp, cloud):
    # points = np.array(cloud.points)
    # color  = np.array(cloud.colors)

    T = g.translation
    R = g.rotation_matrix
    # matrix = np.eye(4)
    # matrix[:3,:3] = R
    # matrix[:3, 3] = T
    # matrix = np.linalg.inv(matrix)
    # points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).T
    # points = np.dot(matrix, points)
    # points = points[:3].T
    points = wall_pts.copy()
    points -= T
    points = np.dot(points, R)

    # pcd = o3d.geometry.PointCloud()
    # pt = np.concatenate([points, np.array([[0, 0, 0]])], axis=0)
    # pt = np.concatenate([pt, np.array([[1, 0, 0]])], axis=0)
    # pt = np.concatenate([pt, np.array([[0, 1, 0]])], axis=0)
    # pt = np.concatenate([pt, np.array([[0, 0, 1]])], axis=0)
    # cl = np.concatenate([color, np.array([[255, 0, 255]])], axis=0)
    # cl = np.concatenate([cl, np.array([[255, 0, 0]])], axis=0)
    # cl = np.concatenate([cl, np.array([[0, 255, 0]])], axis=0)
    # cl = np.concatenate([cl, np.array([[0, 0, 255]])], axis=0)
    # pcd.points = o3d.utility.Vector3dVector(pt)
    # pcd.colors = o3d.utility.Vector3dVector(cl)
    # grasp = copy.deepcopy(g)
    # grasp.translation = np.array([0, 0, 0])
    # grasp.rotation_matrix = np.eye(3)
    # vis_grasp(grasp, pcd)
    wrist1_mask = ((points[:, 0] >  -0.25) & (points[:, 0] <=  0.01)  & \
                   (points[:, 1] >  -0.09) & (points[:, 1] <   0.09)  & \
                   (points[:, 2] > -0.025) & (points[:, 2] <  0.025)) | \
                  ((points[:, 0] >  -0.35) & (points[:, 0] <= -0.25)  & \
                   (points[:, 1] >  -0.04) & (points[:, 1] <   0.04)  & \
                   (points[:, 2] >  -0.04) & (points[:, 2] <   0.04))
    
    x_axis = np.array([-1, 0, 0])
    direction = np.array([0, -1, 0]).reshape((3, 1))
    direction = np.dot(R.T, direction).reshape(3)
    d1 = direction - np.dot(direction, x_axis) * x_axis
    d1 = d1 / np.linalg.norm(d1)
    d2 = np.cross(x_axis, d1)
    d2 = d2 / np.linalg.norm(d2)
    rot = np.array([x_axis, d1, d2]).T
    points = np.dot(rot, points.T).T
    wrist2_mask = (points[:, 0] >  0.25) & (points[:, 0] < 0.32) & \
                  (points[:, 1] > -0.04) & (points[:, 1] < 0.20) & \
                  (points[:, 2] > -0.15) & (points[:, 2] < 0.04)

    mask = wrist1_mask | wrist2_mask
    points = points[mask, :]
    # color = color[mask, :]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([pcd])
    return points.shape[0] > 10

def grasp_object_pcd(g: Grasp, cloud: o3d.geometry.PointCloud):
    """
    return a point cloud of the object grasped by g
    """
    points = np.array(cloud.points)
    colors = np.array(cloud.colors)

    T = g.translation
    R = g.rotation_matrix
    points -= T
    points = np.dot(points, R)

    grasp_mask = (points[:, 0] > -g.depth) & (points[:, 0] < g.depth) & \
                 (points[:, 1] > -g.width/2) & (points[:, 1] < g.width/2) & \
                 (points[:, 2] > -g.height/2) & (points[:, 2] < g.height/2)
    
    if np.sum(grasp_mask) < 300:
        return None

    points = points[grasp_mask]
    colors = colors[grasp_mask]
    points = np.dot(points, R.T)
    points += T
    original_points = points.copy()
    for i in range(1, 11):
        pts_temp = original_points.copy()
        pts_temp[:, 2] += i*0.003
        points = np.concatenate([points, pts_temp], axis=0)
    colors = colors.repeat(11, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def find_nearest_point_and_normal(pcd, query_pt):
    """
    find the nearest point and normal of the point cloud to the query point
    """
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        # pcd.orient_normals_consistent_tangent_plane(100)
        
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    query_pt = np.array(query_pt)
    [k, idx, _] = kdtree.search_knn_vector_3d(query_pt, 1)

    nearest_point = np.array(pcd.points)[idx[0]]
    normal = np.array(pcd.normals)[idx[0]]

    return nearest_point, normal, idx[0]

def grasp_point_normals_dot(g: Grasp, cloud: o3d.geometry.PointCloud):
    """
    get the dot product of the normals of the two grasp points
    """
    pcd = grasp_object_pcd(g, cloud)
    if pcd is None:
        return None

    # matrix = np.eye(4)
    # matrix[:3,:3] = g.rotation_matrix
    # matrix[:3, 3] = g.translation

    res = 0
    for i in range(5):
        # m1 = np.linalg.inv(matrix)
        # m1[0, 3] -= g.depth/2
        # m1[1, 3] += g.width/2
        # m1 = np.linalg.inv(m1)
        # pt1 = m1[:3, 3]
        pt1 = g.translation.copy()
        pt1 -= (g.depth/2) * g.rotation_matrix[:, 0]
        pt1 += (g.width/2) * g.rotation_matrix[:, 1]
        grasp_pt1, normal1, idx1 = find_nearest_point_and_normal(pcd, pt1)

        # m2 = np.linalg.inv(matrix)
        # m2[0, 3] -= g.depth/2
        # m2[1, 3] -= g.width/2
        # m2 = np.linalg.inv(m2)
        # pt2 = m2[:3, 3]
        pt2 = g.translation.copy()
        pt2 -= (g.depth/2) * g.rotation_matrix[:, 0]
        pt2 -= (g.width/2) * g.rotation_matrix[:, 1]
        grasp_pt2, normal2, idx2 = find_nearest_point_and_normal(pcd, pt2)

        gripper_x = g.rotation_matrix[:, 0]
        normal1 -= np.dot(normal1, gripper_x) * gripper_x
        normal1 /= np.linalg.norm(normal1)
        normal2 -= np.dot(normal2, gripper_x) * gripper_x
        normal2 /= np.linalg.norm(normal2)

        res += np.abs(np.dot(normal1, normal2))

    # points = np.array(pcd.points)
    # colors = np.array(pcd.colors)
    # colors[idx1] = np.array([255, 0, 0])
    # colors[idx2] = np.array([0, 0, 255])
    # for i in range(1, 21):
    #     points = np.concatenate([points, (grasp_pt1 + i*0.003*normal1).reshape(-1, 3)], axis=0)
    #     points = np.concatenate([points, (grasp_pt2 + i*0.003*normal2).reshape(-1, 3)], axis=0)
    #     colors = np.concatenate([colors, np.array([[0, 255, 0]])], axis=0)
    #     colors = np.concatenate([colors, np.array([[0, 255, 255]])], axis=0)
    # # print(np.abs(np.dot(normal1, normal2)))
    # # points = np.concatenate([points, (grasp_pt1 + 0.1*normal1).reshape(-1, 3)], axis=0)
    # # points = np.concatenate([points, (grasp_pt2 + 0.1*normal2).reshape(-1, 3)], axis=0)
    # # colors = np.concatenate([colors, np.array([[255, 255, 0]])], axis=0)
    # # colors = np.concatenate([colors, np.array([[0, 255, 255]])], axis=0)
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # # o3d.visualization.draw_geometries([pcd])
    # vis_grasp(g, pcd)

    return res / 5

# def move_grasp_to_center(g: Grasp, cloud_without_wall: o3d.geometry.PointCloud):
def move_grasp_to_center(g: Grasp, cloud: o3d.geometry.PointCloud, cloud_without_wall: o3d.geometry.PointCloud, wall_pts):
    """
    move the grasp to the center of the object
    """
    # height adjustment
    vertex1 = g.translation + g.depth/2 * g.rotation_matrix[:, 0] + g.width/2 * g.rotation_matrix[:, 1]
    vertex2 = g.translation + g.depth/2 * g.rotation_matrix[:, 0] - g.width/2 * g.rotation_matrix[:, 1]
    too_low_1 = vertex1[2] > desk_z
    too_low_2 = vertex2[2] > desk_z
    if too_low_1 or too_low_2:
        theta = -np.arctan(g.rotation_matrix[2, 1] / g.rotation_matrix[2, 0])
        rot_mat = Rotation.from_euler('z', theta, degrees=False).as_matrix()
        g.rotation_matrix = np.dot(g.rotation_matrix, rot_mat.T)
        g.translation[2] = desk_z - g.depth - 0.002

        # if too_low_1 and too_low_2:
        #     g.translation[2] = desk_z - g.depth - 0.002

    # normals adjustment
    dot_product = grasp_point_normals_dot(g, cloud)
    # print(f"{dot_product=}")

    if dot_product is None:
        # print("Empty grasp.")
        return None
    
    thres = 0.8
    found = dot_product > thres
    rot_angle = 5
    g1 = copy.deepcopy(g)
    g2 = copy.deepcopy(g)
    while (not found) and (rot_angle < 46):
        rot_mat1 = Rotation.from_euler('x', rot_angle, degrees=True).as_matrix()
        g1.rotation_matrix = np.dot(g.rotation_matrix, rot_mat1.T)
        dot_product = grasp_point_normals_dot(g1, cloud)
        if dot_product is None:
            rot_angle += 5
            continue
        found = dot_product > thres
        if found:
            g = g1
            break
        
        rot_mat2 = Rotation.from_euler('x', -rot_angle, degrees=True).as_matrix()
        g2.rotation_matrix = np.dot(g.rotation_matrix, rot_mat2.T)
        dot_product = grasp_point_normals_dot(g2, cloud)
        if dot_product is None:
            rot_angle += 5
            continue
        found = dot_product > thres
        if found:
            g = g2
            break
        rot_angle += 5
    if not found:
        # print("Normals adjustment failed")
        return None

    # center adjustment
    points_without_wall = np.array(cloud_without_wall.points)
    transl = g.translation.copy()
    search_range = 0.05
    search_center = g.translation
    search_space = [search_center[0] - search_range, search_center[0] + search_range,
                    search_center[1] - search_range, search_center[1] + search_range,
                    search_center[2] - search_range, search_center[2] + search_range]
    idxs = mask(points_without_wall, search_space)
    points_masked = points_without_wall[idxs]
    colors_masked = np.array(cloud_without_wall.colors)[idxs]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_masked)
    pcd.colors = o3d.utility.Vector3dVector(colors_masked)
    center = pcd.get_center()
    # g.translation = (transl + center) / 2
    g.translation = center
    g.translation[2] = transl[2]
    # g.translation[2] = (transl[2] + center[2]) / 2
    # print(f"{g.translation=}")
    # vis_grasp(g, cloud)

    points = np.array(cloud.points)
    points -= g.translation
    points = np.dot(points, g.rotation_matrix)
    points_without_wall -= g.translation
    points_without_wall = np.dot(points_without_wall, g.rotation_matrix)
    wall = wall_pts.copy()
    wall -= g.translation
    wall = np.dot(wall, g.rotation_matrix)
    left_collision = True
    right_collision = True
    bottom_collision = True
    empty_grasp = True
    trys = 0
    sign = 0
    last_sign = 0
    while (left_collision or right_collision or bottom_collision or empty_grasp) and trys < 36:
    # while (left_collision or right_collision) and trys < 50:
    # while left_collision or right_collision:
        left_mask = (points_without_wall[:, 0] > -g.depth) & (points_without_wall[:, 0] < g.depth) & \
                    (points_without_wall[:, 1] > g.width/2) & (points_without_wall[:, 1] < g.width/2 + 0.01) & \
                    (points_without_wall[:, 2] > -g.height/2) & (points_without_wall[:, 2] < g.height/2)
        left_wall_mask = (wall[:, 0] > -0.10) & (wall[:, 0] < g.depth) & \
                         (wall[:, 1] > g.width/2) & (wall[:, 1] < g.width/2 + 0.035) & \
                         (wall[:, 2] > -g.height/2) & (wall[:, 2] < g.height/2)
        right_mask = (points_without_wall[:, 0] > -g.depth) & (points_without_wall[:, 0] < g.depth) & \
                     (points_without_wall[:, 1] > -g.width/2 - 0.01) & (points_without_wall[:, 1] < -g.width/2) & \
                     (points_without_wall[:, 2] > -g.height/2) & (points_without_wall[:, 2] < g.height/2)
        right_wall_mask = (wall[:, 0] > -0.10) & (wall[:, 0] < g.depth) & \
                          (wall[:, 1] > -g.width/2 - 0.035) & (wall[:, 1] < -g.width/2) & \
                          (wall[:, 2] > -g.height/2) & (wall[:, 2] < g.height/2)
        bottom_mask = (points_without_wall[:, 0] < -0.03) & (points_without_wall[:, 0] > -0.10) & \
                      (points_without_wall[:, 1] > -0.05) & (points_without_wall[:, 1] < 0.05) & \
                      (points_without_wall[:, 2] > -0.027) & (points_without_wall[:, 2] < 0.027)
        grasp_mask = (points_without_wall[:, 0] > -g.depth) & (points_without_wall[:, 0] < g.depth/2) & \
                     (points_without_wall[:, 1] > -g.width/3) & (points_without_wall[:, 1] < g.width/3) & \
                     (points_without_wall[:, 2] > -g.height/2) & (points_without_wall[:, 2] < g.height/2)
        
        # # left mask debug
        # points_left_masked = np.array(cloud_without_wall.points)[left_mask]
        # colors_left_masked = np.array(cloud_without_wall.colors)[left_mask]
        # pcd_left_masked = o3d.geometry.PointCloud()
        # pcd_left_masked.points = o3d.utility.Vector3dVector(points_left_masked)
        # pcd_left_masked.colors = o3d.utility.Vector3dVector(colors_left_masked)
        # vis_grasp(g, pcd_left_masked)

        # # right mask debug
        # points_right_masked = np.array(cloud_without_wall.points)[right_mask]
        # colors_right_masked = np.array(cloud_without_wall.colors)[right_mask]
        # pcd_right_masked = o3d.geometry.PointCloud()
        # pcd_right_masked.points = o3d.utility.Vector3dVector(points_right_masked)
        # pcd_right_masked.colors = o3d.utility.Vector3dVector(colors_right_masked)
        # vis_grasp(g, pcd_right_masked)

        # # bottom mask debug
        # points_bottom_masked = np.array(cloud_without_wall.points)[bottom_mask]
        # colors_bottom_masked = np.array(cloud_without_wall.colors)[bottom_mask]
        # pcd_bottom_masked = o3d.geometry.PointCloud()
        # pcd_bottom_masked.points = o3d.utility.Vector3dVector(points_bottom_masked)
        # pcd_bottom_masked.colors = o3d.utility.Vector3dVector(colors_bottom_masked)
        # vis_grasp(g, pcd_bottom_masked)

        # # grasp mask debug
        # points_grasp_masked = np.array(cloud.points)[grasp_mask]
        # colors_grasp_masked = np.array(cloud.colors)[grasp_mask]
        # pcd_grasp_masked = o3d.geometry.PointCloud()
        # pcd_grasp_masked.points = o3d.utility.Vector3dVector(points_grasp_masked)
        # pcd_grasp_masked.colors = o3d.utility.Vector3dVector(colors_grasp_masked)
        # vis_grasp(g, pcd_grasp_masked)
        
        left_collision = (np.sum(left_mask) + np.sum(left_wall_mask)) > 50
        right_collision = (np.sum(right_mask) + np.sum(right_wall_mask)) > 50
        bottom_collision = np.sum(bottom_mask) > 100
        empty_grasp = np.sum(grasp_mask) < 100
        # print(f"{left_collision=}, {right_collision=}, {bottom_collision=}, {empty_grasp=}")

        if empty_grasp:
            return None

        direction = g.rotation_matrix[:, 1]
        prob1 = 0.45
        prob2 = 0.90
        rot_angle = 0
        if left_collision and right_collision:
            last_sign = sign
            sign = 0
        elif left_collision:
            last_sign = sign
            sign = 1
        elif right_collision:
            last_sign = sign
            sign = -1
        if sign != 0:
            if sign == -1 * last_sign:
                sign = 0
            else:
                # print(f"{left_collision=} and {right_collision=}, moving")
                g.translation += sign * 0.01 * direction
                points -= sign * 0.01 * np.array([0, 1, 0])
                points_without_wall -= sign * 0.01 * np.array([0, 1, 0])
                wall -= sign * 0.01 * np.array([0, 1, 0])
        if sign == 0:
            sample = random.random()
            if g.width <= 0.099 and sample < prob1:
                # print(f"{left_collision=} and {right_collision=}, width up")
                g.width += 0.001
            elif sample < prob2:
                # print(f"{left_collision=} and {right_collision=}, rotating")
                rot_mat = Rotation.from_euler('x', 5, degrees=True).as_matrix()
                g.rotation_matrix = np.dot(g.rotation_matrix, rot_mat.T)
                points = np.dot(points, rot_mat.T)
                points_without_wall = np.dot(points_without_wall, rot_mat.T)
                wall = np.dot(wall, rot_mat.T)
                rot_angle += 5
            else:
                g.translation -= 0.01 * g.rotation_matrix[:, 0]
                points += 0.01 * np.array([1, 0, 0])
                points_without_wall += 0.01 * np.array([1, 0, 0])
                wall += 0.01 * np.array([1, 0, 0])

        if bottom_collision:
            # print(f"{bottom_collision=}, moving up")
            g.translation -= 0.01 * g.rotation_matrix[:, 0]
            points += 0.01 * np.array([1, 0, 0])
            points_without_wall += 0.01 * np.array([1, 0, 0])
            wall += 0.01 * np.array([1, 0, 0])

        # vis_grasp(g, cloud)

        if rot_angle >= 180:
            return None

        trys += 1
        # print(f"{g.translation=}")
        # vis_grasp(g, cloud_without_wall)

    if trys >= 36:
        return None
    # print(f"{trys=}")
    return g

def adjust_grasp(g: Grasp, cloud: o3d.geometry.PointCloud, wall_pts):
    '''
    Adjust the grasp to be more perpendicular to the surface and no collision with the box
    '''
    if not box_detection_pcd(g, np.array(cloud.points)):
    # if not box_detection_pcd(g, cloud):
        return g
    
    # print("Adjusting rotation")
    g_x_axis = g.rotation_matrix[:, 0]
    g_y_axis = g.rotation_matrix[:, 1]
    g_z_axis = g.rotation_matrix[:, 2]
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # pt = g.translation
    # pt -= g_x_axis * 0.22
    # direction = np.cross(g_x_axis, x_axis)
    # direction = direction / np.linalg.norm(direction)
    # pt += direction * 0.17

    if np.abs(np.dot(g_y_axis, y_axis)) < np.abs(np.dot(g_z_axis, y_axis)):
        rot_axis = 'y'
    else:
        rot_axis = 'z'

    found = False
    rot_angle = 5
    g1 = copy.deepcopy(g)
    g2 = copy.deepcopy(g)
    while (not found) and (rot_angle < 46):
        rot_mat1 = Rotation.from_euler(rot_axis, rot_angle, degrees=True).as_matrix()
        g1.rotation_matrix = np.dot(g.rotation_matrix, rot_mat1.T)
        found = not box_detection_pcd(g1, wall_pts)
        # found = not box_detection_pcd(g1, cloud)

        if found:
            return g1
        
        rot_mat2 = Rotation.from_euler(rot_axis, -rot_angle, degrees=True).as_matrix()
        g2.rotation_matrix = np.dot(g.rotation_matrix, rot_mat2.T)
        found = not box_detection_pcd(g2, wall_pts)
        # found = not box_detection_pcd(g2, cloud)

        if found:
            return g2

        rot_angle += 5
    
    # print("Rotation adjustment failed")
    return None