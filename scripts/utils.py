import rospy
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose
import tf2_ros
from tf.transformations import quaternion_matrix

def mask(cloud, work_space):
    # cloud (n*3)
    # between (x_min,x_max),(y_min,y_max),(z_min,z_max)
    x_min, x_max, y_min, y_max, z_min, z_max = work_space
    valid_indices = np.where((cloud[:, 0] < x_max) & (cloud[:, 0] > x_min) &
                             (cloud[:, 1] < y_max) & (cloud[:, 1] > y_min) &
                             (cloud[:, 2] < z_max) & (cloud[:, 2] > z_min))[0]
    
    return valid_indices

def mask_color(cloud, work_space):
    # cloud (n*3)
    # between (x_min,x_max),(y_min,y_max),(z_min,z_max)
    x_min, x_max, y_min, y_max, z_min, z_max = work_space
    valid_indices = np.where((cloud[:, 0] > x_max) | (cloud[:, 0] < x_min) &
                             (cloud[:, 1] > y_max) | (cloud[:, 1] < y_min) &
                             (cloud[:, 2] > z_max) | (cloud[:, 2] < z_min))[0]
    
    return valid_indices

def in_workspace(point, work_space):
    x_min, x_max, y_min, y_max, z_min, z_max = work_space
    return (point[0] < x_max) & (point[0] > x_min) & (point[1] < y_max) & (point[1] > y_min)

def point_camera2robot(tf_buffer, points_cam):
    try:
        transform = tf_buffer.lookup_transform(
            target_frame="base_link",
            source_frame="camera_depth_optical_frame",
            time=rospy.Time(0)
        )
    except (tf2_ros.LookupException, 
            tf2_ros.ConnectivityException, 
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"TF error: {e}")
        return None
    
    trans = transform.transform.translation
    rot = transform.transform.rotation
    T = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
    T[:3, 3] = [trans.x, trans.y, trans.z]

    # Convert 3D coordinates to the robot arm coordinate system
    points_cam = np.reshape(points_cam, (-1, 3))
    homo_coordinates = np.concatenate((np.reshape(points_cam, (-1, 3)), np.ones((points_cam.shape[0], 1))), axis=1)
    points_bot = np.dot(T, homo_coordinates.T)[:3, :]

    return points_bot.T

def rotmat_camera2robot(tf_buffer, rotmat_cam):
    try:
        transform = tf_buffer.lookup_transform(
            target_frame="base_link",
            source_frame="camera_depth_optical_frame",
            time=rospy.Time(0)
        )
    except (tf2_ros.LookupException, 
            tf2_ros.ConnectivityException, 
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"TF error: {e}")
        return None
    
    rot = transform.transform.rotation
    T = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])[:3, :3]
    
    rotmat_bot = T @ rotmat_cam
    
    return rotmat_bot

def pose6D_to_matrix(pose):
    translation = pose[:3]
    rotation_vector = pose[3:]
    rotation_matrix = Rotation.from_rotvec(rotation_vector).as_matrix()

    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation

    return matrix

def matrix_to_pose6D(matrix):
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    rotation_vector = Rotation.from_matrix(rotation_matrix).as_rotvec()

    pose = np.concatenate([translation, rotation_vector])

    return pose

def moveit_target2pose6D(target):
    translation = np.array([target.position.x, target.position.y, target.position.z])
    quat = np.array([target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w])
    rotvec = Rotation.from_quat(quat).as_rotvec()
    pose6D = np.concatenate([translation, rotvec])

    return pose6D

def pose6D2moveit_target(pose6D):
    target = Pose()
    target.position.x = pose6D[0]
    target.position.y = pose6D[1]
    target.position.z = pose6D[2]
    quat = Rotation.from_rotvec(pose6D[3:]).as_quat()
    target.orientation.x = quat[0]
    target.orientation.y = quat[1]
    target.orientation.z = quat[2]
    target.orientation.w = quat[3]

    return target

def matrix_TCP2gripper(matrix_TCP):
    matrix_base = np.linalg.inv(matrix_TCP)

    # matrix_base[0, 3] += 0.01
    # depth = depth if depth < 0.03 else 0.03
    # theta = np.arccos((500 * width - 16.25) / 57)
    # theta = np.arccos((0.5 * width - 16.25) / 57)
    # d = 57e-3 * np.sin(theta) - 9e-3
    # matrix_base[2, 3] -= (d - depth + 0.17)
    matrix_base[2, 3] -= 0.18

    x_modifier = np.array([[1,  0, 0, 0],
                           [0,  0, 1, 0],
                           [0, -1, 0, 0],
                           [0,  0, 0, 1]])
    z_modifier = np.array([[ 0, 1, 0, 0],
                           [-1, 0, 0, 0],
                           [ 0, 0, 1, 0],
                           [ 0, 0, 0, 1]])
    y_modifier = np.array([[0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [-1, 0, 0, 0],
                           [0, 0, 0, 1]])
    matrix_base = x_modifier @ y_modifier @ z_modifier @ matrix_base
    matrix_gripper = np.linalg.inv(matrix_base)

    return matrix_gripper

def pose6D_TCP2gripper(pose6D_TCP):
    matrix_TCP = pose6D_to_matrix(pose6D_TCP)
    matrix_gripper = matrix_TCP2gripper(matrix_TCP)
    pose6D_gripper = matrix_to_pose6D(matrix_gripper)

    return pose6D_gripper

def matrix_gripper2TCP(matrix_gripper):
    matrix_base = np.linalg.inv(matrix_gripper)

    # depth = depth if depth < 0.03 else 0.03
    # theta = np.arccos((500 * width - 16.25) / 57)
    # theta = np.arccos((0.5 * width - 16.25) / 57)
    # print("theta = arccos((500 * width - 16.25) / 57)")
    # d = 57e-3 * np.sin(theta) - 9e-3
    # matrix_base[2, 3] += (d - depth + 0.17)
    # matrix_base[0, 3] += (d - depth + 0.17)
    matrix_base[0, 3] += 0.18

    x_modifier = np.array([[1,  0, 0, 0],
                           [0,  0,-1, 0],
                           [0,  1, 0, 0],
                           [0,  0, 0, 1]])
    y_modifier = np.array([[0, 0, -1, 0],
                           [0, 1, 0, 0],
                           [1, 0, 0, 0],
                           [0, 0, 0, 1]])
    z_modifier = np.array([[ 0, 1, 0, 0],
                           [-1, 0, 0, 0],
                           [ 0, 0, 1, 0],
                           [ 0, 0, 0, 1]])
    matrix_base = z_modifier @ y_modifier @ x_modifier @ matrix_base
    matrix_TCP = np.linalg.inv(matrix_base)

    return matrix_TCP

def pose6D_gripper2TCP(pose6D_gripper):
    matrix_gripper = pose6D_to_matrix(pose6D_gripper)
    matrix_TCP = matrix_gripper2TCP(matrix_gripper)
    pose6D_TCP = matrix_to_pose6D(matrix_TCP)

    return pose6D_TCP

def get_target_grasp_pose6D(tf_buffer, translation, rotation_matrix):
    transl = point_camera2robot(tf_buffer, translation)
    transl = transl[0]
    rotmat = rotmat_camera2robot(tf_buffer, rotation_matrix)
    rotvec = Rotation.from_matrix(rotmat).as_rotvec()
    pose6D = np.concatenate([transl, rotvec])
    target = pose6D_gripper2TCP(pose6D)

    return target

def moveit_target_pose_from_graspnet(tf_buffer, translation, rotation_matrix):
    transl = point_camera2robot(tf_buffer, translation)[0]
    rotmat = rotmat_camera2robot(tf_buffer, rotation_matrix)
    matrix = np.eye(4)
    matrix[:3, :3] = rotmat
    matrix[:3, 3] = transl
    target_matrix = matrix_gripper2TCP(matrix)
    # target_matrix = matrix

    # modifier = np.diag([-1, -1, 1, 1])
    # target_matrix = modifier @ target_matrix

    target = Pose()
    target.position.x = target_matrix[0, 3]
    target.position.y = target_matrix[1, 3]
    target.position.z = target_matrix[2, 3]
    quat = Rotation.from_matrix(target_matrix[:3, :3]).as_quat()
    target.orientation.x = quat[0]
    target.orientation.y = quat[1]
    target.orientation.z = quat[2]
    target.orientation.w = quat[3]

    return target

def moveit_target_pose_from_pose6D(pose6D):
    matrix = pose6D_to_matrix(pose6D)
    modifier = np.diag([-1, -1, 1, 1])
    matrix = modifier @ matrix

    target = Pose()
    target.position.x = matrix[0, 3]
    target.position.y = matrix[1, 3]
    target.position.z = matrix[2, 3]
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    target.orientation.x = quat[0]
    target.orientation.y = quat[1]
    target.orientation.z = quat[2]
    target.orientation.w = quat[3]

    return target

if __name__ == "__main__":
    translation = np.array([0.07492547, 0.16113761, 0.81200004])
    print(point_camera2robot(translation)[0])

    # rotvec = np.array([2.985584712207952, -0.6134870744994786, -0.1187754537936827])
    # quat = Rotation.from_rotvec(rotvec).as_quat()
    # print(quat)