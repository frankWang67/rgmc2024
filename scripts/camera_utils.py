#!/home/wshf/miniconda3/envs/graspnet/bin/python

import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct
import ctypes

NAME_FIELDS = ['x', 'y', 'z', 'rgb']

def camera_shot(mode="both"):
    '''
    Take one shot of color and depth or only color or only depth
    :param mode: str, "both" or "color" or "depth"
    :return color_cv: PIL.Image
    :return depth_cv: PIL.Image
    '''
    bridge = CvBridge()
    try:
        rospy.init_node('shoter')
    except:
        pass
    if mode == "both":
        color = rospy.wait_for_message('/camera/color/image_raw', Image, timeout = None)
        color_cv = bridge.imgmsg_to_cv2(color, "bgr8")
        color_cv = color_cv / 255.0
        color_cv = color_cv.astype(np.float32)
        #深度是连续采集10次取平均
        for i in range(10):
            depth = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image, timeout = None)
            depth_cv = bridge.imgmsg_to_cv2(depth, "16UC1")
            depth_cv = depth_cv /1000.0
            depth_cv = depth_cv.astype(np.float32)
            depth_cv = cv2.GaussianBlur(depth_cv, (5, 5), 0)
            if i == 0:
                depth_sum = depth_cv
            else:
                depth_sum = depth_sum + depth_cv
        depth_cv = depth_sum / 10.0

        return color_cv, depth_cv
    elif mode == "color":
        color = rospy.wait_for_message('/camera/color/image_raw', Image, timeout = None)
        color_cv = bridge.imgmsg_to_cv2(color, "bgr8")
        color_cv = color_cv / 255.0
        color_cv = color_cv.astype(np.float32)
        return color_cv
    elif mode =="depth":
        for i in range(10):
            depth = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image, timeout = None)
            depth_cv = bridge.imgmsg_to_cv2(depth, "16UC1")
            depth_cv = depth_cv /1000.0
            depth_cv = depth_cv.astype(np.float32)
            depth_cv = cv2.GaussianBlur(depth_cv, (5, 5), 0)
            if i == 0:
                depth_sum = depth_cv
            else:
                depth_sum = depth_sum + depth_cv
        depth_cv = depth_sum / 10.0

        return depth_cv

def show(color, depth):
    '''
    Display color and depth image
    :param color: PIL.Image
    :param depth: PIL.Image
    '''
    bridge = CvBridge()
    color_cv = bridge.imgmsg_to_cv2(color, "bgr8")
    depth_cv = bridge.imgmsg_to_cv2(depth, "16UC1")
    print(color_cv.shape)
    print(depth_cv.shape)
    return 

def point_cloud_shot():
    '''
    Shot point cloud in flat shape
    :return pt_cloud: np.array (N, 3)
    :return color_rgb: np.array (N, 3)
    '''
    try:
        rospy.init_node('point_cloud_shoter')
    except:
        pass
    msg = rospy.wait_for_message('/camera/depth/color/points', PointCloud2, timeout = None)
    points = point_cloud2.read_points_list(msg, field_names=('x', 'y', 'z', 'rgb'))
    points = np.array(points)
    pt_cloud = points[:, :3]
    color = points[:, 3]
    print('color', color.shape, color.dtype)
    float2RGB_vec = np.vectorize(float2RGB)
    r, g, b = float2RGB_vec(color)
    color_rgb = np.stack((r, g, b), axis=1)
    print('point cloud shot')

    return pt_cloud, color_rgb

def point_cloud_shotv2():
    '''
    Use structured point cloud
    This point cloud is of shape (H, W, 4) 
    :return pt_cloud: np.array (H, W, 3)
    :return color_rgb: np.array (H, W, 3)
    '''
    try:
        rospy.init_node('point_cloud_shoter')
    except:
        pass
    msg = rospy.wait_for_message('/camera/depth_registered/points', PointCloud2, timeout = None)
    width = msg.width
    height = msg.height
    points = point_cloud2.read_points_list(msg, field_names=('x', 'y', 'z', 'rgb'))
    points = np.array(points, dtype=np.float32)
    # print('points', points.shape, points.dtype)
    pt_cloud = points[:, :3]
    color = points[:, 3]

    # float2RGB_vec = np.vectorize(float2RGBv2)
    # r, g, b = float2RGB_vec(color)
    # color_rgb = np.stack((r, g, b), axis=1)
    color_rgb = float2RGBv2(color)
    pt_cloud = pt_cloud.reshape((height, width, 3))
    color_rgb = color_rgb.reshape((height, width, 3))

    print('point cloud shot')
    return pt_cloud, color_rgb

def float2RGB(float_num):
    '''
    Decode float to RGB for float64
    :param float_num: float
    :return red: float
    :return green: float
    :return blue: float
    '''
    s = struct.pack('>f', float_num)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value
    red = int((pack & 0x00FF0000)>>16)
    green = int((pack & 0x0000FF00)>>8)
    blue = int((pack & 0x000000FF))
    red = red / 255
    green = green / 255
    blue = blue / 255
    
    return red, green, blue

def float2RGBv2(rgb):
    '''
    Decode float to RGB for float64
    '''
    rgb = rgb.copy()
    rgb.dtype = np.uint32
    i = 0
    r = np.asarray((rgb >> 16) & 0xFF, dtype=np.uint8)
    g = np.asarray((rgb >> 8) & 0xFF, dtype=np.uint8)
    b = np.asarray((rgb >> 0) & 0xFF, dtype=np.uint8)

    r = r / 255
    g = g / 255
    b = b / 255
    rgb_array = np.stack((r, g, b), axis=1)
    return rgb_array

if __name__ == "__main__":
    run = 2
    img = camera_shot("color")
    # 归一化的图片转为255
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(f"run_{run}.jpg", img)