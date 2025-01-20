#!/home/wshf/miniconda3/envs/graspnet/bin/python

import rospy
from dh_gripper_msgs.msg import GripperCtrl, GripperState

def gripper_goto(pos, force=100.0, speed=100.0):
    pub = rospy.Publisher('/gripper/ctrl', GripperCtrl, queue_size=10)
    rospy.sleep(0.01)
    msg = GripperCtrl()
    msg.initialize = False
    msg.position = pos
    msg.force = force
    msg.speed = speed
    pub.publish(msg)

def get_gripper_pos():
    state = rospy.wait_for_message('/gripper/states', GripperState)
    pos = state.position
    pos = float(pos)
    return pos

if __name__ == "__main__":
    rospy.init_node('gripper_ctrl')
    rospy.sleep(1)

    gripper_goto(60.0)
    rospy.sleep(1)
    print(get_gripper_pos())
    rospy.sleep(1)
    gripper_goto(0.0)
    rospy.sleep(1)
    print(get_gripper_pos())
