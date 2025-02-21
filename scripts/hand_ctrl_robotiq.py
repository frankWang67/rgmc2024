import rospy
from std_msgs.msg import Float32
from robotiq_2f_gripper_control.robotiq_2f_gripper_driver import Robotiq2FingerGripperDriver
import time

class RobotiqGripper:
    def __init__(self, comport='/dev/ttyUSB0'):
        self.gripper = Robotiq2FingerGripperDriver(comport=comport, init_requested=False)

    def set_pos(self, pos):
        self.gripper.goto(pos=pos, speed=0.1, force=60, block=True)

    def get_pos(self):
        pos = rospy.wait_for_message('/gripper_pos', Float32)
        return pos.data

if __name__ == "__main__":
    rospy.init_node('hand_ctrl_robotiq', anonymous=True)
    gripper = RobotiqGripper()
    time.sleep(1)
    gripper.set_pos(pos=0.0)
    print(f"{gripper.get_pos()}")