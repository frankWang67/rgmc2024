#!/home/wshf/miniconda3/envs/graspnet/bin/python

import rospy
import time
import rtde_receive, dashboard_client
from moveit_commander import MoveGroupCommander

robot_ip = "192.168.0.5"

def prevent_protective_stop(group):
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    rtde_d = dashboard_client.DashboardClient(robot_ip)
    rtde_d.connect()
    rate = rospy.Rate(100)
    rospy.set_param('/pause_node', False)
    while not rospy.is_shutdown():
        unlock = False
        while rtde_r.isProtectiveStopped():
            group.stop()
            print("Protective stop detected, robot stopped")
            rospy.set_param('/pause_node', True)
            rtde_d.unlockProtectiveStop()
            time.sleep(0.2)
            unlock = True
        if unlock:
            rtde_d.play()
            print("Protective stop released")
            time.sleep(0.2)
            rospy.set_param('/pause_node', False)
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node("prevent_protective_stop")
    group = MoveGroupCommander("manipulator")
    try:
        prevent_protective_stop(group)
    except rospy.ROSInterruptException:
        pass