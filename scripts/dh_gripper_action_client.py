#!/home/wshf/miniconda3/envs/graspnet/bin/python

import rospy
import actionlib
from dh_gripper_msgs.msg import GripperCtrl, GripperState

class DHGripperClient:
    def __init__(self, action_server_name="gripper_controller/gripper_cmd", states_topic="/gripper/states"):
        """
        Initialize the gripper client.

        :param action_server_name: The name of the gripper action server.
        :param joint_states_topic: The topic to subscribe for joint states.
        """
        rospy.init_node("dh_gripper_client", anonymous=True)

        # Action client
        self.client = actionlib.SimpleActionClient(action_server_name, GripperCtrl)

        # Wait for the server to start
        rospy.loginfo(f"Waiting for action server '{action_server_name}'...")
        self.client.wait_for_server()
        rospy.loginfo(f"Connected to action server '{action_server_name}'.")

        # Joint states subscriber
        self.states_topic = states_topic
        self.state = None
        rospy.Subscriber(self.states_topic, GripperState, self.states_callback)

    def states_callback(self, msg):
        """Callback to receive joint state updates."""
        self.state = msg

    def send_goal(self, position, force=100.0, speed=100.0):
        """
        Send a goal to the gripper.

        :param position: Target position (radians).
        :param max_effort: Maximum gripping force.
        """
        goal = GripperCtrl()
        goal.initialize = False
        goal.position = position
        goal.force = force
        goal.speed = speed

        rospy.loginfo(f"Sending goal: position={position}, force={force}, speed={speed}")
        self.client.send_goal(goal, done_cb=self.done_callback, feedback_cb=self.feedback_callback)

    def done_callback(self, state, result):
        """Callback when the action is done."""
        rospy.loginfo(f"Action finished with state={state}, result={result}")

    def feedback_callback(self, feedback):
        """Callback for action feedback."""
        rospy.loginfo(f"Feedback received: position={feedback.position}, effort={feedback.effort}")

    def wait_for_result(self, timeout=None):
        """
        Wait for the result of the action.

        :param timeout: Timeout in seconds.
        :return: True if the action finished in time, False otherwise.
        """
        return self.client.wait_for_result(timeout)

    def get_state(self):
        """
        Get the current state.

        :return: The latest GripperState message, or None if no state has been received yet.
        """
        return self.state

if __name__ == "__main__":
    try:
        # Initialize the client
        client = DHGripperClient()

        # Example usage: Move the gripper
        client.send_goal(position=100.0)

        # Wait for the result with a 5-second timeout
        if client.wait_for_result(timeout=rospy.Duration(5.0)):
            rospy.loginfo("Goal reached successfully.")
        else:
            rospy.logwarn("Timed out waiting for result.")

        # Query joint state
        rospy.sleep(1)  # Allow some time for state to update
        state = client.get_state()
        if state:
            rospy.loginfo(f"Current joint state: {state}")
        else:
            rospy.logwarn("No joint state received.")

    except rospy.ROSInterruptException:
        rospy.loginfo("Action client interrupted.")
