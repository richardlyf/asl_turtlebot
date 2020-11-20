#!/usr/bin/env python

from enum import Enum

import rospy
from asl_turtlebot.msg import DetectedObject
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from std_msgs.msg import Float32MultiArray, String
from project_navigator import Navigator
from project_navigator import Mode as NavMode
from marker_pub import MarkerTracker, publish_marker
import tf
import math
import numpy as np

class Mode(Enum):
    """State machine modes. Feel free to change."""
    NAV = 1
    POSE = 2
    STOP = 3
    CROSS = 4
    WAITING = 5
    MANUAL = 6
    READY = 7


class SupervisorParams:

    def __init__(self, verbose=False):
        # If sim is True (i.e. using gazebo), we want to subscribe to
        # /gazebo/model_states. Otherwise, we will use a TF lookup.
        self.use_gazebo = rospy.get_param("sim")

        # How is nav_cmd being decided -- human manually setting it, or rviz
        self.rviz = rospy.get_param("rviz")

        # If using gmapping, we will have a map frame. Otherwise, it will be odom frame.
        self.mapping = rospy.get_param("map")

        # Threshold at which we consider the robot at a location
        self.pos_eps = rospy.get_param("~pos_eps", 0.1)
        self.theta_eps = rospy.get_param("~theta_eps", 0.3)

        # Time to stop at a stop sign
        self.stop_time = rospy.get_param("~stop_time", 3.)

        # Time to wait for loading or unloading
        self.wait_time = rospy.get_param("~wait_time", 3.)
        
        # Minimum distance from a stop sign to obey it
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 0.5)

        # Time taken to cross an intersection
        self.crossing_time = rospy.get_param("~crossing_time", 3.)

        if verbose:
            print("SupervisorParams:")
            print("    use_gazebo = {}".format(self.use_gazebo))
            print("    rviz = {}".format(self.rviz))
            print("    mapping = {}".format(self.mapping))
            print("    pos_eps, theta_eps = {}, {}".format(self.pos_eps, self.theta_eps))
            print("    stop_time, stop_min_dist, crossing_time = {}, {}, {}".format(self.stop_time, self.stop_min_dist, self.crossing_time))


class Supervisor:

    def __init__(self):
        # Initialize ROS node
        rospy.init_node('turtlebot_supervisor', anonymous=True)
        self.params = SupervisorParams(verbose=True)

        # Current mode
        self.mode = Mode.MANUAL
        self.prev_mode = None  # For printing purposes

        # Navigator with planning, tracking, and pose stabilization
        self.navigator = Navigator()
        rospy.on_shutdown(self.navigator.shutdown_callback)
        rospy.loginfo("Storing initial state info...")
        while True:
            if self.navigator.update_state():
                break

        # Create MarkerTracker to remember vendor locations
        self.vendor_names = ["apple", "banana", "pizza", "kite"]
        self.marker_tracker = MarkerTracker(self.vendor_names)

        # Remember stop sign locations
        self.stop_sign_pose = None

        # Remember the current order
        self.orders = []
        self.order_repeat_count = 0
        self.order_repeat_max = 10 # A given order is published 10 times
        self.vendor_dist = 0.2 # How far away to park from vendor
        self.home_pose = (self.navigator.x, self.navigator.y, self.navigator.theta)

        # Whether we should go home if we reach the ready state and there are
        # no more orders.
        self._should_go_home = False

        # Remember the last time cat/dog is seen
        self.seen_cat = 0
        self.seen_dog = 0

        ########## PUBLISHERS ##########
        # Command vel (used for idling)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
        self.cat_publisher = rospy.Publisher('/cat', String, queue_size=10)
        self.dog_publisher = rospy.Publisher('/dog', String, queue_size=10)

        ########## SUBSCRIBERS ##########
        # Stop sign detector
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        # Vendor detectors
        for vendor_name in self.vendor_names:
            rospy.Subscriber('/detector/' + vendor_name, DetectedObject, self.vendor_detected_callback)
        # Request subscriber
        rospy.Subscriber('/delivery_request', String, self.request_callback) 
        # Animal detectors
        rospy.Subscriber('/detector/cat', DetectedObject, self.cat_callback)
        rospy.Subscriber('/detector/dog', DetectedObject, self.dog_callback)

    def publish_goal_pose(self, goal):
        """ sends the current desired pose to the navigator """
        pose_g_msg = Pose2D()
        pose_g_msg.x = goal[0]
        pose_g_msg.y = goal[1]
        pose_g_msg.theta = goal[2]
        self.nav_goal_publisher.publish(pose_g_msg)

    ########## SUBSCRIBER CALLBACKS ##########
    def estimate_pose(self, msg):
        """
        estimates the position of the detected target based on the current robot position
        """
        # compute the angle of the target relative to the robot
        theta_left = msg.thetaleft
        theta_right = msg.thetaright
        if theta_left > math.pi:
            theta_left -= 2. * math.pi
        if theta_right > math.pi:
            theta_right -= 2. * math.pi
        avg_theta = theta_left + theta_right
        if avg_theta != 0:
            avg_theta /= 2.

        # compute position of the detected 
        self.navigator.update_state()
        global_theta = avg_theta + self.navigator.theta
        x = self.navigator.x + np.cos(global_theta) * msg.distance 
        y = self.navigator.y + np.sin(global_theta) * msg.distance 
        # returned theta is the angle of the target facing robot
        return x, y, global_theta - np.pi


    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """
        if self.mode == Mode.STOP or self.mode == Mode.CROSS:
            return
        self.stop_sign_pose = self.estimate_pose(msg)
        publish_marker("stop_sign", (self.stop_sign_pose[0], self.stop_sign_pose[1]), 404, (1, 0, 0))

    def vendor_detected_callback(self, msg):
        # Do not update vendor once we're done exploring
        if self.mode != Mode.MANUAL:
            return

        vendor_x, vendor_y, global_theta = self.estimate_pose(msg)
        # Ad hoc solution for pizza being detected as kite
        if msg.name == "kite":
            msg.name = "pizza"
        self.marker_tracker.place_marker(msg.name, (vendor_x, vendor_y), global_theta)

    def request_callback(self, msg):
        if self.mode == Mode.MANUAL:
            self.mode = Mode.READY
        self.order_repeat_count += 1
        self.order_repeat_count %= self.order_repeat_max
        # Ignore the message if it's not received for the first time
        if self.order_repeat_count != 1:
            return
        # Ignore empty request
        if msg.data == '':
            return
        # Parse the vendors
        new_orders = msg.data.split(",")
        if "home" in new_orders:
            self._should_go_home = True
        self.orders.extend(x for x in new_orders if x != "home")
        rospy.loginfo("Received order %s", new_orders)

    def cat_callback(self, msg):
        
        if rospy.get_time() - self.seen_cat > 3:
            rospy.loginfo("MEOWWWWWWWWWWW")
            self.cat_publisher.publish('MEOWWWWWWWWWWW')
        self.seen_cat = rospy.get_time()

    def dog_callback(self, msg):
        
        if rospy.get_time() - self.seen_dog > 3:
            rospy.loginfo("woof woof woof")
            self.dog_publisher.publish('woof woof woof')
        self.seen_dog = rospy.get_time()

    ########## STATE MACHINE ACTIONS ##########
    def stay_idle(self):
        """ sends zero velocity to stay put """
        vel_g_msg = Twist()
        self.cmd_vel_publisher.publish(vel_g_msg)

    def check_stop_sign(self):
        """ check if close to stop sign and should stop"""
        if self.stop_sign_pose is None:
            publish_marker("stop_sign", (-1, -1), 404, (1, 0, 0))
            return
        self.navigator.update_state()
        # TODO maybe if the stop sign and the robot are not facing each other, ignore
        # The angle can be hard to estimate

        # If the robot is close enough to the stop sign, stop
        stop_sign_loc = np.array([self.stop_sign_pose[0], self.stop_sign_pose[1]]) 
        robot_loc = np.array([self.navigator.x, self.navigator.y])
        dist = np.linalg.norm(stop_sign_loc - robot_loc)
        if dist > 0 and dist < self.params.stop_min_dist and self.mode == Mode.NAV:
            self.init_stop_sign()
            self.stop_sign_pose = None
            publish_marker("stop_sign", (-1, -1), 404, (1, 0, 0))

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """
        self.stop_sign_start = rospy.get_rostime()
        self.mode = Mode.STOP

    def has_stopped(self):
        """ checks if stop sign maneuver is over """
        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.params.stop_time)

    def init_wait(self):
        """ initiates waiting """
        self.waiting_start = rospy.get_rostime()
        self.mode = Mode.WAITING
    
    def is_ready(self):
        """ checks if loading or unloading is over """
        return self.mode == Mode.WAITING and \
               rospy.get_rostime() - self.waiting_start > rospy.Duration.from_sec(self.params.wait_time)
    
    def init_crossing(self):
        """ initiates an intersection crossing maneuver """
        self.cross_start = rospy.get_rostime()
        self.mode = Mode.CROSS

    def has_crossed(self):
        """ checks if crossing maneuver is over """
        return self.mode == Mode.CROSS and \
               rospy.get_rostime() - self.cross_start > rospy.Duration.from_sec(self.params.crossing_time)

    def _pop_next_order(self):
        """Find, remove, and return the next destination we should go to.

        Greedily chooses the next destination based on which is closest.
        Returns a solved Astar plan if one is found, or None otherwise.
        """
        if not self.orders:
            return None
        plans = []
        bad_orders = []
        for i, order in enumerate(self.orders):
            pose = self.marker_tracker.get_goal_pose(order, self.vendor_dist)
            if not pose:
                rospy.loginfo("Vendor '%s' unknown. Skipping.", order)
                bad_orders.append(order)
                continue
            plan = self.navigator.plan_to(pose)
            if not plan:
                rospy.loginfo("Vendor '%s' unreachable. Skipping.", order)
                bad_orders.append(order)
                continue
            plans.append((plan, pose, i))
        if not plans:
            return None
        best_plan, best_pose, i = min(plans, key=lambda x: x[0].cost)
        rospy.loginfo("Moving to %s.", self.orders[i])
        self.orders.pop(i)
        # Drop bad orders with probability 10%, just so we don't keep them
        # around forever, but we still try again to reach them in case a
        # failure was just a fluke.
        for x in bad_orders:
            if np.random.random() < 0.1:
                self.orders.remove(x)
        return best_pose

    ########## STATE MACHINE LOOP ##########
    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """
        self.check_stop_sign()

        # logs the current mode
        if self.prev_mode != self.mode:
            rospy.loginfo("Current mode: %s", self.mode)
            self.prev_mode = self.mode

        # Check for planning errors
        elif self.navigator.planning_error():
            self.mode = Mode.READY
            self.navigator.reset_planning_error()

        # Navigator reached goal 
        elif self.navigator.mode == NavMode.PARK and self.navigator.at_goal():
            self.navigator.clear_goal()
            if self.mode != Mode.MANUAL:
                self.init_wait()

        elif self.mode == Mode.WAITING:
            # Waiting to load or unload food
            self.stay_idle()
            if self.is_ready():
                self.mode = Mode.READY

        elif self.mode == Mode.READY:
            # Ready to move to the next way point
            self.stay_idle()
            next_order = self._pop_next_order()
            if next_order:
                self._should_go_home = True
                self.publish_goal_pose(next_order)
                self.mode = Mode.NAV
            elif self._should_go_home:
                rospy.loginfo("Going home")
                self._should_go_home = False
                self.publish_goal_pose(self.home_pose)
                self.mode = Mode.NAV

        elif self.mode == Mode.NAV or self.mode == Mode.MANUAL:
            # Moving towards a desired pose
            self.navigator.navigate()

        elif self.mode == Mode.STOP:
            # At a stop sign
            self.stay_idle()
            if self.has_stopped():
                self.navigator.replan(force=True)
                self.init_crossing()	

        elif self.mode == Mode.CROSS:
            # Crossing an intersection
            self.navigator.navigate()
            if self.has_crossed():
                self.mode = Mode.NAV

        else:
            raise Exception("This mode is not supported: {}".format(str(self.mode)))


    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()


if __name__ == '__main__':
    sup = Supervisor()
    sup.run()
