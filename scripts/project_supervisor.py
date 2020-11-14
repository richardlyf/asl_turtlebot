#!/usr/bin/env python

from enum import Enum

import rospy
from asl_turtlebot.msg import DetectedObject
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from std_msgs.msg import Float32MultiArray, String
from project_navigator import Navigator
from marker_pub import MarkerTracker
import tf
import math
import numpy as np

class Mode(Enum):
    """State machine modes. Feel free to change."""
    NAV = 1
    POSE = 2
    STOP = 3
    CROSS = 4
    MANUAL = 6


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

        # Minimum distance from a stop sign to obey it
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 1)

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
        self.mode = Mode.NAV
        self.prev_mode = None  # For printing purposes

        # Navigator with planning, tracking, and pose stabilization
        self.navigator = Navigator()
        rospy.on_shutdown(self.navigator.shutdown_callback)

        # Create MarkerTracker to remember vendor locations
        self.vendor_names = ["apple", "banana", "pizza"]
        self.marker_tracker = MarkerTracker(self.vendor_names)

        ########## PUBLISHERS ##########
        # Command vel (used for idling)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        ########## SUBSCRIBERS ##########
        # Stop sign detector
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        # Vendor detectors
        for vendor_name in self.vendor_names:
            rospy.Subscriber('/detector/' + vendor_name, DetectedObject, self.vendor_detected_callback)
                

    ########## SUBSCRIBER CALLBACKS ##########
    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        dist = msg.distance

        # if close enough and in nav mode, stop
        if dist > 0 and dist < self.params.stop_min_dist and self.mode == Mode.NAV:
            self.init_stop_sign()

    def vendor_detected_callback(self, msg):
        # compute the angle of the vendor relative to the robot
        theta_left = msg.thetaleft
        theta_right = msg.thetaright
        if theta_left > math.pi:
            theta_left -= 2. * math.pi
        if theta_right > math.pi:
            theta_right -= 2. * math.pi
        avg_theta = theta_left + theta_right
        if avg_theta != 0:
            avg_theta /= 2.

        # compute position of the detected vendor
        global_theta = avg_theta + self.navigator.theta
        vendor_x = self.navigator.x + np.cos(global_theta) * msg.distance 
        vendor_y = self.navigator.y + np.sin(global_theta) * msg.distance 

        self.marker_tracker.place_marker(msg.name, (vendor_x, vendor_y), np.pi - global_theta)

    ########## STATE MACHINE ACTIONS ##########
    def stay_idle(self):
        """ sends zero velocity to stay put """
        vel_g_msg = Twist()
        self.cmd_vel_publisher.publish(vel_g_msg)

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """
        self.stop_sign_start = rospy.get_rostime()
        self.mode = Mode.STOP

    def has_stopped(self):
        """ checks if stop sign maneuver is over """
        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.params.stop_time)

    def init_crossing(self):
        """ initiates an intersection crossing maneuver """
        self.cross_start = rospy.get_rostime()
        self.mode = Mode.CROSS

    def has_crossed(self):
        """ checks if crossing maneuver is over """
        return self.mode == Mode.CROSS and \
               rospy.get_rostime() - self.cross_start > rospy.Duration.from_sec(self.params.crossing_time)


    ########## STATE MACHINE LOOP ##########
    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """

        # logs the current mode
        if self.prev_mode != self.mode:
            rospy.loginfo("Current mode: %s", self.mode)
            self.prev_mode = self.mode

        elif self.mode == Mode.NAV:
            # Moving towards a desired pose
            self.navigator.navigate()

        elif self.mode == Mode.STOP:
            # At a stop sign
            self.stay_idle()
            if self.has_stopped():
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
