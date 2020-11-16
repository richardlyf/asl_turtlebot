#!/usr/bin/env python

from enum import Enum

import rospy
from asl_turtlebot.msg import DetectedObject
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from std_msgs.msg import Float32MultiArray, String
from project_navigator import Navigator
from project_navigator import Mode as NavMode
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
        self.vendor_names = ["apple", "banana", "pizza"]
        self.marker_tracker = MarkerTracker(self.vendor_names)

        # Remember the current order
        self.order_list = None
        self.order_repeat_count = 0
        self.order_repeat_max = 10 # A given order is published 10 times
        self.vendor_dist = 0.4 # How far away to park from vendor
        self.home_pose = (self.navigator.x, self.navigator.y, self.navigator.theta)

        # Error handling
        self.retry_limit = 10
        self.error_pos = []
        self.cov = [[0.5, 0], [0, 0.5]]

        ########## PUBLISHERS ##########
        # Command vel (used for idling)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)

        ########## SUBSCRIBERS ##########
        # Stop sign detector
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        # Vendor detectors
        for vendor_name in self.vendor_names:
            rospy.Subscriber('/detector/' + vendor_name, DetectedObject, self.vendor_detected_callback)
        # Request subscriber
        rospy.Subscriber('/delivery_request', String, self.request_callback) 
                
    def publish_goal_pose(self, goal):
        """ sends the current desired pose to the navigator """
        pose_g_msg = Pose2D()
        pose_g_msg.x = goal[0]
        pose_g_msg.y = goal[1]
        pose_g_msg.theta = goal[2]
        self.nav_goal_publisher.publish(pose_g_msg)

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
        self.navigator.update_state()
        global_theta = avg_theta + self.navigator.theta
        vendor_x = self.navigator.x + np.cos(global_theta) * msg.distance 
        vendor_y = self.navigator.y + np.sin(global_theta) * msg.distance 

        self.marker_tracker.place_marker(msg.name, (vendor_x, vendor_y), np.pi - global_theta)

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
        self.order_list = msg.data.split(",")
        rospy.loginfo("Received order %s", self.order_list)
        # TODO order_list should be updated to be in pick up order
        self.order_list.append("home")

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


    ########## STATE MACHINE LOOP ##########
    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """

        # logs the current mode
        if self.prev_mode != self.mode:
            rospy.loginfo("Current mode: %s", self.mode)
            self.prev_mode = self.mode

        # Navigator reached goal 
        elif self.navigator.mode == NavMode.PARK and self.navigator.at_goal():
            self.navigator.clear_goal()
            if self.mode != Mode.MANUAL:
                self.init_wait()

        # Error position handling
        elif self.navigator.mode == NavMode.ERROR:
            # TODO: This is buggy still. A nav_cmd is published many times, and we try to
            # send a new cmd too quickly, before the subscriber queue is able to drain?
            return
            pos = (self.navigator.x_g, self.navigator.y_g)
            # If the goal that errored wasn't known, treat it as a new goal
            if pos not in self.error_pos:
                self.error_pos = [pos]
            # If the same goal was retried too many times, give up
            if len(self.error_pos) >= self.retry_limit:
                rospy.loginfo("Retried around %s %s times, giving up...", self.error_pos[0], len(self.error_pos))
                if self.mode != Mode.MANUAL:
                    self.mode = Mode.READY
                self.navigator.switch_mode(NavMode.IDLE)
            else:
                original_pos = self.error_pos[0]
                new_pos = tuple(np.random.multivariate_normal(original_pos, self.cov))
                self.error_pos.append(new_pos)
                self.publish_goal_pose((new_pos[0], new_pos[1], self.navigator.theta_g))

        elif self.mode == Mode.WAITING:
            # Waiting to load or unload food
            self.stay_idle()
            if self.is_ready():
                self.mode = Mode.READY

        elif self.mode == Mode.READY:
            # Ready to move to the next way point
            self.stay_idle()
            if self.order_list and len(self.order_list) > 0:
                next_target_name = self.order_list.pop(0)
                if next_target_name == "home":
                    rospy.loginfo("Moving to home")
                    self.publish_goal_pose(self.home_pose)
                    self.mode = Mode.NAV
                else:
                    # Find pose of vendor
                    rospy.loginfo("res: %s",self.navigator.map_resolution)
                    vendor_pose = self.marker_tracker.get_goal_pose(next_target_name, self.vendor_dist)
                    if vendor_pose is None:
                        rospy.loginfo("Position of %s unknown. Skipping...", next_target_name)
                    else:
                        rospy.loginfo("Moving to %s", next_target_name)
                        self.publish_goal_pose(vendor_pose)
                        self.mode = Mode.NAV

        elif self.mode == Mode.NAV or self.mode == Mode.MANUAL:
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
