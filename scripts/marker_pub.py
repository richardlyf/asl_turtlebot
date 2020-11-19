#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D
import numpy as np
from utils import wrapToPi


class MarkerTracker:

    def __init__(self, target_names):
        self.target_names = target_names
        self.marker_publishers = []
        for i in range(len(target_names)):
            target_name = target_names[i]
            self.marker_publishers.append(rospy.Publisher('marker_' + target_name, Marker, queue_size=10))
        self.marker_locations = [(None, None) for i in range(len(target_names))]
        # Heading theta from vendor to the robot, where the robot observed the vendor
        self.heading = [None for i in range(len(target_names))]


    def place_marker(self, target_name, target_pos, theta):
        if target_name not in self.target_names:
            return

        x, y = target_pos
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()
        marker.id = self.target_names.index(target_name)

        marker.type = 2 # sphere

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0

        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        
        self.marker_locations[marker.id] = target_pos
        self.heading[marker.id] = theta
        self.marker_publishers[marker.id].publish(marker)

    def get_goal_pose(self, target_name, distance):
        """
        Given the target and distance from the vendor, computes the 
        position the robot should travel to.
        """
        if target_name not in self.target_names:
            return
        marker_id = self.target_names.index(target_name)
        theta = self.heading[marker_id]
        if theta is None:
            return
        x, y = self.marker_locations[marker_id]
        x += np.cos(theta) * distance
        y += np.sin(theta) * distance
        robot_heading = wrapToPi(theta + np.pi)
        return (x, y, robot_heading)


    def get_marker_pos(self, target_name):
        if target_name not in self.target_names:
            return
        marker_id = self.target_names.index(target_name)
        return self.marker_locations[marker_id]


def publish_marker(target_name, pos, unique_id, color):
    publisher = rospy.Publisher('marker_' + target_name, Marker, queue_size=10)
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time()
    marker.id = unique_id

    marker.type = 2 # sphere

    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = 0

    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1

    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    
    publisher.publish(marker)


