#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:36:42 2020
@author: gsuveer
"""


import rospy
import actionlib
import hector_uav_msgs
from hector_uav_msgs.msg import PoseAction
from visualization_msgs.msg import Marker
from hector_uav_msgs.msg import PoseActionGoal

z = 5
exploration_waypoints = [[5,5,z],[5,-5,z],[-5,-5,z],[-5,5,z]]
waypoint_index        = 0 
    

if __name__ == '__main__':
  rospy.init_node('exploration_node')
  client = actionlib.SimpleActionClient('/action/pose', PoseAction)
  rospy.loginfo("Waiting for Server")
  client.wait_for_server()
  rospy.loginfo("Server Started")
  pub = rospy.Publisher('/action/pose/goal', PoseActionGoal, queue_size=10)
  while not rospy.is_shutdown():
    try:
        expected_vis_marker = rospy.wait_for_message("/visualization_marker", Marker, timeout=20)
        rospy.loginfo("Marker found, Starting to track")
        rospy.sleep(5)

    except:
        goal_pose = PoseActionGoal()
        rospy.loginfo("Marker not found, Continuing Exploration")
        goal_pose.header.seq= waypoint_index
        goal_pose.header.frame_id= 'world'
        goal_pose.goal.target_pose.header.frame_id = 'world'
        goal_pose.goal.target_pose.pose.position.x = exploration_waypoints[waypoint_index][0]
        goal_pose.goal.target_pose.pose.position.y = exploration_waypoints[waypoint_index][1]
        goal_pose.goal.target_pose.pose.position.z = exploration_waypoints[waypoint_index][2]
        waypoint_index = (waypoint_index + 1)%4 

        #rospy.loginfo(goal_pose)
        #rospy.loginfo(goal_pose.goal_id.stamp)
        pub.publish(goal_pose)
        rospy.sleep(5)
        
        #client.send_goal(goal_pose)
        #client.wait_for_result()
