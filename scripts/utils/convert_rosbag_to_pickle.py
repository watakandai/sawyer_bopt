#! /usr/bin/env python
import os
import csv
import numpy as np
import pickle

import pydmps
import pydmps.dmp_discrete

import rosbag
import rospy
import rospkg

from intera_core_msgs.msg import EndpointState

############################################
############################################
# Global Variables to set
BAG_FILENAME = 'traj.bag'
OUTPUT_FILENAME = 'traj.pickle'
# 
############################################
############################################

pkg_path = rospkg.RosPack().get_path('sawyer_bopt')
log_path = os.path.join(pkg_path, "assets")
bag_path = os.path.join(log_path, BAG_FILENAME)

print('reading bag file ... ' + bag_path)
bag = rosbag.Bag(bag_path)

T = []
X = []
Y = []
Z = []
QX = []
QY = []
QZ = [] 
QW = []
for topic, msg, t in bag.read_messages(topics={'/robot/limb/right/endpoint_state'}):
    print(msg.pose.position, msg.pose.orientation)
    T.append(msg.header.stamp)
    X.append(msg.pose.position.x)
    Y.append(msg.pose.position.y)
    Z.append(msg.pose.position.z)
    QX.append(msg.pose.orientation.x)
    QY.append(msg.pose.orientation.y)
    QZ.append(msg.pose.orientation.z)
    QW.append(msg.pose.orientation.w)

traj = np.array([X, Y, Z])
with open(os.path.join(log_path, OUTPUT_FILENAME), 'wb') as f:
    pickle.dump(traj, f)

dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=500, ay=np.ones(3)*10.0)
dmp.imitate_path(y_des=traj, plot=True)

bag.close()
