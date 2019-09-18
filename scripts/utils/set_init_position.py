#!/usr/bin/env python

# Copyright (c) 2019, HIRO Group at the University of Colorado Boulder.

"""
Sawyer SDK Inverse Kinematics Position Control 
"""
import argparse
import struct
import sys
import copy
import math
import numpy as np

import rospy
import rospkg

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

import intera_interface
from position_control import PositionControl, EulerXYZ2Quaternion


def main():
    rospy.init_node("set_init_position")

    # Manipulate to given position 
    limb = 'right'
    pos_con = PositionControl(limb)
    
    # Initial Position & Orientaion 
    position = Point(x=0.9, y=0.08, z=-0.129)
    q = EulerXYZ2Quaternion(np.array([180, 90, 180])*np.pi/180)
    orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    pose = Pose(
        #position=Point(x=0.45, y=0.155, z=-0.129),
        position=position,
        orientation=orientation
    )

    while not rospy.is_shutdown():
        pos_con.set_position(pose)
    return 0
        

if __name__ == '__main__':
    sys.exit(main())