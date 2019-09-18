#!/usr/bin/env python

# Copyright (c) 2019, HIRO Group at the University of Colorado Boulder.

"""
Sawyer SDK Inverse Kinematics Position Control
"""
import os
import sys
import copy
import csv
import math
import numpy as np
import pickle
import dill
import pydmps
import time
from datetime import datetime

import rospy
import rospkg

import tf
from gazebo_msgs.msg import ModelState, ContactsState
from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
    SetModelState,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from ros_bayesian_optimization import RosBayesianOptimization
from src.sampling import RandomSampling, GeneticAlgorithmSamling, BanditReinforcementLearning 

import intera_interface
from sawyer_pykdl import sawyer_kinematics

SAWYER_HEIGHT_OFFSET = 0.93 # writeen in $(find sawyer_gazebo)/launch/saywer_world.launch

def turn_vector_by_quaternion(v, q):
    v1 = tf.transformations.unit_vector(v)
    q2 = list(v1)
    q2.append(0.0)
    return tf.transformations.quaternion_multiply(
            tf.transformations.quaternion_multiply(q, q2),
            tf.transformations.quaternion_conjugate(q))[:3]

def QuaternionNorm(Q_raw):
    qx_temp,qy_temp,qz_temp,qw_temp = Q_raw[0:4]
    qnorm = math.sqrt(qx_temp*qx_temp + qy_temp*qy_temp + qz_temp*qz_temp + qw_temp*qw_temp)
    qx_ = qx_temp/qnorm
    qy_ = qy_temp/qnorm
    qz_ = qz_temp/qnorm
    qw_ = qw_temp/qnorm
    Q_normed_ = np.array([qx_, qy_, qz_, qw_])
    return Q_normed_

def Quaternion2EulerXYZ(Q_raw):
    Q_list = [Q_raw.x, Q_raw.y, Q_raw.z, Q_raw.w]
    Q_normed = QuaternionNorm(Q_list)
    qx_ = Q_normed[0]
    qy_ = Q_normed[1]
    qz_ = Q_normed[2]
    qw_ = Q_normed[3]

    tx_ = math.atan2((2 * qw_ * qx_ - 2 * qy_ * qz_), (qw_ * qw_ - qx_ * qx_ - qy_ * qy_ + qz_ * qz_))
    ty_ = math.asin(2 * qw_ * qy_ + 2 * qx_ * qz_)
    tz_ = math.atan2((2 * qw_ * qz_ - 2 * qx_ * qy_), (qw_ * qw_ + qx_ * qx_ - qy_ * qy_ - qz_ * qz_))
    EulerXYZ_ = np.array([tx_,ty_,tz_])
    return EulerXYZ_

def EulerXYZ2Quaternion(EulerXYZ_):
    tx_, ty_, tz_ = EulerXYZ_[0:3]
    sx = math.sin(0.5 * tx_)
    cx = math.cos(0.5 * tx_)
    sy = math.sin(0.5 * ty_)
    cy = math.cos(0.5 * ty_)
    sz = math.sin(0.5 * tz_)
    cz = math.cos(0.5 * tz_)

    qx_ = sx * cy * cz + cx * sy * sz
    qy_ = -sx * cy * sz + cx * sy * cz
    qz_ = sx * sy * cz + cx * cy * sz
    qw_ = -sx * sy * sz + cx * cy * cz
    Q_ = np.array([qx_, qy_, qz_, qw_])
    return Q_

def point_to_numpy3d(pose):
    return np.array([pose.x, pose.y, pose.z])

class ArmControl(object):
    def __init__(self, limb="right", tip_name="right_gripper_tip", hover_distance=0.1):
        self._limb_name = limb
        self._tip_name = tip_name
        self._limb = intera_interface.Limb(limb)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        self._joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        self._kin = sawyer_kinematics(limb)
        self.prev_pos = np.zeros(3)
        self.prev_time = None

    def init_pose(self, return_pos_offset=0.5):
        q = EulerXYZ2Quaternion(np.array([180, 90, 180])*np.pi/180)
        # move to position
        self.endpoint_init_pose = Pose(
            position = Point(x=0.9, y=0.08, z=-0.129),
            orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        )
        self.set_position(self.endpoint_init_pose)
        # get endpoint orientation
        self.endpoint_init_quat = self._limb.endpoint_pose()["orientation"]
        self.tmp_return_pos = self.endpoint_init_pose
        self.tmp_return_pos.position.z += return_pos_offset

    def _guarded_move_to_joint_position(self, joint_angles, timeout=5.0):
        if rospy.is_shutdown():
            return
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles, timeout=timeout)
            return True
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")
            return False

    def set_position(self, pose):
        # modify speed according to my simulation
        joint_angles = self._limb.ik_request(pose, self._tip_name)
        self._limb.set_joint_position_speed(0.001)
        success = self._guarded_move_to_joint_position(joint_angles)
        if not success:
            pos = np.array([pose.position.x, pose.position.y, pose.position.z]) 
            self.control(pos)

    def control(self, target):
        # Get current end point position and velocity
        endpoint_pos = self._limb.endpoint_pose()["position"]
        endpoint_pos = point_to_numpy3d(endpoint_pos)
        endpoint_vel = self._limb.endpoint_velocity()["linear"]
        endpoint_vel = point_to_numpy3d(endpoint_vel)

        # PD controller computes required acceleration
        kp = 50
        kd = np.sqrt(50)
        x_des = kp * (target - endpoint_pos) - kd * endpoint_vel

        # Inertia Matrix of Robotic Arms (also known as mass matrix)
        Mq = self._kin.inertia()
        Mq_inv = np.linalg.inv(Mq)

        # Jacobian (= dx/dq)
        Jac = self._kin.jacobian()[:3, :]
        Jac_T = np.transpose(Jac)

        # Inertia Matrix in operational space
        MxEE = np.linalg.inv(np.dot(Jac, np.dot(Mq_inv, Jac_T)))
        # Required Torque
        U = np.dot(Jac_T, np.dot(MxEE, x_des.reshape(3,1)))

        # Dict (joint_names : torques)
        torques = {name : u for name, u in zip(self._joint_names, U)}

        # send command
        self._limb.set_joint_torques(torques)

    def reset(self):
        self.set_position(self.tmp_return_pos)
        self.set_position(self.endpoint_init_pose)
        self.prev_pos = np.zeros(3)

    def get_curr_pose(self):
        endpoint_pose = self._limb.endpoint_pose()
        self.endpoint_pos = point_to_numpy3d(endpoint_pose["position"])
        return endpoint_pose

    def get_init_quat(self):
        return self.endpoint_init_quat

    def is_goal(self, goal, thres=0.03):
        endpoint_pos = self._limb.endpoint_pose()["position"]
        self.endpoint_pos = point_to_numpy3d(endpoint_pos)
        if math.sqrt(math.pow(self.endpoint_pos[0]-goal[0], 2) +
                     math.pow(self.endpoint_pos[1]-goal[1], 2) +
                     math.pow(self.endpoint_pos[2]-goal[2], 2)) < thres:
            return True
        return False

    def is_stuck(self, time_thres=3, pos_thres=0.3):
        if self.prev_time is None:
            self.prev_time = time.time()
            self.prev_pos = self.endpoint_pos

        curr_time = time.time()
        if curr_time-self.prev_time > time_thres:
            if math.sqrt(math.pow(self.endpoint_pos[0]-self.prev_pos[0], 2) +
                         math.pow(self.endpoint_pos[1]-self.prev_pos[1], 2) +
                         math.pow(self.endpoint_pos[2]-self.prev_pos[2], 2)) < pos_thres:
                return True
            self.prev_time = curr_time
            self.prev_pos = self.endpoint_pos

        return False

    def is_running(self, goal):
        return not (self.is_goal(goal) or self.is_stuck())

class Environment(object):
    def __init__(self):
        self._init()

    def _init(self):
        self.load_ball()
        rospy.on_shutdown(self.delete_ball)

    def load_ball(self, ball_pose=Pose(position=Point(x=0.88, y=0.08, z=0.85)),
                   ball_reference_frame="world"):
        model_path = rospkg.RosPack().get_path('sawyer_bopt')+"/models/"
        # Load Ball SDF
        ball_xml = ''
        with open (model_path + "ball_for_cup/model.sdf", "r") as ball_file:
            ball_xml = ball_file.read().replace('\n', '')
            rospy.loginfo(ball_xml)
        # Spawn Ball SDF
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            resp_sdf = spawn_sdf("ball_for_cup", ball_xml, "/",
                                ball_pose, ball_reference_frame)
        except rospy.ServiceException, e:
            rospy.logerr("Spawn SDF service call failed: {0}".format(e))

    def delete_ball(self):
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            resp_delete = delete_model('ball_for_cup')
        except rospy.ServiceException, e:
            print("Delete Model service call failed: {0}".format(e))

    def update_model_state(self, model_name, pose, reference_frame="world"):
        rospy.wait_for_service('/gazebo/set_model_state')

        state_msg = ModelState()
        state_msg.model_name = model_name
        state_msg.pose = pose
        state_msg.reference_frame = reference_frame

        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except rospy.ServiceException, e:
            rospy.loginfo("Service call failed: %s" % e)

    def _calc_ball_pose(self, endpoint_pose,
                   endpoint_init_quat,
                   object_offset=np.array([0, 0, 0.03]),
                   coordinate_offset=SAWYER_HEIGHT_OFFSET):
        p = endpoint_pose["position"]
        q = endpoint_pose["orientation"]

        # Calculate Object Offset Position in Global Coordinate
        offset_global = np.linalg.norm(object_offset)*turn_vector_by_quaternion(object_offset, q)

        # update ball's orientation
        o = tf.transformations.quaternion_multiply(q, tf.transformations.quaternion_conjugate(endpoint_init_quat))
        ball_orientation = Quaternion(x=o[0], y=o[1], z=o[2], w=o[3])
        # update ball's position
        ball_position = Point(
            x=p.x+offset_global[0],
            y=p.y+offset_global[1],
            z=p.z+offset_global[2]+coordinate_offset
        )
        # update ball's pose (integrate position & orientation)
        return Pose(position=ball_position, orientation=ball_orientation)

    def update_ball_pose(self, endpoint_pose, endpoint_init_quat):
        pose = self._calc_ball_pose(
            endpoint_pose = endpoint_pose,
            endpoint_init_quat = endpoint_init_quat
        )
        self.update_model_state('ball_for_cup', pose)

class GoalSearcher(object):
    def __init__(self, bounds, search_type='bopt', save=True):
        self._search_type = search_type

        if self._search_type is 'bopt':
            self.searcher = RosBayesianOptimization(domain=bounds, acquisition_type='LCB', acquisition_weight=10)
            #self.searcher = RosBayesianOptimization(domain=bounds, acquisition_jitter=0.49)
        if self._search_type is 'random':
            self.searcher = RandomSampling(bounds=bounds)
        if self._search_type is 'ga':
            self.searcher = GeneticAlgorithmSamling(bounds=bounds)
        if self._search_type is 'rl':
            self.searcher = BanditReinforcementLearning(bounds=bounds)

        self.save = save
        if self.save:
            self.model_path = rospkg.RosPack().get_path('sawyer_bopt') + "/logs/" + datetime.now().strftime("%Y_%m_%d_%H_%M")
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def compute_first_sample_point(self):
        return self.searcher.init()

    def update(self, reward):
        self.searcher.update(reward)

    def compute_next_sample_point(self):
        return self.searcher.step()

    def save_ith(self, ith_iteration):
        # save bopt parameters
        if self.save:
            if self._search_type is 'bopt':
                path = os.path.join(self.model_path, 'bopt_%i.dill'%(ith_iteration))
                with open(path, mode='wb') as f:
                    dill.dump(self.searcher, f)

    def get_best_reward(self):
        return self.searcher.Y_best 

    def get_best_position(self):
        return self.searcher.x_opt

    def get_all_rewards(self):
        return self.searcher.Y

    def get_all_positions(self):
        return self.searcher.X

class MotionPlanner(object):
    def __init__(self):
        traj = self.load_trajectory()
        self.dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=100, ay=np.ones(3)*10.0)
        self.dmp.imitate_path(y_des=traj, plot=False)

    def load_trajectory(self):
        path = os.path.join(rospkg.RosPack().get_path('sawyer_bopt'), 'assets/traj.pickle')
        with open(path, 'rb') as f:
            traj = pickle.load(f)
        return traj

    def set_goal(self, goal):
        self.dmp.goal = goal

    def step(self, tau=3):
        return self.dmp.step(tau=tau)

    def reset_state(self):
        self.dmp.reset_state()
