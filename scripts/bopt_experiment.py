#!/usr/bin/env python

# Copyright (c) 2019, HIRO Group at the University of Colorado Boulder.

"""
Sawyer SDK Inverse Kinematics Position Control
"""
import argparse
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
from gazebo_msgs.msg import ContactsState

from src.core import (
    ArmControl, 
    Environment, 
    MotionPlanner, 
    GoalSearcher, 
)

WAIT_FOR_ROLLING_BALL = 2
score = INITIAL_SCORE = -20

def score_callback(msg, score_):
    global score
    if msg.states and score == INITIAL_SCORE:
        score = score_

def run_method_until_reaches_goal(env, r, search_type):
    global score

    # RosBayesianOptimization object
    bounds = [{'name':'x', 'type':'continuous', 'domain':(-0.7,  0.7)},
              {'name':'y', 'type':'continuous', 'domain':(-1.1, -0.4)},
              {'name':'z', 'type':'continuous', 'domain':(0.3,  0.5)}]
    searcher = GoalSearcher(bounds=bounds, search_type=search_type, save=False)
    goal = searcher.compute_first_sample_point()

    # DMP Motion Planner
    planner = MotionPlanner()
    planner.set_goal(goal)

    # Controller for arms
    controller = ArmControl()
    controller.init_pose()

    # initialize instances
    score = INITIAL_SCORE
    start_time = None
    i = 1

    while not rospy.is_shutdown():
        # Get Trajectory and update Robot Arm Position
        if controller.is_running(goal):
            pos, _, _, = planner.step(tau=3.0)
            controller.control(pos)
            # Get Endeffector's Pose
            endpoint_pose = controller.get_curr_pose()
            endpoint_init_quat = controller.get_init_quat()
            # update ball's pose according to the endeffector's pose
            env.update_ball_pose(endpoint_pose, endpoint_init_quat)
        else:
            # wait untill ball is rolled ...
            if start_time is None:
                start_time = time.time()

            # After few seconds, record the received score and update bopt model
            elif time.time() - start_time > WAIT_FOR_ROLLING_BALL:
                # update model
                searcher.update(score)
                rospy.logerr('%ith SCORE:     %.2f'%(i, score))
                # calc new goal
                goal = searcher.compute_next_sample_point()
                # set a new goal
                planner.reset_state()
                planner.set_goal(goal)

                # initialize env, contrller, score, ...
                env.delete_ball()
                controller.reset()
                env.load_ball()

                # save bopt
                searcher.save_ith(i)

                if score == 1000:
                # if i == 3:
                    return i 
                score = INITIAL_SCORE
                start_time = None
                i += 1

        r.sleep()

def experiment_comparative_methods():
    # init node
    rospy.init_node("ros_bopt_experiment")
    rospy.logerr("EXPERIMENT FOR COMPARING METHODS")
    rospy.Subscriber("/mattress/score0", ContactsState, score_callback, 0, queue_size=10)
    rospy.Subscriber("/mattress/score10", ContactsState, score_callback, 10, queue_size=10)
    rospy.Subscriber("/mattress/score100", ContactsState, score_callback, 1000, queue_size=10)
    rospy.Subscriber("/mattress/score5", ContactsState, score_callback, 5, queue_size=10)
    rospy.Subscriber("/mattress/score_10", ContactsState, score_callback, -10, queue_size=10)

    # controls over environment
    env = Environment()
    r = rospy.Rate(30)

    EXPERIMENT_ITERATIONS = 100
    iterations = []

    csv_path = rospkg.RosPack().get_path('sawyer_bopt') + "/logs/minimum_iterations_bopt.csv"
    search_types = ['bopt', 'random', 'rl']

    for search_type in search_types:
        for i in range(EXPERIMENT_ITERATIONS):
            rospy.logerr('-'*20 + '%i ith Experiment'%(i+1) + '-'*20)
            iteration = run_method_until_reaches_goal(env, r, search_type)
            iterations.append(iteration)
            rospy.logerr('%i ITERATIONS'%(iteration))

            with open(csv_path, 'a') as f:
                writer = csv.writer(f)
                date_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
                writer.writerow([date_time, search_type, i+1, iteration])

        iters = np.array(iterations)
        rospy.logerr('-'*20 + 'ITERATIONS' + '-'*20)
        rospy.logerr(iters)
        rospy.logerr('-'*20 + 'Mean' + '-'*20)
        rospy.logerr(np.mean(iters))
        rospy.logerr('-'*20 + 'STD' + '-'*20)
        rospy.logerr(np.std(iters))

    return 0

def run_convergence_simulation(env, r):
    global score

    # RosBayesianOptimization object
    bounds = [{'name':'x', 'type':'continuous', 'domain':(-0.7,  0.7)},
              {'name':'y', 'type':'continuous', 'domain':(-1.1, -0.4)},
              {'name':'z', 'type':'continuous', 'domain':(0.3,  0.5)}]
    searcher = GoalSearcher(bounds=bounds)
    goal = searcher.compute_first_sample_point()

    # DMP Motion Planner
    planner = MotionPlanner()
    planner.set_goal(goal)

    # Controller for arms
    controller = ArmControl()
    controller.init_pose()

    # initialize instances
    score = INITIAL_SCORE
    start_time = None
    i = 1

    while not rospy.is_shutdown():
        # Get Trajectory and update Robot Arm Position
        if controller.is_running(goal):
            pos, _, _, = planner.step(tau=3.0)
            controller.control(pos)
            # Get Endeffector's Pose
            endpoint_pose = controller.get_curr_pose()
            endpoint_init_quat = controller.get_init_quat()
            # update ball's pose according to the endeffector's pose
            env.update_ball_pose(endpoint_pose, endpoint_init_quat)
        else:
            # wait untill ball is rolled ...
            if start_time is None:
                start_time = time.time()

            # After few seconds, record the received score and update bopt model
            elif time.time() - start_time > WAIT_FOR_ROLLING_BALL:
                # update model
                searcher.update(score)
                m_diff, v_diff = searcher.searcher.mean_var_difference()
                rospy.logerr('%ith SCORE:     %.2f'%(i, score))
                # calc new goal
                goal = searcher.compute_next_sample_point()
                # set a new goal
                planner.reset_state()
                planner.set_goal(goal)

                # initialize env, contrller, score, ...
                env.delete_ball()
                controller.reset()
                env.load_ball()

                # save bopt
                searcher.save_ith(i)

                rospy.logerr('DIFF')
                #max_m = np.max(np.abs(m_diff.reshape(-1,1)))
                #min_m = np.min(np.abs(m_diff.reshape(-1,1)))
                diff = np.mean(np.abs(m_diff.reshape(-1,1)))
                rospy.logerr(diff)
                #max_v = np.max(np.abs(v_diff.reshape(-1,1)))
                #min_v = np.min(np.abs(v_diff.reshape(-1,1)))
                diff = np.mean(np.abs(v_diff.reshape(-1,1)))
                rospy.logerr(diff)

                #if diff < 0.001:
                #    return i

                score = INITIAL_SCORE
                start_time = None
                i += 1

        r.sleep()

def experiment_convergence():
    # init node
    rospy.init_node("ros_bopt_experiment")
    rospy.logerr("EXPERIMENT FOR CONVERGENCE")
    #rospy.Subscriber("/mattress/score0", ContactsState, score_callback, 0, queue_size=10)
    rospy.Subscriber("/mattress/score10", ContactsState, score_callback, 10, queue_size=10)
    rospy.Subscriber("/mattress/score100", ContactsState, score_callback, 1000, queue_size=10)
    rospy.Subscriber("/mattress/score5", ContactsState, score_callback, 5, queue_size=10)
    rospy.Subscriber("/mattress/score_10", ContactsState, score_callback, -10, queue_size=10)

    # controls over environment
    env = Environment()
    r = rospy.Rate(30)

    EXPERIMENT_ITERATIONS = 100
    iterations = []
    
    csv_path = rospkg.RosPack().get_path('sawyer_bopt') + "/logs/convergence.csv"

    for i in range(EXPERIMENT_ITERATIONS):
        rospy.logerr('-'*20 + '%i th Experiment'%(i+1) + '-'*20)
        iteration = run_convergence_simulation(env, r)
        iterations.append(iteration)
        rospy.logerr('%i ITERATIONS'%(iteration))

        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([i+1, iteration])
        
        iters = np.array(iterations)
        rospy.logerr('-'*20 + 'ITERATIONS' + '-'*20)
        rospy.logerr(iters)
        rospy.logerr('-'*20 + 'Mean' + '-'*20)
        rospy.logerr(np.mean(iters))
        rospy.logerr('-'*20 + 'STD' + '-'*20)
        rospy.logerr(np.std(iters))

    return 0

def parse_args():
    parser = argparse.ArgumentParser(description="Options Below ....")
    parser.add_argument('--experiment', '-e', type=str, default='comparison', help='Choose Experiment from ["comparison", "convergence"]')
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unknown = parse_args()
    if args.experiment == 'comparison':
        sys.exit(experiment_comparative_methods())
    if args.experiment == 'convergence':
        sys.exit(experiment_convergence())
