#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""

import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import cv2
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, realpath

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/'
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'


def create_poses(start_index, end_index):
    itera = 1

    # create pose estimator
    #image_size = image.shape
    #print image_size
    image_size = [720, 1280, 3]
    pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()

    for index in range(start_index, end_index):
        image = cv2.imread(IMAGE_FILE_PATH +"run/run_"+ "{0:03}".format(index) +".jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb


        # estimation
        pose_2d, visibility, pose_3d = pose_estimator.estimate(image)
        print("iteration :"+ str(itera))


        #print (pose_3d)
        np.save("../joints/joint_list_" + str(itera), pose_3d)


        # Show 2D and 3D poses
        display_results(image, pose_2d, visibility, pose_3d, itera)
        itera += 1

    # close model
    pose_estimator.close()

def display_results(in_image, data_2d, joint_visibility, data_3d, index):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')
    plt.savefig("../result_2d_pose/run_"+str(index))
    # Show 3D poses
    for single_3D in data_3d:
        #plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D,index)
    #plt.savefig('../result_images/walking2_'+str(index))
    plt.close()

if __name__ == '__main__':
    import sys
    create_poses(1,55)
