import numpy as np
from driver import Driver
from pr3_utils import *
import os

if __name__ == '__main__':
    DIRECTORY = "../data/"
    NUM = "00"
    DATASET = f"dataset{NUM}/dataset{NUM}.npy"
    FILE = os.path.join(DIRECTORY, DATASET)

    VELOCITY, ANGULAR, TIMESTAMPS, FEATURES, KL, KR, EXTL, EXTR = load_data(FILE)
	# (a) IMU Localization via EKF Prediction
    theDriver = Driver(VELOCITY, ANGULAR, TIMESTAMPS, FEATURES, KL, KR, EXTL, EXTR)
    #POSE = theDriver.predict()
    #visualize_trajectory_2d(POSE, path_name="IMU Trajectory", show_ori=True)

	# (b) Landmark Mapping via EKF Update
    #MEAN, COVARIANCE = theDriver.update()
    #visualize_landmarks(MEAN,POSE)
	# (c) Visual-Inertial SLAM
    CORRECTED_POSE, LANDMARKS = theDriver.vSLAM()

    visualize_trajectory_2d(CORRECTED_POSE, path_name="Corrected Trajectory", show_ori=True)

	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


