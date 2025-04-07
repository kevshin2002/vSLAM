import numpy as np
from scipy.spatial.transform import Rotation
from pr3_utils import *
import scipy
from scipy import sparse
import matplotlib.pyplot as plt

class Driver:
    def __init__(self, V, W, T, F, KL, KR, EXT_L, EXT_R):
        self.VELOCITY = V
        self.ANGULAR = W
        self.T = T
        self.F = F
        self.KL = KL
        self.KR = KR
        self.EXT_L = EXT_L
        self.EXT_R = EXT_R
        
        self.COV_PRIOR = np.eye(6)
        self.MOTION_NOISE = np.eye(6)
        self.OBSERVATION_NOISE = np.eye(4) * 2

    def predict(self):
        self.POSE = np.zeros((len(self.T), 4, 4))
        self.POSE[0] = np.eye(4)

        for theRun in range(1, len(self.T)):
            DT = self.T[theRun] - self.T[theRun - 1]
            self.POSE[theRun], self.COV_PRIOR = self._EKF(self.POSE[theRun - 1], self.VELOCITY[theRun - 1], self.ANGULAR[theRun - 1], DT, self.COV_PRIOR, self.MOTION_NOISE)
        return self.POSE

    def _EKF(self, POSE, VELOCITY, ANGULAR, DT, PRIORI, NOISE):
        ANGULAR_SKEW = axangle2skew(ANGULAR)
        ANGULAR_VELOCITY = axangle2skew(VELOCITY)

        MU = np.block([
                      [ANGULAR_SKEW, ANGULAR_VELOCITY],
                      [np.zeros((3, 3)), ANGULAR_SKEW]
                      ])
        
        MOTION = np.concatenate([VELOCITY * DT, ANGULAR * DT])
        UPDATE_POSE = POSE @ axangle2pose(MOTION)
        
        EXP_MU = exp_ad_twist(MU * -DT)
        UPDATE_COVARIANCE = EXP_MU @ PRIORI @ EXP_MU.T + DT * NOISE

        return UPDATE_POSE, UPDATE_COVARIANCE

    def update(self):
        _, NUM_LANDMARKS, NUM_FRAMES = self.F.shape
        MU_COORD = np.zeros(3 * NUM_LANDMARKS)
        MU = np.zeros(4 * NUM_LANDMARKS)
        COVARIANCE = scipy.sparse.lil_matrix((3 * NUM_LANDMARKS, 3 * NUM_LANDMARKS))
        COVARIANCE.setdiag(np.ones(3 * NUM_LANDMARKS))

        LANDMARKS = np.zeros(NUM_LANDMARKS, dtype=bool)
        LANDMARK_CLOSE = np.zeros(NUM_LANDMARKS, dtype=bool)
        LANDMARKS_DEPTH = np.zeros(NUM_LANDMARKS, dtype=bool)

        STEREO = np.linalg.norm(self.EXT_L[:3, 3] - self.EXT_R[:3, 3])

        Ks = np.array([
                        [self.KR[0, 0], 0, self.KR[0, 2], 0],
                        [0, self.KR[1, 1], self.KR[1, 2], 0],
                        [self.KR[0, 0], 0, self.KR[0, 2], -self.KR[0, 0] * STEREO],
                        [0, self.KR[1, 1], self.KR[1, 2], 0]
                    ])
       
        for theFrame in range(NUM_FRAMES):
            if theFrame % 100 == 0:
                print(f"Processing {theFrame}/{NUM_FRAMES}")
            LANDMARK_INDICES = np.where(self.F[0, :, theFrame] != -1)[0]
            if not LANDMARK_INDICES.size:
                continue

            for theIndex in LANDMARK_INDICES:
                if not LANDMARKS[theIndex]:
                    LANDMARK_COORD = self._triangulate(self.F[:, theIndex, theFrame], self.POSE[theFrame], self.KR, STEREO) 
                    TRAJECTORY_COORD = self.POSE[:, :2, 3]
                    DISTANCE = np.linalg.norm(TRAJECTORY_COORD - LANDMARK_COORD[:2], axis=1)
                    MIN_DIST = np.min(DISTANCE)
                    LANDMARKS[theIndex] = True

                    if 0.1 < LANDMARK_COORD[2] < 10.0:
                        LANDMARKS_DEPTH[theIndex] = True

                    if MIN_DIST <= 20:
                        MU_COORD[3*theIndex:3*theIndex+3] = LANDMARK_COORD[:3]
                        MU[4*theIndex:4*theIndex+4] = np.append(LANDMARK_COORD[:3], 1)
                        LANDMARK_CLOSE[theIndex] = True

            VALID = [theIndex for theIndex in LANDMARK_INDICES if LANDMARKS[theIndex] and LANDMARK_CLOSE[theIndex] and LANDMARKS_DEPTH[theIndex]]
            if not VALID:
                continue

            Z = self.F[:, VALID, theFrame].T.reshape(-1)
            PREDICTION, J = self._observe(MU, VALID, self.POSE[theFrame], Ks, STEREO)
            R = scipy.sparse.block_diag([self.OBSERVATION_NOISE for _ in range(len(VALID))]).tocsr()

            INNOVATION = J @ COVARIANCE @ J.T + R
            INNOVATION += scipy.sparse.eye(INNOVATION.shape[0]) * 1e-6
            
            GAIN = COVARIANCE @ J.T @ scipy.sparse.linalg.inv(INNOVATION)
            MU_COORD += GAIN @ (Z - PREDICTION)
            COVARIANCE = (scipy.sparse.eye(3 * NUM_LANDMARKS) - GAIN @ J) @ COVARIANCE
        
        return MU_COORD, COVARIANCE

   
    def vSLAM(self, motion_noise=1.0, obs_noise=2.0):
        self.MOTION_NOISE = np.eye(6) * motion_noise
        self.OBSERVATION_NOISE = np.eye(4) * obs_noise

        imu_trajectory = self.predict()
        corrected_poses = np.copy(imu_trajectory)
        state_cov = self.COV_PRIOR
        landmarks, landmark_cov = self.update()

        for t in range(1, len(self.T)):
            _, J = self._observe(landmarks, np.arange(len(landmarks)), corrected_poses[t], self.KR, np.linalg.norm(self.EXT_L[:3, 3] - self.EXT_R[:3, 3]))
            R = sparse.block_diag([self.OBSERVATION_NOISE] * (len(landmarks) // 4)).tocsc()

            innovation = J @ state_cov @ J.T + R
            innovation += sparse.eye(innovation.shape[0]) * 1e-6
            kalman_gain = state_cov @ J.T @ sparse.linalg.inv(innovation)

            observation_error = landmarks - self.F[:, :, t].flatten()
            correction_vector = kalman_gain @ observation_error
            delta_pose = exp_ad_twist(correction_vector)

 
            corrected_poses[t] = corrected_poses[t] @ delta_pose
            state_cov = (np.eye(6) - kalman_gain @ J) @ state_cov

        return corrected_poses, landmarks


    def _triangulate(self, FEATURES, POSE, K, STEREO): 
        UL, VL, UR, VR = FEATURES
        K_inv = np.linalg.inv(K)
    
        PL = K_inv @ np.array([UL, VL, 1])
        PR = K_inv @ np.array([UR, VR, 1])
        XL, YL = PL[0], PL[1]
        XR, YR = PR[0], PR[1]
       
        DISPARITY = XL - XR
        if abs(DISPARITY) < 0.0001:  
            DISPARITY = 0.0001 if DISPARITY >= 0 else -0.0001
    
        Z = STEREO / DISPARITY
        if Z > 10.0:
            Z = 10.0
        
        M = np.array([XL * Z, YL * Z, Z, 1])
        return POSE @ self.EXT_L @ M

    def _observe(self, MEAN, INDICES, POSE, Ks, STEREO):
        num_landmarks = len(MEAN) // 4
        num_visible = len(INDICES)
    
        PREDICTIONS = np.zeros(4 * num_visible)
        row_indices, col_indices, values = [], [], []
    
        wTi_inv = np.linalg.inv(POSE)
        projection_matrix = np.hstack([np.eye(3), np.zeros((3, 1))])
    
        for idx, landmark_id in zip(range(num_visible), INDICES):
            landmark_position = MEAN[4 * landmark_id: 4 * landmark_id + 4]
        
            transformed_point = self.EXT_L @ wTi_inv @ landmark_position
            projected_point = projection(transformed_point)
        
            PREDICTIONS[4 * idx: 4 * idx + 4] = Ks @ projected_point
        
            proj_jacobian = projectionJacobian(transformed_point.reshape(1, 4))[0]
            H = Ks @ proj_jacobian @ self.EXT_L @ wTi_inv @ projection_matrix.T
            
            row_grid, col_grid = np.meshgrid(range(4), range(3), indexing='ij')
            row_indices.extend((4 * idx + row_grid).flatten())
            col_indices.extend((3 * landmark_id + col_grid).flatten())
            values.extend(H.flatten())
    
        J = sparse.csr_matrix((values, (row_indices, col_indices)), shape=(4 * num_visible, 3 * num_landmarks))
    
        return PREDICTIONS, J
