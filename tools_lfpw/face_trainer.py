import re
import sys
from os import listdir
from os.path import isfile, join

import numpy as np


def horizontal_flip_landmarks(landmarks):
    assert len(landmarks.shape) == 2 # Assume to get 2D vector

    swap = [
        [0, 1], [2, 3], [4, 6], [5, 7], [8, 9], [10, 11], [12, 14], [13, 15],
        [16, 17], [18, 19], [22, 23], [24, 26], [25, 27], [28, 29], [30, 31],  [32, 33],
    ]

    res = landmarks.copy()
    for (a, b) in swap:
        res[a] = landmarks[b].copy()
        res[b] = landmarks[a].copy()

    return (res * np.array((-1, 1, 1)))


def print_usage():
    usage = """
Usage:
    %(cmd)s <input_dir>
            """ % {"cmd":sys.argv[0]}
    print(usage)

def translate(arr, M):
    r = np.array([
            [1.0, 0.0, arr[0]],
            [0.0, 1.0, arr[1]],
            [0.0, 0.0, 1.]
        ])
    return r.dot(M)

def rotate(angle, M):
    rad = np.deg2rad(angle)
    r = np.array([
            [np.cos(rad),  -np.sin(rad), 0.0],
            [np.sin(rad),   np.cos(rad), 0.0],
            [        0.0,           0.0, 1.0]
        ])
    return r.dot(M)

def scale(fa, M):
    r = np.array([
            [fa, 0., 0.],
            [0., fa, 0.],
            [0., 0., 1.]
        ])
    return r.dot(M)


def flip_y(M):
    return np.array([
            [-1., 0., 0.],
            [0.,  1., 0.],
            [0.,  0., 1.]
        ]).dot(M)


def emit(idx, res, opp, landmarks_t):
    # Compute the normalized landmarks - that means:
    # 1. Center the landmarks to the center of the face
    # 2. Normalize the size of the detected face to 100
    # normalized_landmarks = (landmarks - face_center) / scale
    normalized_landmarks = opp.dot(landmarks_t.T).T

    res.append(np.r_[idx, normalized_landmarks[:, 0:2].reshape((-1)), np.linalg.inv(opp).reshape((-1))])

    # Flip the normalized landmarks around the y-axis.
    normalized_landmarks = horizontal_flip_landmarks(normalized_landmarks)
    opp = flip_y(opp)
    res.append(np.r_[idx, normalized_landmarks[:, 0:2].reshape((-1)), np.linalg.inv(opp).reshape((-1))])

if __name__ == '__main__':
    # if (len(sys.argv) == 1):
    #     print_usage()
    #     sys.exit(1)

    # input_dir = sys.argv[1]
    input_dir = 'img_sub'

    # Read the data with the landmarks from 'data.csv' and add extra columns
    # to store the position of the face.
    data = np.genfromtxt(join(input_dir, 'data_faces.csv'), delimiter=',')

    # Look for entries that have data for the face_x coordinate set.
    face_idx = np.where(data[:, 70] != 0)[0]

    normalized = []
    res = []

    ROT_ANGLE = 20.

    variations = [
        lambda M: M,

        lambda M: rotate(ROT_ANGLE, M),
        lambda M: rotate(-ROT_ANGLE, M),

        # lambda M: translate([ 5.,  5.], M),
        # lambda M: translate([-5.,  5.], M),
        # lambda M: translate([-5., -5.], M),
        # lambda M: translate([ 5., -5.], M),

        # lambda M: translate([ 5.,  5.], rotate(ROT_ANGLE, M)),
        # lambda M: translate([-5.,  5.], rotate(ROT_ANGLE, M)),
        # lambda M: translate([-5., -5.], rotate(ROT_ANGLE, M)),
        # lambda M: translate([ 5., -5.], rotate(ROT_ANGLE, M)),
    ]

    for idx in face_idx:
        # NOTE: The entries 0:70 contain the landmarks provided by the LFPW db
        #       The entries 70:74 contain the (x, y, w, h) of the detected face
        landmarks = data[idx, 0:70].reshape((-1, 2))
        face_center = data[idx, 70:72] + (data[idx, 72:74] / 2)

        scale_factor = 100./ data[idx, 72] # '100 virtual size' / 'width of detected face'
        opp = scale(scale_factor, translate(-face_center, np.diag((1, 1, 1))))

        # Create a temporary version of the landmarks with a constant factor
        # to support working with the 3D matrix
        landmarks_t = np.c_[landmarks, np.ones(landmarks.shape[0])]

        for variation in variations:
            emit(idx, res, variation(opp), landmarks_t)


    res = np.array(res)

    # Remove the features for the ear.
    res = np.c_[res[:,0:28*2+1], res[:, 69:]]

    np.savetxt(join(input_dir, 'data_faces_train.csv'), res, fmt="%0.3f", delimiter=',')
    print
    print '--> Wrote out face information for training -> `data_faces_train.csv`'



