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

    return (res * np.array((-1, 1)))


def print_usage():
    usage = """
Usage:
    %(cmd)s <input_dir>
            """ % {"cmd":sys.argv[0]}
    print(usage)


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

    for idx in face_idx:
        # NOTE: The entries 0:70 contain the landmarks provided by the LFPW db
        #       The entries 70:74 contain the (x, y, w, h) of the detected face
        landmarks = data[idx, 0:70].reshape((-1, 2))
        face_center = data[idx, 70:72] + (data[idx, 72:74] / 2)

        scale = data[idx, 72] / 100. # 'width of detected face' / '100 virtual size'

        # Compute the normalized landmarks - that means:
        # 1. Center the landmarks to the center of the face
        # 2. Normalize the size of the detected face to 100
        normalized_landmarks = (landmarks - face_center) / scale

        # Create the translation matrix that allows to remap the normalized
        # landmarks to the physical image coordinate space.
        (tx, ty) = face_center; sx = sy = scale
        m = np.array([
                [sx, 0., tx],
                [0., sy, ty],
                [0., 0., 1.]
            ])

        res.append(np.r_[idx, normalized_landmarks.reshape((-1)), m.reshape((-1))])

        # Flip the normalized landmarks around the y-axis.
        normalized_landmarks = horizontal_flip_landmarks(normalized_landmarks)
        m = np.array([
                [-1. * sx, 0., tx],
                [      0., sy, ty],
                [      0., 0., 1.]
            ])
        res.append(np.r_[idx, normalized_landmarks.reshape((-1)), m.reshape((-1))])

    np.savetxt(join(input_dir, 'data_faces_train.csv'), res, fmt="%0.3f", delimiter=',')
    print
    print '--> Wrote out face information for training -> `data_faces_train.csv`'



