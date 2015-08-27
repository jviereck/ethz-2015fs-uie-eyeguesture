import re
import sys
from os import listdir
from os.path import isfile, join

import numpy as np


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

    for idx in face_idx:
        landmarks = data[idx, 0:70].reshape((-1, 2))
        face_center = data[idx, 70:72] + (data[idx, 72:74] / 2)

        # Compute the normalized landmarks - that means:
        # 1. Center the landmarks to the center of the face
        # 2. Normalize the size of the detected face to 100
        normalized_landmarks = (landmarks - face_center) / (data[idx, 72] / 100.)

        normalized.append(normalized_landmarks)

    print np.mean(np.array(normalized), axis=0)

    # Write out the data file with additional face information.
    # np.savetxt(join(input_dir, 'data_faces.csv'), data, fmt="%0.3f", delimiter=',')
    # print
    # print '--> Finished face detection and wrote data with faces to `data_faces.csv`'



