import re
import sys
from os import listdir
from os.path import isfile, join

import numpy as np


# Landmarks idx and their interpretation. (Taken from http://neerajkumar.org/databases/lfpw/)
#
#   00. left_eyebrow_out
#   01. right_eyebrow_out
#   02. left_eyebrow_in
#   03. right_eyebrow_in
#   04. left_eyebrow_center_top
#   05. left_eyebrow_center_bottom
#   06. right_eyebrow_center_top
#   07. right_eyebrow_center_bottom
#   08. left_eye_out
#   09. right_eye_out
#   10. left_eye_in
#   11. right_eye_in
#   12. left_eye_center_top
#   13. left_eye_center_bottom
#   14. right_eye_center_top
#   15. right_eye_center_bottom
#   16. left_eye_pupil
#   17. right_eye_pupil
#   18. left_nose_out
#   19. right_nose_out
#   20. nose_center_top
#   21. nose_center_bottom
#   22. left_mouth_out
#   23. right_mouth_out
#   24. mouth_center_top_lip_top
#   25. mouth_center_top_lip_bottom
#   26. mouth_center_bottom_lip_top
#   27. mouth_center_bottom_lip_bottom
#   28. left_ear_top
#   29. right_ear_top
#   30. left_ear_bottom
#   31. right_ear_bottom
#   32. left_ear_canal
#   33. right_ear_canal
#   34. chin
def horizontal_flip_landmarks(landmarks):
    assert len(landmarks.shape) == 2 # Assume to get 2D vector

    swap = [
        [0, 1],
        # [1, 0],
        [2, 3],
        # [3, 2],
        [4, 6],
        [5, 7],
        # [6, 4],
        # [7, 5],
        [8, 9],
        # [9, 8],
        [10, 11],
        # [11, 10],
        [12, 14],
        [13, 15],
        # [14, 12],
        # [15, 13],
        [16, 17],
        # [17, 16],
        [18, 19],
        # [19, 18],
        # [20, 20],
        # [21, 21],
        [22, 23],
        # [23, 22],
        [24, 26],
        [25, 27],
        # [26, 24],
        # [27, 25],
        [28, 29],
        # [29, 28],
        [30, 31],
        # [31, 30],
        [32, 33],
        # [33, 32],
        # [34, 34]
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

    for idx in face_idx:
        landmarks = data[idx, 0:70].reshape((-1, 2))
        face_center = data[idx, 70:72] + (data[idx, 72:74] / 2)

        # Compute the normalized landmarks - that means:
        # 1. Center the landmarks to the center of the face
        # 2. Normalize the size of the detected face to 100
        normalized_landmarks = (landmarks - face_center) / (data[idx, 72] / 100.)

        normalized.append(normalized_landmarks)
        normalized.append(horizontal_flip_landmarks(normalized_landmarks))

    print np.mean(np.array(normalized), axis=0)

    # Write out the data file with additional face information.
    # np.savetxt(join(input_dir, 'data_faces.csv'), data, fmt="%0.3f", delimiter=',')
    # print
    # print '--> Finished face detection and wrote data with faces to `data_faces.csv`'



