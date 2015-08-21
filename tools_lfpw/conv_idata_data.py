import re
import sys
from os import listdir
from os.path import isfile, join

import numpy as np
import cv2


def read_file_to_string(filename):
    f = open(filename, 'rt')
    s = f.read()
    f.close()
    return s

def flatten_list(aList):
    return [y for x in aList for y in x]

def face_contains(face, px, py):
    (x, y, w, h) = face
    return x <= px and px <= (x + w) and y <= py and y <= (py + h)

def print_usage():
    usage = """
Usage:
    %(cmd)s <input_dir>
            """ % {"cmd":sys.argv[0]}
    print(usage)


if __name__ == '__main__':
    if (len(sys.argv) == 1):
        print_usage()
        sys.exit(1)

    input_dir = sys.argv[1]
    idata = np.genfromtxt(join(input_dir, 'idata.csv'), delimiter='\t', dtype='str')

    t = idata.shape[1] / 3
    x_index = np.linspace(0, t - 1, t).astype('int') * 3 + 2
    y_index = np.linspace(0, t - 1, t).astype('int') * 3 + 3

    # Concat the x and y indices and then sort them.
    index = np.sort(np.r_[x_index, y_index])

    # Do the final lookup, which yields the x and y coordinates of the landmarks
    # nicely one after the other :)
    np.savetxt(join(input_dir, 'data.csv'), idata[:, index].astype('float'), delimiter=",", fmt="%0.3f")
