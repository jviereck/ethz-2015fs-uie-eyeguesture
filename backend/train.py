import matplotlib.pyplot as plt
from os.path import isfile, join
from scipy import misc
import numpy as np
import sys

import time

# A set of basic configurations for the learning
param = {
    # Training image dimensions.
    'width': 384,
    'height': 286
}

_last_log = None
_last_task = None

def log_finish():
    if _last_log != None:
        print '(FINISHED "%s" in %0.2f sec)' % (_last_task, time.time() - _last_log)
        print ''

def log(task_name):
    global _last_log, _last_task
    log_finish()

    print '=== STARTING "%s"' % (task_name)
    _last_log = time.time();
    _last_task = task_name


def read_landmark_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    img_ids = data[:,1]
    train_size = len(img_ids) # Number of images we use for training.

    # Get the landmarks and reshape to a N * 20-landmarks * (x, y) dimensional array
    landmarks = data[:,1:].reshape(-1, 20, 2)

    # Initialize the 'approximated_landmarks' to the mean of all the landmarks
    mean = np.mean(landmarks, axis=0)
    mean_flat = mean.reshape(40)
    approx_landmarks = np.tile(mean_flat, train_size).reshape(-1, 20, 2)

    return img_ids, landmarks, approx_landmarks

def read_images(img_ids, img_root_path):
    def read_image(img_id):
        # Read the image data.
        data = misc.imread(join(img_root_path, 'BioID_%04d.pgm' % (img_id)))

        # Check the image we get fits the expected dimensions.
        assert data.shape[0] == param['height']
        assert data.shape[1] == param['width']

        # Convert the image data to signed-int16 type and reshape to a single
        # dimensional array to make indexing during pixel-feature computation
        # easier.
        return data.astype('int16').reshape(data.shape[0] * data.shape[1])

    return np.vstack(map(read_image, img_ids))

def print_usage():
    usage = """
Usage:
    %(cmd)s <landmarks.csv-file> <image_root_path>
            """ % {"cmd":sys.argv[0]}
    print(usage)

if __name__ == '__main__':
    if (len(sys.argv) == 2):
        print_usage()
        sys.exit(1)


    log('Processing landmarks')
    img_ids, landmarks, approx_landmarks = read_landmark_csv(sys.argv[1])

    log('Loading image data')
    img_data = read_images(img_ids, sys.argv[2])

    log_finish()


# Commands to show an image
# plt.imshow(rd)
# plt.show()
