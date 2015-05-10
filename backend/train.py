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

    return np.vstack(map(read_image, img_ids)).reshape(-1)

# ------------------------------------------------------------------------------
# Random Tree Classifier

def sample_offset_vector(radius):
    radius = np.random.rand() * radius
    angle = np.random.rand() * 2. * np.pi
    return np.array((radius * np.cos(angle), radius * np.sin(angle)))

def clip_cord(coordinate, img_width, img_height):
    return np.fmin(np.fmax(coordinate, [0, 0]), [img_width - 1, img_height - 1])

def get_random_offset_pixel_values(img_data, indices, approx_landmark, radius,
    img_width, img_height):


    img_pixels = img_width * img_height

    # Sample a random offset vector within the given radius.
    offset = sample_offset_vector(radius)

    # Apply the offets to the approximated landmark positions and ensure the
    # resulting positions are inside of the image.
    pixel_targets = clip_cord(approx_landmark + offset, img_width, img_height)

    # Convert the 2d vector into a 1d vector for the image data lookup.
    transform_1d = np.array((1, img_width))
    pixel_targets_1d = np.fmin(np.round(pixel_targets.dot(transform_1d)), img_pixels - 1)

    # Shift 1d vector from per-image coordinates to full img_data indices.
    pixel_img_targets = pixel_targets_1d + (indices * img_pixels)

    # Lookup the values in the image data for the computed indices.
    return img_data[pixel_img_targets.astype('int64')], offset

def compute_var(vector_2d):
    # Compute first the variance along the
    res = np.var(vector_2d, axis=1)
    return np.sum(res) * len(vector_2d)

def var_red(arr):
    # Computes a single term in the formular at
    # http://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction

    # import pdb
    # pdb.set_trace()

    N = float(len(arr))

    if N == 0:
        return 0.0

    n = np.tile(arr, N).reshape(-1, N)

    return 0.5 * (1.0/N) * np.sum((n - n.T)**2)

def var_red_xy(arr_xy):
    return np.sqrt(var_red(arr_xy[:,0]) ** 2 + var_red(arr_xy[:,1]) ** 2)

def flatten_list(aList):
    return [y for x in aList for y in x]

def compute_split_node(img_data, indices, landmark_residual, approx_landmark,
        radius, num_sample, img_width, img_height):
    """Comptues a split node using random sampling.

    """

    assert indices.__class__ == np.ndarray, "Expect indices to be an np.array."

    # Part 1: Compute random offsets and gather the pixel differences from the
    #   image data based on the offsets.

    offsets = []
    pixel_diffs = []

    for i in range(num_sample):
        pixel_values_a, offset_a = get_random_offset_pixel_values(
            img_data, indices, approx_landmark, radius, img_width, img_height)

        pixel_values_b, offset_b = get_random_offset_pixel_values(
            img_data, indices, approx_landmark, radius, img_width, img_height)

        pixel_diff = pixel_values_a - pixel_values_b

        offsets.append((offset_a, offset_b))
        pixel_diffs.append(pixel_diff)


    # Part 2: Look for the offset / trashold combination, that yields the best
    #   variance reduction.

    # To compute the variance reductinon, see the forumular here:
    # http://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction

    # var_total = compute_var(landmark_residual)

    var_red_total = var_red_xy(landmark_residual)

    var_reduce_best = 0
    best_result = None

    for i in range(num_sample):
        # Pick the threshold randomly from the pixel values.
        threshold = np.random.choice(pixel_diffs[i])

        lhs_indices = np.where(pixel_diffs[i] < threshold)
        rhs_indices = np.where(pixel_diffs[i] >= threshold)

        var_reduce = var_red_total - \
            var_red_xy(landmark_residual[lhs_indices]) - \
            var_red_xy(landmark_residual[rhs_indices])

        if var_reduce > var_reduce_best:
            var_reduce_best = var_reduce

            best_result = (i, threshold, lhs_indices, rhs_indices)

    assert best_result != None, "A best choice for the threshold was not found."

    return (best_result[1], offsets[best_result[0]][0], offsets[best_result[1]][1]),  \
        best_result[2], best_result[3]


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

    log('Compute single split node')

    # Compute the initial residual landmarks.
    landmark_residual = landmarks - approx_landmarks

    # Example training for the first landmark over all images:
    indices = np.array(range(landmarks.shape[0]))
    landmark_residual = landmark_residual[:,0]
    approx_landmark = approx_landmarks[:,0]
    radius = 10
    num_sample = 500

    res = compute_split_node(img_data, indices, landmark_residual, approx_landmark, \
        radius, num_sample, param['width'], param['height'])

    print res

    log_finish()


# Commands to show an image
# plt.imshow(rd)
# plt.show()
