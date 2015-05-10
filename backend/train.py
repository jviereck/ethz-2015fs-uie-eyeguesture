import matplotlib.pyplot as plt
from os.path import isfile, join
from scipy import misc
import numpy as np
import sys

from liblinear.liblinearutil import train as liblinear_train
from liblinear.liblinear import problem as liblinear_problem
from liblinear.liblinear import parameter as liblinear_parameter

import time

# A set of basic configurations for the learning
param = {
    # Training image dimensions.
    'img_width': 384,
    'img_height': 286,
    'num_sample': 100
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
        assert data.shape[0] == param['img_height']
        assert data.shape[1] == param['img_width']

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
    # return np.sqrt(var_red(arr_xy[:,0]) ** 2 + var_red(arr_xy[:,1]) ** 2)
    return var_red(arr_xy[:,0]) + var_red(arr_xy[:,1])

def flatten_list(aList):
    return [y for x in aList for y in x]

def compute_split_node(img_data, indices, full_landmark_residual, full_approx_landmark,
        radius, num_sample, img_width, img_height):
    """Comptues a split node using random sampling.

    """

    # Create a copy of the landmark data that is indiced by this function call.
    landmark_residual = full_landmark_residual[indices]
    approx_landmark = full_approx_landmark[indices]

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

    var_red_total = var_red_xy(landmark_residual)

    var_reduce_best = 0
    best_result = None

    for i in range(num_sample):
        # Pick the threshold randomly from the pixel values.
        threshold = np.random.choice(pixel_diffs[i])

        lhs_indices = np.where(pixel_diffs[i] < threshold)[0]
        rhs_indices = np.where(pixel_diffs[i] >= threshold)[0]

        var_reduce = var_red_total - \
            var_red_xy(landmark_residual[lhs_indices]) - \
            var_red_xy(landmark_residual[rhs_indices])

        if var_reduce > var_reduce_best or best_result == None:
            var_reduce_best = var_reduce

            best_result = (i, threshold, lhs_indices, rhs_indices)

    assert best_result != None, "A best choice for the threshold was not found."

    # Convert the local indices to global-all-images indices back again.
    best_offsets = offsets[best_result[0]]
    return [int(best_result[1]), best_offsets[0][0], best_offsets[0][1], \
        best_offsets[1][0], best_offsets[1][0]],  \
        indices[best_result[2]], indices[best_result[3]]

def assert_single_landmark(landmark):
    assert len(landmark.shape) == 2
    assert landmark.shape[1] == 2

class TreeClassifier:
    def __init__(self, depth, debug=False):
        self.depth = depth
        self.debug = debug

        self.node_data = None
        self.train_data_leafs = None

    def train_node(self, level, idx, img_data, param, radius, indices, \
        landmark_residual, landmark_approx):

        if self.debug:
            print ''
            print '=== Train node: level=%d idx=%d' % (level, idx)

        split_data, lhs_indices, rhs_indices = compute_split_node( \
            img_data, indices, landmark_residual, landmark_approx, \
            radius, param['num_sample'], param['img_width'], param['img_height'])

        if self.debug:
            print split_data, lhs_indices, rhs_indices

        self.node_data[idx] = split_data

        nodes_level_left = idx - (pow(2, level) - 1)
        nodes_level_right = (pow(2, level + 1) - 2) - idx

        idx_lhs = idx + nodes_level_right + 2 * nodes_level_left + 1
        idx_rhs = idx_lhs + 1

        # Early exit if the full depth of the tree was learned.
        if (level + 1 == self.depth):
            # Compute the offsets to set the a one for the local binary features.
            offset_lhs = idx_lhs - (pow(2, level + 1) - 1)
            offset_rhs = offset_lhs + 1

            # Set the binary features for the lhs and rhs indices.
            self.train_data_leafs[lhs_indices, offset_lhs] = 1
            self.train_data_leafs[rhs_indices, offset_rhs] = 1
            return

        self.train_node(level + 1, idx_lhs, img_data, param, radius, lhs_indices,
            landmark_residual, landmark_approx)
        self.train_node(level + 1, idx_rhs, img_data, param, radius, rhs_indices,
            landmark_residual, landmark_approx)

    def fit(self, img_data, param, radius, landmark, landmark_approx):
        assert_single_landmark(landmark)
        assert_single_landmark(landmark_approx)

        # Need to have 'depth - 1' node_data only, as the leafs don't have
        self.node_data = range(pow(2, self.depth) - 1) # Allocate list for later node data

        # Allocate matrix that will hold the binary local features for this tree.
        # The entires are set when the leafs are reached during the call to 'train_node'.
        self.train_data_leafs = np.zeros((landmark.shape[0], pow(2, self.depth + 1)))

        landmark_residual = landmark - landmark_approx

        # The root note tains over all possible images - therefore use all indices.
        indices = np.array(range(landmark.shape[0]))

        # Start training the root note.
        self.train_node(0, 0, img_data, param, radius, indices, \
            landmark_residual, landmark_approx)

        return self.train_data_leafs

class RandomForestClassifier:
    def __init__(self, depth=5, n_tree=5, debug=False):
        self.depth = depth
        self.n_tree = n_tree
        self.debug = debug

    def fit(self, img_data, param, radius, landmark, landmark_approx):
        self.tree_classifiers = []
        for i in range(self.n_tree):
            self.tree_classifiers.append(TreeClassifier(self.depth, self.debug))

        res = []
        for classifier in self.tree_classifiers:
            res.append(classifier.fit(img_data, param, radius, landmark, landmark_approx))

        # Return the concatinated binary features of all the tree classifiers.
        return np.hstack(res);

    def serialize(self):
        # TODO: Convert the node_data on the trees into a JSON format for later
        #   reloading / classification in the browser.
        pass



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

    # Fix the random seed to get reproducable results.
    np.random.seed(42)

    log('Processing landmarks')
    img_ids, landmarks, landmarks_approx = read_landmark_csv(sys.argv[1])

    log('Loading image data')
    img_data = read_images(img_ids, sys.argv[2])

    log('Compute single split node')


    # Example training for the first landmark over all images:
    radius = 10.0

    for iter in range(5):
        res = []
        for mark_idx in range(landmarks.shape[1]):
        # for mark_idx in range(2):
            print 'Train landmark %d/%d' % (mark_idx, landmarks.shape[1])

            # NOTE: depth = number of split nodes -> the leafs are on 'depth + 1' level!
            rf = RandomForestClassifier(depth=3, n_tree=5)
            res.append(rf.fit(img_data, param, radius, landmarks[:, mark_idx], landmarks_approx[:,mark_idx]))

        # Get the concatinated global feature mapping PHI over all the single
        # landmarks local binary features
        global_feature_mapping = np.hstack(res)

        landmarks_residual = (landmarks - landmarks_approx).reshape(-1, 40)

        # How to call into liblinear is mostly inspired by:
        # https://github.com/jwyang/face-alignment/blob/master/src/globalregression.m
        cost = 1.0/global_feature_mapping.shape[1]
        x_list = global_feature_mapping.tolist()

        # Note: Instead of solving the entire matrix system at once here, solving
        #       one residual coordinate after the other to obtain a single column
        #       of the final 'W' matrix at the end. Glying the 'W' matrix together
        #       on line (*) below.
        res = []
        for i in range(landmarks_residual.shape[1]):
            y_list = landmarks_residual[:,i].tolist()

            model = liblinear_train(y_list, x_list, '-s 12 -p 0 -c %f -q' % (cost))

            # Copy the result model 'w' data to an numpy array and append the obtained
            # 'w' column to the 'res' array.
            res.append(np.fromiter(model.w, dtype=np.double, count=global_feature_mapping.shape[1]))

        W = np.vstack(res).T  # Glue together the entire 'W' matrix (*).

        # Now that the global 'W' matrix was calculated, compute the shifts of the
        # landmarks by applying the binary global feature mapping to the matrix.
        # Need to reshape the result to a 2d vector again.
        landmarks_shifts = np.dot(global_feature_mapping, W).reshape((-1, 20,2))

        # Update the landmark approximations
        landmarks_approx = landmarks_approx + landmarks_shifts

    log_finish()


# Commands to show an image
# plt.imshow(rd)
# plt.show()