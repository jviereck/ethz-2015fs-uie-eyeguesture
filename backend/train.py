import matplotlib.pyplot as plt
from os.path import isfile, join
from scipy import misc
import numpy as np
import sys

import time
import unittest

import matplotlib.pyplot as plt
from multiprocessing import Pool

from liblinear.liblinearutil import train as liblinear_train
from liblinear.liblinear import problem as liblinear_problem
from liblinear.liblinear import parameter as liblinear_parameter

# ------------------------------------------------------------------------------
# Import the optimized var-red function
import ctypes
var_red_lib = ctypes.CDLL('var_red.so')

# IMPORTANT: Must specify the return type of the c-function. Otherwise python
#    doesn't really pick it up correctly and the result is garbage!
var_red_lib_fn = var_red_lib.var_red
var_red_lib_fn.restype = ctypes.c_float
# ------------------------------------------------------------------------------


def flatten_list(aList):
    return [y for x in aList for y in x]

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


def read_train_csv(input_dir):
    img_data = []
    data = np.genfromtxt(join(input_dir, 'data_faces_train.csv'), delimiter=',', dtype='f32')

    num_features = (data.shape[1] - 1 - 9) / 2

    # Read the actual image data.
    img_data_dict = dict()
    for img_id in np.unique(data[:, 0]):
        gray_img = misc.imread(join(input_dir, '%04d.jpg' % img_id), flatten=True)
        (height, width) = gray_img.shape
        gray_img = gray_img.astype('int16').reshape(gray_img.shape[0] * gray_img.shape[1])
        img_data_dict[img_id] = (gray_img, width, height)


    # Create an VirtualImage for each entry.
    # A 'VirtualImage' holds the image-pixel data and the translation matrix to
    # map virtual coordinates to the underlying pixel coordinate system.
    for entry in data:
        # The last 9 entries of the data contain the transformation matrix.
        t = img_data_dict[entry[0]]
        img_data.append(VirtualImage(t[0], t[1], t[2], entry[-9:].reshape((3, 3))))

    # Get the landmarks and reshape to a N * (num_landmarks) * (x, y) dimensional array
    landmarks = data[:, 1:-9].reshape(-1, num_features, 2)

    # Get the mean of the read landmarks and use this as the initial guess for training.
    mean = np.mean(landmarks, axis=0)
    mean_flat = mean.reshape(2 * num_features)
    approx_landmarks = np.tile(mean_flat, data.shape[0]).reshape(-1, num_features, 2)

    return img_data, landmarks, approx_landmarks


def read_landmark_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype='f32')
    img_ids = data[:,0]
    train_size = len(img_ids) # Number of images we use for training.

    # Get the landmarks and reshape to a N * 20-landmarks * (x, y) dimensional array
    landmarks = data[:,1:].reshape(-1, 20, 2)

    # Initialize the 'approximated_landmarks' to the mean of all the landmarks
    mean = np.mean(landmarks, axis=0)
    mean_flat = mean.reshape(40)
    approx_landmarks = np.tile(mean_flat, train_size).reshape(-1, 20, 2)

    return img_ids, landmarks, approx_landmarks

def store_rf_get_global_features(iter, res):
    features = []
    serialized_trees = []

    for entry in res:
        feature, ser = entry
        features.append(feature)
        serialized_trees.append(ser)

    np.savez_compressed(join('trained_output', '%d_rf' % (iter)), np.array(features).reshape(-1).astype(np.float32))

    return np.hstack(features)

# ------------------------------------------------------------------------------
# Random Tree Classifier

def compute_var(vector_2d):
    # Compute first the variance along the
    res = np.var(vector_2d, axis=1)
    return np.sum(res) * len(vector_2d)

def var_red(arr):
    # Computes a single term in the formular at
    # http://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction

    # import pdb; pdb.set_trace()
    N = float(len(arr))

    if N == 0.0:
        return 0.0

    # n = np.tile(arr, N).reshape(-1, N)
    # return 0.5 * (1.0/N**2) * np.sum((n - n.T)**2)
    return var_red_lib_fn(ctypes.c_int(len(arr)), ctypes.c_void_p(arr.ctypes.data))

def var_red_xy(arr_xy):
    # return np.sqrt(var_red(arr_xy[:,0]) ** 2 + var_red(arr_xy[:,1]) ** 2)
    return var_red(arr_xy[:,0]) + var_red(arr_xy[:,1])

def get_last_index(lst, elm):
    return (len(lst) - 1) - lst[::-1].index(elm)

# A set of very simple unit tests for the 'split_by_threshold' function.
class SplitByThresholdTest(unittest.TestCase):
    def testZero(self):
        pixel_diffs = np.array([-193, -171, -128, -125, -123, -122, -119, -118,
            -117, -116, -114, -103, -102, -101, -100, -99, -98, -95, -87, -86,
            -85, -82, -71, -69, -55, -55, -52, -51, -51, -50, -50, -46, -45, -44,
            -41, -41, -39, -38, -37, -33, -31, -30, -20, -14, -11, -10, -10, -6,
            -4, -2, 0, 5, 8, 11, 13, 25, 27, 28, 29, 30, 32, 33, 42, 43, 43, 43,
            46, 46, 50, 51, 51, 52, 54, 57, 61, 61, 68, 69, 71, 72, 73, 74, 81,
            86, 87, 88, 90, 91, 116])

        split_by_threshold(pixel_diffs, 4, 0.0)
        split_by_threshold(pixel_diffs, 4, 1.0)


    def testOne(self):
        pixel_diffs = np.array([-108, -103, -91, -51, -39, -39, -38, -36, -31,
            -28, -27, -26, -24, -24, -23, -22, -21, -21, -20, -19, -18, -9, -6,
            -5, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
            6, 6, 7, 7, 12, 20, 22, 36, 37, 63, 63, 87, 89, 89, 90, 91, 92, 95,
            97, 97, 98, 100, 100, 111, 123, 129, 129, 131, 131, 132, 133, 133,
            134, 136, 179, 180, 181, 182, 184, 184, 184, 193, 197])

        split_by_threshold(pixel_diffs, 4, 0.0)
        split_by_threshold(pixel_diffs, 4, 1.0)


# @profile
def split_by_threshold(pixel_diffs, min_split, rand_val):
    sorted_pixel_diffs = np.sort(pixel_diffs).tolist()

    # Find the 'threshold' value for splitting the lower values, such that
    # 'len(lhs) >= min_split'. Due to redundency in the pixel values getting
    # just the index at 'sorted_pixel_diffs[min_split]' is not good enough
    # as a 'threshold' value!

    # A trivial implementation would go ahead and yield as 'threshold_low_idx'
    # the index '0' BUT that is not correct due to the redundency!
    # 'sorted_pixel_diffs = [0, 0, 1, 2, 3], min_split = 1'
    # -> 'threshold_low_idx = 1'
    threshold_low_idx = 0
    while True:
        # Look for the last index that suites the current 'threshold_low_idx'.
        threshold_low_idx = get_last_index(sorted_pixel_diffs, sorted_pixel_diffs[threshold_low_idx])
        # Check if there are more values found than required by 'min_split'.
        if (threshold_low_idx + 1) >= min_split:  break;
        # IF not enough values, then look at the next 'threshold' value
        threshold_low_idx += 1


    # Similar to the above for 'threshold_low_idx' but for the high-index-value.
    threshold_hih_idx = len(sorted_pixel_diffs) - 1
    while True:
        threshold_hih_idx = sorted_pixel_diffs.index(sorted_pixel_diffs[threshold_hih_idx])
        # NOTE: Do a '- 1' in the following as the 'rhs_indices' are computed
        #       with 'pixel_diffs > threshold'.
        if (len(sorted_pixel_diffs) - threshold_hih_idx) >= min_split:  break;
        # IF not enough values, then look at the next 'threshold' value
        threshold_hih_idx -= 1



    # Check the computed 'threshold_low_idx' and 'threshold_hih_idx'.
    # There EXISTS a valid threshold value IF the low and high indices do not overlap.
    if threshold_hih_idx < threshold_low_idx: return None, None, None

    # === COMPUTE THE NEW threshold
    # EXAMPLE:
    #          l        h        << threshold_{[l]ow,[h]ih}_idx markers
    #          |        |
    #    0  1  2  3  4  5  6     << INDICES
    #          |        |
    #   [0, 0, 0, 1, 2, 3, 3]    << sorted_pixel_diffs
    #          |        |
    #          0  1  2  3
    #          |<--->|           << SPAN
    #
    # NOTE: The 'threshold' value splits the values in
    #
    #         'lhs = val <= threshold'
    #         'rhs = val >  threshold'
    #
    #       Therefore, the span should be '- 1' of the distance 'high - low'
    span = threshold_hih_idx - threshold_low_idx - 1 # EXAMPLE_VAL=3
    threshold_idx = threshold_low_idx + np.round(span * rand_val)

    threshold = sorted_pixel_diffs[int(threshold_idx)]
    lhs_indices = np.where(pixel_diffs <= threshold)[0]
    rhs_indices = np.where(pixel_diffs > threshold)[0]

    if len(lhs_indices) < min_split or len(rhs_indices) < min_split:
        print '=== compute_split_node -> ERROR'
        print 'threshold_low_idx=%d threshold_hih_idx=%d -> low_val=%d, hih_val=%d' % (
            threshold_low_idx, threshold_hih_idx, sorted_pixel_diffs[threshold_low_idx], sorted_pixel_diffs[threshold_hih_idx])
        print 'ind=%d of len(sorted_pixel_diffs)=%d, min_split=%d, threshold=%d' % (
            threshold_idx, len(sorted_pixel_diffs), min_split, threshold)
        print 'len(lhs)=%d len(rhs)=%d' % (len(lhs_indices), len(rhs_indices))
        print sorted_pixel_diffs
        assert False, "Got less splits than there should be!"

    return threshold, lhs_indices, rhs_indices


def sample_offset_vector(radius):
    radius = np.random.rand() * radius
    angle = np.random.rand() * 2. * np.pi
    return np.array((radius * np.cos(angle), radius * np.sin(angle), 1.0))


def get_pixel_diffs(img_data, indices, approx_landmark, radius, num_sample):
    offsets = []
    pixel_diffs = np.zeros((num_sample, len(indices)))

    for i in range(2 * num_sample):
        offsets.append(sample_offset_vector(radius))

    offset_vec = np.array(offsets)

    for i, idx in enumerate(indices):
        # HACK: Convert the 2D landmark vector to a 3D one here.
        landmark = np.array([approx_landmark[idx][0], approx_landmark[idx][1], 0.0])
        pixvals = img_data[idx].get_virt_pixel_values(landmark + offset_vec)
        pixel_diffs[:,i] = pixvals[0:num_sample] - pixvals[num_sample:]


    # HACK: Conver the 3D offset vector back to a 2D one and concat it.
    offset_vec = offset_vec[:, 0:2]
    return pixel_diffs, np.hstack((offset_vec[0:num_sample], offset_vec[num_sample:]))


# To enable profiling on this function, use the 'line_profiler' package.
# SEE: http://www.huyng.com/posts/python-performance-analysis/
# @profile
def compute_split_node(min_split, indices, full_landmark_residual, pixel_diffs, offsets):
    """Comptues a split node using random sampling.
    """

    assert len(indices) >= 2 * min_split

    # Create a copy of the landmark data that is indiced by this function call.
    landmark_residual = full_landmark_residual[indices]

    assert indices.__class__ == np.ndarray, "Expect indices to be an np.array."

    # Part 1: Compute random offsets and gather the pixel differences from the
    #   image data based on the offsets.

    # Part 2: Look for the offset / trashold combination, that yields the best
    #   variance reduction.

    # To compute the variance reductinon, see the forumular here:
    # http://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction

    var_red_total = var_red_xy(landmark_residual)

    var_reduce_best = 0
    best_result = None

    for i in range(len(pixel_diffs)):
        threshold, lhs_indices, rhs_indices = \
            split_by_threshold(pixel_diffs[i], min_split, np.random.rand())

        # IN CASE no 'threshold' satisfing the 'min_split' requirement was
        #         able to be computed -> exit.
        if threshold is None: continue

        var_reduce = var_red_total - \
            var_red_xy(landmark_residual[lhs_indices]) - \
            var_red_xy(landmark_residual[rhs_indices])

        if var_reduce > var_reduce_best or best_result == None:
            var_reduce_best = var_reduce
            best_result = (i, threshold, lhs_indices, rhs_indices)

    assert best_result != None, "A best choice for the threshold was not found."
    assert len(best_result[2]) >= min_split and len(best_result[3]) >= min_split, "Achieved a split with minimum number of nodes."

    # Convert the local indices to global-all-images indices back again.
    best_offsets = offsets[best_result[0]]
    # import pdb; pdb.set_trace()
    return [int(best_result[1]), best_offsets[0], best_offsets[1], \
        best_offsets[2], best_offsets[3]],  \
        indices[best_result[2]], indices[best_result[3]]


def assert_single_landmark(landmark):
    assert len(landmark.shape) == 2
    assert landmark.shape[1] == 2
    # The external C-interfaces assume to get float32 values. Therefore, check
    # if the type of the landmark array is also float32 to avoid future problems.
    assert landmark.dtype == np.float32


class TreeClassifier:
    """
    @param(depth): The depth counts the first root note as well.
        EXAMPLE for depth=3:
           0
         1   2
        3 4 5 6
    """
    def __init__(self, depth, debug=False):
        self.depth = depth
        self.debug = debug

        self.node_data = None
        self.train_data_leafs = None

    def get_child_idx(self, level, idx):
        nodes_level_left = idx - (pow(2, level) - 1)
        nodes_level_right = (pow(2, level + 1) - 2) - idx

        idx_lhs = idx + nodes_level_right + 2 * nodes_level_left + 1
        idx_rhs = idx_lhs + 1
        return idx_lhs, idx_rhs

    def train_node(self, level, idx, img_data, param, radius, indices, \
        landmark_residual, landmark_approx):

        if self.debug:
            print ''
            print '=== Train node: level=%d idx=%d' % (level, idx)

        # The number of minimal elements splitted to the right and to the left.
        min_split = pow(2, self.depth - level - 1)


        pixel_diffs, offsets = get_pixel_diffs(img_data, indices, landmark_approx,
            radius, param['num_sample'])

        split_data, lhs_indices, rhs_indices = compute_split_node(\
            min_split, indices, landmark_residual, pixel_diffs, offsets)

        if self.debug:
            print split_data, lhs_indices, rhs_indices

        self.split_data[idx*5:((idx+1)*5)] = split_data

        idx_lhs, idx_rhs = self.get_child_idx(level, idx)

        # Early exit if the full depth of the tree was learned.
        if (level + 1 == self.depth):
            # Compute the offsets to set the a one for the local binary features.
            offset_lhs = idx_lhs - (pow(2, level + 1) - 1)
            offset_rhs = offset_lhs + 1

            # Set the binary features for the lhs and rhs indices.
            self.train_data_leafs[lhs_indices, offset_lhs] = 1
            self.train_data_leafs[rhs_indices, offset_rhs] = 1
        else:
            self.train_node(level + 1, idx_lhs, img_data, param, radius,
                lhs_indices, landmark_residual, landmark_approx)
            self.train_node(level + 1, idx_rhs, img_data, param, radius,
                rhs_indices, landmark_residual, landmark_approx)

    def fit(self, img_data, param, radius, landmark, landmark_approx):
        assert_single_landmark(landmark)
        assert_single_landmark(landmark_approx)

        # Need to have 'depth - 1' split_data only, as the leafs don't have
        self.split_data = np.zeros((pow(2, self.depth) - 1) * 5) # Allocate list for later node data

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
        res = []
        for tree in self.tree_classifiers:
            res.append(tree.split_data)

        return np.array(tree.split_data).reshape(-1)


class VirtualImage:
    """A image consuming coordinates in the virtual coordinate space."""

    def __init__(self, data, width, height, translation=None):
        self.data = data
        self.width = width
        self.height = height

        if translation is None:
            self.translation = np.diag([1, 1, 1])
        else:
            self.translation = translation

    def virt2phys_2d(self, arr_virt):
        assert arr_virt.shape[1] == 2 # Expecting a 2D vector with constant 1

        return self.virt2phys(np.c_[arr_virt, np.ones(arr_virt.shape[0])])

    """Converts an 2D array with virtual coordiantes into physical coordinates"""
    def virt2phys(self, arr_virt):
        assert arr_virt.shape[1] == 3 # Expecting a 3D vector with constant 1
        assert len(np.where(arr_virt[:, 2] != 1.0)[0]) == 0

        # import pdb; pdb.set_trace()
        p = self.translation.dot(arr_virt.T)[0:2].T

        # Clip the x and y coordinates to the width and height of the picture.
        return np.fmin(np.fmax(p, [0, 0]), [self.width - 1, self.height - 1])


    def get_virt_pixel_values(self, arr_virt):
        p = self.virt2phys(arr_virt)

        # Convert the 2d vector into a 1d vector for the image data lookup.
        transform_1d = np.array((1, self.width))
        pixel_targets_1d = np.fmin(np.round(p.dot(transform_1d)), (self.width * self.height) - 1)

        return self.data[pixel_targets_1d.astype('int64')] # Convert to int64 to get non-float indices


    def debug(self, ax, img_dim, landmarks, landmark_approx):
        img_data = self.data.reshape(self.height, self.width)

        # The following defines the window to plot around the face. For example
        # the value '70' means to plot |--70--x--70--| (so 140) virtual sized
        # window around the center of the detected face.
        win = np.array([70, 70])
        bbox = self.virt2phys_2d(np.array([-win, win]))

        (l, t, r, b) = bbox.reshape(-1)
        if l > r:
            k = l; l = r; r = k;
            bbox[0, 0] = bbox[1, 0]

        # HACK: Tried to scale the image using matplotlib - however, that didn't
        #   worked out at all :/ Therefore, doing manually scaling here.
        scale = img_dim / (r - l)
        img_resized = misc.imresize(img_data[t:b, l:r], scale)
        ax.axis([0, img_dim, img_dim, 0])
        ax.imshow(img_resized, cmap=plt.get_cmap('gray'), aspect='auto')

        # Also need to adjust the scaling of the landmarks and shift to the
        # bounding box of the detected face.
        landmarks = (self.virt2phys_2d(landmarks) - bbox[0]) * scale
        ax.plot(landmarks[:,0], landmarks[:,1], 'x')

        landmark_approx = (self.virt2phys_2d(landmark_approx) - bbox[0]) * scale
        ax.plot(landmark_approx[:,0], landmark_approx[:,1], 'o')


def plot_data(img_index, img_data, landmarks, landmark_approx, name):
    DPI = 80
    img_width, img_height = 200., 200.
    fig_size = (img_width / DPI * len(img_index), img_height / DPI)
    # fig_size = (len(img_index) * , )
    fig, axes = plt.subplots(1, len(img_index), sharey=True, figsize=fig_size) # , figsize=fig_size
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    for idx, ax in enumerate(axes):
        img_idx = img_index[idx]
        img_data[img_idx].debug(ax, img_width, landmarks[img_idx], landmark_approx[img_idx])

    plt.savefig('train_round_%02d.png' % int(name), dpi=80)


def train_random_forest(mark_idx):
    print 'Train landmark %d/%d' % (mark_idx + 1, landmarks.shape[1])

    # NOTE: depth = number of split nodes -> the leafs are on 'depth' level!
    rf = RandomForestClassifier(depth=TRAIN_PARAM['tree_depth'], n_tree=TRAIN_PARAM['tree_count'])
    res = rf.fit(img_data, TRAIN_PARAM, radius, landmarks[:, mark_idx], landmarks_approx[:,mark_idx])

    return res, rf.serialize()


def mean_var(vec_arr):
    arr = np.linalg.norm(vec_arr.reshape(-1, 2), axis=1)
    return np.mean(arr), np.var(arr)

def print_usage():
    usage = """
Usage:
    %(cmd)s <input-folder>
            """ % {"cmd":sys.argv[0]}
    print(usage)

if __name__ == '__main__':
    np.seterr(all='raise')

    if (len(sys.argv) <= 1):
        print_usage()
        print ''

        print '(Running unit tests...)'
        unittest.main()

        sys.exit(1)

    # Fix the random seed to get reproducable results.
    np.random.seed(42)

    log('Reading landmarks and images')
    img_data, landmarks, landmarks_approx = read_train_csv(sys.argv[1])
    num_features = landmarks.shape[1]


    TRAIN_PARAM = {
        'radii': [20., 17., 5., 5., 5.],
        'num_sample': 100,
        'tree_depth': 3,
        'tree_count': 5
    }

    IMG_DEBUG_INDEX = np.array([0, 2, 4])
    # IMG_DEBUG_INDEX = np.array([0, 1, 2, 3, 4, 5]) * 25
    plot_data(IMG_DEBUG_INDEX, img_data, landmarks, landmarks_approx, '0')

    np.savez_compressed(join('trained_output', 'initial_shape'), landmarks_approx[0].reshape(-1))
    np.savez_compressed(join('trained_output', 'configuration'), np.array([
        num_features, TRAIN_PARAM['tree_depth'], TRAIN_PARAM['tree_count']
    ]))

    # Example training for the first landmark over all images:
    for (iter, radius) in enumerate(TRAIN_PARAM['radii']):
        log('Construct RandomForestClassifier (iter=%d/%d, radius=%.3f)' % (iter + 1, len(TRAIN_PARAM['radii']), radius))

        # NOTE: Creating the pool object here, such that ALL the local and
        #       global variables *BEFORE* this invocation are also available
        #       to the forked child processes.
        # HACK: Work using 'map_async' to work around ctrl+c not terminating [1]
        pool = Pool(processes=7)
        res = pool.map_async(train_random_forest, range(num_features)).get(9999999)

        # res = []
        # for mark_idx in range(num_features):
        #     res.append(train_random_forest(mark_idx))

        # Get the concatinated global feature mapping PHI over all the single
        # landmarks local binary features
        global_feature_mapping = store_rf_get_global_features(iter, res)

        log('Compute global regression matrix')

        # import pdb; pdb.set_trace()

        landmarks_residual = (landmarks - landmarks_approx).reshape(-1, num_features * 2)

        # How to call into liblinear is mostly inspired by:
        # https://github.com/jwyang/face-alignment/blob/master/src/globalregression.m
        cost = 10.0 / global_feature_mapping.shape[1]
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

        # Glue together the entire 'W' matrix (*).
        W = np.vstack(res).T.astype(np.float32)

        np.savez_compressed(join('trained_output', '%d_w' % iter), W)

        # Now that the global 'W' matrix was calculated, compute the shifts of the
        # landmarks by applying the binary global feature mapping to the matrix.
        # Need to reshape the result to a 2d vector again.
        landmarks_shifts = np.dot(global_feature_mapping, W).reshape((-1, num_features, 2)).astype(np.float32)

        # Update the landmark approximations
        landmarks_approx = landmarks_approx + landmarks_shifts

        print '--> landmarks_shifts: mean=%.3f var=%.3f' % mean_var(landmarks_shifts)
        print '--> landmarks error:  mean=%.3f var=%.3f' % mean_var((landmarks - landmarks_approx))

        # Create image log
        plot_data(IMG_DEBUG_INDEX, img_data, landmarks, landmarks_approx, str(iter + 1))

    log_finish()


# Commands to show an image
# plt.imshow(rd)
# plt.show()

# [1]: http://stackoverflow.com/q/1408356
