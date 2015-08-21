# This script takes an image and generates a grayscale 384 x 286 pixel png image
# from it.

import sys
from os import listdir
from os.path import isfile, join

import numpy as np
import scipy


# # Based on: http://stackoverflow.com/a/12201744
# def rgb2gray(rgb):
#     return np.clip(np.dot(rgb[...,:3], [0.299, 0.587, 0.144]), 0, 255).astype('uint8')


def show_img(img):
    plt.imshow(img, cmap = 'gray')
    # TRICK to show the figure but NOT make it modal and continue with execution.
    #   Taken from: http://stackoverflow.com/a/22899859
    # plt.show()
    plt.draw()
    plt.pause(0.01)


def print_usage():
    usage = """
Usage:
    %(cmd)s <input_dir>
            """ % {"cmd":sys.argv[0]}
    print(usage)


def clip_img(img, bbox):
    (t, b, l, r) = bbox
    t = max(0, t)
    b = min(img.shape[0] - 1, b)
    l = max(0, l)
    r = min(img.shape[1] - 1, r)
    return (t, b, l, r)


def extract_face(data, input_dir, file_name, img_id):
    img = scipy.misc.imread(join(input_dir, file_name), flatten=True)

    d = data[img_id]

    # NOTE: These numbers are zero-indexed
    LEFT_EYE_IN = 10
    RIGHT_EYE_IN = 11
    CHIN = 34
    X = 0; Y = 1

    eye_center = (d[RIGHT_EYE_IN] + d[LEFT_EYE_IN]) / 2

    chin = np.copy(d[CHIN])
    # Add a little bit of extra y-offset to make sure the chin-end is also covered.
    chin[1] += np.linalg.norm(eye_center - chin)*0.1

    # Compute the top of the face. This is assumed to be rougly twice the distance
    # from the chin to the center of the eye upwards from the chin.
    top = chin - (chin - eye_center) * 2

    # Compute now the length of the face - that is the distance between the chin
    # and the point where the face ends on the top. Scale the vector between the
    # eyes to this distance and then use it to compute the left and right edge
    # of the face.
    length = np.linalg.norm(chin - top)
    t = d[RIGHT_EYE_IN] - d[LEFT_EYE_IN]
    k = t / np.linalg.norm(t)
    k = k * length / 2

    left = eye_center - k; right = eye_center + k

    # Compute the bounding box of the face by taking the X and Y coordindates.
    (t, b, l, r) = clip_img(img, np.round([top[Y], chin[Y], left[X], right[X]]))
    sub = img[t:b, l:r]

    # Show the extracted image.
    # show_img(sub)
    scipy.misc.imsave(join(input_dir, 'face_%04d.png' % (img_id)), sub)

    shifted_d = d - np.array([l, t])
    np.savetxt(join(input_dir, 'face_meta_%04d.csv' % img_id), shifted_d, fmt="%0.3f", delimiter=",")



if __name__ == '__main__':
    # if (len(sys.argv) == 1):
    #     print_usage()
    #     sys.exit(1)

    # input_dir = sys.argv[1]

    input_dir = 'img_sub'
    data = np.genfromtxt(join(input_dir, 'data.csv'), delimiter=',').reshape((-1, 35, 2))


    file_names = [f
        for f in listdir(input_dir)
        if isfile(join(input_dir, f)) and f.endswith('.jpg') ]

    for file_name in file_names:
        img_id = int(file_name[0:4])
        extract_face(data, input_dir, file_name, img_id)

















