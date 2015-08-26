import re
import sys
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import scipy

WIDTH = 384; HEIGHT = 286


def get_noise_img(width, height):
    return np.random.uniform(0, 255, width * height).astype('uint8').reshape((height, width))


def scale(img, landmarks, scale):
    return scipy.misc.imresize(img, scale), (landmarks * scale)


def plot_data(img_data, landmarks):
    DPI = 80
    img_height = img_data.shape[0]
    img_width = img_data.shape[1]
    fig_size = (img_width / DPI, img_height / DPI)
    fig, axes = plt.subplots(1, 1, sharey=True, figsize=fig_size) # , figsize=fig_size
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    axes.axis([0, img_width, img_height, 0])
    axes.set_aspect('equal')
    axes.imshow(img_data, cmap=plt.get_cmap('gray'))
    axes.plot(landmarks[:,0], landmarks[:,1], '+')

    plt.savefig('debug.png', dpi=80)


def emit_train_img(img_id, sub_id, img, landmarks):
    (h, w) = img.shape

    assert h < HEIGHT; assert w < WIDTH

    # Get a noisy background and then place the image in the center.
    res = get_noise_img(WIDTH, HEIGHT)
    l = int((WIDTH - w) / 2.0); t = int((HEIGHT - h) / 2.0)
    res[t:t+h, l:l+w] = img

    # Shift the landmarks and save them to a file.
    new_landmarks = landmarks + np.array([l, t])
    np.savetxt(join(input_dir, 'train_meta_%04d-%02d.csv' % (img_id, sub_id)), new_landmarks, fmt="%0.3f", delimiter=",")

    # Save the shifted image as well.
    scipy.misc.imsave(join(input_dir, 'train_%04d-%02d.png' % (img_id, sub_id)), res)

    # DEBUG: Uncomment the following line to dump the landmarks
    # plot_data(res, new_landmarks)
    return sub_id + 1


def gen_train_img(input_dir, img_id):
    landmarks_org = np.genfromtxt(join(input_dir, 'face_meta_%04d.csv' % img_id), delimiter=',')
    img_org = scipy.misc.imread(join(input_dir, 'face_%04d.png' % img_id), flatten=True)

    # Rescale the image in case it is too large for the current viewpoint.
    s = h = img_org.shape[0]
    if h > 280:
        img, landmarks = scale(img_org, landmarks_org, 260.0/h)
        s = img.shape[0]
    else:
        img, landmarks = img_org, landmarks_org

    sub_id = 0
    sub_id = emit_train_img(img_id, sub_id, img, landmarks)

    # TODO: Rescale the original image in contrast to t
    while s > 180:
        s -= 30.0
        img, landmarks = scale(img_org, landmarks_org, s / img_org.shape[0])
        sub_id = emit_train_img(img_id, sub_id, img, landmarks)



if __name__ == '__main__':
    # if (len(sys.argv) == 1):
    #     print_usage()
    #     sys.exit(1)

    # input_dir = sys.argv[1]

    input_dir = 'img_sub'

    file_names = [f
        for f in listdir(input_dir)
        if isfile(join(input_dir, f)) and f.endswith('.jpg') ]

    for file_name in file_names:
        img_id = int(file_name[0:4])

        gen_train_img(input_dir, img_id)

















