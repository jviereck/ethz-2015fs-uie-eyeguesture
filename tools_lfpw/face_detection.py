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


def find_face(faces, nose_x, nose_y):
    candidate = None

    for face in faces:
        # import pdb; pdb.set_trace()
        if not face_contains(face, nose_x, nose_y): # and face_contains(face, chin_x, chin_y)):
            continue

        # If there is already a candidate face AND the detected face is larger,
        # then use the larger face.
        # From experiments it turns out the larger face is normally better.
        if not (candidate is None) and face[3] > candidate[3]:
            candidate = None

        if candidate is None:
            candidate = face

    return candidate


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
    data = np.genfromtxt(join(input_dir, 'data.csv'), delimiter=',')
    data = np.c_[data, np.zeros((data.shape[0], 4))]

    # Run the face detection for all the image files found in the folder.
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    file_names = [f
        for f in listdir(input_dir)
        if isfile(join(input_dir, f)) and f.endswith('.jpg') ]


    face_res = []
    for file_name in file_names:
        print 'Inspecting face for %s' % (file_name)

        img = cv2.imread(join(input_dir, file_name))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 3)

        if len(faces) == 0:
            print 'Could not find face for %s' % (file_name)
            continue

        id = int(file_name[0:4])  # Extract the photo-id from the filename.
        nose_x = data[id][40]
        nose_y = data[id][41]
        face = find_face(faces, nose_x, nose_y)

        if face is None:
            print 'Could not find a matching face for %s' % file_name
            print faces
            print nose_x, nose_y
        else:
            (x,y,w,h) = data[id, 70:] = face

            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    # Write out the data file with additional face information.
    np.savetxt(join(input_dir, 'data_faces.csv'), data, fmt="%0.3f", delimiter=',')
    print
    print '--> Finished face detection and wrote data with faces to `data_faces.csv`'



