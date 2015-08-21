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
    idata = np.genfromtxt(join(input_dir, 'idata.csv'), delimiter=',', dtype='str')

    # Take the x and the y coordinates. For this, compute the indices to keep.
    t = idata.shape[1] / 3
    x_index = np.linspace(0, t-1, t).astype('int') * 3 + 2
    y_index = np.linspace(0, t-1, t).astype('int') * 3 + 3

    # Concat the x and y indices and then sort them.
    index = np.sort(np.r_[x_index, y_index])

    # Do the final lookup, which yields the x and y coordinates of the landmarks
    # nicely one after the other :)
    data = np.c_[idata[:, index], np.zeros((1132, 4))].astype('float')


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

        faces = face_cascade.detectMultiScale(gray, 1.5, 3)

        if len(faces) == 0:
            # Retry with a smaller scaling factor.
            faces = face_cascade.detectMultiScale(gray, 1.1, 3)

        if len(faces) == 0:
            print 'Could not find face for %s' % (file_name)
            continue

        # for (x,y,w,h) in faces:
        #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    #     face_res.append(faces)
    # for idx, file_name in enumerate(file_names):

        id = int(file_name[0:4])  # Extract the photo-id from the filename.
        nose_x = data[id][40]
        nose_y = data[id][41]

        chin_x = data[id][68]
        chin_y = data[id][69]

        found = False

        for face in faces:
            # import pdb; pdb.set_trace()
            if face_contains(face, nose_x, nose_y): # and face_contains(face, chin_x, chin_y)):
                data[id][70] = face[0]
                data[id][71] = face[1]
                data[id][72] = face[2]
                data[id][73] = face[3]
                found = True
                print 'Update id=%d' % id
                break;

        if not found:
            print 'Could not find a matching face for %s' % file_name
            print faces
            print nose_x, nose_y, chin_x, chin_y


    np.savetxt('faces.csv', data, fmt="%0.3f")



