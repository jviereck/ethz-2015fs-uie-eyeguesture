from os.path import isfile, join
import numpy as np
import shutil


if __name__ == '__main__':
    # Read the landmarks.csv file and ignore the header to end up with a 2d array.
    data = np.genfromtxt('landmarks_filtered.csv', delimiter=';', skip_header=1)

    for bioid_number in data[:,0]:
        bioid_file_name = 'BioID_%04d.pgm.png' % bioid_number
        shutil.copyfile(join('output', bioid_file_name), join('output_filtered', bioid_file_name))
