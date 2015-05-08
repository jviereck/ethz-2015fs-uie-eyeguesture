import numpy as np
from numpy import linalg

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def plot_histogram(eye_distance):
    # the histogram of the data
    n, bins, patches = plt.hist(eye_distance, 50, facecolor='green')

    plt.xlabel('Eye Distance [px]')
    plt.ylabel('#BioID images')
    plt.title('Eye Distances among BioID Faces')
    plt.grid(True)

    plt.show()

# MIN_EYE_DISTANCE = 42
# MAX_EYE_DISTANCE = 48

MIN_EYE_DISTANCE = 80
MAX_EYE_DISTANCE = 90

CSV_HEADER = "bioid_number;0x;0y;1x;1y;2x;2y;3x;3y;4x;4y;5x;5y;6x;6y;7x;7y;8x;8y;9x;9y;10x;10y;11x;11y;12x;12y;13x;13y;14x;14y;15x;15y;16x;16y;17x;17y;18x;18y;19x;19y"

if __name__ == '__main__':
    # Read the landmarks.csv file and ignore the header to end up with a 2d array.
    data = np.genfromtxt('landmarks.csv', delimiter=',', skip_header=1)

    eye_vector = data[:,1:3] - data[:,3:5]

    eye_distance = linalg.norm(eye_vector, axis=1)

    # Uncomment the following line to get a
    #plot_histogram(eye_distance)

    target_indices = np.where((eye_distance >= MIN_EYE_DISTANCE) & (eye_distance <= MAX_EYE_DISTANCE))[0]

    print 'Found %d faces for eye-distance MIN=%d and MAX=%d' % (
        target_indices.shape[0], MIN_EYE_DISTANCE, MAX_EYE_DISTANCE)

    # Write out the filtered landmarks into a new CSV file.
    np.savetxt("landmarks_filtered.csv", data[target_indices], delimiter=",", fmt="%.3f", header=CSV_HEADER)



