import sys
from os import listdir
from os.path import isfile, join

# labels = [
#     "file_name",
#     "00_right_eye_pupil",
#     "01_left_eye_pupil",
#     "02_right_mouth_corner",
#     "03_left_mouth_corner",
#     "04_outer_end_of_right eye brow",
#     "05_inner_end_of_right eye brow",
#     "06_inner_end_of_left eye brow",
#     "07_outer_end_of_left eye brow",
#     "08_right_temple",
#     "09_outer_corner_of_right eye",
#     "10_inner_corner_of_right eye",
#     "11_inner_corner_of_left eye",
#     "12_outer_corner_of_left eye",
#     "13_left_temple",
#     "14_tip_of_nose",
#     "15_right_nostril",
#     "16_left_nostril",
#     "17_centre_point_on_outer_edge_of_upper_lip",
#     "18_centre_point_on_outer_edge_of_lower_lip",
#     "19_tip_of_chin"
# ]

labels = ["bioid_number"]

for idx in range(20):
    labels.append(str(idx) + 'x')
    labels.append(str(idx) + 'y')

def read_file_to_string(filename):
    f = open(filename, 'rt')
    s = f.read()
    f.close()
    return s

def flatten_list(aList):
    return [y for x in aList for y in x]

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
    file_names = [f
        for f in listdir(input_dir)
        if isfile(join(input_dir, f)) and f.endswith('.pts') ]

    res = [';'.join(labels)]

    for file_name in file_names:
        file_content = read_file_to_string(join(input_dir, file_name))
        points = [f.strip().split(' ') for f in file_content.split('\n')[3:23]]

        res.append(file_name[6:10] + ',' + ','.join(flatten_list(points)))

    f = open('landmarks.csv', 'w')
    f.write('\n'.join(res))
    f.close();
