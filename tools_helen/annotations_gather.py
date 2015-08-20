import re
import sys
from os import listdir
from os.path import isfile, join

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
        if isfile(join(input_dir, f)) and f.endswith('.txt') ]

    res = []

    for file_name in file_names:
        file_content = read_file_to_string(join(input_dir, file_name))
        res.append(','.join(re.split('[^0-9._]+', file_content.strip())))

    f = open('annotations.csv', 'w')
    f.write('\n'.join(res))
    f.close();
