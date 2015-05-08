import sys
from os import listdir
from os.path import isfile, join

from PIL import Image

def print_usage():
    usage = """
Usage:
    %(cmd)s <input_dir>
            """ % {"cmd":sys.argv[0]}
    print(usage)

if __name__ == '__main__':
    print 'hello world'
    if (len(sys.argv) == 1):
        print_usage()
        sys.exit(1)

    input_dir = sys.argv[1]
    file_names = [f
        for f in listdir(input_dir)
        if isfile(join(input_dir, f)) and f.endswith('.pgm') ]

    print 'Reading .pgm files from: ' + input_dir
    print 'Got files: ' + str(file_names)

    for file_name in file_names:
        im = Image.open(join(input_dir, file_name))
        # im = im.convert('RGB')
        im.save(file_name + '.png')



