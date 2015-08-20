from os.path import isfile, join
import numpy as np
import sys
import requests
import urllib2


def print_usage():
    usage = """
Usage:
    %(cmd)s <input_file>
            """ % {"cmd":sys.argv[0]}
    print(usage)

if __name__ == '__main__':
    if (len(sys.argv) == 1):
        print_usage()
        sys.exit(1)

    input_file = sys.argv[1]

    # Read the input file and keep only the average annotated landmarks
    data = np.genfromtxt(input_file, delimiter='\t', skip_header=1, dtype='str')

    average_indices = np.where(data[:,1] == 'average')

    adata = data[average_indices]              # average data
    idata = np.c_[range(len(adata)), adata]    # id-annotated data

    # Start with the download procedure
    for t in idata:
        print 'Try to download %d / %d' % (int(t[0]) + 1, len(idata))

        # SEE this StackOverflow about request library: http://stackoverflow.com/a/10744565
        r = requests.get(t[1])
        if r.status_code != 200:
            continue

        output = open('lfpw_downloads/%04d.jpg'%(t[0]),'wb')
        output.write(r.content)
        output.close()


