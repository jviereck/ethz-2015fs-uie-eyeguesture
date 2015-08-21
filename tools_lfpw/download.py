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
    idata = np.c_[range(len(adata)), adata[:,1:]]    # id-annotated data

    np.savetxt('lfpw_downloads/idata.csv', idata, delimiter=",", fmt="%s")

    # Start with the download procedure
    for t in idata:
        file_id = int(t[0])
        print 'Try to download %d / %d: %s' % (file_id + 1, len(idata), t[1])

        try:
            response = urllib2.urlopen(t[1], timeout=4)
        except:
            print '(... failed to download)'
            continue

        output = open('lfpw_downloads/%04d.jpg' % (file_id),'wb')
        output.write(response.read())
        output.close()



