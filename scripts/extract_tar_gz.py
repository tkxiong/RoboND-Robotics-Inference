# Extract tar file
# extract_tar_gz.py <file.tar.gz>
#

import os
import tarfile
import sys

filename = sys.argv[1]

if os.path.exists(filename) is False:
    print "File does not exist..."
    sys.exit()

if tarfile.is_tarfile(filename) is False:
    print "File is not tar.gz"
    sys.exit()

set_dir = "/tmp/model"
try:
    os.makedirs(set_dir)
except:
    print "Folder exist..."
tf = tarfile.open(filename)
tf.extractall(set_dir)