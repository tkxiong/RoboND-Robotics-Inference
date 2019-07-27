## RoboND-Robotics-Inference

docker run -it --runtime=nvidia --rm --net host -v /home/kx/Robotics/RoboND-Robotics-Inference:/workspace nvcr.io/nvidia/caffe:19.05-py2 bash

python extract_tar_gz.py ../model.tar.gz 

python caffe_classification.py ../images/marker.png /tmp/model/deploy.prototxt /tmp/model/snapshot_iter_1650.caffemodel mean.npy /tmp/model/labels.txt
