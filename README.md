## RoboND-Robotics-Inference

docker run -it --runtime=nvidia --rm --net host -v /home/kx/Robotics/RoboND-Robotics-Inference:/workspace nvcr.io/nvidia/caffe:19.05-py2 bash

python scripts/extract_tar_gz.py model.tar.gz 

python scripts/caffe_classification.py images/game_pad.png /tmp/model/deploy.prototxt /tmp/model/snapshot_iter_1530.caffemodel scripts/mean.npy /tmp/model/labels.txt
