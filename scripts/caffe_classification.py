# Load model and perform inference on user image
# python caffe_classification.py <input image name> <deploy.prototxt> <model.caffemodel> <mean.npy> <labels.txt> <output image name>
#

import os
import caffe
import numpy as np
import sys
import glob
import cv2

input_image = caffe.io.load_image(sys.argv[1])
model_def = sys.argv[2]
model_weights = sys.argv[3]
model_mean = sys.argv[4]
labels_file = sys.argv[5]
output_image_name = sys.argv[6]

# Using GPU
caffe.set_mode_gpu()
# Using CPU
#caffe.set_mode_cpu()

net = caffe.Classifier(model_def, model_weights, mean = np.load(model_mean).mean(1).mean(1),
                        channel_swap=(2,1,0),
                        raw_scale=255,
                        image_dims=(256, 256))

prediction = net.predict([input_image])

labels = np.loadtxt(labels_file, str, delimiter='\t')

print "output label:", labels[prediction[0].argmax()]
print ""

# sort top five predictions from softmax output
top_inds = prediction[0].argsort()[::-1][:5]  # reverse sort and take five largest items

for x in top_inds:
    print str(prediction[0][x]) + " : " + labels[x]

result_text = "Predict : " + str(int(prediction[0][top_inds[0]] * 100)) + "%" + " Label : " + labels[top_inds[0]]

img = cv2.imread(sys.argv[1])
output_img = img.copy()
cv2.putText(output_img, result_text,(0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
cv2.imwrite(output_image_name, output_img)