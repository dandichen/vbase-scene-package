import sys
sys.path.insert(0, '/mnt/scratch/third-party-packages/mxnet-0.5.0-cpu/python')

from pipeline.pipeline import Pipeline
from classifier.training_util import TrainingUtil
import ConfigParser
import cv2
import numpy as np

# p = ConfigParser.ConfigParser()
# p.read('/mnt/scratch/chenyangli/vbase-scene/config/classifier/training.cfg')
#
# training = TrainingUtil(p)
#
# training_list = '/mnt/scratch/chenyangli/vbase-scene/train_list.lst'
#
# training.learn(training_list)
# #gt = training.learner.classifier


p = ConfigParser.ConfigParser()
p.read('/mnt/scratch/chenyangli/vbase-scene/config/pipeline/pipeline.cfg')

# in_video = '/mnt/scratch/chenyangli/vbase-scene/4.mp4'
in_video = 'test.avi'
in_images = '/mnt/scratch/sync_sd/car_record/demo/20160621-2/binocular_camera/2016-06-21_124511/2016-06-21_1246/'

pipe = Pipeline(p)
"""
f = open(training_list, 'r')
labels = []
frames = []
for line in f:
    param = line.split('\t')
    img_file = param[1][:-1]
    im = cv2.imread(img_file)
    frames.append(im)
    labels.append(int(param[0]))

labels = np.array(labels)
predicted = pipe.classifier.predict(frames)
print ((predicted > 0.5) != labels)
print (predicted > 0.5)
print labels
"""

# print pipe.get_result(in_video, 'video)
# pipe.visualize_result()
print pipe.get_result(in_images, 'images')
pipe.visualize_result()
