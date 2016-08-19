import ConfigParser
from scene.utils.video_file_tools import extract_frames, video_meta_data
from scene.classifier import trad_classifier, end2end_classifier
from scene.aggregator import simple
import os
import cv2
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
classifier_selector = {'trad': trad_classifier.TradClassifier,
                       'end2end': end2end_classifier.End2EndClassifer}

aggregator_selector = {'simple': simple.SimpleAggregator}


# TODO: Add multiple classifier support
class Pipeline(object):
    def __init__(self, config):
        classifier_name = config.get('classifier', 'classifier_name')
        classifier_config_path = config.get('classifier', 'config_path')
        if not (classifier_name in classifier_selector):
            raise ValueError('Please provide a correct classifier name.')

        option_parser = ConfigParser.ConfigParser()
        option_parser.read(classifier_config_path)

        self.classifier = classifier_selector[classifier_name](option_parser)

        aggregator_name = config.get('aggregator', 'aggregator')
        aggregator_config_path = config.get('aggregator', 'config_path')
        if not (aggregator_name in aggregator_selector):
            raise ValueError('Please provide a correct aggregator name.')

        option_parser = ConfigParser.ConfigParser()
        option_parser.read(aggregator_config_path)

        self.aggregator = aggregator_selector[aggregator_name](option_parser)

        self.sampling_rate = config.getint('sampling', 'sampling_rate')

        self.threshold = config.get('aggregator', 'threshold').split(' ')
        self.threshold = map(float, self.threshold)
        self.threshold = np.array(self.threshold)

        self.pred = np.array([])

        self.result = []

    def get_result(self, in_video, flag='video'):
        if flag == 'video':
            frames = extract_frames(in_video=in_video, space=self.sampling_rate, flag='video')
            self.pred = self.classifier.predict(frames)
            self.result = self.aggregator.get_result_clip(in_pred=self.pred, in_video=in_video, length=np.shape(self.aggregator.aggregation_result)[0], in_threshold=self.threshold, flag='video')
        elif flag == 'images':
            frames = extract_frames(in_video=in_video, space=self.sampling_rate, flag='images')
            self.pred = self.classifier.predict(frames)
            length = 0
            for dir_name, _, file_list in sorted(os.walk(in_video)):
                print dir_name
                for name in sorted(file_list):
                    print name
                    if fnmatch.fnmatch(name, '*cam1*.jpg'):
                        length += 1
            self.result = self.aggregator.get_result_clip(in_pred=self.pred, in_video=frames, length=length, in_threshold=self.threshold, flag='images')
        else:
            raise ValueError
        return self.result

    def get_result_sec(self, in_video):
        if len(self.result) == 0:
            self.get_result(in_video)

        _, _, _, fps = video_meta_data(in_video)
        return (np.array(self.result) / float(fps)).tolist()

    def visualize_result(self):
        if np.size(self.pred) == 0:
            raise ValueError
        result = self.aggregator.aggregation_result
        tag_num = np.shape(result)[1]
        frame_num = np.shape(result)[0]
        x_array = np.arange(frame_num).astype(float)
        np.save('temp.npy', result)
        for i in range(tag_num):
            plt.fill_between(x_array, 0, result[:, i].reshape(frame_num),
                             alpha=0.3)
        plt.show()
        return 0
