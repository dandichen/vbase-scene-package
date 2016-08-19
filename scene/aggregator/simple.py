import numpy as np
from scene.utils import video_file_tools as tools
from scipy import signal
import logging
import os
from aggregator import Aggregator

__author__ = 'chenyangli'


class SimpleAggregator(Aggregator):
    def __init__(self, config, config_label='aggregation'):
        super(SimpleAggregator, self).__init__()
        try:
            self.max_window_size = config.getint(config_label, 'max_window_size')
            self.smooth_window_size = config.getfloat(config_label, 'smooth_window_size')
            self.smooth_std = config.getfloat(config_label, 'smooth_std')
        except ValueError:
            logging.error('Config parameter error')
            raise ValueError
        self.aggregation_result = []
        self.labels = []

    @staticmethod
    def _smoothing_prediction(preds, length, std):
        kernel = signal.gaussian(length, std=std)
        kernel = kernel / np.sum(kernel)
        ret_val = np.zeros_like(preds)
        num_tags = np.shape(preds)[1]

        for i in range(num_tags):
            ret_val[:, i] = np.convolve(preds[:, i], kernel, 'same')
        return ret_val

    @staticmethod
    def _max_pool_prediction(preds, window_size):
        ret_val = preds
        window_size = int(window_size)
        half_window_size = int(window_size) / 2
        padded = np.lib.pad(preds,
                            ((half_window_size, window_size - half_window_size),
                             (0, 0)),
                            'constant',
                            constant_values=0)
        for i in range(window_size):
            ret_val = np.maximum(ret_val, padded[i:-window_size + i, :])
        return ret_val

    def get_result(self, in_pred, in_video, length, flag='video'):
        if flag == 'video':
            _, _, length, fps = tools.video_meta_data(in_video)
        elif flag == 'images':
            fps = 15
        else:
            raise ValueError
        sampling_rate = length / float(np.shape(in_pred)[0])

        fps /= float(sampling_rate)
        if len(np.shape(in_pred)) == 1:
            in_pred = np.reshape(in_pred, (np.size(in_pred), 1))

        num_tags = np.shape(in_pred)[1]
        preds = self._smoothing_prediction(in_pred,
                                          fps * self.smooth_window_size,
                                          fps * self.smooth_std)
        preds = self._max_pool_prediction(preds, fps*self.max_window_size)

        ret_val = np.zeros((length, num_tags))
        xp = np.arange(start=0, stop=length, step=sampling_rate)
        for i in range(num_tags):
            ret_val[:, i] = np.interp(np.arange(start=0, stop=length), xp, preds[:, i])
        return ret_val

    def get_result_clip(self, in_pred, in_video, length, in_threshold=[], flag='video'):
        ret_val = []

        self.aggregation_result = self.get_result(in_pred, in_video, length, flag)

        if flag == 'video':
            _, _, _, fps = tools.video_meta_data(in_video)
        elif flag == 'images':
            fps = 15
        else:
            raise ValueError
        labels = self.aggregation_result
        length = np.shape(labels)[0]
        num_tags = np.size(labels)/length

        if type(in_threshold) is list:
            in_threshold = np.full(num_tags, 0.5, dtype=float)

        labels = np.subtract(labels, in_threshold)
        labels = np.array(labels > 0).astype(int)
        self.labels = labels

        length = np.shape(labels)[0]
        num_tags = np.size(labels)/length

        for tag in range(num_tags):
            agg = []
            prev = 1
            for time in range(1, length):
                if labels[time-1, tag] != labels[time, tag]:
                    if labels[time-1, tag] == 1:
                        agg.append((prev, time))
                    prev = time
            if (prev != len(labels)) and (labels[prev, tag] == 1):
                agg.append((prev, len(labels)))
            ret_val.append(agg)
        return ret_val





