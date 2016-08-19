from feature_base import FeatureExtractor
import os
import numpy as np
import mxnet as mx
import logging
from PIL import Image

__author__ = 'xhou, chenyangli'


class DeepFeatExtractor(FeatureExtractor):

    def __init__(self, config):
        super(DeepFeatExtractor, self).__init__()
        prefix = config.get('model', 'prefix')
        model_iter = config.getint('model', 'model_iter')
        try:

            if config.getboolean('model', 'use_gpu'):
                self.ctx = mx.gpu()
            else:
                self.ctx = mx.cpu()
            self.numpy_batch_size = config.getint('model', 'numpy_batch_size')
        except ValueError:
            logging.error('Config parameter error!')

        # automatic
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.get('model', 'model_dir'))
        mean_file = os.path.join(model_dir, 'mean_224.nd')
        model_prefix = os.path.join(model_dir, prefix)

        self.model = mx.model.FeedForward.load(model_prefix, model_iter, ctx=self.ctx)
        self.mean_img = mx.nd.load(mean_file)['mean_img'].asnumpy()
        _, self.h, self.w = self.mean_img.shape

        try:
            feat_layer = config.get('model', 'feat_layer')
            # out_layer = config.get('model', 'output_layer')
            internals = self.model.symbol.get_internals()
            self.group_symbol = internals[feat_layer]
            self.extractor = mx.model.FeedForward(
                ctx=self.ctx, symbol=self.group_symbol, numpy_batch_size=self.numpy_batch_size,
                arg_params=self.model.arg_params, aux_params=self.model.aux_params,
                allow_extra_params=True)
        except IOError:
            logging.error('Deep model init failed!')
            raise ValueError

    def pre_proc_img(self, in_file):
        img = Image.fromarray(in_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_w, img_h = img.size
        short_egde = min(img_w, img_h)
        h_margin = int((img_h - short_egde) / 2)
        w_margin = int((img_w - short_egde) / 2)
        crop_img = img.crop((w_margin, h_margin, w_margin + short_egde, h_margin + short_egde))
        resized_img = np.array(crop_img.resize((self.w, self.h), Image.BICUBIC))
        sample = np.transpose(resized_img, (2, 0, 1))

        normed_img = sample - self.mean_img
        normed_img.resize(1, 3, self.h, self.w)
        return normed_img

    def frame_to_feature(self, in_img):
        preds = []
        num_img = len(in_img)
        batch_size = self.numpy_batch_size
        num_batch = (num_img - 1) / self.numpy_batch_size + 1
        for idx_batch in range(num_batch):
            cur_batch_size = min(num_img-idx_batch*batch_size, batch_size)
            batch = np.ndarray((cur_batch_size,
                                3, self.h, self.w))
            for idx_img in range(cur_batch_size):
                batch[idx_img, :, :, :] = self.pre_proc_img(
                                          in_img[idx_batch*batch_size+idx_img])
            out = self.extractor.predict(batch)
            preds.append(out)

        preds = np.vstack(preds)
        return preds
