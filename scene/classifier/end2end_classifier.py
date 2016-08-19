import mxnet as mx
import logging
import os
import numpy as np
from classifier import Classifier
from PIL import Image

__author__ = 'chenyangli'


# TODO: Comments and interfaces, removing extra functions
class End2EndClassifer(Classifier):
    def __init__(self, config, config_label='tag_mxnet'):
        super(End2EndClassifer, self).__init__()
        try:
            prefix = config.get(config_label, 'prefix')
            model_iter = config.getint(config_label, 'model_iter')
            if config.getboolean(config_label, 'use_gpu'):
                self.ctx = mx.gpu()
            else:
                self.ctx = mx.cpu()
            self.numpy_batch_size = config.getint(config_label, 'numpy_batch_size')
        except ValueError:
            logging.error('Config parameter error!')
            raise ValueError

        model_dir = config.get(config_label, 'model_dir')
        self.mean_file = os.path.join(model_dir, prefix+'-mean.nd')
        model_prefix = os.path.join(model_dir, prefix)

        self.model = mx.model.FeedForward.load(model_prefix, model_iter, ctx=mx.gpu())
        self.mean_img = mx.nd.load(self.mean_file)['mean_img'].asnumpy()
        _, self.h, self.w = self.mean_img.shape

        try:
            tag_layer = config.get(config_label, 'feat_layer')
            internals = self.model.symbol.get_internals()
            output_symbol = internals[tag_layer]

            self.tagger = mx.model.FeedForward(ctx=self.ctx,
                                               symbol=output_symbol,
                                               arg_params=self.model.arg_params,
                                               aux_params=self.model.aux_params,
                                               numpy_batch_size=self.numpy_batch_size,
                                               allow_extra_params=True)
        except IOError:
            logging.error('Deep model init failed!')
            raise ValueError

# A bunch of image preprocessing interface
    def pre_proc_img_numpy(self, in_numpy):
        img = Image.fromarray(in_numpy)
        return self.pre_proc_img(img)

    def pre_proc_img_file(self, in_file):
        img = Image.open(in_file)
        return self.pre_proc_img(img)

    def pre_proc_img(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_w, img_h = img.size
        short_edge = min(img_w, img_h)
        h_margin = int((img_h - short_edge) / 2)
        w_margin = int((img_w - short_edge) / 2)
        crop_img = img.crop((w_margin, h_margin, w_margin + short_edge, h_margin + short_edge))
        resized_img = np.array(crop_img.resize((self.w, self.h), Image.BICUBIC))
        sample = np.transpose(resized_img, (2, 0, 1))

        normed_img = sample - self.mean_img
        normed_img.resize((1, 3, self.h, self.w))
        return normed_img

    def predict(self, in_list):
        preds = []
        num_img = len(in_list)
        batch_size = self.numpy_batch_size
        num_batch = (num_img - 1) / self.numpy_batch_size + 1
        for idx_batch in range(num_batch):
            cur_batch_size = min(num_img-idx_batch*batch_size, batch_size)
            batch = np.ndarray((cur_batch_size,
                                3, self.h, self.w))
            for idx_img in range(cur_batch_size):
                batch[idx_img, :, :, :] = self.pre_proc_img_numpy(
                                          in_list[idx_batch*batch_size+idx_img])
            out = self.tagger.predict(batch)
            preds.append(out)

        preds = np.vstack(preds)
        return preds

    def predict_lst(self, in_list):
        """
        This function predicts confidence of each tags for given images, taking a list of
        strings indicating image file names as input.
        Parameters
        ----------
        in_list:
            A list containing the image paths.
        -------
        Return:
            A n*m numpy array containing confidence of each classifier to each frame, where
            n equals the number of frames and m equals the number of classifier

        """
        preds = []
        num_img = len(in_list)
        batch_size = self.numpy_batch_size
        num_batch = (num_img - 1) / self.numpy_batch_size + 1
        for idx_batch in range(num_batch):
            cur_batch_size = min(num_img - idx_batch * batch_size, batch_size)
            batch = np.ndarray((cur_batch_size,
                                3, self.h, self.w))
            for idx_img in range(cur_batch_size):
                batch[idx_img, :, :, :] = self.pre_proc_img_file(
                    in_list[idx_batch * batch_size + idx_img])
            out = self.tagger.predict(batch)
            preds.append(out)

        preds = np.vstack(preds)
        return preds

    # predict labels from A PACKED .REC FILE
    def predict_rec(self, in_rec):
        data_shape = self.mean_img.shape
        data_iter = mx.io.ImageRecordIter(path_imgrec=in_rec,
                                          mean_img=self.mean_file,
                                          data_shape=data_shape,
                                          batch_size=self.numpy_batch_size,
                                          shuffle=False)
        preds = self.tagger.predict(data_iter)
        return preds



