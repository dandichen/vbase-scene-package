
class FeatureExtractor(object):
    def __init__(self, **config):
        # TODO: config feature base class
        pass

    def frame_to_feature(self, image):
        raise NotImplementedError
