import numpy as np
import cv2
import ConfigParser
from learning.classify import Learner
from sklearn.externals import joblib
import feature

classifier_selector = {'random_forest': Learner}
feature_selector = {'deep': feature.deep.DeepFeatExtractor}


# TODO: A more sofiscated design, save meta data, automatic config file generation,
# multiple classifier support
class TrainingUtil(object):
    def __init__(self, config):
        self.save_path = config.get('learner', 'save_path')
        learner_name = config.get('learner', 'learner_name')

        learner_config_path = config.get('learner', 'learner_config_path')
        option_parser = ConfigParser.ConfigParser()
        option_parser.read(learner_config_path)
        self.learner = classifier_selector[learner_name](option_parser)

        feature_name = config.get('feature', 'feature_name')
        feature_config_path = config.get('feature', 'feature_config_path')
        if not (feature_name in feature_selector):
            raise ValueError('Please provide a correct feature name.')

        option_parser = ConfigParser.ConfigParser()
        option_parser.read(feature_config_path)

        self.feature_extractor = feature_selector[feature_name](option_parser)

    def learn(self, data_list, validation_rate=0.1):
        f = open(data_list, 'r')
        labels = []
        frames = []
        for line in f:
            param = line.split('\t')
            img_file = param[1][:-1]
            im = cv2.imread(img_file)
            frames.append(im)
            labels.append(int(param[0]))
        features = self.feature_extractor.frame_to_feature(frames)
        labels = np.array(labels)

        self.learner.load_data(features, labels, 1-validation_rate)
        validation_score = self.learner.train_test()
        print validation_score

        joblib.dump(self.learner.get_classifier(), self.save_path)

        return validation_score
