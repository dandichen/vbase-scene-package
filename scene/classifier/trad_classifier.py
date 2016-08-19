from classifier import Classifier
from feature import deep
import ConfigParser
from learning import classify


feature_selector = {'deep': deep.DeepFeatExtractor}
classifier_selector = {'sklearn': classify.Predictor}


# TODO: Multiple predictor support
class TradClassifier(Classifier):
    def __init__(self, config):
        super(TradClassifier, self).__init__()
        feature_name = config.get('feature', 'feature_name')
        feature_config_path = config.get('feature', 'config_path')
        if not (feature_name in feature_selector):
            raise ValueError('Please provide a correct feature name.')

        option_parser = ConfigParser.ConfigParser()
        option_parser.read(feature_config_path)

        self.feature_extractor = feature_selector[feature_name](option_parser)

        predictor_name = config.get('predictor', 'predictor_name')
        predictor_config_path = config.get('predictor', 'config_path')
        if not (predictor_name in classifier_selector):
            raise ValueError('Please provide a correct learner name.')

        option_parser = ConfigParser.ConfigParser()
        option_parser.read(predictor_config_path)

        self.predictor = classifier_selector[predictor_name](option_parser)

    def predict(self, in_numpy_list):
        features = self.feature_extractor.frame_to_feature(in_numpy_list)
        return self.predictor.predict_prob(features)

