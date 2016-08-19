from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import cross_validation
import numpy as np
import ast
import ConfigParser
from sklearn.externals import joblib

__author__ = 'tongjiang, yiwang'

classifier_selector = {
    'random_forest': RandomForestClassifier,
    'LSVC': LinearSVC
}


#  TODO: wrap when necessary!
class Learner:
    def __init__(self, config):
        classifier_name = config.get('classifier', 'classifier_name')
        if not (classifier_name in classifier_selector):
            raise ValueError('Please provide a correct classifier name.')
        options = dict(config.items(classifier_name))
        if len(options) != 0:
            for i in options:
                options[i] = ast.literal_eval(options[i])
        else:
            options = dict()

        self.classifier = classifier_selector[classifier_name](**options)
        self.features = []
        self.lbl = []
        self.n = 0
        self.train = 0

    def load_data(self, features, lbl, rate=0.7, shuffle=False):
        if len(features) != len(lbl):
            raise ValueError('Can\'t match features and labels')
        self.n = len(features)
        self.train = rate
        if not shuffle:
            self.features = features
            self.lbl = lbl
        else:
            idx = np.arange(self.n)
            np.random.shuffle(idx)
            self.features = np.array([features[idx[i]] for i in range(self.n)])
            self.lbl = np.array([lbl[idx[i]] for i in range(self.n)])

    def train_test(self):
        split = int(self.n * self.train)
        self.classifier.fit(self.features[:split],
                            self.lbl[:split])
        result = self.classifier.predict(self.features[:split])
        error = np.array(result != self.lbl[:split])

        print "Train Error:" + str(float(np.sum(error)) / split)

        result = self.classifier.predict(self.features[split:])
        error = np.array(result != self.lbl[split:])
        fn = np.array(np.logical_and(result == 0, self.lbl[split:] == 1))
        fp = np.array(np.logical_and(result == 1, self.lbl[split:] == 0))
        correct_rate = 1 - float(np.sum(error))/(self.n-split)
        print "Precision: " + str(1 - float(np.sum(fp)) / np.sum(result))
        print "Recall: " + str(1 - float(np.sum(fn)) / np.sum(self.lbl[split:]))
        print "Validation Error:" + str(float(np.sum(error)) / (self.n-split))
        print "Total:" + str(np.sum(self.lbl[split:])) + ' ' + str(self.n-split)
        return correct_rate

    def cross_validation(self, k):
        return cross_validation.cross_val_score(self.classifier,
                                                self.features,
                                                self.lbl,
                                                cv=k)

    def get_classifier(self):
        return self.classifier


class Predictor:
    def __init__(self, config):
        predictor_path = config.get('predictor', 'predictor_path')
        self.classifier = joblib.load(predictor_path)

    def predict(self, features):
        result = self.classifier.predict(features)
        return result

    def predict_prob(self, features):
        result = self.classifier.predict_proba(features)
        result = np.vstack(result)[:, 1]
        return result

