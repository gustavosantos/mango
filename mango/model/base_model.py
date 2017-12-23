"""Module that contains BaseModel abstract class"""
from abc import ABCMeta, abstractmethod
import numpy as np

class BaseModel(object):
    """Abstract class that defines common methods for all models"""
    __metaclass__ = ABCMeta
    def __init__(self, configs, score_method, predict_as_probability,
                 num_cv_folds):
        self.configs_ = configs
        self.score_method_ = score_method
        self.predict_as_probability_ = predict_as_probability
        self.num_cv_folds_ = num_cv_folds

    @abstractmethod
    def init(self):
        """Abstract method. Should initialize all components from a model"""
        pass

    @abstractmethod
    def fit(self, input_data, targets):
        """Abstract method. Should find the parameters of the model from the
        input_data"""
        pass

    @abstractmethod
    def predict(self, input_data):
        """Abstract method. Should make predictions for the input_data from the
        trained model"""
        pass

    @abstractmethod
    def get_name(self):
        """Abstract method. Should return the label associated with a model
        (used for printing)"""
        pass

    def estimate_performance(self, data):
        """Method used to estimate the performance of the model"""
        scores = []
        for train_index, test_index in data.split_data(self.num_cv_folds_):
            fit_inputs = data.get_features(train_index)
            fit_targets = data.get_targets(train_index)
            self.fit(fit_inputs, fit_targets)

            score_inputs = data.get_features(test_index)
            score_targets = data.get_targets(test_index)
            predictions = self.predict(score_inputs)
            training_predictions = self.predict(fit_inputs)

            if data.is_log_target():
                score_targets = np.exp(score_targets)
                predictions = np.exp(predictions)

            scores.append(self.score_method_(predictions, score_targets))

            #TODO: Time series predictions have a fixed size (equals the test
            #size). Make it flexible.
            if not data.is_time_series():
                print self.get_name(), 'Training score:',\
                  self.score_method_(training_predictions, fit_targets)

            print self.get_name(), 'Last score:', scores[-1]

        print self.get_name(), 'scores:', scores
        print self.get_name(), 'final performance:', np.mean(scores), '\n'
        return np.mean(scores)
