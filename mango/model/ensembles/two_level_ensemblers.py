"""Module that contains the two level ensemblers: StackingEnsembler and
BlendingEnsembler classes"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from mango.model.ensembles.base_two_level_ensembler import BaseTwoLevelEnsembler

class StackingEnsembler(BaseTwoLevelEnsembler):
    """Class that combines models using the stacking method"""
    def __init__(self, configs, score_method, predict_as_probability,
                 num_cv_folds):
        BaseTwoLevelEnsembler.__init__(self, configs, score_method,
                                       predict_as_probability, num_cv_folds)
        self.num_training_folds_ = self.configs_['number_training_folds']

    def fit(self, input_data, targets):
        """Train the full ensembler"""
        splitter = StratifiedKFold(self.num_training_folds_)
        combiner_target = np.zeros((len(input_data),))
        formatted_combiner_input = np.zeros((len(input_data),\
          len(self.first_level_models_)))

        for train_index, test_index in splitter.split(input_data, targets):
            first_level_input = input_data[train_index]
            first_level_target = targets[train_index]
            combiner_input = input_data[test_index]

            self.fit_first_level_models(first_level_input, first_level_target)
            formatted_combiner_input[test_index] =\
              self.predict_first_level_models(combiner_input)
            combiner_target[test_index] = targets[test_index]

        self.fit_combiner_model(formatted_combiner_input, combiner_target)

class BlendingEnsembler(BaseTwoLevelEnsembler):
    """Class that combines models using the stacking method"""
    def __init__(self, configs, score_method, predict_as_probability,
                 num_cv_folds):
        BaseTwoLevelEnsembler.__init__(self, configs, score_method,
                                       predict_as_probability, num_cv_folds)
        self.holdout_set_fraction = self.configs_['holdout_set_fraction']

    def fit(self, input_data, targets):
        """Train the full ensembler"""
        first_level_input, combiner_input, first_level_target, combiner_target =\
          train_test_split(input_data, targets, stratify=targets,\
            test_size=self.holdout_set_fraction)

        self.fit_first_level_models(first_level_input, first_level_target)
        formatted_combiner_input = self.predict_first_level_models(combiner_input)
        self.fit_combiner_model(formatted_combiner_input, combiner_target)
