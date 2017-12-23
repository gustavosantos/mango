"""Module that contains Data class"""
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold, PredefinedSplit
from mango.data.engines.engine_factory import EngineFactory

class Data(object):
    """Class used to manipulate data"""
    def __init__(self, configs, data_formatter):
        self.configs_ = configs
        self.data_formatter_ = data_formatter
        self.engine_ = self.build_engine()
        self.targets_ = self.init_target()
        self.features_names_, self.features_ = self.init_features()
        self.is_time_series_ = self.configs_.get('time_series_feature', False)

    def build_engine(self):
        """Build the configured engine"""
        engine_factory = EngineFactory(self.configs_['engine'])
        return engine_factory.build()

    def init_target(self):
        """Initialize the target array"""
        if 'target_column' in self.configs_:
            target_column = self.configs_['target_column']
            targets = self.engine_.get_column_as_array(target_column)
            self.engine_ = self.engine_.erase_columns([target_column])
            return self.data_formatter_.get_formatted_target(targets)
        return None

    def init_features(self):
        """Initialize the features matrix"""
        return self.data_formatter_.get_formatted_features(self.engine_)

    def stratified_split(self, num_bins):
        """Perform a stratified split of the data"""
        splitter = StratifiedKFold(num_bins)
        return splitter.split(self.features_, self.targets_)

    def grouped_split(self, num_bins):
        """Perform a split of the data based on a group"""
        group_column = str(self.configs_['group_column'])
        groups = self.engine_.get_column_as_array(group_column)
        splitter = GroupKFold(num_bins)
        return splitter.split(self.features_, self.targets_, groups)

    def random_split(self, num_bins):
        """Perform a totally random split of the data"""
        splitter = KFold(num_bins)
        return splitter.split(self.features_, self.targets_)

    def fixed_split(self):
        """Perform a fixed data split with a fixed size for the test fold"""
        test_size = self.configs_['split_parameters']['test_size']

        data_size = 0
        if self.is_time_series_:
            #Time series split data by columns
            data_size = self.features_.shape[1]
        else:
            data_size = self.features_.shape[0]

        test_fold = [-1 for i in range(0, data_size - test_size)]
        test_fold += [0 for i in range(data_size - test_size, data_size)]
        splitter = PredefinedSplit(test_fold=test_fold)
        return splitter.split()

    def split_data(self, num_bins):
        """Split data in num_bins bins"""
        if self.configs_['split_method'] == 'group':
            return self.grouped_split(num_bins)
        elif self.configs_['split_method'] == 'stratified':
            return self.stratified_split(num_bins)
        elif self.configs_['split_method'] == 'random':
            return self.random_split(num_bins)
        elif self.configs_['split_method'] == 'fixed':
            return self.fixed_split()
        return None

    def split_array_by_column(self, input_data, indexes):
        """Split an array by column using a sequential index array"""
        if indexes is None:
            return input_data
        beg = indexes[0]
        end = indexes[-1] + 1
        return np.array([data[beg:end] for data in input_data])

    def get_targets(self, indexes=None):
        """Return the target array"""
        if self.is_time_series_:
            return self.split_array_by_column(self.features_, indexes)

        if indexes is not None:
            return self.targets_[indexes]
        return self.targets_

    def get_features(self, indexes=None):
        """Return the features matrix"""
        if self.is_time_series_:
            return self.split_array_by_column(self.features_, indexes)

        feature_array = self.features_
        if 'text_feature' in self.configs_ and self.configs_['text_feature']:
            feature_array = np.array([text_feature[0] \
                for text_feature in self.features_])
        if indexes is not None:
            return feature_array[indexes]
        return feature_array

    def is_log_target(self):
        """Returns true when the target is in log scale"""
        return self.data_formatter_.is_log_target()

    def is_time_series(self):
        """Return true when data is a time series"""
        return self.is_time_series_

    def get_engine(self):
        """Return the internal engine"""
        return self.engine_

    def get_feature_names(self):
        """Return a list with all feature names obtained from data"""
        return self.features_names_
