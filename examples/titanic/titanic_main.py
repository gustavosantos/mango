"""Module that contains main"""
import json
import numpy as np

from mangoml.data.data import Data
from mangoml.data.result import Result
from mangoml.data.data_formatter import DataFormatter
from mangoml.model.model_pool import ModelPool

def read_config(config_filename):
    """Read JSON configuration file"""
    config_file = open(config_filename)
    json_data = json.load(config_file)
    return json_data

def train_best_model(configs, data_formatter):
    """Choose and fit the best model configured"""
    train_data = Data(configs['data']['inputs']['train'], data_formatter)

    model_pool = ModelPool(configs['model_pool'],\
      configs['model_pool']['scorer']['is_score_ascending'],\
      train_data.get_feature_names())
    model_pool.pick_best_model(train_data)
    model_pool.fit_best_model(train_data)

    return model_pool

def get_predictions(configs, data_formatter, model_pool):
    """Get the best predictions"""
    test_data = Data(configs['data']['inputs']['test'], data_formatter)
    predictions = model_pool.get_best_model_predictions(test_data)

    return test_data, predictions

def create_output_file(configs, test_data, predictions):
    """Generate an output file with the predictions obtained from best model"""
    result = Result(configs['data']['output'], test_data.get_engine())
    predictions = np.array(predictions)

    result.add_prediction('Survived', predictions)
    result.build_file()

def main():
    """Main function"""
    configs = read_config('config/titanic_parameters.json')
    data_formatter = DataFormatter(configs['data'])

    model_pool = train_best_model(configs, data_formatter)
    test_data, predictions = get_predictions(configs, data_formatter,
                                             model_pool)
    create_output_file(configs, test_data, predictions)

if __name__ == "__main__":
    main()
