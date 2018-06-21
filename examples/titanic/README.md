# Example: Kaggle's Titanic Survival Competition

The best way to exemplify how to use mango is through a Kaggle competition. I will use the classic [Titanic Competition](https://www.kaggle.com/c/titanic) to show how the API can make it easier to experiment with different models and achieve interesting results.

## Configuration file

Let's take a look at *titanic\_parameters.json*, the configuration file with all the parameters used during data reading and model building. This JSON file is composed by two main objects: *data* and *model\_pool*. The first one is responsible for data manipulation while the second one is responsible for model configuration.

### data

```json
"data": {
  "data_formatter": {
    "columns_to_remove": ["PassengerId", "Name", "Ticket", "Cabin"],
    "categorical_encoder": "sklearn_LabelEncoder",
    "numeric_fill_value": "column_mean"
  },
  "inputs": {
    "train": {
      "engine": {
        "id": "pandas",
        "mode": "read",
        "parameters": {
          "filepath_or_buffer": "input/train.csv"
        }
      },
      "target_column": "Survived"
    },
    "test": {
      "engine": {
        "id": "pandas",
        "mode": "read",
        "parameters": {
          "filepath_or_buffer": "input/test.csv"
        }
      }
    }
  },
  "output": {
    "columns_to_remove": ["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"],
    "engine": {
        "id": "pandas",
        "mode": "write",
        "parameters": {
          "path_or_buf": "titanic_submission.csv",
          "index": false
        }
      }
  }
}
```

There are three main objects inside Data: *data\_formatter*, *inputs* and *output*.

In the object *data\_formatter* we configure operations that will be performed for both training and testing data processing. For this example we configured three parameters: the CSV columns that won't be used during our modelling (*columns\_to\_remove*), the object that will encode categorical data (*categorical\_encoder*) and how we will fill missing numeric data (*numeric\_fill\_value*). It's important to use an encoder and fill missing numeric values because some models are not capable of dealing with raw string fields or missing data.

The object *inputs* is where we configure the data reading through a pandas wrapper. For the training data we must specify the target column (*target\_column*). The *parameters* attribute inside both the *train* and *test* objects are just the arguments for the pandas method [read\_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html).

In the object *output* we configure how the model predictions will be formatted when we create a CSV output file using the test data. For the titanic submissions we will remove all the columns from the test data, except the PassengerId. The *parameters* attribute is the argument used by the pandas method [to\_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html).

### model\_pool

```json
"model_pool": {
  "predict_as_probability": false,
  "splitter":{
    "id": "mangoml_sklearn_StratifiedKFold",
    "parameters": {
      "n_splits": 5
    }
  },
  "pool": {
    "pipeline_models": [
      {
        "model": {
          "id": "xgboost_XGBClassifier",
          "label": "XGBoost classifier",
          "print_feature_importance": true,
          "parameters": {
          }
        }
      },
      {
        "model": {
          "id": "sklearn_LogisticRegressionCV",
          "label": "Logistic Regression + Hyperparameter search with CV",
          "print_feature_importance": true,
          "parameters": {
          }
        }
      }
    ],
    "ensemble_models": [
    ]
  },
  "scorer": {
    "id": "accuracy",
    "is_score_ascending": true
  }
}
```

Here, we configure a pool of models that will have their performance compared through cross-validation. After the comparison, we pick the best model and train it with all the available training data. The best model is then used to make predictions for the test data.

The splitting method used for cross-validation is configured using the *splitter* object. In this example we use a stratified K-fold cross-validation with 5 bins. We configure if the models will predict probabilities using the attribute *predict\_as\_probability*. We also configure the models that will be compared using cross-validation (inside the *pool* object) and the scoring method (inside the *scorer* object).

We can use two types of models inside the pool: *pipeline\_models* and *ensemble\_models*. For this tutorial we will use only *pipeline\_models*. These models are inspired by the [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) object from scikit-learn (in fact, they are implemented using it). The idea is simple: we configure all the steps used during a machine learning pipeline, from preprocessors like [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to calibrators like [CalibratedClassifierCV](http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html). We then configure the type of model that will be used. In this example we will compare two models: the [XGBClassifier](http://xgboost.readthedocs.io/en/latest/python/python_api.html) from xgboost and the [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html) from scikit-learn.

All the objects in *pool* have a *parameters* object. The attributes configured using this JSON object are directly passed as arguments for the original object constructor, so you can just read in scikit or xgboost documentation which arguments are available and configure them using this field.

Finally, we have to choose how the models will be compared using the object *scorer*. For the titanic competition we will use the classic accuracy score.

And that's it: the data reading and models are configured! Let's take a look at the python code needed for the titanic competition (*titanic\_main.py*).

## Python code

The first step is to read the JSON with all the parameters using the default JSON reader from python.

```
def read_config(config_filename):
    """Read JSON configuration file"""
    config_file = open(config_filename)
    json_data = json.load(config_file)
    return json_data
```

Next we instantiate a DataFormatter object that will be used to format both the train and test data.

```
configs = read_config('config/titanic_parameters.json')
data_formatter = DataFormatter(configs['data'])
```

The next step is to compare the configured models and pick the best one as our winner. We instantiate a Data object to deal with the training data and a ModelPool object to compare all the models and pick the best one. The winner model is then trained with all the available training data.

```
def train_best_model(configs, data_formatter):
    """Choose and fit the best model configured"""
    train_data = Data(configs['data']['inputs']['train'], data_formatter)
    model_pool = ModelPool(configs['model_pool'],\
      configs['model_pool']['scorer']['is_score_ascending'],\
      train_data.get_feature_names())
    model_pool.pick_best_model(train_data)
    model_pool.fit_best_model(train_data)
    return model_pool
```

It's time to obtain some predictions. We instantiate a Data object to deal with the test data and get the predictions from the ModelPool's winner.

```
def get_predictions(configs, data_formatter, model_pool):
    """Get the best predictions"""
    test_data = Data(configs['data']['inputs']['test'], data_formatter)
    predictions = model_pool.get_best_model_predictions(test_data)
    return test_data, predictions
```

Finally we generate an output file using the predictions from the model. We create a Result object that will read the JSON config file to choose which columns will be exported and how the output CSV file will be formatted.

```
def create_output_file(configs, test_data, predictions):
    """Generate an output file with the predictions obtained from best model"""
    result = Result(configs['data']['output'], test_data.get_engine())
    predictions = np.array(predictions)
    result.add_prediction('Survived', predictions)
    result.build_file()
```

And we're done! After running the code we will obtain an output file that can be submitted to Kaggle as our model's prediction. The cross-validation results will be printed and you will have an idea of how well the model will perform with the test data (and hopefully with any unseen data!).

We can do more than we did in this example: we could try using different models, configuring the pipeline\_model parameters to obtain better results or preprocessing the input columns. Maybe we should try ensembling some models? All of this can be done through the JSON configuration file. 

The interesting part is that the python code is almost the same for different experiments. The main changes would be the path for the JSON config file and the name of the output column in the output CSV. The core of all modifications will always be resumed to the JSON config file.
