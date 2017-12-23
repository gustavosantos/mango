"""Module that contains ModelFactory class"""
import operator
from sklearn import neural_network
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import svm
from sklearn import dummy
import xgboost as xgb
from mango.model.models.regressors.prophet_wrapper import ProphetWrapper

class ModelFactory(object):
    """Class used to build models"""
    def build_mango(self, model_id, model_params):
        """Method that builds models/wrappers implemented by mango"""
        if model_id == 'mango_fbprophet_Prophet':
            return ProphetWrapper(**model_params)
        return None

    def build_sklearn(self, model_id, model_params):
        """Method that builds models implemented in sklearn"""
        if model_id == 'sklearn_LogisticRegressionCV':
            return linear_model.LogisticRegressionCV(**model_params)
        elif model_id == 'sklearn_MLPClassifier':
            return neural_network.MLPClassifier(**model_params)
        elif model_id == 'sklearn_GaussianNB':
            return naive_bayes.GaussianNB(**model_params)
        elif model_id == 'sklearn_MultinomialNB':
            return naive_bayes.MultinomialNB(**model_params)
        elif model_id == 'sklearn_BernoulliNB':
            return naive_bayes.BernoulliNB(**model_params)
        elif model_id == 'sklearn_RandomForestClassifier':
            return ensemble.RandomForestClassifier(**model_params)
        elif model_id == 'sklearn_SVC':
            return svm.SVC(**model_params)
        elif model_id == 'sklearn_AdaBoostClassifier':
            return ensemble.AdaBoostClassifier(**model_params)
        elif model_id == 'sklearn_SGDClassifier':
            return linear_model.SGDClassifier(**model_params)
        elif model_id == 'sklearn_PassiveAggressiveClassifier':
            return linear_model.PassiveAggressiveClassifier(**model_params)
        elif model_id == 'sklearn_RidgeClassifier':
            return linear_model.RidgeClassifier(**model_params)
        elif model_id == 'sklearn_DummyClassifier':
            return dummy.DummyClassifier(**model_params)
        elif model_id == 'sklearn_LinearRegression':
            return linear_model.LinearRegression(**model_params)
        elif model_id == 'sklearn_LassoCV':
            return linear_model.LassoCV(**model_params)
        elif model_id == 'sklearn_DummyRegressor':
            return dummy.DummyRegressor(**model_params)
        return None

    def build_xgboost(self, model_id, model_params):
        """Method that builds models implemented in xgboost"""
        if model_id == 'xgboost_XGBClassifier':
            return xgb.XGBClassifier(**model_params)
        elif model_id == 'xgboost_XGBRegressor':
            return xgb.XGBRegressor(**model_params)
        return None

    def build(self, model_id, model_params):
        """Method that builds models using the specified id"""
        if 'mango' in model_id:
            return self.build_mango(model_id, model_params)
        elif 'sklearn' in model_id:
            return self.build_sklearn(model_id, model_params)
        elif 'xgboost' in model_id:
            return self.build_xgboost(model_id, model_params)
        return None

    def print_sklearn_feature_importance(self, model, column_names):
        """Method used to print the feature importance of sklearn models"""
        if model.coef_.ndim == 1:
            coefs = dict(zip(column_names, model.coef_))
        else:
            coefs = dict(zip(column_names, model.coef_[0]))
        sorted_keys = sorted(coefs, key=lambda dict_key: abs(coefs[dict_key]))
        print 'Feature coefficients:', [(key, coefs[key])\
                                        for key in sorted_keys]

    def print_xgboost_feature_importance(self, model, column_names):
        """Method used to print the feature importance of xbgoost models"""
        weights = model.booster().get_score(importance_type='weight')
        weights = {column_names[int(key[1:])]: value
                   for key, value in weights.items()}
        print 'Feature coefficients:', sorted(weights.items(),
                                              key=operator.itemgetter(1))

    def print_feature_importance(self, model, model_id, column_names):
        """Method used to print the obtained importance of each feature
        after training"""
        if 'xgb' in model_id:
            self.print_xgboost_feature_importance(model, column_names)
        elif 'LinearRegression' in model_id or 'LogisticRegression' in model_id:
            self.print_sklearn_feature_importance(model, column_names)
        else:
            print 'Model', model_id, 'cannot display coefficients'
