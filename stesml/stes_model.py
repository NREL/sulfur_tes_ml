import joblib
import datetime
import xgboost as xgb
import pandas as pd
import numpy as np
from tensorflow import keras

from stesml.model_tools import get_predictions

class stes_model:
    # Parameters for full training sets
    NN_parameters = {'n_layers': 1, 'n_hidden_units': 82, 'batch_size': 2809, 'epochs': 10}
    # Parameters for datasets liimited to t <= 360
    NN_trunc_parameters = {'n_layers': 1, 'n_hidden_units': 82, 'batch_size': 10, 'epochs': 9}
    
    XGB_parameters = {'learning_rate': 0.3, 'subsample': 1, 'colsample_bytree': 1, 'num_boost_round': 200}
    RF_parameters = {'n_estimators': 150, 'max_depth': 64, 'max_samples': 0.8785156026362354}
    
    optimized_model_parameters = {
        'NN': NN_parameters, 'NN_trunc': NN_trunc_parameters,
        'XGBoost': XGB_parameters,
        'RandomForest': RF_parameters
    }
    
    @classmethod
    def get_parameters(cls, model_type='NN', truncated=False):
        if truncated and model_type == 'NN':
            return cls.optimized_model_parameters[model_type + '_trunc']
        return cls.optimized_model_parameters[model_type]
    @classmethod
    def set_parameters(cls, model_type, parameters, truncated=False):
        if truncated:
            cls.optimized_model_parameters[model_type + '_trunc'] = parameters
        cls.optimized_model_parameters[model_type] = parameters
    
    @classmethod
    def save_model(cls, model, model_type, addendum):
        if model_type == 'NN':
            model.save("../models/" + model_type + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M"))
        elif model_type == 'XGBoost':
            model.save_model("../models/" + model_type + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + ".json")
        joblib.dump(addendum, "../addenda/addendum_" + model_type + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + ".pkl")
    @classmethod
    def load_model(cls, model_type='NN', model_name=None):
        if model_type == 'NN':
            model = keras.models.load_model("../models/" + model_name)
        elif model_type == 'XGBoost':
            model = xgb.Booster()
            model.load_model("../models/" + model_name + ".json")
        addendum = joblib.load("../addenda/addendum_" + model_name + ".pkl")
        return model, addendum
    
    @classmethod
    def get_htc(cls, Ti=500, Tw=600, end_time=7200, stepsize=0.1):
        model_name = 'XGBoost_h_model'
        model_type = 'XGBoost' # Options: NN, XGBoost, RandomForest
        model, addendum = stes_model.load_model(model_type, model_name)
        flow_time = np.arange(0, end_time, stepsize).tolist()
        Ti_list = [Ti] * len(flow_time)
        Tw_list = [Tw] * len(flow_time)
        X_df = pd.DataFrame()
        X_df['flow-time'] = flow_time
        X_df['Tw'] = Tw_list
        X_df['Ti'] = Ti_list
        X = X_df.to_numpy()
        y_hat = get_predictions(model, X, model_type='XGBoost')
        X_df['h_hat'] = y_hat
        return X_df
    
    @classmethod
    def get_htc_at_time_t(cls, Ti=500, Tw=600, time=2000):
        model_name = 'XGBoost_h_model'
        model_type = 'XGBoost' # Options: NN, XGBoost, RandomForest
        model, addendum = stes_model.load_model(model_type, model_name)
        X = [[time, Tw, Ti]]
        y_hat = get_predictions(model, X, model_type=model_type)
        return y_hat[0]