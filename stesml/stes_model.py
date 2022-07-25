import joblib
import datetime
import xgboost as xgb
import pandas as pd
import numpy as np
from tensorflow import keras

from stesml.model_tools import get_predictions
from stesml.model_tools import get_T_from_h_results

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
    def predict_h(cls, Ti=500, Tw=600, end_time=7200, stepsize=0.1):
        model_name = 'XGBoost_h_model'
        model_type = 'XGBoost'
        model, addendum = stes_model.load_model(model_type, model_name)
        flow_time = np.arange(0, end_time, stepsize).tolist()
        Ti_list = [Ti] * len(flow_time)
        Tw_list = [Tw] * len(flow_time)
        h_df = pd.DataFrame()
        h_df['flow-time'] = flow_time
        h_df['Tw'] = Tw_list
        h_df['Ti'] = Ti_list
        X = h_df.to_numpy()
        y_hat = get_predictions(model, X, model_type='XGBoost')
        h_df['h_hat'] = y_hat
        return h_df
    
    @classmethod
    def predict_h_at_time_t(cls, Ti=500, Tw=600, time=2000):
        model_name = 'XGBoost_h_model'
        model_type = 'XGBoost'
        model, addendum = stes_model.load_model(model_type, model_name)
        X = [[time, Tw, Ti]]
        y_hat = get_predictions(model, X, model_type=model_type)
        return y_hat[0]
    
    @classmethod
    def predict_T(cls, Ti=500, Tw=600, end_time=7200, stepsize=0.1):
        h_df = cls.predict_h(Ti, Tw, end_time, stepsize)
        model_name = 'NN_T_model_tLessThan360'
        model_type = 'NN'
        model, addendum = stes_model.load_model(model_type, model_name)
        flow_time = np.arange(0, end_time, stepsize).tolist()
        Ti_list = [Ti] * len(flow_time)
        Tw_list = [Tw] * len(flow_time)
        T_df = pd.DataFrame()
        T_df['flow-time'] = flow_time
        T_df['Tw'] = Tw_list
        T_df['Ti'] = Ti_list
        X = T_df.to_numpy()
        X_scaled = addendum['scaler_X'].transform(X)
        y_hat = get_predictions(model, X_scaled, model_type='NN', scale=True, scaler_y=addendum['scaler_y'])
        T_df['Tavg_hat'] = y_hat
        h_df['Tavg_hat'] = T_df['Tavg_hat']
        T_df = get_T_from_h_results(h_df, plot=False, hybrid_model=True, hybrid_split_time=360)
        T_df.drop('h_hat',axis=1,inplace=True)
        return T_df
    
    @classmethod
    def predict_T_at_time_t(cls, Ti=500, Tw=600, time=2000):
        if time <= 360:
            model_name = 'NN_T_model_tLessThan360'
            model_type = 'NN'
            model, addendum = stes_model.load_model(model_type, model_name)
            X = np.array([[time, Tw, Ti]])
            X_scaled = addendum['scaler_X'].transform(X)
            y_hat = get_predictions(model, X_scaled, model_type='NN', scale=True, scaler_y=addendum['scaler_y'])
            return y_hat[0]
        else:
            T_df = cls.predict_T(Ti, Tw, end_time=time, stepsize=0.1)
            Tavg_hat = T_df.iloc[-1]['Tavg_hat']
            return Tavg_hat