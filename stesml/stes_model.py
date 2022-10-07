import joblib
import datetime
import xgboost as xgb
import pandas as pd
import numpy as np
from tensorflow import keras

from stesml.model_tools import get_predictions
from stesml.model_tools import get_T_from_h_results

class stes_model:
    # Parameters for t=[0,7200]
    NN_parameters = {'n_layers': 1, 'n_hidden_units': 469, 'batch_size': 1471, 'learning_rate': 0.1249326681143701, 'epochs': 6}
    
    # Parameters for datasets liimited to t <= 360
    NN_trunc_parameters = {'n_layers': 2, 'n_hidden_units': 72, 'batch_size': 13, 'learning_rate': 0.00048181964195323425, 'epochs': 6}
    
    XGB_trunc_parameters = { # XGBoost Parameters for t_max = 360, t_min = 0.000001 (to eliminate t=0 datapoint)
        'learning_rate': 0.30286086065588325,
        'subsample': 0.10056651740231178,
        'max_depth': 4,
        'num_boost_round': 250
    }
    
    XGB_parameters = { # XGBoost Parameters for t_min = 360
        'learning_rate': 0.2744925045562501,
        'subsample': 0.49442114581173074,
        'max_depth': 4,
        'num_boost_round': 250
    }
    
    RF_parameters = {'n_estimators': 150, 'max_depth': 64, 'max_samples': 0.8785156026362354}
    
    optimized_model_parameters = {
        'NN': NN_parameters, 'NN_trunc': NN_trunc_parameters,
        'XGBoost': XGB_parameters, 'XGBoost_trunc': XGB_trunc_parameters,
        'RandomForest': RF_parameters
    }
    
    @classmethod
    def get_parameters(cls, model_type='NN', truncated=False):
        if truncated:
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
        joblib.dump(addendum, "../models/addenda/addendum_" + model_type + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + ".pkl")
        
    @classmethod
    def load_model(cls, model_type='NN', model_name=None):
        if model_type == 'NN':
            model = keras.models.load_model("../models/" + model_name)
        elif model_type == 'XGBoost':
            model = xgb.Booster()
            model.load_model("../models/" + model_name + ".json")
        addendum = joblib.load("../models/addenda/addendum_" + model_name + ".pkl")
        return model, addendum
    
    @classmethod
    def predict_h(cls, Ti=500, Tw=600, start_time=360, end_time=7200, stepsize=0.1, model_name='XGBoost_h_model', model_type='XGBoost'):
        # Load the model
        model, addendum = stes_model.load_model(model_type, model_name)
        # Format the input data
        flow_time = np.arange(start_time, end_time, stepsize).tolist()
        Ti_list = [Ti] * len(flow_time)
        Tw_list = [Tw] * len(flow_time)
        h_df = pd.DataFrame()
        h_df['flow-time'] = flow_time
        h_df['Tw'] = Tw_list
        h_df['Ti'] = Ti_list
        X = h_df.to_numpy()
        if model_type == 'NN':
            # Scale the input data
            X = addendum['scaler_X'].transform(X)
            scale = True
        else:
            scale = False
        # Get predictions
        y_hat = get_predictions(model, X, model_type=model_type, scale=scale, scaler_y=addendum['scaler_y'])
        h_df['h_hat'] = y_hat
        return h_df
    
    @classmethod
    def predict_h_at_time_t(cls, Ti=500, Tw=600, time=2000, model_name='XGBoost_h_model', model_type='XGBoost'):
        # Load the model
        model, addendum = stes_model.load_model(model_type, model_name)
        # Format the input data
        X = [[time, Tw, Ti]]
        if model_type == 'NN':
            # Scale the input data
            X = addendum['scaler_X'].transform(X)
            scale = True
        else:
            scale = False
        # Get prediction
        y_hat = get_predictions(model, X, model_type=model_type, scale=scale, scaler_y=addendum['scaler_y'])
        return y_hat[0]
    
    @classmethod
    def predict_average_h(cls, Ti=500, Tw=600, start_time=0, end_time=7200, stepsize=0.1, model_name='XGBoost_h_model', model_type='XGBoost'):
        # Simulate up until 'time' with a default stepsize of 0.1 seconds
        h_df = cls.predict_h(Ti, Tw, end_time=end_time, stepsize=stepsize, model_name=model_name, model_type=model_type)
        h_df.set_index('flow-time', inplace=True)
        h_df = h_df.truncate(before=start_time, after=end_time)
        print(h_df)
        # And return only the prediction for t=time (the last prediction)
        average_h_hat = h_df['h_hat'].mean()
        return average_h_hat
    
    @classmethod
    def predict_T(cls, Ti=500, Tw=600, end_time=7200, stepsize=0.1, T_model_name='NN_T_model_tLessThan360', T_model_type='NN',  
                  hybrid_model=False, h_model_name='XGBoost_h_model', h_model_type='XGBoost', hybrid_split_time=360):
        if hybrid_model:
            # This is the hybrid model, so first we need to get h predictions
            h_df = cls.predict_h(Ti, Tw, 0, end_time, stepsize, h_model_name, h_model_type)
            # Load the Temperature model
            model, addendum = stes_model.load_model(T_model_type, T_model_name)
            # Format the input data
            flow_time = np.arange(0, end_time, stepsize).tolist()
            Ti_list = [Ti] * len(flow_time)
            Tw_list = [Tw] * len(flow_time)
            T_df = pd.DataFrame()
            T_df['flow-time'] = flow_time
            T_df['Tw'] = Tw_list
            T_df['Ti'] = Ti_list
            X = T_df.to_numpy()
            if T_model_type == 'NN':
                # Scale the input data
                X = addendum['scaler_X'].transform(X)
                scale = True
            else:
                scale = False
            # Get predictions
            y_hat = get_predictions(model, X, model_type=T_model_type, scale=scale, scaler_y=addendum['scaler_y'])
            T_df['Tavg_hat'] = y_hat
            h_df['Tavg_hat'] = T_df['Tavg_hat']
            # Combine T predictions with T calculations from h
            T_df = get_T_from_h_results(h_df, plot=False, hybrid_model=True, hybrid_split_time=hybrid_split_time, predictions=True)
            T_df.drop('h_hat',axis=1,inplace=True)
            return T_df
        else:
            # Load the model
            model, addendum = stes_model.load_model(T_model_type, T_model_name)
            # Format the input data
            flow_time = np.arange(0, end_time, stepsize).tolist()
            Ti_list = [Ti] * len(flow_time)
            Tw_list = [Tw] * len(flow_time)
            T_df = pd.DataFrame()
            T_df['flow-time'] = flow_time
            T_df['Tw'] = Tw_list
            T_df['Ti'] = Ti_list
            X = T_df.to_numpy()
            if T_model_type == 'NN':
                # Scale the input data
                X = addendum['scaler_X'].transform(X)
                scale = True
            else:
                scale = False
            # Get predictions
            y_hat = get_predictions(model, X, model_type=T_model_type, scale=scale, scaler_y=addendum['scaler_y'])
            T_df['Tavg_hat'] = y_hat
            return T_df
    
    
    @classmethod
    def predict_T_at_time_t(cls, Ti=500, Tw=600, time=7200, T_model_name='NN_T_model_tLessThan360', T_model_type='NN',  
                            hybrid_model=False, h_model_name='XGBoost_h_model', h_model_type='XGBoost', hybrid_split_time=360):
        # Simulate up until 'time' with a default stepsize of 0.1 seconds
        T_df = cls.predict_T(Ti, Tw, end_time=time, stepsize=0.1, T_model_name=T_model_name, T_model_type=T_model_type, 
                             hybrid_model=hybrid_model, h_model_name=h_model_name,  h_model_type=h_model_type, hybrid_split_time=hybrid_split_time)
        # And return only the prediction for t=time (the last prediction)
        Tavg_hat = T_df.iloc[-1]['Tavg_hat']
        return Tavg_hat
    
    @classmethod
    def predict_average_T(cls, Ti=500, Tw=600, start_time=0, end_time=7200, stepsize=0.1, T_model_name='NN_T_model_tLessThan360', T_model_type='NN',  
                            hybrid_model=False, h_model_name='XGBoost_h_model', h_model_type='XGBoost', hybrid_split_time=360):
        # Simulate up until 'time' with a default stepsize of 0.1 seconds
        T_df = cls.predict_T(Ti, Tw, end_time=end_time, stepsize=stepsize, T_model_name=T_model_name, T_model_type=T_model_type, 
                             hybrid_model=hybrid_model, h_model_name=h_model_name,  h_model_type=h_model_type, hybrid_split_time=hybrid_split_time)
        T_df.set_index('flow-time', inplace=True)
        T_df = T_df.truncate(before=start_time, after=end_time)
        print(T_df)
        # And return only the prediction for t=time (the last prediction)
        average_Tavg_hat = T_df['Tavg_hat'].mean()
        return average_Tavg_hat
    
    