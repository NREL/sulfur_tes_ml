import joblib
import datetime
import xgboost as xgb
from tensorflow import keras

class stes_model:
    ## For the NN model, the first set of parameters here is for training on full datasets
    ## The second set of parameters is for training the model with datasets limited to t <= 360
    NN_parameters = {'n_layers': 1, 'n_hidden_units': 82, 'batch_size': 2809, 'epochs': 10} # Parameters for full training sets
    NN_truncated_parameters = {'n_layers': 1, 'n_hidden_units': 82, 'batch_size': 10, 'epochs': 9} # Parameters for datasets liimited to t <= 360
    XGB_parameters = {'learning_rate': 0.06600212850505194, 'subsample': 0.6242681848206246, 'colsample_bytree': 0.7982472652709917, 'num_boost_round': 160}
    RF_parameters = {'n_estimators': 150, 'max_depth': 64, 'max_samples': 0.8785156026362354}
    optimized_model_parameters = {'NN': NN_parameters, 'NN_truncated': NN_truncated_parameters, 'XGBoost': XGB_parameters, 'RandomForest': RF_parameters}
    
    @classmethod
    def get_parameters(cls, model_type='NN', truncated=False):
        if truncated:
            return cls.optimized_model_parameters[model_type + '_truncated']
        return cls.optimized_model_parameters[model_type]
    @classmethod
    def set_parameters(cls, model_type, parameters, truncated=False):
        if truncated:
            cls.optimized_model_parameters[model_type + '_truncated'] = parameters
        cls.optimized_model_parameters[model_type] = parameters
    
    @classmethod
    def save_model(cls, model, model_type, addendum):
        if model_type == 'NN':
            model.save("../models/" + model_type + "_" + datetime.datetime.now().strftime("%Y%m%d-%H"))
        elif model_type == 'XGBoost':
            model.save_model("../models/" + model_type + "_" + datetime.datetime.now().strftime("%Y%m%d-%H") + ".json")
        joblib.dump(addendum, "../addenda/addendum_" + model_type + "_" + datetime.datetime.now().strftime("%Y%m%d-%H") + ".pkl")
    @classmethod
    def load_model(cls, model_type='NN', model_name=None):
        if model_type == 'NN':
            model = keras.models.load_model("../models/" + model_name)
        elif model_type == 'XGBoost':
            model = xgb.Booster()
            model.load_model("../models/" + model_name + ".json")
        addendum = joblib.load("../addenda/addendum_" + model_name + ".pkl")
        return model, addendum
    

    
    