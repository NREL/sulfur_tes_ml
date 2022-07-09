import joblib

class stes_model:
    NN_parameters = {'n_layers': 1, 'n_hidden_units': 82, 'batch_size': 2809, 'epochs': 10}
    XGB_parameters = {'learning_rate': 0.06600212850505194, 'subsample': 0.6242681848206246, 'colsample_bytree': 0.7982472652709917, 'num_boost_round': 160}
    RF_parameters = {'n_estimators': 150, 'max_depth': 64, 'max_samples': 0.8785156026362354}
    optimized_model_parameters = {'NN': NN_parameters, 'XGBoost': XGB_parameters, 'RandomForest': RF_parameters}
    
    NN_addendum = {'model_name': None}
    XGB_addendum = {'model_name': None}
    RF_addendum = {'model_name': None}
    final_model_addendum = {'NN': NN_addendum, 'XGBoost': XGB_addendum, 'RandomForest': RF_addendum}
    
    @classmethod
    def get_parameters(cls, model_type='NN'):
        return cls.optimized_model_parameters[model_type]
    
    @classmethod
    def set_parameters(cls, model_type, parameters):
        cls.optimized_model_parameters[model_type] = parameters
    
    @classmethod
    def get_val_index(cls, model_type='NN', model_name=None):
        return cls.final_model_addendum[model_type][model_name]['val_index']
    
    @classmethod
    def get_scaler_x(cls, model_type='NN', model_name=None):
        return cls.final_model_addendum[model_type][model_name]['scaler_x']
    
    @classmethod
    def get_scaler_x(cls, model_type='NN', model_name=None):
        return cls.final_model_addendum[model_type][model_name]['scaler_x']
    
    @classmethod
    def set_addendum(cls, model_type='NN', model_name=None, addendum):
        cls.final_model_addendum[model_type][model_name] = addendum
    
    @classmethod
    def get_addendum(cls, model_type='NN', model_name=None):
        return cls.final_model_addendum[model_type][model_name]
    
    