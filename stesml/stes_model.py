import joblib

class stes_model:
    NN_parameters = {'n_layers': 1, 'n_hidden_units': 82, 'batch_size': 2809, 'epochs': 10}
    XGB_parameters = {'learning_rate': 0.06600212850505194, 'subsample': 0.6242681848206246, 'colsample_bytree': 0.7982472652709917, 'num_boost_round': 160}
    RF_parameters = {'n_estimators': 150, 'max_depth': 64, 'max_samples': 0.8785156026362354}
    optimized_model_parameters = {'NN': NN_parameters, 'XGBoost': XGB_parameters, 'RandomForest': RF_parameters}
    
    @classmethod
    def get_parameters(cls, model_type='NN'):
        return cls.optimized_model_parameters[model_type]
    @classmethod
    def set_parameters(cls, model_type, parameters):
        cls.optimized_model_parameters[model_type] = parameters
    
    @classmethod
    def save_model(cls, model, model_type):
        if model_type == 'NN':
            model.save("../models/" + model_type + "_" + datetime.datetime.now().strftime("%Y%m%d-%H"))
        elif model_type == 'XGBoost':
            model.save_model("../models/" + model_type + "_" + datetime.datetime.now().strftime("%Y%m%d-%H") + ".json")
        joblib.dump(addendum, "../addenda/addendum_" + model_type + "_" + datetime.datetime.now().strftime("%Y%m%d-%H") + ".pkl")
    @classmethod
    def load_model(cls, model_type='NN', model_name=None):
        if model_type == 'NN':
            model = keras.models.load_model(model_name)
        elif model_type == 'XGBoost':
            model = xgb.Booster()
            model.load_model(model_name)
        addendum - joblib.load("addendum_" + model_name)
        return model, addendum
    

    
    