import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from stesml.data_tools import get_scenario_index
from stesml.data_tools import get_index_splits
from stesml.data_tools import get_cv
from stesml.data_tools import get_dataframe
from stesml.data_tools import get_split_data
from stesml.data_tools import get_scaled_data

from stesml.postprocessing_tools import get_T_from_h
from stesml.postprocessing_tools import get_h_from_T

from tensorflow.keras.callbacks import EarlyStopping

def get_model(model_type, parameters):
    if model_type == "XGBoost":
        # No need to return model for XGBoost
        # Model is instantiated and trained via xgboost.train in the fit_model method below
        model = None
    elif model_type == "RandomForest":
        n_estimators = parameters['n_estimators']
        max_depth = parameters['max_depth']
        max_samples = parameters['max_samples']
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples, n_jobs=-1)
    elif model_type == "NN":
        n_layers = parameters['n_layers']
        n_hidden_units = parameters['n_hidden_units']
        model = Sequential()
        model.add(Dense(n_hidden_units, activation='relu', input_shape=(3,)))
        for i in range(n_layers-1):
            model.add(Dense(n_hidden_units, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.build()
    else:
        print("Please choose either XGBoost, RandomForest, or NN for model type")
        return None
    return model

def fit_model(model, model_type, X_train, y_train, X_val=None, y_val=None, parameters=None):
    if X_val is None: # If no validation data is passed, validate with training data
        X_val = X_train
        y_val = y_train
        eval_name = 'train'
    else:
        eval_name = 'val'
    if model_type == "NN":
        earlystopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=2,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        batch_size = parameters['batch_size']
        epochs = parameters['epochs']
        model.fit(x=X_train, 
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs, # If training ever reaches 100 epochs without early stopping, this should be increased
                  validation_data=(X_val, y_val),
                  callbacks=[earlystopping_callback])
    elif model_type == "XGBoost":
        parameters['eval_metric'] = 'rmse'
        num_boost_round = parameters['num_boost_round']
        dtrain = xgb.DMatrix(data=X_train,
                             label=y_train)
        dval = xgb.DMatrix(data=X_val,
                             label=y_val)
        model = xgb.train(params=parameters,
                        dtrain=dtrain,
                        num_boost_round=num_boost_round, # If training ever reaches 10000 rounds without early stopping, this should be increased
                        early_stopping_rounds=20,
                        evals=[(dval,eval_name)],
                        verbose_eval=20)
    elif model_type == "RandomForest":
        model.fit(X_train, y_train)
    return model
    
def get_predictions(model, X, y=None, scale=False, scaler_y=None, model_type='NN'):
    if model_type == 'XGBoost':
        X = xgb.DMatrix(data=X)
    y_hat = model.predict(X)
    if scale:
        y_hat = scaler_y.inverse_transform(y_hat.reshape(-1,1)).reshape(1,-1)[0]
        y = scaler_y.inverse_transform(y.reshape(-1,1)).reshape(1,-1)[0]
        return y_hat, y
    else:
        return y_hat

def evaluate_results(metric, y, y_hat):
    if metric == 'rmse':
        result = mean_squared_error(y, y_hat, squared=False)
    elif metric == 'r2':
        result = r2_score(y, y_hat)
    else:
        print('Metric must either be rmse or r2')
        return None
    return result
    
def train_and_validate_model(data_dir=None, model_type='NN', target='Tavg', metric='rmse', scale=True, parameters=None, n_repeats=1, random_state=5, t_min=-1, t_max=-1, split_test_data=False, features=["flow-time", "Tw", "Ti"]):
    result_tot = 0
    addenda = list()
    
    # Get a dataframe with the filepaths of each file in the data directory
    scenario_index = get_scenario_index(data_dir)
    
    # If requested, break out test set
    if split_test_data:
        train_and_val_index, test_index = get_index_splits(scenario_index, random_state)
    else:
        train_and_val_index = scenario_index.index
        test_index = None

    # Generate cross-validation object (80-20 train-val split)
    cv = get_cv(n_repeats, random_state)
    
    # Loop through the splits in cv
    for i, (train_index, val_index) in enumerate(cv.split(train_and_val_index)):

        # Get train and val data
        X_train, y_train = get_split_data(scenario_index, train_index, target, t_min, t_max, features)
        X_val, y_val = get_split_data(scenario_index, val_index, target, t_min, t_max, features)
    
        # If requested, scale data
        if scale:
            X_train, y_train, scaler_X, scaler_y = get_scaled_data(X_train, y_train)
            X_val, y_val = get_scaled_data(X_val, y_val, scaler_X, scaler_y)
        else:
            scaler_X, scaler_y = None, None

        # Get the model
        model = get_model(model_type, parameters)

        # Fit the model to training data
        model = fit_model(model, model_type, X_train, y_train, X_val, y_val, parameters)

        # Get predictions for validation data
        if scale:
            y_hat, y_val = get_predictions(model, X_val, y_val, scale, scaler_y, model_type)
        else:
            y_hat = get_predictions(model, X_val, model_type=model_type)

        # Evaluate results
        result = evaluate_results(metric, y_val, y_hat)
        result_tot += result
        result_avg = result_tot/(i+1)
        print(f'Split #{i}, This Result: {result:.4f}, Average Result: {result_avg:.4f}')
        
        # Provide addendum for this model
        addendum = {
            'y_val': y_val,
            'y_hat': y_hat,
            'scenario_index': scenario_index,
            'train_index': train_index, 
            'val_index': val_index, 
            'test_index': test_index, 
            'result': result,
            'scaler_X': scaler_X, 
            'scaler_y': scaler_y
            }
        addenda.append(addendum)
    
    return result_avg, addenda

def train_model(data_dir=None, model_type='NN', target='Tavg', scale=True, parameters=None, random_state=5, t_min=-1, t_max=-1, features=["flow-time", "Tw", "Ti"]):
    # Get a dataframe with the filepaths of each file in the data directory
    scenario_index = get_scenario_index(data_dir)
    
    # Break out test set
    train_index, test_index = get_index_splits(scenario_index, random_state)

    # Get train data, and test data for validation while training
    X_train, y_train = get_split_data(scenario_index, train_index, target, t_min, t_max, features)
    X_test, y_test = get_split_data(scenario_index, test_index, target, t_min, t_max, features)
    
    # If requested, scale data
    if scale:
        X_train, y_train, scaler_X, scaler_y = get_scaled_data(X_train, y_train)
        X_test, y_test = get_scaled_data(X_test, y_test, scaler_X, scaler_y)
    else:
        scaler_X, scaler_y = None, None
    
    # Get model
    model = get_model(model_type, parameters)
    
    # Train model
    model = fit_model(model, model_type, X_train, y_train, X_test, y_test, parameters=parameters)
    
    # Populate addendum
    addendum = {'train_index': train_index, 'test_index': test_index, 'scaler_X': scaler_X, 'scaler_y': scaler_y}
    
    return model, addendum

def test_model(model, model_type='NN', data_dir=None, target='Tavg', scale=True, addendum=None, t_min=-1, t_max=-1, features=["flow-time", "Tw", "Ti"]):
    # Get test data
    scenario_index = get_scenario_index(data_dir)
    if 'test_index' not in addendum: # This is here for backwards compatibility, so older models still work
        test_index = addendum['val_index']
    else:
        test_index = addendum['test_index']
    X_test, y_test = get_split_data(scenario_index, test_index, target, t_min, t_max, features)
    
    # If requested, scale data
    if scale:
        if 'scaler_X' not in addendum: # This is here for backwards compatibility, so older models still work
            scaler_X = addendum['scaler_x']
        else:
            scaler_X = addendum['scaler_X']
        scaler_y = addendum['scaler_y']
        X_test, y_test = get_scaled_data(X_test, y_test, scaler_X, scaler_y)
    
    # get predictions for test data
    if scale:
        y_hat, y_test = get_predictions(model, X_test, y_test, scale, scaler_y, model_type)
    else:
        y_hat = get_predictions(model, X_test, model_type=model_type)
    
    # evaluate results
    rmse = evaluate_results('rmse', y_test, y_hat)
    r2 = evaluate_results('r2', y_test, y_hat)
    print(f'RMSE: {rmse:.7f}, R2: {r2:.7f}')
    
    # return results
    test_df = get_dataframe(scenario_index, test_index, t_min, t_max)
    test_df[target+"_hat"] = y_hat
    
    results = {
        'test_df': test_df,
        'rmse': rmse, 
        'r2': r2
    }

    return results

def get_T_from_h_results(test_df, plot=False, hybrid_model=False, hybrid_split_time=-1):
    T_hat = np.array([])
    T_expected = np.array([])
    for idx, grp in test_df.groupby(["Tw", "Ti"]):
        T_hat_grp = get_T_from_h(grp, hybrid_model, hybrid_split_time)
        if plot:
            # Plotting results
            grp["T_hat"] = T_hat_grp
            ax = grp.plot(x="flow-time", y='Tavg', c='DarkBlue', linewidth=2.5, label="Expected")
            plot = grp.plot(x="flow-time", y='T_hat', c='DarkOrange', linewidth=2.5, label="Predicted", ax=ax)
            plt.title('Tw = {Tw}  Ti = {Ti}'.format(Tw=idx[0], Ti=idx[1]))
            plt.show()
        T_hat = np.concatenate((T_hat, T_hat_grp))
        T_expected = np.concatenate((T_expected, grp["Tavg"]))
        
    rmse = mean_squared_error(T_expected, T_hat, squared=False)
    r2 = r2_score(T_expected, T_hat)
    
    return rmse, r2

def get_h_from_T_results(test_df, plot=False):
    h_hat = np.array([])
    h_expected = np.array([])
    for idx, grp in test_df.groupby(["Tw", "Ti"]):
        h_hat_grp = get_h_from_T(grp)
        if plot:
            # Plotting results
            grp["h_hat"] = h_hat_grp
            ax = grp.plot(x="flow-time", y='h', c='DarkBlue', linewidth=2.5, label="Expected")
            plot = grp.plot(x="flow-time", y='h_hat', c='DarkOrange', linewidth=2.5, label="Predicted", ax=ax)
            ax.set_yscale('log')
            ax.set_xlim(0.001,7199)
            plot.set_yscale('log')
            plt.title('Tw = {Tw}  Ti = {Ti}'.format(Tw=idx[0], Ti=idx[1]))
            plt.show()
        h_hat = np.concatenate((h_hat, h_hat_grp))
        h_expected = np.concatenate((h_expected, grp["h"]))
        
    rmse = mean_squared_error(h_expected, h_hat, squared=False)
    r2 = r2_score(h_expected, h_hat)
    
    return rmse, r2