import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from stesml.data_tools import get_scenario_index
from stesml.data_tools import get_train_and_val_index
from stesml.data_tools import get_cv
from stesml.data_tools import load_data
from stesml.data_tools import get_train_data
from stesml.data_tools import get_test_data
from stesml.data_tools import get_train_and_test_data

from stesml.postprocessing_tools import get_T
from stesml.postprocessing_tools import get_h

from stesml.data_tools import get_train_and_test_index_short
from stesml.data_tools import get_train_data_short

from optuna.integration import TFKerasPruningCallback

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler

earlystopping_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=2,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
        
def build_NN_model(n_layers=3, n_hidden_units=50):
    model = Sequential()
    model.add(Dense(n_hidden_units, activation='relu', input_shape=(3,)))
    for i in range(n_layers-1):
        model.add(Dense(n_hidden_units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.build()
    return model

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
        model = build_NN_model(n_layers, n_hidden_units)
    else:
        print("Please choose either XGBoost, RandomForest, or NN for model type")
        return None
    return model

def fit_model(model, model_type, X_train, y_train, X_test=None, y_test=None, parameters=None):
    if X_test is None: # If no validation data is passed, validate with training data
        X_test = X_train
        y_test = y_train
        eval_name = 'train'
    else:
        eval_name = 'test'
    if model_type == "NN":
        batch_size = parameters['batch_size']
        epochs = parameters['epochs']
        model.fit(x=X_train, 
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs, # If training ever reaches 100 epochs without early stopping, this should be increased
                  validation_data=(X_test, y_test),
                  callbacks=[earlystopping_callback])
    elif model_type == "XGBoost":
        parameters['eval_metric'] = 'rmse'
        num_boost_round = parameters['num_boost_round']
        dtrain = xgb.DMatrix(data=X_train,
                             label=y_train)
        dtest = xgb.DMatrix(data=X_test,
                             label=y_test)
        model = xgb.train(params=parameters,
                        dtrain=dtrain,
                        num_boost_round=num_boost_round, # If training ever reaches 10000 rounds without early stopping, this should be increased
                        early_stopping_rounds=20,
                        evals=[(dtest,eval_name)],
                        verbose_eval=20)
    elif model_type == "RandomForest":
        model.fit(X_train, y_train)
    return model
    
def get_predictions(model, X_test, y_test=None, scale=False, scaler_y=None, model_type='NN'):
    if model_type == 'XGBoost':
        X_test = xgb.DMatrix(data=X_test)
    y_hat = model.predict(X_test)
    if scale:
        y_hat = scaler_y.inverse_transform(y_hat.reshape(-1,1)).reshape(1,-1)[0]
        y_test = scaler_y.inverse_transform(y_test.reshape(-1,1)).reshape(1,-1)[0]
        return y_hat, y_test
    else:
        return y_hat

def evaluate_results(metric, y_test, y_hat):
    if metric == 'rmse':
        result = mean_squared_error(y_test, y_hat, squared=False)
    elif metric == 'r2':
        result = r2_score(y_test, y_hat)
    else:
        print('Metric must either be rmse or r2')
        return None
    return result
    
def build_train_test_model(data_dir=None, model_type='NN', target='Tavg', metric='rmse', scale=True, parameters=None, n_repeats=1, random_state=5, t_min=-1, t_max=-1):
    result_tot = 0
    addendum = list()
    
    # Get a dataframe with the filepaths of each file in the data directory
    scenario_index = get_scenario_index(data_dir)
    
    # Break out validation set
    train_and_test_index, val_index = get_train_and_val_index(scenario_index, random_state)

    # Split data into train and test sets for cross-validation (80-20 train-test split)
    cv = get_cv(n_repeats, random_state)
    
    # Loop through the splits in cv
    for i, (train_index, test_index) in enumerate(cv.split(train_and_test_index)):

        # Get train and test data
        if scale:
            X_train, y_train, X_test, y_test, scaler_x, scaler_y = get_train_and_test_data(scenario_index, train_index, test_index, target, scale, t_min=t_min, t_max=t_max)
        else:
            X_train, y_train, X_test, y_test = get_train_and_test_data(scenario_index, train_index, test_index, target, t_min=t_min, t_max=t_max)

        # Get the model
        model = get_model(model_type, parameters)

        # Fit the model to training data
        model = fit_model(model, model_type, X_train, y_train, X_test, y_test, parameters)

        # Get predictions for test data
        if scale:
            y_hat, y_test = get_predictions(model, X_test, y_test, scale, scaler_y, model_type)
        else:
            y_hat = get_predictions(model, X_test, model_type=model_type)

        # Evaluate results
        result = evaluate_results(metric, y_test, y_hat)
        result_tot += result
        result_avg = result_tot/(i+1)
        print(f'Split #{i}, This Result: {result:.4f}, Average Result: {result_avg:.4f}')
        
        # Provide addendum for the last trained model
        if scale:
            addendum.append([y_test, y_hat, scenario_index, train_index, test_index, val_index, result, scaler_x, scaler_y])
        else:
            addendum.append([y_test, y_hat, scenario_index, train_index, test_index, val_index, result])

    return result_avg, addendum

def final_train(data_dir=None, model_type='NN', target='Tavg', scale=True, parameters=None, random_state=5, t_min=-1, t_max=-1):
    # Get a dataframe with the filepaths of each file in the data directory
    scenario_index = get_scenario_index(data_dir)
    
    # Break out validation set
    train_index, val_index = get_train_and_val_index(scenario_index, random_state)

    # Get train data
    X_train, y_train = get_train_data(scenario_index, train_index, target, t_min, t_max)
    
    # If requested, scale data
    if scale:
        scaler_x = StandardScaler().fit(X_train)
        X_train = scaler_x.transform(X_train)
        scaler_y = StandardScaler().fit(y_train.reshape(-1,1))
        y_train = scaler_y.transform(y_train.reshape(-1,1)).reshape(1,-1)[0] 
    
    # Get model
    model = get_model(model_type, parameters)
    
    # Train model
    model = fit_model(model, model_type, X_train, y_train, parameters=parameters)
    
    # Return model
    if scale:
        addendum = {'train_index': val_index, 'val_index': val_index, 'scaler_x': scaler_x, 'scaler_y': scaler_y}
    else:
        addendum = {'train_index': val_index, 'val_index': val_index}
        
    return model, addendum

def validate_model(model, model_type='NN', data_dir=None, target='Tavg', scale=True, addendum=None, t_min=-1, t_max=-1):
    
    # get validation data
    scenario_index = get_scenario_index(data_dir)
    val_index = addendum['val_index']
    X_val, y_val = get_train_data(scenario_index, val_index, target, t_min, t_max)
    
    if scale:
        scaler_x = addendum['scaler_x']
        scaler_y = addendum['scaler_y']
        X_val = scaler_x.transform(X_val)
        y_val = scaler_y.transform(y_val.reshape(-1,1)).reshape(1,-1)[0] 
    
    # get predictions for validation data
    if scale:
        y_hat, y_val = get_predictions(model, X_val, y_val, scale, scaler_y, model_type)
    else:
        y_hat = get_predictions(model, X_val, model_type=model_type)
    
    # evaluate results
    rmse = evaluate_results('rmse', y_val, y_hat)
    r2 = evaluate_results('r2', y_val, y_hat)
    print(f'RMSE: {rmse:.7f}, R2: {r2:.7f}')
    
    # return results
    val_df = load_data(scenario_index, val_index, t_min, t_max)
    val_df[target+"_hat"] = y_hat
    
    results = {
        'val_df': val_df,
        'rmse': rmse, 
        'r2': r2
    }

    return results












def get_shuffle_results(model_type, scenario_index, n_estimators, n_shuffle_iterations=1, verbose=0, target='Tavg', calc_T_from_h=False, short=False, t=100):
    
    rmse = 0
    r2 = 0
    rmse_T_tot = 0 # Only used if target is 'h'
    r2_T_tot = 0 # Only used if target is 'h'
    
    for j in range(n_shuffle_iterations):
        
        model = get_model(model_type)
        model.set_params(n_estimators=n_estimators)
        
        if short:
            train_index, train_index_short, test_index = get_train_and_test_index_short(scenario_index)
            X_train, y_train = get_train_data_short(scenario_index, train_index, train_index_short, target, t=t)
        else:
            train_index, test_index = get_train_and_test_index(scenario_index)
            X_train, y_train = get_train_data(scenario_index, train_index, target)
            
        X_test, y_test = get_test_data(scenario_index, test_index, target)

        model.fit(X_train, y_train)

        y_hat = get_predictions(model, X_test)
        
        if calc_T_from_h:
            test_df = load_data(scenario_index, test_index, x=x)
            test_df["h_hat"] = y_hat
            rmse_T, r2_T = get_T_from_h_results(test_df, plot=False)
            rmse_T_tot += rmse_T
            r2_T_tot += r2_T
            rmse_T_cur_avg = rmse_T_tot/(j + 1)
            r2_T_cur_avg = r2_T_tot/(j + 1)
        rmse += mean_squared_error(y_test, y_hat, squared=False)
        r2 += r2_score(y_test,y_hat)
        rmse_cur_avg = rmse/(j + 1)
        r2_cur_avg = r2/(j + 1)

        if verbose >= 1:
            print('Estimators:',n_estimators,'Shuffle:',j,'RMSE:',rmse_cur_avg,'R2:',r2_cur_avg)
            if calc_T_from_h:
                print('RMSE_T:',rmse_T_cur_avg,'R2_T:',r2_T_cur_avg)
        if verbose >= 2:
            print('Predicted:',y_hat)
            print('Expected:',y_test)
    
    rmse = rmse_cur_avg
    r2 = r2_cur_avg
    
    print('# of Estimators: {n_estimators}, RMSE = {rmse:.5f}, r2 = {r2:.5f}'.format(n_estimators=n_estimators, rmse=rmse, r2=r2))
    
    if calc_T_from_h:
        rmse_T = rmse_T_cur_avg
        r2_T = r2_T_cur_avg
        print('RMSE_T:',rmse_T,'R2_T:',r2_T)
        
    return rmse, r2

def get_progress(model_type, scenario_index, min_estimators, max_estimators, step_size, n_shuffle_iterations=1, verbose=0, target='Tavg'):
    rmse_history = list()
    r2_history = list()
    for n_estimators in range(min_estimators, max_estimators + 1, step_size):
        rmse, r2 = get_shuffle_results(model_type, scenario_index, n_estimators, n_shuffle_iterations, verbose, target) 
        rmse_history.append((n_estimators, rmse))
        r2_history.append((n_estimators, r2))
    return rmse_history, r2_history


def get_T_from_h_results(test_df, plot=False):
    T_hat = np.array([])
    T_expected = np.array([])
    for idx, grp in test_df.groupby(["Tw", "Ti"]):
        T_hat_grp = np.array([])
        Ti = grp["Ti"][0]
        Tw = grp["Tw"][0]
        #T_hat_grp = np.append(T_hat_grp, Ti)
        T_prev = Ti
        for i, h in enumerate(grp["h_hat"]):
            if grp['flow-time'][i] < 360:
                T = grp["Tavg_hat"][i]
                T_hat_grp = np.append(T_hat_grp, T)
                T_prev = T
                h_prev = h
                continue
            timestep = grp["flow-time"][i] - grp["flow-time"][i-1]
            T = get_T(T_prev, h_prev, Tw, timestep)
            T_hat_grp = np.append(T_hat_grp, T)
            T_prev = T
            h_prev = h
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
        h_hat_grp = get_h(grp)
        if plot:
            # Plotting results
            grp["h_hat"] = h_hat_grp
            ax = grp.plot(x="flow-time", y='h', c='DarkBlue', linewidth=2.5, label="Expected")
            plot = grp.plot(x="flow-time", y='h_hat', c='DarkOrange', linewidth=2.5, label="Predicted", ax=ax)
            #ax.set_xscale('log')
            ax.set_yscale('log')
            #plot.set_xscale('log')
            plot.set_yscale('log')
            plt.title('Tw = {Tw}  Ti = {Ti}'.format(Tw=idx[0], Ti=idx[1]))
            plt.show()
        h_hat = np.concatenate((h_hat, h_hat_grp))
        h_expected = np.concatenate((h_expected, grp["h"]))
        
    rmse = mean_squared_error(h_expected, h_hat, squared=False)
    r2 = r2_score(h_expected, h_hat)
    
    return rmse, r2











def focal_obj(X_train):
    def custom_obj(y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows = y.shape[0]
        zeros = np.zeros((rows), dtype=float)
        ones = np.ones((rows), dtype=float)
        grad = np.zeros((rows), dtype=float)
        hess = np.zeros((rows), dtype=float)
        Tw = X_train[:,1]
        Ti = X_train[:,2]
        # (y_hat - y)^2 + relu(Ti-y_hat) + relu(y_hat - Tw)
        grad = 4*(y_hat - y) - 0*np.maximum((Ti - y_hat),zeros) + 0*np.maximum((y_hat - Tw),zeros)
        hess = 4*ones + 0*np.maximum(np.sign(Ti - y_hat),zeros) + 0*np.maximum(np.sign(y_hat - Tw),zeros)
        grad = grad.reshape((rows, 1))
        hess = hess.reshape((rows, 1))
        return grad, hess
    return custom_obj

