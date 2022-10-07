import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime

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
        if y is not None:
            y = scaler_y.inverse_transform(y.reshape(-1,1)).reshape(1,-1)[0]
            return y_hat, y
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
    rmse_tot = 0
    r2_tot = 0
    result = {'rmse': list(), 'r2': list()}
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
        rmse = evaluate_results('rmse', y_val, y_hat)
        r2 = evaluate_results('r2', y_val, y_hat)
        rmse_tot += rmse
        r2_tot += r2
        rmse_avg = rmse_tot/(i+1)
        r2_avg = r2_tot/(i+1)
        print(f'Split #{i}, This RMSE: {rmse:.6f}, Average RMSE: {rmse_avg:.4f}')
        print(f'Split #{i}, This R2: {r2:.6f}, Average R2: {r2_avg:.4f}')
        result['rmse'].append(rmse)
        result['r2'].append(r2)
        
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
    
    return result, addenda

def train_and_validate_hybrid_model(data_dir=None, parameters_xgb=None, parameters_nn=None, n_repeats=1, random_state=5, hybrid_split_time=360, features=["flow-time", "Tw", "Ti"]):
    h_r2_tot = 0
    h_rmse_tot = 0
    Tavg_r2_tot = 0
    Tavg_rmse_tot = 0
    hybrid_r2_tot = 0
    hybrid_rmse_tot = 0
    addenda = list()
    
    # Get a dataframe with the filepaths of each file in the data directory
    scenario_index = get_scenario_index(data_dir)
    
    train_and_val_index = scenario_index.index

    # Generate cross-validation object (80-20 train-val split)
    cv = get_cv(n_repeats, random_state)
    
    # Loop through the splits in cv
    for i, (train_index, val_index) in enumerate(cv.split(train_and_val_index)):

        #Get scaled train and val data for NN model
        X_train_nn, y_train_nn = get_split_data(scenario_index, train_index, 'Tavg', t_max=hybrid_split_time, features=features)
        X_val_nn, y_val_nn = get_split_data(scenario_index, val_index, 'Tavg', t_max=hybrid_split_time, features=features)
        X_train_nn, y_train_nn, scaler_X, scaler_y = get_scaled_data(X_train_nn, y_train_nn)
        X_val_nn, y_val_nn = get_scaled_data(X_val_nn, y_val_nn, scaler_X, scaler_y)
        
        # Get train and val data for XGBoost model
        X_train_xgb, y_train_xgb = get_split_data(scenario_index, train_index, 'h', t_min=hybrid_split_time, features=features)
        X_val_xgb, y_val_xgb = get_split_data(scenario_index, val_index, 'h', t_min=hybrid_split_time, features=features)
    
        # Get the models
        model_NN = get_model('NN', parameters_nn)
        model_XGB = get_model('XGBoost', parameters_xgb)
    
        # Fit the models to training data
        model_NN = fit_model(model_NN, 'NN', X_train_nn, y_train_nn, X_val_nn, y_val_nn, parameters_nn)
        model_XGB = fit_model(model_XGB, 'XGBoost', X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb, parameters_xgb)

        # Get predictions for validation data
        y_hat_nn, y_val_nn = get_predictions(model_NN, X_val_nn, y_val_nn, scale=True, scaler_y=scaler_y, model_type='NN')
        y_hat_xgb = get_predictions(model_XGB, X_val_xgb, model_type='XGBoost')
        
        addendum_nn = {
            'y_val': y_val_nn,
            'y_hat': y_hat_nn,
            'scenario_index': scenario_index,
            'train_index': train_index, 
            'val_index': val_index,
            'result': None,
            'scaler_X': scaler_X, 
            'scaler_y': scaler_y
            }
        
        addendum_xgb = {
            'y_val': y_val_xgb,
            'y_hat': y_hat_xgb,
            'scenario_index': scenario_index,
            'train_index': train_index, 
            'val_index': val_index,
            'result': None,
            'scaler_X': None, 
            'scaler_y': None
            }
        
        # Evaluate xgb model and get results
        h_results = test_model(model_XGB, model_type='XGBoost', data_dir=data_dir, target='h', scale=False, addendum=addendum_xgb, t_min=hybrid_split_time)
        h_df = h_results['test_df']
        h_r2 = h_results['r2']
        h_r2_tot += h_r2
        h_r2_avg = h_r2_tot/(i+1)
        h_rmse = h_results['rmse']
        h_rmse_tot += h_rmse
        h_rmse_avg = h_rmse_tot/(i+1)
        print(f'Split #{i}, XGB h RMSE: {h_rmse:.6f}, XGB h RMSE Average: {h_rmse_avg:.6f}')
        print(f'Split #{i}, XGB h R2: {h_r2:.6f}, XGB h R2 Average: {h_r2_avg:.6f}')
        
        
        # Evaluate nn model and get results
        Tavg_results = test_model(model_NN, model_type='NN', data_dir=data_dir, target='Tavg', scale=True, addendum=addendum_nn, t_max=hybrid_split_time)
        Tavg_r2 = Tavg_results['r2']
        Tavg_r2_tot += Tavg_r2
        Tavg_r2_avg = Tavg_r2_tot/(i+1)
        Tavg_rmse = Tavg_results['rmse']
        Tavg_rmse_tot += Tavg_rmse
        Tavg_rmse_avg = Tavg_rmse_tot/(i+1)
        print(f'Split #{i}, NN Tavg RMSE: {Tavg_rmse:.6f}, NN Tavg RMSE Average: {Tavg_rmse_avg:.6f}')
        print(f'Split #{i}, NN Tavg R2: {Tavg_r2:.6f}, NN Tavg R2 Average: {Tavg_r2_avg:.6f}')
        
        # Get predictions for all time (for buiilding the huybrid model)
        h_results = test_model(model_XGB, model_type='XGBoost', data_dir=data_dir, target='h', scale=False, addendum=addendum_xgb)
        h_df = h_results['test_df']
        Tavg_results = test_model(model_NN, model_type='NN', data_dir=data_dir, target='Tavg', scale=True, addendum=addendum_nn)
        Tavg_df = Tavg_results['test_df']
        
        # Add nn results for Tavg to XGB (h) results dataframe
        h_df['Tavg_hat'] = Tavg_df['Tavg_hat']
        
        # Get hybrid model results
        hybrid_df = get_T_from_h_results(h_df, plot=False, hybrid_model=True, hybrid_split_time=hybrid_split_time)
        y_val_hybrid = hybrid_df['Tavg']
        y_hat_hybrid = hybrid_df['Tavg_hat']
        hybrid_r2 = r2_score(y_val_hybrid, y_hat_hybrid)
        hybrid_r2_tot += hybrid_r2
        hybrid_r2_avg = hybrid_r2_tot/(i+1)
        hybrid_rmse = mean_squared_error(y_val_hybrid, y_hat_hybrid, squared=False)
        hybrid_rmse_tot += hybrid_rmse
        hybrid_rmse_avg = hybrid_rmse_tot/(i+1)
        print(f'Split #{i}, Hybrid RMSE: {hybrid_rmse:.6f}, Hybrid RMSE Average: {hybrid_rmse_avg:.6f}')
        print(f'Split #{i}, Hybrid R2: {hybrid_r2:.6f}, Hybrid R2 Average: {hybrid_r2_avg:.6f}')
        
        # Provide addendum for this model
        addendum_hybrid = {
            'y_val': y_val_hybrid,
            'y_hat': y_hat_hybrid,
            'scenario_index': scenario_index,
            'train_index': train_index, 
            'val_index': val_index,
            'result': None,
            'scaler_X': scaler_X, 
            'scaler_y': scaler_y
            }
        
        addendum_composite = {'NN': addendum_nn, 'XGBoost': addendum_xgb, 'Hybrid': addendum_hybrid}
        
        addenda.append(addendum_composite)
    
    return addenda

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
    if 'test_index' not in addendum: # If test set not split out, test with validation data
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

def train_final_model(data_dir=None, model_type='NN', target='Tavg', scale=True, parameters=None, t_min=-1, t_max=-1, features=["flow-time", "Tw", "Ti"]):
    # Get a dataframe with the filepaths of each file in the data directory
    scenario_index = get_scenario_index(data_dir)

    # Get train data, and test data for validation while training
    X_train, y_train = get_split_data(scenario_index, scenario_index.index, target, t_min, t_max, features)
    
    # If requested, scale data
    if scale:
        X_train, y_train, scaler_X, scaler_y = get_scaled_data(X_train, y_train)
    else:
        scaler_X, scaler_y = None, None
    
    # Get model
    model = get_model(model_type, parameters)
    
    # Train model
    model = fit_model(model, model_type, X_train, y_train, parameters=parameters)
    
    # Populate addendum
    addendum = {'scaler_X': scaler_X, 'scaler_y': scaler_y}
    
    return model, addendum

def get_T_from_h_results(test_df, plot=False, hybrid_model=False, hybrid_split_time=-1, predictions=False):
    scenario_features=["Tw", "Ti"]
    T_hat = np.array([])
    T_expected = np.array([])
    figures = {}
    if plot:
        for idx, grp in test_df.groupby(["Tw", "Ti"]):
            T_hat_grp = get_T_from_h(grp, hybrid_model, hybrid_split_time)
            # Plotting results
            grp["T_hat"] = T_hat_grp
            ax = grp.plot(x="flow-time", y='Tavg', c='DarkBlue', linewidth=2.5, label="Expected")
            plot = grp.plot(x="flow-time", y='T_hat', c='DarkOrange', linewidth=2.5, label="Predicted", ax=ax)
            plt.title('Hybrid Model Predictions for Tw = {Tw}  Ti = {Ti}'.format(Tw=idx[0], Ti=idx[1]))
            plt.show()
            fig = plot.get_figure()
            key = 'Tfromh_'
            for i, sf in enumerate(scenario_features):
                key += f'{scenario_features[i]}_{idx[i]}'
                if i != len(scenario_features) - 1:
                    key += '_'
            figures[key] = fig
            T_hat = np.concatenate((T_hat, T_hat_grp))
            T_expected = np.concatenate((T_expected, grp["Tavg"]))

        rmse = mean_squared_error(T_expected, T_hat, squared=False)
        r2 = r2_score(T_expected, T_hat)
        return rmse, r2, figures
    elif predictions:
        for idx, grp in test_df.groupby(["Tw", "Ti"]):
            T_hat_grp = get_T_from_h(grp, hybrid_model, hybrid_split_time)
            T_hat = np.concatenate((T_hat, T_hat_grp))
        test_df['Tavg_hat'] = T_hat
        return test_df
    else:
        for idx, grp in test_df.groupby(["Tw", "Ti"]):
            T_hat_grp = get_T_from_h(grp, hybrid_model, hybrid_split_time)
            T_hat = np.concatenate((T_hat, T_hat_grp))
            T_expected = np.concatenate((T_expected, grp["Tavg"]))
        test_df['Tavg_hat'] = T_hat
        test_df['Tavg'] = T_expected
        return test_df
    

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

def analyze_CV_results(addenda, t_min=-1, t_max=-1, target='Tavg', hybrid=False):
    scenario_dict = {}
    # These keys are specific to the scenarios used in the representative training set
    # scenario_dict has a key for each scenario and is used to keep track of the results
    # for every scenario.
    for Tw in range(420,661,20):
        for Ti in range(400,641,20):
            if Ti < Tw:
                scenario_dict[str(Tw) + '_' + str(Ti)] = list()
    
    # Loop through each addendum (one for each CV split)
    # Get the results for each scenario in the validation set
    # Append the results to scenario_dict
    for addendum in addenda:
        y_val = addendum['y_val']
        y_hat = addendum['y_hat']
        scenario_index = addendum['scenario_index']
        val_index = addendum['val_index']
        val_df = get_dataframe(scenario_index, val_index, t_min=t_min, t_max=t_max)
        if hybrid:  # If this is the hybrid model, change the target column,
                    # because the data was reordered when making hybrid model predictions
            val_df[target] = y_val
        val_df[target+'_hat'] = y_hat
        
        for idx, grp in val_df.groupby(['Tw','Ti']):
            rmse = mean_squared_error(grp[target], grp[target+'_hat'], squared=False)
            scenario_dict[str(int(idx[0])) + '_' + str(int(idx[1]))].append(rmse)
    
    # Get average RMSE for each scenario
    scenario_avg_rmse = {}
    for scenario, rmse_list in scenario_dict.items():
        if len(rmse_list) > 0:
            scenario_avg_rmse[scenario] = sum(rmse_list)/len(rmse_list)

    # Save average RMSE to csv file
    with open('scenario_results_' + target + '_hybrid_' + str(hybrid) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for key, val in scenario_avg_rmse.items():
            writer.writerow([key,val])