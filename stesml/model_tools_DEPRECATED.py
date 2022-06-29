import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from stesml.data_tools import get_train_and_test_index
from stesml.data_tools import load_data
from stesml.data_tools import series_to_supervised
from stesml.data_tools import get_train_data
from stesml.data_tools import get_test_data

from stesml.postprocessing_tools import get_T
from stesml.postprocessing_tools import get_h

from stesml.data_tools import get_train_and_test_index_short
from stesml.data_tools import get_train_data_short


def get_model(model_type="XGBoost", n_estimators=1000):
    if model_type == "XGBoost":
        model = XGBRegressor(n_estimators=n_estimators, colsample_bylevel=.75, n_jobs=6)
    elif model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=30)
    else:
        print("Please choose either XGBoost or RandomForest for model type")
        return None
    return model

def walk_forward_validation(X_test_sample, model):
    predictions = list()
    Tc = X_test_sample[0][2]
    for time_step in X_test_sample:
        time_step = np.append(time_step,Tc)
        Tc = model.predict(time_step.reshape(1, -1))
        predictions.append(Tc.tolist()[0])
    #predictions.append(predictions[-1])
    return predictions

def get_predictions(model, X_test, is_recurrent=False):
    if is_recurrent:
        y_hat = list()
        for X_test_sample in X_test:
            y_hat_sample = walk_forward_validation(X_test_sample, model)
            y_hat += y_hat_sample
        y_hat = np.array(y_hat)
    else:
        y_hat = model.predict(X_test)
    
    return y_hat

def get_shuffle_results(model_type, scenario_index, n_estimators, n_shuffle_iterations=1, is_recurrent=False, verbose=0, target='Tavg', per_case=False, x=0, calc_T_from_h=False, short=False, t=100):
    
    rmse = 0
    r2 = 0
    rmse_T_tot = 0 # Only used if target is 'h'
    r2_T_tot = 0 # Only used if target is 'h'
    count = 0
    
    for j in range(n_shuffle_iterations):
        count += 1
        
        model = get_model(model_type)
        model.set_params(n_estimators=n_estimators)
        
        if short:
            train_index, train_index_short, test_index = get_train_and_test_index_short(scenario_index)
            X_train, y_train = get_train_data_short(scenario_index, train_index, train_index_short, target, t=t)
        else:
            train_index, test_index = get_train_and_test_index(scenario_index)
            X_train, y_train = get_train_data(scenario_index, train_index, test_index, is_recurrent, target, per_case, x=x)
            
        X_test, y_test = get_test_data(scenario_index, test_index, is_recurrent, target, x=x)

        model.fit(X_train, y_train)

        y_hat = get_predictions(model, X_test, is_recurrent)
        
        if calc_T_from_h:
            test_df = load_data(scenario_index, test_index, x=x)
            test_df["h_hat"] = y_hat
            rmse_T, r2_T = get_T_from_h_results(test_df, plot=False)
            rmse_T_tot += rmse_T
            r2_T_tot += r2_T
            rmse_T_cur_avg = rmse_T_tot/count
            r2_T_cur_avg = r2_T_tot/count

        rmse += mean_squared_error(y_test, y_hat, squared=False)
        r2 += r2_score(y_test,y_hat)

        rmse_cur_avg = rmse/count
        r2_cur_avg = r2/count

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

def get_progress(model_type, scenario_index, min_estimators, max_estimators, step_size, n_shuffle_iterations=1, is_recurrent=False, verbose=0, target='Tavg', per_case=False, x=0):
    rmse_history = list()
    r2_history = list()
    
    for n_estimators in range(min_estimators, max_estimators + 1, step_size):
        rmse, r2 = get_shuffle_results(model_type, scenario_index, n_estimators, n_shuffle_iterations, is_recurrent, verbose, target, per_case, x) 
        
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
        T_hat_grp = np.append(T_hat_grp, Ti)
        T_prev = Ti
        for i, h in enumerate(grp["h_hat"]):
            if i == len(grp["h_hat"]) - 1:
                continue
            if i < 1:
                T = grp["Tavg"][i]
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

