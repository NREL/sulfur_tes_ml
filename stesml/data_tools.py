import random
import pandas as pd
import numpy as np
import glob
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold

def get_scenario_index(data_dir):
    scenario_index = pd.DataFrame({"filepath": glob.glob(os.path.join(data_dir, "ML_*_*.csv"))})
    return scenario_index

def get_train_and_val_index(scenario_index, random_state=-1):
    if random_state == -1:
        random_state = random.randrange(2652124)
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=random_state)
    train_and_test_index, val_index  = next(cv.split(scenario_index.index))
    return train_and_test_index, val_index

def get_cv(n_repeats=1, random_state=-1):
    if random_state == -1:
        random_state = random.randrange(2652124)
    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=random_state)
    return cv

def load_data(scenario_index, selected_index, t_min=-1, t_max=-1):
    """ Load data from files in scenario_index with indices matching ones in selected_index"""
    df_arr = []
    for f in scenario_index.loc[selected_index].filepath:
        Tw = float(f.split("/")[-1].split("_")[1])
        Ti = float(f.split("/")[-1].split("_")[2].replace(".csv", ""))
        f_df = pd.read_csv(f)
        f_df["Ti"] = Ti
        if t_min > 0:
            f_df = f_df[f_df['flow-time'] >= t_min]
        if t_max > 0:
            f_df = f_df[f_df['flow-time'] <= t_max]
        df_arr.append(f_df)
    combined_df = pd.concat(df_arr)
    return combined_df

def get_train_data(scenario_index, train_index, target='Tavg', t_min=-1, t_max=-1):
    train_df = load_data(scenario_index, train_index, t_min, t_max)
    X_train = train_df[["flow-time", "Tw", "Ti"]].to_numpy()
    y_train = train_df[[target]].to_numpy().reshape(-1,)
    return X_train, y_train

def get_test_data(scenario_index, test_index, target='Tavg', t_min=-1, t_max=-1):
    test_df = load_data(scenario_index, test_index, t_min, t_max)
    X_test = test_df[["flow-time", "Tw", "Ti"]].to_numpy()
    y_test = test_df[[target]].to_numpy().reshape(-1,)
    return X_test, y_test

def get_train_and_test_data(scenario_index, train_index, test_index, target='Tavg', scale=False, t_min=-1, t_max=-1):
    X_train, y_train = get_train_data(scenario_index, train_index, target, t_min, t_max)
    X_test, y_test = get_test_data(scenario_index, test_index, target, t_min, t_max)
    
    if scale:
        scaler_x = StandardScaler().fit(X_train)
        X_train = scaler_x.transform(X_train)
        X_test = scaler_x.transform(X_test)
        
        scaler_y = StandardScaler().fit(y_train.reshape(-1,1))
        y_train = scaler_y.transform(y_train.reshape(-1,1)).reshape(1,-1)[0]
        y_test = scaler_y.transform(y_test.reshape(-1,1)).reshape(1,-1)[0] 
        
        return X_train, y_train, X_test, y_test, scaler_x, scaler_y
    
    return X_train, y_train, X_test, y_test