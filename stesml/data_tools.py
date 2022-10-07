import random
import pandas as pd
import glob
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold

def get_scenario_index(data_dir, filename_pattern="ML_*_*.csv"):
    scenario_index = pd.DataFrame({"filepath": glob.glob(os.path.join(data_dir, filename_pattern))})
    return scenario_index

def get_cv(n_repeats=1, random_state=-1):
    if random_state == -1:
        random_state = random.randrange(2652124)
    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=random_state)
    return cv

def get_index_splits(scenario_index, random_state=-1):
    if random_state == -1:
        random_state = random.randrange(2652124)
    cv = get_cv(random_state=random_state)
    train_index, test_index  = next(cv.split(scenario_index.index))
    return train_index, test_index

def get_dataframe(scenario_index, split_index, t_min=-1, t_max=-1, include_Ti=True):
    """ Load data from files in scenario_index with indices matching ones in split_index"""
    df_arr = []
    for f in scenario_index.loc[split_index].filepath:
        f_df = pd.read_csv(f)
        if include_Ti:
            Ti = float(f.split("/")[-1].split("_")[2].replace(".csv", ""))
            f_df["Ti"] = Ti
        if t_min > 0:
            f_df = f_df[f_df['flow-time'] >= t_min]
        if t_max > 0:
            f_df = f_df[f_df['flow-time'] <= t_max]
        df_arr.append(f_df)
    combined_df = pd.concat(df_arr)
    return combined_df

def get_split_data(scenario_index, split_index, target='Tavg', t_min=-1, t_max=-1, features=["flow-time", "Tw", "Ti"]):
    split_df = get_dataframe(scenario_index, split_index, t_min, t_max)
    X_split = split_df[features].to_numpy()
    y_split = split_df[[target]].to_numpy().reshape(-1,)
    return X_split, y_split

def get_scaled_data(X, y, scaler_X=None, scaler_y=None):
    return_scaler_X = False
    return_scaler_y = False
    
    if scaler_X == None:
        scaler_X = StandardScaler().fit(X)
        return_scaler_X = True
    if scaler_y == None:
        scaler_y = StandardScaler().fit(y.reshape(-1,1))
        return_scaler_y = True
        
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1,1)).reshape(1,-1)[0]
    
    if return_scaler_X and return_scaler_y:
        return X_scaled, y_scaled, scaler_X, scaler_y
    elif return_scaler_X:
        return X_scaled, y_scaled, scaler_X
    elif return_scaler_y:
        return X_scaled, y_scaled, scaler_y
    else:
        return X_scaled, y_scaled