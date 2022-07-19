import random
import pandas as pd
import glob
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold

def get_scenario_index(data_dir):
    scenario_index = pd.DataFrame({"filepath": glob.glob(os.path.join(data_dir, "ML_*_*.csv"))})
    return scenario_index

def get_index_splits(scenario_index, random_state=-1):
    if random_state == -1:
        random_state = random.randrange(2652124)
    cv = get_cv(random_state=random_state)
    train_and_val_index, test_index  = next(cv.split(scenario_index.index))
    return train_and_val_index, test_index

def get_cv(n_repeats=1, random_state=-1):
    if random_state == -1:
        random_state = random.randrange(2652124)
    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=random_state)
    return cv

def get_dataframe(scenario_index, selected_index, t_min=-1, t_max=-1):
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

def get_split_data(scenario_index, split_index, target='Tavg', t_min=-1, t_max=-1):
    split_df = get_dataframe(scenario_index, split_index, t_min, t_max)
    X_split = split_df[["flow-time", "Tw", "Ti"]].to_numpy()
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