import random
import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedKFold

def get_train_and_test_index(scenario_index):
    random_state = random.randrange(2652124)
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=random_state)

    train_index, test_index  = next(cv.split(scenario_index.index))
    
    return train_index, test_index

def load_data(scenario_index, selected_index, is_recurrent_test_data=False):
    """ Load data from files in scenario_index with indices matching ones in selected_index"""
    
    df_arr = []
    for f in scenario_index.loc[selected_index].filepath:
        Tw = float(f.split("/")[-1].split("_")[1])
        Ti = float(f.split("/")[-1].split("_")[2].replace(".csv", ""))

        f_df = pd.read_csv(f, skiprows=12)
        f_df["Tw"] = Tw
        f_df["Ti"] = Ti
        df_arr.append(f_df)
    
    if is_recurrent_test_data:
        return df_arr
    
    combined_df = pd.concat(df_arr)
    return combined_df

# transform a time series dataset into a supervised learning dataset
# source: https://machinelearningmastery.com/random-forest-for-time-series-forecasting/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    df = pd.DataFrame(data)
    cols = list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        
    # put it all together
    combined_df = pd.concat(cols, axis=1)
    
    # drop rows with NaN values
    if dropnan:
        combined_df.dropna(inplace=True)
        
    return combined_df.values

def get_train_data(scenario_index, train_index, is_recurrent=False):
    train_df = load_data(scenario_index, train_index)
    if is_recurrent:
        train = train_df[["Time", "Tw", "Ti", "Tc"]].to_numpy()
        train_shift = pd.DataFrame(series_to_supervised(train))
        # Get rid of last prediction for each set, 
        # because the next datapoint is the first datapoint of the next set
        train_shift.drop(train_shift[train_shift[0] == 21.15].index, inplace = True)
        train_shift = train_shift.to_numpy()
        X_train = train_shift[:, 0:4]
        y_train = train_shift[:, -1]
    else:
        X_train = train_df[["Time", "Tw", "Ti"]].to_numpy()
        y_train = train_df[["Tc"]].to_numpy().reshape(-1,)
    return X_train, y_train

def get_test_data(scenario_index, test_index, is_recurrent=False):
    if is_recurrent:
        test_data = load_data(scenario_index, test_index, is_recurrent_test_data=True)
        test_df_list = list()
        for test_df in test_data:
            test = test_df[["Time", "Tw", "Ti", "Tc"]].to_numpy()
            test_shift = series_to_supervised(test)
            test_df_list.append(test_shift)
        X_test = list()
        y_test = list()
        for test_shift in test_df_list:
            X, Y = test_shift[:, 0:-5], test_shift[:, -1].tolist()
            X_test.append(X)
            y_test += Y
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    else:
        test_df = load_data(scenario_index, test_index)
        X_test = test_df[["Time", "Tw", "Ti"]].to_numpy()
        y_test = test_df[["Tc"]].to_numpy().reshape(-1,) 
        
    return X_test, y_test