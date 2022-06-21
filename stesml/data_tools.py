import random
import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedKFold

def get_train_and_test_index(scenario_index):
    random_state = random.randrange(2652124)
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=random_state)

    train_index, test_index  = next(cv.split(scenario_index.index))
    
    return train_index, test_index

def load_data_sulfur(scenario_index, selected_index, is_recurrent_test_data=False, per_case=False, x=0):
    """ Load data from files in scenario_index with indices matching ones in selected_index"""
    
    df_arr = []
    for f in scenario_index.loc[selected_index].filepath:
        Tw = float(f.split("/")[-1].split("_")[1])
        Ti = float(f.split("/")[-1].split("_")[2].replace(".csv", ""))
        
        f_df = pd.read_csv(f)
        
        ### ADDITIONS FOR BASE CASE MODEL
        if x > 0:
            for i in range(10*x,20*x):
                if f_df["flow-time"][i] >= x:
                    f_df["Tx"] = f_df["Tavg"][i]
                    break
        ### END ADDITIONS
        
        f_df["Ti"] = Ti
        
        if per_case:
            f_df = f_df.head(x*10)
        
        df_arr.append(f_df)
    
    if is_recurrent_test_data:
        return df_arr
    
    combined_df = pd.concat(df_arr)
    return combined_df

# transform a time series dataset into a supervised learning dataset
# source: https://machinelearningmastery.com/random-forest-for-time-series-forecasting/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, per_case=False):
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

def get_train_data_sulfur(scenario_index, train_index, test_index, is_recurrent=False, target='T',  per_case=False, x=0):
    train_df = load_data_sulfur(scenario_index, train_index, x=x)
    if is_recurrent:
        if target == 'T':
            if x > 0:
                train = train_df[["flow-time", "Tw", "Ti", "Tx", "Tavg"]]
            else:
                train = train_df[["flow-time", "Tw", "Ti", "Tavg"]]
        elif target == 'h':
            if x > 0:
                train = train_df[["flow-time", "Tw", "Ti", "Tx", "h"]]
            else:
                train = train_df[["flow-time", "Tw", "Ti", "h"]]
        else:
            print('Target must be T or h.\n')
            return None
        if per_case:
            test_df = load_data_sulfur(scenario_index, test_index, per_case=per_case, x=x)
            if target == 'T':
                if x > 0:
                    test = test_df[["flow-time", "Tw", "Ti", "Tx", "Tavg"]]
                else:
                    test = test_df[["flow-time", "Tw", "Ti", "Tavg"]]
            elif target == 'h':
                if x > 0:
                    test = test_df[["flow-time", "Tw", "Ti", "Tx", "h"]]
                else:
                    test = test_df[["flow-time", "Tw", "Ti", "h"]]
            train = pd.concat([train, test]).to_numpy()
        else:
            train = train.to_numpy()
        train_shift = pd.DataFrame(series_to_supervised(train))
        print(train_shift)
        # Get rid of last prediction for each set, 
        # because the next datapoint is the first datapoint of the next set
        train_shift.drop(abs(train_shift[train_shift[0] == 0].index - 1), inplace = True)
        # Get rid of samples before 1 second to ensure all timesteps are equal to 0.1
        train_shift.drop(train_shift[train_shift[0] < 1].index, inplace = True)
        train_shift = train_shift.to_numpy()
        # For training, get Tw, Ti, Tx, & Tavg_t-1 for X
        X_train = train_shift[:, 1:5]
        # Get Tavg_t for y
        y_train = train_shift[:, -1]
    else:
        if x > 0:
            X_train = train_df[["flow-time", "Tw", "Ti", "Tx"]]
        else:
            X_train = train_df[["flow-time", "Tw", "Ti"]]
        if target == 'T':
            y_train = train_df[["Tavg"]]
        elif target == 'h':
            y_train = train_df[["h"]]
        else:
            print('Target must be T or h.\n')
            return None
        if per_case:
            test_df = load_data_sulfur(scenario_index, test_index, per_case=per_case, x=x)
            X_test = test_df[["flow-time", "Tw", "Ti", "Tx"]]
            if target == 'T':
                y_test = test_df[["Tavg"]]
            elif target == 'h':
                y_test = test_df[["h"]]
            X_train = pd.concat([X_train, X_test]).to_numpy()
            y_train = pd.concat([y_train, y_test]).to_numpy().reshape(-1,)
        else:
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy().reshape(-1,)
    return X_train, y_train

def get_test_data_sulfur(scenario_index, test_index, is_recurrent=False, target='T', x=0):
    if is_recurrent:
        test_data = load_data_sulfur(scenario_index, test_index, is_recurrent_test_data=True, x=x)
        test_df_list = list()
        for test_df in test_data:
            if target == 'T':
                if x > 0:
                    test = test_df[["flow-time", "Tw", "Ti", "Tx", "Tavg"]].to_numpy()
                else:
                    test = test_df[["flow-time", "Tw", "Ti", "Tavg"]].to_numpy()
            elif target == 'h':
                if x > 0:
                    test = test_df[["flow-time", "Tw", "Ti", "Tx", "h"]].to_numpy()
                else:
                    test = test_df[["flow-time", "Tw", "Ti", "h"]].to_numpy()
            else:
                print('Target must be T or h.\n')
                return None
            test_shift = series_to_supervised(test)
            test_df_list.append(test_shift)
        print(test_df_list)
        X_test = list()
        y_test = list()
        for test_shift in test_df_list:
            # For testing, get only Tw, Ti, & Tx. Tavg_t-1 will be supplied by
            # the previous prediction in walk-forward validation
            X = test_shift[:, 1:4]
            # Get T_avg_t for y
            y = test_shift[:, -1].tolist()
            X_test.append(X)
            y_test += y
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    else:
        test_df = load_data_sulfur(scenario_index, test_index, x=x)
        if x > 0:
            X_test = test_df[["flow-time", "Tw", "Ti", "Tx"]].to_numpy()
        else:
            X_test = test_df[["flow-time", "Tw", "Ti"]].to_numpy()
        if target == 'T':
            y_test = test_df[["Tavg"]].to_numpy().reshape(-1,)
        elif target == 'h':
            y_test = test_df[["h"]].to_numpy().reshape(-1,)
        else:
            print('Target must be T or h.\n')
            return None
        
    return X_test, y_test