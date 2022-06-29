import random
import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedKFold

def get_train_and_test_index(scenario_index, random_state=-1):
    if random_state == -1:
        random_state = random.randrange(2652124)
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=random_state)

    train_index, test_index  = next(cv.split(scenario_index.index))
    
    return train_index, test_index

def get_train_and_test_index_short(scenario_index, random_state=-1):
    if random_state == -1:
        random_state = random.randrange(2652124)
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=random_state)

    train_index, test_index  = next(cv.split(scenario_index.index))
    
    #random_state = random.randrange(2652124)
    cv = RepeatedKFold(n_splits=2, n_repeats=1, random_state=random_state)
    train_sub_index = next(cv.split(pd.DataFrame(train_index).index))
    train_index_short = train_index[train_sub_index[0]]
    train_index = train_index[train_sub_index[1]]
    
    return train_index, train_index_short, test_index

def load_data(scenario_index, selected_index, is_recurrent_test_data=False, per_case=False, x=0):
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

def load_data_short(scenario_index, selected_index, t=100):
    """ Load data from files in scenario_index with indices matching ones in selected_index"""
    
    df_arr = []
    for f in scenario_index.loc[selected_index].filepath:
        Tw = float(f.split("/")[-1].split("_")[1])
        Ti = float(f.split("/")[-1].split("_")[2].replace(".csv", ""))
        
        f_df = pd.read_csv(f)

        f_df["Ti"] = Ti
        f_df = f_df[f_df['flow-time'] < t]
        
        df_arr.append(f_df)
    
    combined_df = pd.concat(df_arr)
    return combined_df

def get_train_data_short(scenario_index, train_index, train_index_short, target='Tavg', t=100):
    train_df = load_data(scenario_index, train_index)
    train_df_short = load_data_short(scenario_index, train_index_short, t)
    X_train = train_df[["flow-time", "Tw", "Ti"]]
    X_train_short = train_df_short[["flow-time", "Tw", "Ti"]]
    X_train = pd.concat((X_train, X_train_short)).to_numpy()
    y_train = train_df[[target]]
    y_train_short = train_df_short[[target]]
    y_train = pd.concat((y_train, y_train_short)).to_numpy().reshape(-1,)
    return X_train, y_train

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

def get_train_data(scenario_index, train_index, test_index, is_recurrent=False, target='Tavg', per_case=False, x=0):
    train_df = load_data(scenario_index, train_index, x=x)
    if is_recurrent:
        if x > 0:
            train = train_df[["flow-time", "Tw", "Ti", "Tx", target]]
        else:
            train = train_df[["flow-time", "Tw", "Ti", target]]
        if per_case:
            test_df = load_data(scenario_index, test_index, per_case=per_case, x=x)
            if x > 0:
                test = test_df[["flow-time", "Tw", "Ti", "Tx", target]]
            else:
                test = test_df[["flow-time", "Tw", "Ti", target]]
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
        y_train = train_df[[target]]
        if per_case:
            test_df = load_data(scenario_index, test_index, per_case=per_case, x=x)
            X_test = test_df[["flow-time", "Tw", "Ti", "Tx"]]
            y_test = test_df[[target]]
            X_train = pd.concat([X_train, X_test]).to_numpy()
            y_train = pd.concat([y_train, y_test]).to_numpy().reshape(-1,)
        else:
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy().reshape(-1,)
    return X_train, y_train

def get_test_data(scenario_index, test_index, is_recurrent=False, target='Tavg', x=0):
    if is_recurrent:
        test_data = load_data(scenario_index, test_index, is_recurrent_test_data=True, x=x)
        test_df_list = list()
        for test_df in test_data:
            if x > 0:
                test = test_df[["flow-time", "Tw", "Ti", "Tx", target]].to_numpy()
            else:
                test = test_df[["flow-time", "Tw", "Ti", target]].to_numpy()
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
        test_df = load_data(scenario_index, test_index, x=x)
        if x > 0:
            X_test = test_df[["flow-time", "Tw", "Ti", "Tx"]].to_numpy()
        else:
            X_test = test_df[["flow-time", "Tw", "Ti"]].to_numpy()
        y_test = test_df[[target]].to_numpy().reshape(-1,)
        
    return X_test, y_test