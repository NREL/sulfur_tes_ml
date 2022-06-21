import numpy as np

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from stesml.data_tools import get_train_and_test_index
from stesml.data_tools import load_data_sulfur
from stesml.data_tools import series_to_supervised
from stesml.data_tools import get_train_data_sulfur
from stesml.data_tools import get_test_data_sulfur

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

def get_progress(model_type, scenario_index, min_estimators, max_estimators, step_size, num_shuffle_iterations=1, is_recurrent=False, verbose=False, target='T', per_case=False, x=1):
    model = get_model(model_type)
    if model_type == "RandomForest" and num_shuffle_iterations == 1:
        model.set_params(warm_start=True)
    
    rmse_history = list()
    r2_history = list()
    num_iterations = 10
    
    for i in range(min_estimators, max_estimators + 1, step_size):
        model.set_params(n_estimators=i)
        rmse = 0
        r2 = 0
        
        for j in range(num_shuffle_iterations):
            print(j)
            train_index, test_index = get_train_and_test_index(scenario_index)

            X_train, y_train = get_train_data_sulfur(scenario_index, train_index, test_index, is_recurrent, target, per_case, x=x)
            X_test, y_test = get_test_data_sulfur(scenario_index, test_index, is_recurrent, target, x=x)
            
            model.fit(X_train, y_train)

            y_hat = get_predictions(model, X_test, is_recurrent)
            
            if verbose:
                print('Predicted:',y_hat)
                print('Expected:',y_test)

            rmse += mean_squared_error(y_test, y_hat, squared=False)
            
            r2 += r2_score(y_test,y_hat)
            
        rmse /= num_shuffle_iterations
        r2 /= num_shuffle_iterations
        
        rmse_history.append((i, rmse))
        r2_history.append((i, r2))
        print('# of Estimators: {i}, RMSE = {rmse:.5f}, r2 = {r2:.5f}'.format(i=i, rmse=rmse, r2=r2))
        
    return rmse_history, r2_history