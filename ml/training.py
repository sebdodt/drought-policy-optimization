from sklearn.metrics import mean_squared_error
import math
import numpy as np

def train_baseline(X_train, y_train, X_test, y_test):
    y_pred = 0
    y_test = np.array(y_test)
    y_test = y_test[~np.isnan(y_test)]
    mse = mean_squared_error(y_test, [y_pred]*len(y_test))
    return math.sqrt(mse)





def train(datalist):
    X_trains, y_trains, X_tests, y_tests, groups = datalist

    rmse = []
    for i in range(len(X_trains)):
        rmse.append(
            train_baseline(X_trains[i], y_trains[i], X_tests[i], y_tests[i])
        )
    return rmse