from sklearn.metrics import mean_squared_error
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

def train_baseline(X_train, y_train, X_test, y_test):
    y_pred = 0
    y_test = np.array(y_test['y'])
    y_test = y_test[~np.isnan(y_test)]
    mse = mean_squared_error(y_test, [y_pred]*len(y_test))
    return math.sqrt(mse)


def score_cv(model, X_train, y_train, X_test, y_test):
    fitted_model = model.fit(X_train, y_train)
    y_pred = fitted_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return math.sqrt(mse)


def define_models():
    models = dict()

    ## Linear
    linear = make_pipeline(RobustScaler(), LinearRegression())
    models['linear'] = linear

    ## Lasso
    alpha = [0.1, 0.2, 0.4, 0.7, 1, 2, 4, 8, 16]
    for a in alpha:
        lasso = make_pipeline(RobustScaler(), Lasso(random_state=1, alpha=a))
        models['lasso, alpha={a}'.format(a=a)] = lasso

    ## Elastic Net
    ENet = make_pipeline(RobustScaler(), ElasticNet(random_state=1))

    ## Bayesian Ridge
    bayesian = make_pipeline(RobustScaler(), BayesianRidge())

    # ## Random Forest
    # max_depth = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # min_samples_split = [2]# 4, 8, 16, 32, 64, 128, 256, 512]
    # min_samples_leaf = [0, 1]#, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # for d in max_depth:
    #     for s in min_samples_split:
    #         for l in min_samples_leaf:
    #             rf = RandomForestRegressor(random_state=1, max_depth=d, min_samples_split=s)
    #             models['lasso, max_depth={d}, min_samples_split={s}, min_samples_leaf={l}'.format(d=d, s=s, l=l)] = rf

    ## Gradient Boost

    learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    for l in learning_rate:
        GBoost = GradientBoostingRegressor(random_state=1, learning_rate=l)
        models['Gradient Boosting, lr={l}'.format(l=l)] = GBoost

    model_xgb = xgb.XGBRegressor(random_state =1)
    model_lgb = lgb.LGBMRegressor(objective='regression')

    return models


def train_pipeline(X_train, y_train, X_test, y_test, models):

    # prepare models
    y_train = np.array(y_train['y'])
    y_test = np.array(y_test['y'])
    X_train = np.array(X_train.drop('fips', axis=1))
    X_test = np.array(X_test.drop('fips', axis=1))

    scores = []
    for m in models.keys():
        scores.append(score_cv(models[m], X_train, y_train, X_test, y_test))
    return scores, models


def train(datalist):
    print("Start training...")
    X_trains, y_trains, X_tests, y_tests, groups = datalist

    models = define_models()

    n_models = len(models.keys())
    rmse = np.zeros((len(X_trains),n_models+1))
    for i in range(len(X_trains)):
        if (len(X_trains[i])!=0) & (len(X_tests[i])!=0):
            rmse[i,0] = train_baseline(X_trains[i], y_trains[i], X_tests[i], y_tests[i])
            scores, models = train_pipeline(X_trains[i], y_trains[i], X_tests[i], y_tests[i], models)
            rmse[i,1:len(scores)+1] = scores
    return rmse, list(models.keys())