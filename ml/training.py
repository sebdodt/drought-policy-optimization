








def training(datalist):
    X_trains, y_trains, X_tests, y_tests, groups = datalist


    for i in range(len(X_trains)):
        X_trains = train_baseline()