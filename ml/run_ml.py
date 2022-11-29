import labels, features, time_splits, training
import numpy as np

# labels
supply = labels.read_data()
supply = labels.data_cleaning(supply)
label_df = labels.label_construction(supply)
label_df = labels.match_counties(label_df)

# features
feature_df = features.read_data()
validation_sets = time_splits.split_data(feature_df, label_df)
validation_sets_eng = features.generate_features(validation_sets)
validation_sets_bound = time_splits.rbind_df(validation_sets_eng)

# model training
rmse, models = training.train(validation_sets_bound)
# print(rmse)
# print(rmse.mean(axis=0))

# for i in range(len(rmse.mean(axis=0))):
#     print(models[i])
#     print(rmse.mean(axis=0)[i])

print("The best model is:")
print(models[np.argmin(rmse.mean(axis=0))])
print(rmse.mean(axis=0)[np.argmin(rmse.mean(axis=0))])
print("Done.")