import labels, features, time_splits

supply = labels.read_data()
supply = labels.data_cleaning(supply)
label_df = labels.label_construction(supply)
label_df = labels.match_counties(label_df)

feature_df = features.read_data()
validation_sets = time_splits.split_data(feature_df, label_df)
validation_sets_eng = features.generate_features(validation_sets)
print(validation_sets_eng)