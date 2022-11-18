import labels

supply = labels.read_data()
supply = labels.data_cleaning(supply)
label_df = labels.label_construction(supply)
label_df = labels.match_counties(label_df)
print(label_df)