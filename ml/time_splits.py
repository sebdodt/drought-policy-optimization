import pandas as pd
import yaml
import numpy as np


def convert_date_to_period(date):
    datetime = pd.to_datetime(date)
    months_since_jan_2014 = (datetime.year - 2014) * 12 + datetime.month - 1
    return months_since_jan_2014

def determine_splits(config):
    time_config = config['temporal_config']

    feature_start = convert_date_to_period(time_config['feature_start_time'])
    feature_end = convert_date_to_period(time_config['feature_end_time'])
    label_start = convert_date_to_period(time_config['label_start_time'])
    label_end = convert_date_to_period(time_config['label_end_time'])


    split_df = pd.DataFrame(columns=[
        'feature_start',
        'feature_end',
        'train_label_start',
        'train_label_end',
        'test_label_start',
        'test_label_end'])

    test_label_starts = np.arange(
        start=label_start,
        stop=label_end - time_config['test_label_timespans'],
        step=time_config['model_update_frequency'])

    j=0
    for test_start in test_label_starts:
        train_label_ends = np.arange(
            test_start - time_config['max_training_histories'] + time_config['training_label_timespans'],
            test_start + 1,
            step = time_config['training_as_of_date_frequencies'])
        test_label_starts = train_label_ends - time_config['training_label_timespans']
        feature_ends = test_label_starts
        feature_starts = np.maximum(test_start - time_config['max_training_histories'], feature_start)
        feature_starts = np.minimum(feature_end - time_config['min_training_histories'], feature_starts)
        
        
        for i in range(len(feature_ends)):
            new_split = {
                'feature_start': feature_starts,
                'feature_end': feature_ends[i],
                'train_label_start': test_label_starts[i],
                'train_label_end': train_label_ends[i],
                'test_label_start': test_start,
                'test_label_end': test_start + time_config['test_label_timespans']
            }
            new_split = pd.DataFrame(new_split, index=[j])
            j+=1
            split_df = pd.concat([split_df,new_split], ignore_index=True)

    split_df[split_df.select_dtypes(include=[np.number]).ge(0).all(1)]
    return split_df



def split_data():
    config_path = 'ml/config.yaml'
    with open(config_path, 'r') as dbf:
        config = yaml.safe_load(dbf)

    split_df = determine_splits(config)
    return split_df
