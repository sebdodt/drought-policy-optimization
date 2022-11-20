import pandas as pd
import addfips

def read_data():
    # read in features and label data
    path = 'data/weather_data.csv'
    df = pd.read_csv(path)

    # create column that counts the months since January 2014
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['period'] = (df['year'] - 2014) * 12 + df['month'] - 1

    # rename fips column
    df.rename({'name': 'fips'}, axis=1, inplace=True)
    # df['fips'] = df['fips'].astype(int)
    return df
    



def generate_features(datalist):
    X_trains, y_trains, X_tests, y_tests, groups = datalist

    X_trains_eng = []
    y_trains_eng = []
    X_tests_eng = []
    y_tests_eng = []
    for i in range(len(X_trains)):
        train = X_trains[i][['fips','period', 'latitude', 'longitude']].copy()
        test = X_tests[i][['fips', 'period', 'latitude', 'longitude']].copy()
        train.drop_duplicates(inplace=True)
        test.drop_duplicates(inplace=True)


        continuous = [
            'tempmax', 'tempmin', 'temp', 'feelslikemax',
            'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob',
            'precipcover', 'preciptype', 'snow', 'snowdepth',
            'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
            'solarradiation', 'solarenergy', 'uvindex'
        ]
        categorical = [
            'conditions', 'icon'
        ]
        imputations = ['snow', 'snowdepth']

        
        ## imputations
        X_trains[i][imputations].fillna(0, inplace=True)
        

        ## add avg
        avgs = X_trains[i] \
            .groupby(['fips', 'period'])[continuous] \
            .mean() \
            .reset_index()
        train = train.merge(avgs, on=['fips', 'period'], how='left')
        for var in continuous:
            train.rename({var:'avg_'+var}, axis=1, inplace=True)


        ## add avg
        avg_cols = [
            'tempmax',
            'tempmin',
            'temp',
            'feelslikemax',
            'feelslikemin',
            'feelslike',
            'dew',
            'humidity',
            ''
            ]
        

        ## add max
        max_cols = ['tempmax']


        ## add min
        min_cols = ['tempmax']


        ## add since
        since_cols = []


        X_trains_eng.append(train)
        return X_trains_eng



if __name__=='__main__':
    df = read_data()

