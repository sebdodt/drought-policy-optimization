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
    

if __name__=='__main__':
    df = read_data()

