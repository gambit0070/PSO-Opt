import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("btcusd_1-min_data.csv", parse_dates=['datetime'])
df.set_index('datetime', inplace=True)

# split the data
train_df = df[df.index < '2020-01-01']
test_df = df[df.index >= '2020-01-01']

def data_resampling(frequency):
    # resample data frequency
    if frequency == 'month':
        train_df_month = train_df.resample('ME').mean()
        test_df_month = test_df.resample('ME').mean()
        return train_df_month, test_df_month # (96, 6) (63, 6)
    
    if frequency == 'day':
        train_df_day = train_df.resample('D').mean()
        test_df_day = test_df.resample('D').mean()
        return train_df_day, test_df_day # (2922, 6) (2922, 6)
    
    if frequency == 'hour':
        train_df_hour = train_df.resample('h').mean()
        test_df_hour = test_df.resample('h').mean()
        return train_df_hour, test_df_hour # (70118, 6) (70118, 6)


