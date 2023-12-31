import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class RollingDatasetForexNY(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        selected_time = self.df.loc[idx]['Date']

        # we want to pick the date from the NY forex market in UTC time (data is in UTC time) 1pm to 10pm
        # select the nearest 1pm to 10pm data point from our selected_time

        # get the data from the same day
        current_data = self.df[(self.df['Date'].dt.year == selected_time.year) &
                               (self.df['Date'].dt.day == selected_time.day) &
                               (self.df['Date'].dt.month == selected_time.month)]

        # get the data from 1pm to 10pm
        current_data = current_data[(current_data['Date'].dt.hour >= 13) & (current_data['Date'].dt.hour <= 22)]

        # feature is the current data until 9pm
        feature = current_data[current_data['Date'].dt.hour < 21]

        if feature.empty or pd.isna(feature.index.min()) or pd.isna(feature.index.max()):
            # return numpy arrays of correct size
            return np.zeros((480, 4)), 0

        if len(feature) < 480:

            # Ensure time-based indexing
            feature = feature.set_index('Date')
            all_times = pd.date_range(start=feature.index.min().replace(hour=13, minute=0),
                                      end=feature.index.max().replace(hour=21, minute=0),
                                      freq='T')  # 'T' for minute frequency
            feature = feature.reindex(all_times)

            # Fill missing values
            feature = feature.fillna(method='ffill').fillna(method='bfill')
            feature = feature.reset_index()
            feature = feature[:-1]

        feature = feature[['Open', 'High', 'Low', 'Close']]

        # label is True if the latest close price of current_data is higher than the close price of the feature
        label = current_data.iloc[-1]['Close'] > feature.iloc[-1]['Close']
        # label to 0 or 1
        label = 1 if label else 0

        return np.array(feature), np.array(label)
