import numpy as np
from torch.utils.data import Dataset


class RollingDataset(Dataset):
    def __init__(self, items, seq_len, prediction_length):
        self.items = items
        self.seq_len = seq_len
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.items) - (self.seq_len + self.prediction_length)

    def __getitem__(self, idx):
        # return (torch.tensor(self.features[idx], dtype=torch.float32),
        #         torch.tensor(self.labels[idx], dtype=torch.long))

        # past values, time features (day of month, day of year, etc), past_observed_values (usually 1)
        current_df = self.items[idx: idx + self.seq_len]
        indicator_cols = [x for x in current_df.columns if 'indicator' in x]
        # past_values = np.array(current_df[['Close', 'Open', 'High', 'Low'] + indicator_cols])
        past_values = np.array(current_df['Close'])
        time_features = np.array(current_df[['Day', 'Hour', 'Minute']])
        past_observed_values = np.array(current_df['past_observed'])

        future_df = self.items[idx + self.seq_len: idx + self.seq_len + self.prediction_length]

        future_values = np.array(future_df['Close'])
        future_time_features = np.array(future_df[['Day', 'Hour', 'Minute']])
        return past_values, time_features, past_observed_values, future_values, future_time_features
