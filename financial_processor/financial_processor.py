import pandas as pd
import pandas_ta


class FinancialProcessor:

    @staticmethod
    def load_data(data_file):
        # load csv data using pandas
        df = pd.read_csv(data_file)
        print(f'total data rows: {df.shape[0]}')
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df['date'] = pd.to_datetime(df['date'])
        # df = df.drop(columns=['date', 'volume', 'open', 'high', 'low'])
        return df

    @staticmethod
    def load_indicators(df):
        # macd
        df.ta.macd(fast=12, slow=26, append=True)
        # rsi
        df.ta.rsi(length=14, append=True)
        # williams %r
        df.ta.willr(length=14, append=True)
        # awesome oscillator
        df.ta.ao(append=True)
        # adx
        df.ta.adx(append=True)
        # atr
        df.ta.atr(append=True)
        df.dropna(inplace=True)
        return df

    @staticmethod
    def load_time_data(df, date_col='date'):
        df[date_col] = pd.to_datetime(df[date_col])
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month
        df['Day'] = df[date_col].dt.day
        df['Hour'] = df[date_col].dt.hour
        df['Minute'] = df[date_col].dt.minute
        df['Second'] = df[date_col].dt.second
        df['Weekday'] = df[date_col].dt.weekday
        return df

    @staticmethod
    def rename_indicator_columns(df):
        df.columns = [
            f'indicator_{i}' if col.lower() not in ['date', 'close', 'open', 'high', 'low', 'volume'] else col.lower()
            for i, col in enumerate(df.columns)]
        return df
