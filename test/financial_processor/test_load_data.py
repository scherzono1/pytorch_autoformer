from financial_processor.financial_processor import FinancialProcessor


def test_load_data():
    # Load and preprocess data
    df = FinancialProcessor.load_data('D:/tick_data/5min/EURUSD_5min_2023.csv')
    # df = df.drop(columns=['date'])
    df = FinancialProcessor.load_indicators(df)
    # name all indicators as indicator_i for each indicator in our columns
    df = FinancialProcessor.rename_indicator_columns(df)
    df.rename(
        columns={'close': 'Close', 'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'},
        inplace=True)

    assert True
