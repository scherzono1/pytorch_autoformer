import joblib
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from custom_transformer.transformer import TransformerClassifier
from data_processor.rolling_dataset_forex_ny import RollingDatasetForexNY
from financial_processor.financial_processor import FinancialProcessor
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 512  # Number of samples per batch
window_size = 480  # Number of previous samples to use as input to prediction model (8 hours with 1min data minus 1)


def test_init_autoformer():

    # Load and preprocess data
    df = FinancialProcessor.load_data('D:/tick_data/1min/EURUSD_ALL.csv')
    # load data until 2021 (training data)
    df = df[df['date'] < '2021-01-01']
    df.rename(
        columns={'close': 'Close', 'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'},
        inplace=True)

    df = df[['Date', 'Open', 'High', 'Low', 'Close']]

    # scale all columns except date
    scaler = MinMaxScaler()
    scale_cols = [x for x in df.columns if x.lower() != 'date']
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    # save the scaler weights
    joblib.dump(scaler, 'scaler.save')

    df = FinancialProcessor.load_time_data(df, date_col='Date')

    # Load the data in batches
    train_loader = DataLoader(dataset=RollingDatasetForexNY(df), batch_size=batch_size,
                              shuffle=True)

    model = TransformerClassifier(input_feature_size=4, num_classes=2, num_heads=4, num_encoder_layers=2,
                                  dim_feedforward=512, max_seq_length=window_size)
    model.to(device)

    # loss and optimizer (for classification considering the final output is softmax)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):

        for i, (features, labels) in enumerate(train_loader):

            # convert to pytorch tensors
            features = torch.tensor(features).float()
            labels = torch.tensor(labels).long()

            # Move tensors to the configured device (GPU)
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{1000}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            if (i+1) % 5 == 0:
                torch.save(model.state_dict(), 'custom_transformer.pt')
