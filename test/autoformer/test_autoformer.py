import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from transformers import AutoformerConfig, AutoformerModel, AutoformerForPrediction

from data_processor.rolling_dataset import RollingDataset
from financial_processor.financial_processor import FinancialProcessor
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

prediction_length = 30  # The length of the forecasted period
input_size = 14  # Assuming OHLC data has 4 features: Open, High, Low, Close
d_model = 64  # Dimension of the model embeddings
encoder_layers = 2  # Number of encoder layers
decoder_layers = 2  # Number of decoder layers
batch_size = 64  # Number of samples per batch
window_length = 307

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_init_autoformer():
    # Load and preprocess data
    df = FinancialProcessor.load_data('D:/tick_data/5min/EURUSD_5min.csv')
    # load data until 2021 (training data)
    df = df[df['date'] < '2021-01-01']
    # df = df.drop(columns=['date'])
    df = FinancialProcessor.load_indicators(df)
    # name all indicators as indicator_i for each indicator in our columns
    df = FinancialProcessor.rename_indicator_columns(df)
    df = FinancialProcessor.load_time_data(df)
    df.rename(
        columns={'close': 'Close', 'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'},
        inplace=True)
    df['past_observed'] = 1

    # df = df.drop(columns=['Date'])
    #
    # # scale the df
    # scaler = MinMaxScaler()
    # df[df.columns] = scaler.fit_transform(df[df.columns])

    context_length = 300

    # Load the data in batches
    train_loader = DataLoader(dataset=RollingDataset(df, window_length, prediction_length), batch_size=batch_size, shuffle=True)

    # Define Autoformer Configuration
    config = AutoformerConfig(
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=1,
        d_model=d_model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_time_features=3
    )

    # Initialize Autoformer Model
    model = AutoformerForPrediction(config)
    model.to(device)

    # load model weights
    model.load_state_dict(torch.load('autoformer.pt'))

    for epoch in range(1000):

        for i, (features, time_data, past_obv, future_values, future_time_feats) in enumerate(train_loader):

            print(f'Epoch {epoch}, Current training iteration: {i}/{len(train_loader)}')

            features = torch.tensor(features, dtype=torch.float).to(device)
            time_data = torch.tensor(time_data, dtype=torch.long).to(device)
            past_obv = torch.tensor(past_obv, dtype=torch.long).to(device)
            future_values = torch.tensor(future_values, dtype=torch.float).to(device)
            future_time_feats = torch.tensor(future_time_feats, dtype=torch.long).to(device)

            # Forward Pass through the model
            outputs = model(past_values=features, past_time_features=time_data, past_observed_mask=past_obv,
                            future_values=future_values,
                            future_time_features=future_time_feats)

            loss = outputs.loss
            loss.backward()

            if i % 100 == 0:
                # predict a single element
                predicted_out = model.generate(
                    past_values=features[-1].reshape(1, -1),
                    past_time_features=time_data[-1].unsqueeze(0),
                    past_observed_mask=past_obv[-1].reshape(1, -1),
                    future_time_features=future_time_feats[-1].unsqueeze(0),
                )
                predicted_out = predicted_out['sequences']
                last_predicted_prices = np.array(predicted_out[-1][-1].cpu())
                real_prices = np.array(future_values[-1].cpu())

                # Create a figure and axis
                plt.plot(last_predicted_prices, label='Predicted', color='red')
                plt.legend()
                plt.show()

                plt.plot(real_prices, label='Real', color='blue')
                plt.legend()
                plt.show()

                print(f'Loss: {loss}')
                # save the model
                torch.save(model.state_dict(), 'autoformer.pt')


def test_example():
    from huggingface_hub import hf_hub_download
    import torch
    from transformers import AutoformerForPrediction

    file = hf_hub_download(
        repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
    )
    batch = torch.load(file)

    model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly")

    # during training, one provides both past and future values
    # as well as possible additional features
    outputs = model(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        static_categorical_features=batch["static_categorical_features"],
        static_real_features=batch["static_real_features"],
        future_values=batch["future_values"],
        future_time_features=batch["future_time_features"],
    )

    loss = outputs.loss
    loss.backward()

    # during inference, one only provides past values
    # as well as possible additional features
    # the model autoregressively generates future values
    outputs = model.generate(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        static_categorical_features=batch["static_categorical_features"],
        static_real_features=batch["static_real_features"],
        future_time_features=batch["future_time_features"],
    )

    mean_prediction = outputs.sequences.mean(dim=1)
