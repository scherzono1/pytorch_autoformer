import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier(nn.Module):
    def __init__(self, input_feature_size, num_classes, num_heads, num_encoder_layers, dim_feedforward, max_seq_length):
        super(TransformerClassifier, self).__init__()

        # Initial linear layer to match the input features to dim_feedforward
        self.input_linear = nn.Linear(input_feature_size, dim_feedforward)

        # Adjusted to use the new positional encoding function
        self.positional_encoding = self.create_positional_encoding(max_seq_length, dim_feedforward)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(dim_feedforward, num_classes)

    def create_positional_encoding(self, seq_len, feature_size):
        pe = torch.zeros(seq_len, feature_size)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / feature_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape becomes (1, seq_len, feature_size)
        return pe

    def forward(self, x):
        x = self.input_linear(x)
        encoding = self.positional_encoding[:, :x.size(1), :]
        encoding = encoding.to(x.device)
        x = x + encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Aggregating over all time-steps
        x = self.fc(x)
        return F.softmax(x, dim=1)
