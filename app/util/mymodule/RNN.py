import torch.nn as nn


class RNN(nn.Module):
    """
    RNN with Encoder + Linear Decoder for multi-step forecasting
    """

    def __init__(self, configs):
        super(RNN, self).__init__()
        self.seq_len = configs["seq_len"]  # 25
        self.pred_len = configs["pred_len"]  # 30
        self.in_features = configs["enc_in"]  # 6
        self.hidden_dim = configs["hidden_dim"]
        self.num_layers = configs["num_layers"]
        self.fcst_type = configs["fcst_type"]
        # Encoder LSTM
        self.encoder = nn.RNN(
            input_size=self.in_features,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Linear Decoder: [hidden_dim] -> [pred_len, in_features]
        self.decoder = nn.Linear(self.hidden_dim, self.pred_len * self.in_features)

    def forward(self, x):
        # x: [Batch, Input length, Channel] -> [32, 25, 6]

        # Encoder
        _, h = self.encoder(x)  # h: [num_layers, Batch, hidden_dim]

        # 마지막 layer의 hidden state 사용
        hidden_state = h[-1]  # [Batch, hidden_dim]

        # Decoder (Linear)
        predictions = self.decoder(hidden_state)  # [Batch, pred_len * in_features]

        # Reshape: [Batch, pred_len, in_features] -> [32, 30, 6]
        predictions = predictions.view(-1, self.pred_len, self.in_features)
        if self.fcst_type == "MS":
            return predictions[:, :, 0].unsqueeze(-1)  # [Batch, pred_len, 1]
        else:
            return predictions
