import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    """
    MLP 기반 Forecasting 모델
    - 입력: [B, L, C]
    - 출력:
      - S: [B, pred_len]
      - M: [B, pred_len, C]
      - MS: [B, pred_len, 1]
    """
    def __init__(self, configs):
        super(Linear, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.enc_in = configs['enc_in']  # channel 수
        self.fcst_type = configs['fcst_type']

        hidden_dims = configs['hidden_dims']  # e.g., [64, 32]
        dropout = configs['dropout']
        activation = getattr(F, configs['activation']) if isinstance(configs['activation'], str) else F.relu

        self.activation_fn = activation

        input_dim = self.seq_len * self.enc_in
        output_dim = {
            "S": 1,
            "M": self.enc_in,
            "MS": 1
        }[self.fcst_type]

        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, self.pred_len * output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, seq_len, C]
        b, l, c = x.shape
        x = x.reshape(b, -1)  # flatten to [B, L*C]

        out = self.model(x)  # [B, pred_len * out_dim]
        out_dim = {
            "S": 1,
            "M": self.enc_in,
            "MS": 1
        }[self.fcst_type]

        out = out.view(b, self.pred_len, out_dim)

        return out