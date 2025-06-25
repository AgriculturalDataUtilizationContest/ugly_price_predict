import torch.nn as nn
class NLinear(nn.Module):
    """
    Normalization-based Linear Forecasting
    """
    def __init__(self, configs):
        super(NLinear, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.fcst_type = configs['fcst_type']
        self.enc_in = configs['enc_in']  # Channel 수

        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [B, L, C]
        seq_last = x[:, -1:, :].detach()  # [B, 1, C]
        x = x - seq_last  # normalization

        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.Linear(x)      # [B, C, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, C]

        x = x + seq_last  # de-normalization (broadcast)

        if self.fcst_type == 'MS':
            return x[:, :, 0].unsqueeze(-1)  # 단변수 다스텝
        else:
            return x  # S or M