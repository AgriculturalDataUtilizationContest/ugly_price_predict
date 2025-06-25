import torch
import torch.nn as nn
import torch.nn.functional as F

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x_padded = torch.cat([front, x, end], dim=1)  # [B, L + 2*pad, C]

        x_pooled = self.avg(x_padded.permute(0, 2, 1))  # → [B, C, L']
        x_pooled = x_pooled.permute(0, 2, 1)  # → [B, L', C]

        # 강제 interpolation로 L 맞춤 (x와 shape 동일하게)
        if x_pooled.shape[1] != L:
            x_pooled = F.interpolate(x_pooled.permute(0, 2, 1), size=L, mode='linear', align_corners=False)
            x_pooled = x_pooled.permute(0, 2, 1)

        return x_pooled


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size, stride):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.kernel_size = configs['Decomp_kernel_size']
        self.stride = configs['Decomp_stride']

        self.decomposition = series_decomp(self.kernel_size, self.stride)
        self.individual = configs['individual']
        self.channels = configs['enc_in']
        self.fcst_type = configs['fcst_type']

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [B, L, C]
        seasonal_init, trend_init = self.decomposition(x)  # [B, L, C]
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)  # [B, C, L]

        if self.individual:
            seasonal_output = torch.zeros(seasonal_init.size(0), seasonal_init.size(1), self.pred_len,
                                          dtype=seasonal_init.dtype, device=seasonal_init.device)
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output  # [B, C, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, C]

        if self.fcst_type == 'MS':
            return x[:, :, 0].unsqueeze(-1)
        else:
            return x