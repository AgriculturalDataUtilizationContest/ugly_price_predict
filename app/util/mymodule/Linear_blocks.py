from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from app.util.mymodule.layers.PatchTST_layers import series_decomp

##Nlinear
class NLinear_block(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(NLinear_block, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.fcst_type = configs['fcst_type']
        self.fcstLinear = nn.Linear(self.seq_len, self.pred_len)
        self.bcstLinear = nn.Linear(self.seq_len, self.seq_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        fcst = self.fcstLinear(x.permute(0,2,1)).permute(0,2,1)
        fcst = fcst + seq_last
        bcst = self.bcstLinear(x.permute(0,2,1)).permute(0,2,1)
        bcst = bcst + seq_last
    
        return bcst, fcst # [Batch, Output length, Channel]


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size, stride):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

### Dlinear
class DLinear_block(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(DLinear_block, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.Decomp_kernel_size = configs['Decomp_kernel_size']
        self.Decomp_stride = configs['Decomp_stride']

        # Decompsition Kernel Size
        self.decompsition = series_decomp(self.Decomp_kernel_size, self.Decomp_stride)
        self.individual = configs['individual']
        self.channels = configs['enc_in']
        self.fcst_type =configs['fcst_type']
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.fcstLinear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.fcstLinear_Trend = nn.Linear(self.seq_len,self.pred_len)
            self.bcstLinear_Seasonal = nn.Linear(self.seq_len,self.seq_len)
            self.bcstLinear_Trend = nn.Linear(self.seq_len,self.seq_len)


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_fcst = self.fcstLinear_Seasonal(seasonal_init)
            trend_fcst = self.fcstLinear_Trend(trend_init)
            seasonal_bcst = self.bcstLinear_Seasonal(seasonal_init)
            trend_bcst = self.bcstLinear_Trend(trend_init)

            bcst = seasonal_bcst + trend_bcst
            fcst = seasonal_fcst + trend_fcst
            fcst = fcst.permute(0,2,1)
            bcst = bcst.permute(0,2,1)
        return bcst, fcst

