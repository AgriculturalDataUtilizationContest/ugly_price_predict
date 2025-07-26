import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from app.util.mymodule.layers.PatchTST_layers import series_decomp
from app.util.mymodule.Linear_blocks import NLinear_block
from app.util.mymodule.LSTM import LSTM
from app.util.mymodule.Nbeats import Nbeats
from app.util.mymodule.TCN import TCN
from app.util.mymodule.PatchTST import PatchTST
from app.util.mymodule.NLinear import NLinear

class BaseDecbcstModel(nn.Module):
    def __init__(self, configs, model_class):
        super(BaseDecbcstModel, self).__init__()
        self.Decomp_kernel_size = configs['Decomp_kernel_size']
        self.Decomp_stride = configs['Decomp_stride']
        self.fcst_type = configs['fcst_type']
    
        self.decompsition = series_decomp(self.Decomp_kernel_size, self.Decomp_stride)
        self.Model = model_class(configs)  # LSTM 또는 Nbeats
        self.NL = NLinear_block(configs)

    def forward(self, x):
        x = x.permute(0,2,1)  # x: [Batch, Channel, Input length]
        res_init, trend_init = self.decompsition(x)
        res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)

        _, fcst = self.NL(trend_init)
        bcst1, fcst1 = self.NL(res_init)
        x_res = x.permute(0,2,1) - bcst1
        fcst2 = self.Model(x_res)

        out = fcst1 + fcst2 + fcst  # 최종 예측값

        if self.fcst_type == 'MS':
            return out[:, :, 0].unsqueeze(-1)
        else:
            return out


# LSTM
class decbcstLSTM(BaseDecbcstModel):
    def __init__(self, configs):
        super().__init__(configs, LSTM)

# Nbeats
class decbcstNbeats(BaseDecbcstModel):
    def __init__(self, configs):
        super().__init__(configs, Nbeats)

# TCN
class decbcstTCN(BaseDecbcstModel):
    def __init__(self, configs):
        super().__init__(configs, TCN)

# PatchTST
class decbcstPatchTST(BaseDecbcstModel):
    def __init__(self, configs):
        super().__init__(configs, PatchTST)

# Nlinear
class decbcstNLinear(BaseDecbcstModel):
    def __init__(self, configs):
        super().__init__(configs, NLinear)