import torch.nn as nn
import torch.nn.utils.parametrizations as param


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            param.weight_norm(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            param.weight_norm(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                layer.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self, in_channels, channels_list, kernel_size=2, stride=1, dropout=0.2
    ):
        super().__init__()
        layers = []
        for i in range(len(channels_list)):
            dilation = 2**i
            input_c = in_channels if i == 0 else channels_list[i - 1]
            output_c = channels_list[i]
            layers.append(
                TemporalBlock(
                    input_c,
                    output_c,
                    kernel_size,
                    stride,
                    dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.in_channels = configs["enc_in"]
        self.pred_len = configs["pred_len"]
        self.output_type = configs["fcst_type"]  # "S", "MS", "M"
        if isinstance(configs["TCN_num_features"], str):
            configs["TCN_num_features"] = list(
                map(int, configs["TCN_num_features"].split(","))
            )

        self.tcn = TemporalConvNet(
            in_channels=self.in_channels,
            channels_list=configs["TCN_num_features"],
            kernel_size=configs.get("TCN_kernel_size", 2),
            stride=configs.get("TCN_stride", 1),
            dropout=configs.get("dropout", 0.2),
        )

        last_channel = configs["TCN_num_features"][-1]
        if self.output_type == "M":
            self.linear = nn.Linear(last_channel, self.pred_len * self.in_channels)
        else:  # "S" or "MS"
            self.linear = nn.Linear(last_channel, self.pred_len)

        self._init_weights()

    def _init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        out = self.tcn(x)
        last_step = out[:, :, -1]  # (B, C)
        pred = self.linear(last_step)

        if self.output_type == "M":
            return pred.view(-1, self.pred_len, self.in_channels)
        elif self.output_type == "MS":
            return pred.unsqueeze(-1)
        else:  # "S"
            return pred
