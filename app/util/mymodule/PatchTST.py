__all__ = ["PatchTST"]

from typing import Optional
from torch import nn, Tensor
from app.util.mymodule.PatchTST_backbone import PatchTST_backbone
from app.util.mymodule.PatchTST_layers import series_decomp


class PatchTST(nn.Module):
    def __init__(
        self,
        configs,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type="flatten",
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()

        # Configs
        self.fcst_type = configs.get("fcst_type", "MS")
        self.decomposition = configs.get("decomposition", False)

        # Common input configs
        self.in_features = configs.get("enc_in", 1)
        self.lookback = configs.get("seq_len", 36)
        self.horizon = configs.get("pred_len", 12)
        self.patch_len = configs.get("patch_len", 3)
        self.stride = configs.get("patch_stride", 8)
        self.padding_patch = configs.get("padding_patch", "end")
        self.individual = configs.get("individual", False)
        self.revin = configs.get("revin", False)
        self.affine = configs.get("affine", False)
        self.subtract_last = configs.get("subtract_last", False)

        # Backbone hyperparams
        self.n_layers = configs.get("e_layers", 3)
        self.n_heads = configs.get("n_heads", 8)
        self.d_model = configs.get("d_model", 512)
        self.d_ff = configs.get("d_ff", 1024)
        self.dropout = configs.get("dropout", 0.1)
        self.fc_dropout = configs.get("fc_dropout", 0.1)
        self.head_dropout = configs.get("head_dropout", 0.0)

        # Decomposition
        if self.decomposition:
            kernel_size = configs.get("Decomp_kernel_size", 25)
            decomp_stride = configs.get("Decomp_stride", 1)
            self.decomp_module = series_decomp(kernel_size, decomp_stride)

            self.model_trend = self._build_backbone(
                max_seq_len,
                d_k,
                d_v,
                norm,
                attn_dropout,
                act,
                key_padding_mask,
                padding_var,
                attn_mask,
                res_attention,
                pre_norm,
                store_attn,
                pe,
                learn_pe,
                pretrain_head,
                head_type,
                verbose,
                **kwargs
            )
            self.model_res = self._build_backbone(
                max_seq_len,
                d_k,
                d_v,
                norm,
                attn_dropout,
                act,
                key_padding_mask,
                padding_var,
                attn_mask,
                res_attention,
                pre_norm,
                store_attn,
                pe,
                learn_pe,
                pretrain_head,
                head_type,
                verbose,
                **kwargs
            )
        else:
            self.model = self._build_backbone(
                max_seq_len,
                d_k,
                d_v,
                norm,
                attn_dropout,
                act,
                key_padding_mask,
                padding_var,
                attn_mask,
                res_attention,
                pre_norm,
                store_attn,
                pe,
                learn_pe,
                pretrain_head,
                head_type,
                verbose,
                **kwargs
            )

    def _build_backbone(
        self,
        max_seq_len,
        d_k,
        d_v,
        norm,
        attn_dropout,
        act,
        key_padding_mask,
        padding_var,
        attn_mask,
        res_attention,
        pre_norm,
        store_attn,
        pe,
        learn_pe,
        pretrain_head,
        head_type,
        verbose,
        **kwargs
    ):
        return PatchTST_backbone(
            in_features=self.in_features,
            lookback=self.lookback,
            horizion=self.horizon,
            patch_len=self.patch_len,
            stride=self.stride,
            max_seq_len=max_seq_len,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=self.d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=self.dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            fc_dropout=self.fc_dropout,
            head_dropout=self.head_dropout,
            padding_patch=self.padding_patch,
            pretrain_head=pretrain_head,
            head_type=head_type,
            individual=self.individual,
            revin=self.revin,
            affine=self.affine,
            subtract_last=self.subtract_last,
            verbose=verbose,
            **kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.decomposition:
            res_x, trend_x = self.decomp_module(x)
            res_x = self.model_res(res_x.permute(0, 2, 1))
            trend_x = self.model_trend(trend_x.permute(0, 2, 1))
            x = (res_x + trend_x).permute(0, 2, 1)
        else:
            x = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x[:, :, 0].unsqueeze(-1) if self.fcst_type == "MS" else x
