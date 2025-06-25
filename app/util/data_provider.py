import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        data = np.asarray(data)
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

# Todo Grain_id가 여러개인 경우 'M"으로 받을 수 있게끔 작업하기.. 근데 굳이 필요할까? MS
class DatasetCustom(Dataset):
    def __init__(
        self,
        grain_id: str,
        target_path: str,
        exo_path: str,
        start_date: str,
        end_date: str,
        seq_len: int,
        pred_len: int,
        x_features: list,
        target: str,
        target_scaler: StandardScaler,
        exo_scaler: StandardScaler,
        fit_scaler: bool = False,
        stride: int = 1
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.x_features = x_features
        self.target = target
        self.target_scaler = target_scaler
        self.exo_scaler = exo_scaler
        self.stride = stride

        # Load & merge
        df_t = pd.read_parquet(target_path)
        df_e = pd.read_parquet(exo_path)
        df_t['dt'] = pd.to_datetime(df_t['dt'])
        df_e['dt'] = pd.to_datetime(df_e['dt'])

        df_t = df_t[df_t['grain_id'] == grain_id].copy()
        df_t.drop(columns='grain_id', inplace=True)
        df_e = df_e[['dt'] + [col for col in df_e.columns if col in x_features]]

        df = pd.merge(df_t, df_e, on='dt', how='left')
        df = df.sort_values('dt')
        df = df[(df['dt'] >= start_date) & (df['dt'] <= end_date)].copy().reset_index(drop=True)

        full_x_cols = [target] + x_features
        df = df.dropna(subset=full_x_cols)

        self.date = df['dt'].values  # 날짜 정보 보존

        target_vals = df[[target]].values  # (N, 1)
        exo_vals = df[x_features].values  # (N, F)

        if fit_scaler:
            self.target_scaler.fit(target_vals)
            self.exo_scaler.fit(exo_vals)

        target_scaled = self.target_scaler.transform(target_vals)  # (N, 1)
        exo_scaled = self.exo_scaler.transform(exo_vals)          # (N, F)

        self.data_x = np.concatenate([target_scaled, exo_scaled], axis=1)  # (N, 1+F)
        self.data_y = target_scaled  # (N, 1)

        # Calculate all valid starting indices
        self.indices = []
        total_len = len(self.data_x)
        max_start = total_len - (self.seq_len + self.pred_len) + 1
        for start_idx in range(0, max_start, self.stride):
            self.indices.append(start_idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]

        x_start = index
        x_end = x_start + self.seq_len
        y_start = x_end
        y_end = y_start + self.pred_len

        seq_x = self.data_x[x_start:x_end]
        seq_y = self.data_y[y_start:y_end]

        x_dates = self.date[x_start:x_end]
        y_dates = self.date[y_start:y_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            x_dates,
            y_dates
        )

    def inverse_transform(self, data):
        if torch.is_tensor(data):
            data = data.numpy()
        return self.target_scaler.inverse_transform(data)