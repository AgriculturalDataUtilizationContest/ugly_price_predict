import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch import optim
from torch.utils.data import DataLoader
from app.util.data_provider import DatasetCustom, StandardScaler

from app.util.mymodule.Linear import Linear
from app.util.mymodule.DLinear import DLinear
from app.util.mymodule.NLinear import NLinear
from app.util.mymodule.Nbeats import Nbeats
from app.util.mymodule.TCN import TCN
from app.util.mymodule.LSTM import LSTM
from app.util.mymodule.PatchTST import PatchTST
from app.util.mymodule.RNN import RNN
from app.util.mymodule.GRU import GRU
from app.util.mymodule.decbcst import decbcstLSTM

def run_prediction_pipeline(configs):
    # 재현성 보장을 위한 코드
    seed_v = configs["seed_v"]
    random.seed(seed_v)
    np.random.seed(seed_v)
    torch.manual_seed(seed_v)
    torch.cuda.manual_seed_all(seed_v)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed_v)

    scaler_target = StandardScaler()
    scaler_exo = StandardScaler()

    train_ds = DatasetCustom(
        grain_id=configs["train_grain_ids"],
        target_path=configs["input"]["target_path"],
        exo_path=configs["input"]["exo_path"],
        start_date=configs["train_range"][0],
        end_date=configs["train_range"][1],
        seq_len=configs["seq_len"],
        pred_len=configs["pred_len"],
        x_features=configs["x_features"]["direct_horizon_1"],
        target=configs["target"],
        stride=1,
        target_scaler=scaler_target,
        exo_scaler=scaler_exo,
        fit_scaler=True,
    )
    # TODO: 'direct_horizon_1' key가 계속 불필요하게 들어가는데 제거하는 방향이 좋지 않을까
    val_ds = DatasetCustom(
        grain_id=configs["train_grain_ids"],
        target_path=configs["input"]["target_path"],
        exo_path=configs["input"]["exo_path"],
        start_date=configs["valid_range"][0],
        end_date=configs["valid_range"][1],
        seq_len=configs["seq_len"],
        pred_len=configs["pred_len"],
        x_features=configs["x_features"]["direct_horizon_1"],
        target=configs["target"],
        stride=configs["stride"],
        target_scaler=scaler_target,
        exo_scaler=scaler_exo,
        fit_scaler=False,
    )

    test_ds = DatasetCustom(
        grain_id=configs["train_grain_ids"],
        target_path=configs["input"]["target_path"],
        exo_path=configs["input"]["exo_path"],
        start_date=configs["test_range"][0],
        end_date=configs["test_range"][1],
        seq_len=configs["seq_len"],
        pred_len=configs["pred_len"],
        x_features=configs["x_features"]["direct_horizon_1"],
        target=configs["target"],
        stride=configs["stride"],
        target_scaler=scaler_target,
        exo_scaler=scaler_exo,
        fit_scaler=False,
    )

    def custom_collate_fn(batch):
        xs, ys, x_dt, y_dt = zip(*batch)
        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return xs, ys, x_dt, y_dt

    train_dataloader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn,
        drop_last=False,
        generator=g,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn,
        drop_last=False,
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn,
        drop_last=False,
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configs["model"] == "DLinear":
        model = DLinear(configs).to(device)
    elif configs["model"] == "NLinear":
        model = NLinear(configs).to(device)
    elif configs["model"] == "Linear":
        model = Linear(configs).to(device)
    elif configs["model"] == "TCN":
        model = TCN(configs).to(device)
    elif configs["model"] == "PatchTST":
        model = PatchTST(configs).to(device)
    elif configs["model"] == "LSTM":
        model = LSTM(configs).to(device)
    elif configs["model"] == "Nbeats":
        model = Nbeats(configs).to(device)
    elif configs["model"] == "RNN":
        model = RNN(configs).to(device)
    elif configs["model"] == "GRU":
        model = GRU(configs).to(device)
    elif configs["model"] == "decbcstLSTM":
        model = decbcstLSTM(configs).to(device)
    else:
        raise ValueError("Invalid model")

    # TODO: LOSS FN 추가하기
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=lambda epochs: 0.95**epochs, last_epoch=-1
    )

    # Early Stopping
    best_loss = float("inf")
    patience_check = 0

    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(configs["epochs"]):
        model.train()
        train_losses = []

        for x, y, x_dates, y_dates in train_dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)

            loss = criterion(pred.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        loss_per_epoch = np.mean(train_losses)
        train_loss_hist.append(loss_per_epoch)
        # print(f"[Epoch {epoch+1}] Train Loss: {loss_per_epoch:.4f}")

        # Validation per every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y, x_dates, y_dates in val_dataloader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = criterion(pred.squeeze(), y.squeeze())
                    val_losses.append(loss.item())

            val_loss_per_epoch = np.mean(val_losses)
            val_loss_hist.append(val_loss_per_epoch)
            # print(f"[Epoch {epoch+1}] Val Loss: {val_loss_per_epoch:.4f}")

            # Early stopping
            if val_loss_per_epoch > best_loss:
                patience_check += 1
                if patience_check >= configs["patience"]:
                    print("Early stopping triggered.")
                    break
            else:
                best_loss = val_loss_per_epoch
                patience_check = 0
    # Test
    model.eval()
    test_losses, test_pred, test_actual = [], [], []
    test_x_dates, test_y_dates = [], []

    with torch.no_grad():
        for x, y, x_dates, y_dates in test_dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred.squeeze(), y.squeeze())
            test_losses.append(loss.item())

            # Inverse transform
            pred_np = scaler_target.inverse_transform(pred.cpu().numpy())
            y_np = scaler_target.inverse_transform(y.cpu().numpy())

            test_pred.append(pred_np)
            test_actual.append(y_np)
            test_x_dates.extend(x_dates)  # list of list
            test_y_dates.extend(y_dates)

    # Concatenate prediction & actual
    all_pred = np.concatenate(test_pred, axis=0)  # (N, pred_len)
    all_actual = np.concatenate(test_actual, axis=0)  # (N, pred_len)

    flattened_y_dates = np.array(
        [date for batch in test_y_dates for date in batch]
    )  # (N * pred_len,)

    # 각 row = 하나의 timestamp (flatten)
    records = []
    for pred_seq, actual_seq, date_seq in zip(all_pred, all_actual, test_y_dates):
        for d, p, a in zip(date_seq, pred_seq, actual_seq):
            records.append({"date": pd.to_datetime(d), "pred": p[0], "actual": a[0]})

    df_results = pd.DataFrame(records)

    return df_results, (model.state_dict(), configs)


def custom_collate_fn_predict(batch):
    xs, ys, x_dates, y_dates = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys, x_dates, y_dates  # 날짜는 그대로 리스트로 유지


def predict_future(configs, model_path):
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps")  # macbook
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = pd.to_datetime(configs["dt"])
    seq_len = configs["seq_len"]
    pred_len = configs["pred_len"]
    grain_id = configs["train_grain_ids"]# 단일 grain 기준

    # 모델 로드
    # checkpoint = torch.load(model_path, map_location=device, weights_only=False)   # macbook 한정
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    model_config = checkpoint["model_config"]

    model_config.update({
    'freq': configs["freq"],
    'input': configs["input"],
    'dt': configs["dt"],
})
    model_class = {
        "Linear": Linear,
        "DLinear": DLinear,
        "NLinear": NLinear,
        "Nbeats": Nbeats,
        "TCN": TCN,
        "LSTM": LSTM,
        "PatchTST": PatchTST,
        "GRU": GRU,
        "RNN": RNN,
        "decbcstLSTM": decbcstLSTM,
    }[model_config["model"]]
    model = model_class(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler_target = StandardScaler()
    scaler_exo = StandardScaler()

    # 예측용 Dataset 생성: 기준일 전 데이터만 사용
    df_target = pd.read_parquet(configs["input"]["target_path"])
    df_exo = pd.read_parquet(configs["input"]["exo_path"])
    df_target["dt"] = pd.to_datetime(df_target["dt"])
    df_exo["dt"] = pd.to_datetime(df_exo["dt"])

    if configs["freq"] == "M":
        end_date = (dt - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    elif configs["freq"] == "W":
        end_date = (dt - pd.DateOffset(weeks=1)).strftime("%Y-%m-%d")
    else:
        end_date = (dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(end_date)

    ds = DatasetCustom(
        grain_id=grain_id,
        target_path=configs["input"]["target_path"],
        exo_path=configs["input"]["exo_path"],
        start_date="2000-01-01",
        end_date=end_date,
        seq_len=seq_len,
        pred_len=pred_len,
        x_features=configs["x_features"]["direct_horizon_1"],
        target=configs["target"],
        target_scaler=scaler_target,
        exo_scaler=scaler_exo,
        fit_scaler=True,
        stride=1,
    )

    if len(ds) == 0:
        raise ValueError("예측에 사용할 수 있는 시계열 구간이 없습니다.")

    loader = DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=custom_collate_fn_predict
    )
    x, _, x_dates, _ = next(iter(loader))

    x = x.to(device)

    with torch.no_grad():
        y_pred = model(x).cpu().numpy()  # shape: [1, pred_len, 1] or [1, pred_len, C]

    # 역변환
    y_pred = scaler_target.inverse_transform(y_pred[0])

    # 결과 반환
    if configs["freq"] == "M":
        future_dates = pd.date_range(start=dt, periods=pred_len, freq="MS")
    elif configs["freq"] == "W":
        future_dates = pd.date_range(start=dt, periods=pred_len, freq="W")
    else:
        future_dates = pd.date_range(start=dt, periods=pred_len, freq="D")
        print(future_dates)
    y_pred_flat = y_pred.squeeze()

    df_result = pd.DataFrame({"date": future_dates, "pred": y_pred_flat})


    print(df_result)    # 예측 결과 확인용
    return df_result