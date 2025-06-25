# api_pred_all.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import date, timedelta
import os, torch

from app.util.predict import predict_future    # ← 기존 모델 예측 함수 그대로 사용

router = APIRouter()

# ─────────────────── 설정 ────────────────────
GRAIN_IDS = [
    "415_4","415_5","152_4","152_5","151_4","151_5","226_4","226_5",
    "231_4","231_5","412_4","412_5","211_4","211_5","411_4","411_5",
    "214_4","214_5","221_4","221_5","212_4","212_5","245_4","245_5",
    "222_4","222_5","246_4","246_5","224_4","224_5",
]

# ─────────────────── 응답 모델 ─────────────────
class PredFile(BaseModel):
    grain_id: str
    saved_path: str

class PredAllResponse(BaseModel):
    generated: List[PredFile]

# ─────────────────── 디바이스 선택 ──────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ─────────────────── 엔드포인트 ─────────────────
@router.get("/pred_all", response_model=PredAllResponse, tags=["pred_all"])
def run_all_predictions():
    """
    • today(포함) 기준 1 달 전(start_dt) ~ today(end_dt) 구간 예측
    • 모든 grain_id 대해 순차 실행 → `./results/{grain_id}_{end_dt}.csv` 저장
    """
    today = date.today()
    start_dt = (today - timedelta(days=30)).isoformat()
    end_dt   = today.isoformat()

    saved_files: List[PredFile] = []

    for gid in GRAIN_IDS:
        try:
            configs = {
                "start_dt": start_dt,
                "end_dt":   end_dt,
                "model":    "LSTM",
                "seq_len":  5,
                "pred_len": 5,
                "freq":     "D",
                "train_grain_ids": gid,
                "input": {
                    "target_path": "./app/data/retail.parquet",
                    "exo_path":    "./app/data/weather.parquet",
                },
                "target": "v",
                "x_features": {
                    "direct_horizon_1": ["TA_AVG","RN_DAY","HM_AVG","SS_DAY","WS_AVG"]
                },
                "dt": end_dt,
                "device": str(DEVICE),
            }

            ckpt_path = f"./app/models/LSTM_{gid}.pth"
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"모델 없음: {ckpt_path}")

            df_pred = predict_future(configs, ckpt_path)   # ← 예측 실행
            out = f"./app/results/{gid}_{end_dt}.csv"
            os.makedirs(os.path.dirname(out), exist_ok=True)
            df_pred.to_csv(out, index=False)

            saved_files.append(PredFile(grain_id=gid, saved_path=out))

        except Exception as e:
            # 한 grain_id에서 실패해도 나머지는 계속 진행
            saved_files.append(PredFile(grain_id=gid, saved_path=f"ERROR: {e}"))

    return {"generated": saved_files}
