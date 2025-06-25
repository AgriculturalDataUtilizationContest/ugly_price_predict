from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd   # ← rename·타입 변환용
import os
from app.util.predict import predict_future

router = APIRouter()

# ---------- 요청·응답 모델 ----------
class PredictRequest(BaseModel):
    grain_id: str
    start_dt: Optional[date] = None
    end_dt:   Optional[date] = None
    dt:       Optional[date] = None

class PredictionRecord(BaseModel):
    dt: str   # ISO-8601 날짜 문자열
    v: float  # 예측값

class PredictResponse(BaseModel):
    data: List[PredictionRecord]

# ---------- 라우터 ----------
@router.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict_grain(req: PredictRequest):
    # 0) 날짜 기본값 처리
    today = date.today()
    end_dt   = today
    start_dt = (today - relativedelta(months=1))
    dt       = end_dt

    # 1) 모델 파일 경로
    ckpt_path = f"./models/LSTM_{req.grain_id}.pth"
    if not os.path.exists(ckpt_path):
        raise HTTPException(404, f"모델 파일이 없습니다: {ckpt_path}")

    # 2) configs 구성
    configs = {
        "start_dt": start_dt.isoformat(),
        "end_dt":   end_dt.isoformat(),
        "model":    "LSTM",
        "seq_len":  5,
        "pred_len": 5,
        "freq":     "D",
        "train_grain_ids": req.grain_id,
        "input": {
            "target_path": "./data/retail.parquet",
            "exo_path":    "./data/weather.parquet",
        },
        "target": "v",
        "x_features": {
            "direct_horizon_1": ["TA_AVG", "RN_DAY", "HM_AVG", "SS_DAY", "WS_AVG"]
        },
        "dt": dt.isoformat(),
    }

    # 3) 예측 실행
    try:
        df_pred = predict_future(configs, ckpt_path)  # → columns: ['date', 'pred']

        # 컬럼명 통일: date → dt, pred → v
        df_pred = df_pred.rename(columns={"date": "dt", "pred": "v"})

        # dt를 문자열(YYYY-MM-DD)로, 필요 시 정렬
        df_pred["dt"] = pd.to_datetime(df_pred["dt"]).dt.strftime("%Y-%m-%d")
        df_pred = df_pred.sort_values("dt")

        df_pred.to_csv(f"./results/{req.grain_id}_{dt.isoformat()}.csv", index=False)

        records = df_pred[["dt", "v"]].to_dict(orient="records")
        return {"data": records}


    except Exception as e:
        raise HTTPException(500, f"예측 중 오류: {e}")



