from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import date
import pandas as pd
import os
from pydantic import BaseModel

# 라우터 인스턴스
router = APIRouter()

class UglyResponse(BaseModel):
    date: str
    pred: float
    pred_ugly: float

# 계산 함수
def calculate_ugly(item_number: str, dt: str) -> pd.DataFrame:
    orig_cost_path = './app/data/orig_cost.xlsx'
    top_path = f'./app/results/{item_number}_4_{dt}.csv'
    mid_path = f'./app/results/{item_number}_5_{dt}.csv'

    for path in (orig_cost_path, top_path, mid_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {path}")

    df_top = pd.read_csv(top_path).rename(columns={'dt': 'date', 'v': 'pred'})
    df_mid = pd.read_csv(mid_path).rename(columns={'dt': 'date', 'v': 'pred'})
    df_cost = pd.read_excel(orig_cost_path)

    merged = pd.merge(df_top, df_mid, on='date', suffixes=('_top', '_medium'))
    if merged.empty:
        raise ValueError(f"date 병합 결과가 비어있습니다.")

    q_decline = merged['pred_top'] / merged['pred_medium']

    try:
        unit_cost_arr = df_cost.loc[df_cost['grain_id'] == int(item_number), '단위조정가'].values
    except Exception:
        unit_cost_arr = df_cost.loc[df_cost['grain_id'] == item_number, '단위조정가'].values

    if unit_cost_arr.size == 0:
        raise ValueError(f"grain_id={item_number}에 해당하는 단위조정가가 없습니다.")

    unit_cost = unit_cost_arr[0]
    ugly_cost = q_decline * unit_cost

    ugly = df_top[['date', 'pred']].copy()
    ugly['pred_ugly'] = ugly_cost.values

    output_path = f'./results/LSTM_{item_number}_ugly.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ugly.to_csv(output_path, index=False)

    return ugly

#  GET + Path Variable + Query Param 사용
@router.get("/future_calc/{grain_id}", response_model=List[UglyResponse], tags=["future_calc"])
def get_ugly_result(grain_id: str):
    try:
        dt = date.today().isoformat()
        df_result = calculate_ugly(grain_id, dt)
        df_result['date'] = pd.to_datetime(df_result['date']).dt.strftime('%Y-%m-%d')
        return df_result.to_dict(orient="records")

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ugly 계산 중 오류: {e}")
