import pandas as pd
from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel
from typing import List
from datetime import date

router = APIRouter()

def past_ugly(grain_id: int) -> pd.DataFrame:
    """
    주어진 grain_id에 대해
    1) orig_cost.xlsx에서 단위조정가를 가져오고,
    2) kamis.parquet에서 해당 grain_id(등급 4와 5)를 필터링한 뒤
    3) decline_ratio, ugly_cost를 계산하여 DataFrame으로 반환합니다.
    """
    # 1) 단위조정가 로드
    try:
        orig_cost_df = pd.read_excel('./app/data/orig_cost.xlsx')
    except FileNotFoundError:
        raise FileNotFoundError("orig_cost.xlsx 파일을 찾을 수 없습니다.")
    # 2) kamis 데이터 로드
    try:
        kamis_df = pd.read_parquet('kamis.parquet', engine='pyarrow')
    except FileNotFoundError:
        raise FileNotFoundError("kamis.parquet 파일을 찾을 수 없습니다.")
    except Exception as e:
        raise RuntimeError(f"kamis.parquet 파일 로드 중 오류: {e}")

    # 3) 단위조정가 추출
    orig_cost_series = orig_cost_df.loc[
        orig_cost_df['grain_id'] == grain_id,
        '단위조정가'
    ]
    if orig_cost_series.empty:
        raise ValueError(f"grain_id {grain_id}에 해당하는 단위조정가가 없습니다.")
    orig_cost = float(orig_cost_series.values[0])

    # 4) kamis 데이터 필터링
    df_filtered = kamis_df.loc[
        kamis_df['grain_id'].astype(str).str.startswith(str(grain_id))
    ]
    # 등급 4, 5 존재 여부 확인
    val4_exists = f"{grain_id}_04" in df_filtered['grain_id'].values
    val5_exists = f"{grain_id}_05" in df_filtered['grain_id'].values
    if not val4_exists or not val5_exists:
        missing = []
        if not val4_exists:
            missing.append(f"{grain_id}_04")
        if not val5_exists:
            missing.append(f"{grain_id}_05")
        raise ValueError(f"다음 등급 데이터가 없습니다: {', '.join(missing)}")

    # 5) 등급별 시계열 분리
    val_4 = (
        df_filtered
        .loc[df_filtered['grain_id'] == f"{grain_id}_04", ['dt', 'v']]
        .rename(columns={'v': 'v_4'})
    )
    val_5 = (
        df_filtered
        .loc[df_filtered['grain_id'] == f"{grain_id}_05", ['dt', 'v']]
        .rename(columns={'v': 'v_5'})
    )

    # 6) 병합 및 계산
    merged = val_4.merge(val_5, on='dt', how='inner')
    if merged.empty:
        raise ValueError("등급 4와 5의 공통 날짜가 없습니다.")
    merged['decline_ratio'] = merged['v_4'] / merged['v_5']
    merged['ugly_cost']     = merged['decline_ratio'] * orig_cost
    merged['dt']            = pd.to_datetime(merged['dt']).dt.date

    return merged

# Pydantic 모델
class UglyRecord(BaseModel):
    dt: date
    v_4: float
    v_5: float
    decline_ratio: float
    ugly_cost: float

class PastUglyResponse(BaseModel):
    data: List[UglyRecord]

# 엔드포인트 정의
@router.get("/past_ugly/{grain_id}", response_model=PastUglyResponse, tags=["ugly"])
def read_past_ugly(grain_id: int = Path(..., description="품목 코드 (예: 151)")):
    """
    grain_id에 해당하는 가격 비율 및 조정비용 상위 7개 레코드를 반환합니다.
    """
    try:
        df = past_ugly(grain_id)
        records = df.sort_values('dt', ascending=False).head(7).to_dict(orient="records")
        return {"data": records}
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=500, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"알 수 없는 오류 발생: {e}")


@router.get("/today_v5/{grain_id}", tags=["ugly"])
def get_today_v5(grain_id: int = Path(..., description="품목 코드 (예: 151)")):
    """
    grain_id에 해당하는 오늘 날짜의 v_5 값을 반환합니다.
    """
    try:
        df = past_ugly(grain_id)
        today = date.today()

        today_row = df.loc[df['dt'] == today]

        if today_row.empty:
            raise HTTPException(status_code=404, detail=f"{today} 날짜에 대한 데이터가 없습니다.")

        v5_value = float(today_row['v_5'].values[0])
        return {"v_5": v5_value}

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=500, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"알 수 없는 오류 발생: {e}")