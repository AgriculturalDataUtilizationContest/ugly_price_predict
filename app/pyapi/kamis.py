import datetime
from dateutil.relativedelta import relativedelta
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
from app.core.config import settings
import time
import pytz

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

def get_kamis_data(start_dt: str, end_dt: str) -> pd.DataFrame:
    API_KEY  = settings.kamis_api_key
    USER_KEY = settings.kamis_user_key
    target_items = {
        '151': '고구마', '152': '감자', '211': '배추', '212': '양배추',
        '214': '상추', '221': '수박', '222': '참외', '224': '호박',
        '226': '딸기', '231': '무', '245': '양파', '246': '파',
        '411': '사과', '412': '배', '415': '감귤'
    }
    BASE_URL = 'https://www.kamis.or.kr/service/price/xml.do'
    region_codes = [
        '1101','2100','2200','2300','2401','2501','2601','2701',
        '3111','3112','3113','3138','3145','3211','3214','3311',
        '3411','3511','3613','3711','3714','3814','3818','3911'
    ]
    product_rank_map = {"04": "상품", "05": "중품"}

    # 1년 단위로 구간 분할
    segments = []
    s_date = datetime.date.fromisoformat(start_dt)
    e_date = datetime.date.fromisoformat(end_dt)
    cursor = s_date
    while cursor < e_date:
        nxt = min(cursor + relativedelta(years=1), e_date)
        segments.append((cursor, nxt))
        cursor = nxt

    # HTTP 세션 & retry
    session = requests.Session()
    session.mount('https://', HTTPAdapter(
        max_retries=Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429,500,502,503,504],
            allowed_methods=frozenset({"GET"})
        )
    ))

    all_records = []
    for seg_start, seg_end in segments:
        for code, name in target_items.items():
            for rank, rank_name in product_rank_map.items():
                for region in region_codes:
                    params = {
                        'action': 'periodRetailProductList',
                        'p_cert_key': API_KEY,
                        'p_cert_id': USER_KEY,
                        'p_returntype': 'json',
                        'p_startday': seg_start.isoformat(),
                        'p_endday': seg_end.isoformat(),
                        'p_countrycode': region,
                        'p_itemcategorycode': code[0] + '00',
                        'p_itemcode': code,
                        'p_productrankcode': rank,
                        'p_convert_kg_yn': 'N'
                    }
                    # retry logic
                    backoff, ok = 5, False
                    for _ in range(4):
                        try:
                            resp = session.get(BASE_URL, params=params, timeout=(5,60))
                            resp.raise_for_status()
                            ok = True
                            break
                        except:
                            time.sleep(backoff)
                            backoff *= 2
                    if not ok:
                        continue

                    data = resp.json().get('data', {})
                    if data.get('error_code') != '000':
                        continue

                    items = data.get('item') or []
                    if isinstance(items, dict):
                        items = [items]
                    for it in items:
                        if it.get('countyname') != '평균':
                            continue
                        try:
                            price = float(it.get('price','').replace(',','').strip())
                        except ValueError:
                            continue
                        dstr = f"{it.get('yyyy')}/{it.get('regday').strip()}"
                        dt = pd.to_datetime(dstr, format='%Y/%m/%d', errors='coerce')
                        if pd.isna(dt):
                            continue
                        all_records.append({
                            '품목코드': code,
                            '품목명': name,
                            '등급코드': rank,
                            '등급코드명': rank_name,
                            '가격': price,
                            'full_date': dt.normalize()
                        })
                    time.sleep(1)

    # DataFrame 정리
    df = pd.DataFrame(all_records)
    df['가격'] = pd.to_numeric(df['가격'], errors='coerce')
    df.dropna(subset=['가격'], inplace=True)

    idx = pd.date_range(start_dt, end_dt, freq='B')
    parts = []
    
    df_grouped = (
        df.groupby(['품목코드', '등급코드', 'full_date'])['가격']
        .mean()
        .round(0)
        .astype(int)
        .reset_index()
    )
	
    for (code, rank), grp in df_grouped.groupby(['품목코드', '등급코드']):
	ts = grp.set_index('full_date')['가격']
	ts = ts.reindex(idx).ffill().bfill()
	tmp = ts.reset_index().rename(columns={'index': 'dt', '가격': 'v'})
        tmp['grain_id'] = f"{code}_{rank}"
        parts.append(tmp[['grain_id', 'dt', 'v']])
   
    df_final = pd.concat(parts, ignore_index=True).sort_values(['grain_id','dt'])
    df_final.to_parquet('kamis.parquet', index=False)
    return df_final

# Pydantic 모델
class Record(BaseModel):
    grain_id: str
    dt: datetime.date
    v: float

class KamisResponse(BaseModel):
    data: List[Record]

# FastAPI 엔드포인트
@router.get(
    "/kamis",
    response_model=KamisResponse,
    tags=["kamis"]
)
def read_kamis(
    start_dt: Optional[datetime.date] = Query(
        None, description="시작일 (YYYY-MM-DD). 미지정 시 1년 전"),
    end_dt: Optional[datetime.date] = Query(
        None, description="종료일 (YYYY-MM-DD). 미지정 시 오늘")
):
    #    now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)

    kst = pytz.timezone('Asia/Seoul')
    now = datetime.datetime.now(kst)

    today = now.date()
    if end_dt is None:
        end_dt = today
    if start_dt is None:
        start_dt = today - relativedelta(years=1)

    if start_dt >= end_dt:
        raise HTTPException(400, "start_dt는 end_dt보다 이전이어야 합니다.")

    try:
        df = get_kamis_data(start_dt.isoformat(), end_dt.isoformat())
        records = df.to_dict(orient="records")
        return {"data": records}
    except Exception as e:
        raise HTTPException(500, f"KAMIS 데이터 처리 중 오류: {e}")
