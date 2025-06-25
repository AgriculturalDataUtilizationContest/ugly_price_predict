from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
import requests
import random
import os

router = APIRouter()

# Spring Boot 파일 업로드 엔드포인트
SPRING_IMAGE_URL = "http://localhost:8080/api/image"
font_path = os.path.join(os.path.dirname(__file__), "../font/NotoSansKR-Regular.ttf")
font_path = os.path.abspath(font_path)

class WordRequest(BaseModel):
    cropName: str

@router.get("/generate")
def generate_and_upload(cropName: str):
    # 1) 관련 키워드 리스트 + 랜덤 가중치
    keywords = [
        cropName,             # 예: "사과"
        "직거래", "산지직송", "도매시장", "온라인판매", "신선도",
        "가격투명", "품질인증", "물류비절감", "농민직접", "플랫폼",
        "유통망", "소비자", "물류자동화", "수기예측", "계절수요"
    ]
    freqs = { w: (200 if w == cropName else random.randint(5, 100))
              for w in keywords }

    # 2) 워드클라우드 생성 (generate_from_frequencies 사용)
    wc = WordCloud(
        font_path=font_path,
        width=576,
        height=185,
        background_color="white",
        stopwords=STOPWORDS,
        max_words=len(freqs)
    ).generate_from_frequencies(freqs)

    # 3) 메모리버퍼에 PNG로 저장
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)

    # 4) Spring Boot에 파일 전송
    files = {"images": ("cloud.png", buf, "image/png")}
    resp = requests.post(SPRING_IMAGE_URL, files=files)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Spring API error: {resp.text}")

    # 5) S3 URL 파싱해서 반환
    # 만약 Spring이 plain text로 URL만 돌려준다면 resp.text, JSON이라면 resp.json()
    s3_url = resp.json().get("url") if resp.headers.get("content-type","").startswith("application/json") else resp.text
    return {s3_url}