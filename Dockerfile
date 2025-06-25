# 1단계: 의존성만 설치 (휠 캐시)
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 2단계: 실행용 경량 이미지
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 빌드한 휠 파일 설치
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# 전체 소스 복사 (.env는 dockerignore로 제외)
COPY . .

# 로그 실시간 출력
ENV PYTHONUNBUFFERED=1

# FastAPI 실행 (main.py 안의 app 기준)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
