# 1단계: 의존성만 설치 (휠 캐시)
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip wheel --wheel-dir=/build/wheels -r requirements.txt

# 2단계: 실행용 경량 이미지
FROM python:3.12-slim

WORKDIR /app

# 휠 설치
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# 소스 복사
COPY . .

# 로그 실시간 출력
ENV PYTHONUNBUFFERED=1

# FastAPI 실행 (main.py는 app 디렉토리 안에 있음)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
