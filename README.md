# 농산물 가격 예측 API

## 1. 프로젝트 개요 (Overview)

- **프로젝트 이름:** 농산물 가격 예측 API
- **한 줄 요약:** 농산물(일반/못난이)의 과거 및 미래 가격 정보를 제공하고, 시계열 예측 모델을 통해 미래 가격을 예측하는 FastAPI 기반 API 서버입니다.
- **핵심 가치:** 농산물 유통 시장의 가격 변동성을 예측하여 생산자와 소비자 모두에게 안정적인 가격 정보를 제공하고, 특히 '못난이 농산물'의 가치를 재평가하여 시장 활성화에 기여합니다.

## 2. 주요 기능 (Features)

- **데이터 파이프라인 자동화:** Spring Boot의 `@Scheduled` 기능을 통해 매일 자정, 데이터 수집부터 가격 예측까지의 전체 파이프라인을 완전 자동으로 실행합니다.
- **일일 가격 데이터 수집:** `kamis.or.kr`의 Open API를 활용하여 **매일 약 5,000건의** 최신 소매가격 데이터를 수집하고 Parquet 형식으로 저장합니다.
- **과거 못난이 농산물 가격 분석:** 수집된 데이터를 기반으로 상품/중품 가격 비율을 계산하여 과거 '못난이 농산물'의 추정 가격을 제공합니다.
- **미래 가격 예측:** PyTorch로 구현된 `decbcstLSTM` 시계열 모델을 사용하여 주요 농산물의 미래 5일치 가격(상품/중품)을 예측합니다.
- **미래 못난이 농산물 가격 계산:** 예측된 상품/중품 가격과 기존 단위 조정가를 바탕으로 미래 '못난이 농산물'의 가격을 계산합니다.
- **Word Cloud 생성:** 특정 농산물과 관련된 키워드로 Word Cloud 이미지를 동적으로 생성하고, 외부 이미지 서버에 업로드하여 URL을 반환합니다.

### API 주요 엔드포인트 예시

- **미래 못난이 농산물 가격 예측:**
  ```http
  GET /pyapi/future_calc/{grain_id}
  ```
- **과거 못난이 농산물 가격 조회:**
  ```http
  GET /pyapi/past_ugly/{grain_id}
  ```
- **개별 품목 가격 예측 실행 (수동):**
  ```http
  POST /pyapi/predict
  ```
  **Body:**
  ```json
  {
    "grain_id": "151_4"
  }
  ```
- **Word Cloud 이미지 URL 생성:**
  ```http
  GET /pyapi/generate?cropName=사과
  ```

## 3. 설치 및 실행 방법 (Installation & Usage)

### 최소 요구사항

- Python 3.9+
- Docker (권장)

### 의존성 설치

프로젝트 루트 디렉터리에서 아래 명령어를 실행하여 필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

### 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래와 같이 KAMIS API 키를 설정해야 합니다.

```
KAMIS_API_KEY="<YOUR_KAMIS_API_KEY>"
KAMIS_USER_KEY="<YOUR_KAMIS_USER_KEY>"
```

### 로컬 실행

Uvicorn을 사용하여 FastAPI 서버를 실행합니다.

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

서버가 실행되면 `http://127.0.0.1:8000/docs`에서 API 문서를 확인할 수 있습니다.

### Docker를 이용한 실행

Docker를 사용하여 프로젝트를 컨테이너 환경에서 실행할 수 있습니다.

1. **Docker 이미지 빌드:**
   ```bash
   docker build -t ugly-price-predict .
   ```

2. **Docker 컨테이너 실행:**
   ```bash
   docker run -d -p 8000:8000 --env-file ./.env ugly-price-predict
   ```

## 4. 아키텍처 & 기술 스택

### 사용 기술

- **언어:** Python 3
- **웹 프레임워크:** FastAPI
- **머신러닝/데이터 분석:**
  - PyTorch
  - Pandas
  - NumPy
  - Scikit-learn
- **기타:**
  - WordCloud: 키워드 시각화
  - Requests: 외부 API 통신
  - Uvicorn: ASGI 서버

### 프로젝트 구조

```
/
├── app/
│   ├── main.py             # FastAPI 앱 초기화 및 라우터 설정
│   ├── pyapi/              # API 엔드포인트 구현
│   │   ├── kamis.py        # KAMIS 데이터 수집
│   │   ├── past_ugly.py    # 과거 데이터 기반 가격 계산
│   │   ├── predict.py      # 모델 예측 실행
│   │   ├── future_calc.py  # 예측 기반 못난이 가격 계산
│   │   └── ...
│   ├── util/               # 모델 및 데이터 처리 유틸리티
│   │   ├── predict.py      # 예측 파이프라인
│   │   ├── data_provider.py# 데이터셋 클래스
│   │   └── mymodule/       # PyTorch 모델 아키텍처 (LSTM, DLinear 등)
│   ├── models/             # 학습된 모델 파라미터 (.pth)
│   ├── data/               # 원본 데이터 (Parquet, Excel)
│   └── results/            # 예측 결과 저장 (CSV)
├── Dockerfile              # Docker 이미지 빌드 설정
├── requirements.txt        # Python 의존성 목록
└── README.md               # 프로젝트 문서
```

### 시스템 아키텍처 및 자동화

이 시스템은 본 FastAPI 프로젝트와 외부 Spring Boot 애플리케이션이 연동하여 동작합니다.

- **FastAPI (본 프로젝트):**
  - 데이터 처리, 모델 예측, 가격 계산 등 핵심 비즈니스 로직을 수행합니다.
  - 외부에서 호출할 수 있는 HTTP API 엔드포인트를 제공합니다.

- **Spring Boot (외부 연동):**
  - **데이터 파이프라인 오케스트레이터** 역할을 수행합니다.
  - 내부적으로 `@Scheduled` 어노테이션을 사용하여, 매일 정해진 시간에 본 FastAPI 프로젝트의 `/pyapi/kamis` (데이터 수집), `/pyapi/pred_all` (일괄 예측) 엔드포인트를 순차적으로 호출하여 전체 데이터 파이프라인을 자동화합니다.
  - 또한, FastAPI에서 생성된 Word Cloud 이미지를 받아 S3에 업로드하고 URL을 반환하는 역할을 담당합니다.

- **외부 API:**
  - **KAMIS 농산물 유통정보(www.kamis.or.kr):** Open API를 통해 농산물 소매가격 데이터를 수집합니다.
