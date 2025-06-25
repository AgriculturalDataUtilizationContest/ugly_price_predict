from fastapi import FastAPI
from app.api.kamis      import router as kamis_router
from app.api.past_ugly  import router as past_ugly_router
from app.api.predict    import router as predict_router
from app.api.future_calc import router as future_calc_router  # 파일 이름이 future_calc.py일 경우
from app.api.results_update import router as pred_all_router
from app.api.generate import router as generate_router

app = FastAPI(
    title="농산물 가격 정보 & 예측 API",
    description="KAMIS 조회, 단위조정가 기반 비율 계산, 학습된 모델 예측 서비스 통합",
    version="1.0.0"
)

# KAMIS 가격 정보 조회
app.include_router(kamis_router,     prefix="/api", tags=["kamis"])
# 과거 비율 기반 조정비용 계산
app.include_router(past_ugly_router, prefix="/api", tags=["ugly"])
# 학습된 모델을 이용한 예측
app.include_router(predict_router,   prefix="/api", tags=["predict"])
app.include_router(future_calc_router, prefix="/api", tags=["future_calc"])
app.include_router(pred_all_router, prefix="/api", tags=["pred_all"])
# wordcloud 생성 및 Spring Boot로 전송
app.include_router(generate_router, prefix="/api", tags=["generate"])
