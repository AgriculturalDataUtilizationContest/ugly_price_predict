'''
실행 순서!

일단 dockefile에서 실행이 안 되어서 뺀 pip 실행하기
pip

/pyapi/kamis 로 매일매일 데이터 불러와야함. (00시 00분마다 실행)
-> 이 api에서 요청하는 parameter는 start_dt, end_dt
(안 보내주어도 작동 되도록 했음.)
-> 이걸 실행하면, kamis_new.parquet 파일이 생성됨.

/pred_all 로 매일매일 예측한 결과 저장해야함. (00시 10분마다 실행)
-> 이 api에서 요청하는 parameter는 없음.

---

/pyapi/patst_ugly/{grain_id}  :: pathvariable로 grain_id를 받아.
반환 데이터 : 금일 포함 7일 정보 (과거 6 + 금일 1)
v_4 : 중품, v_5 : 상품, ugly_cost: 못난이

/pyapi/future_calc/{grain_id} :: pathvariable로 grain_id를 받아.
반환 데이터 : 금일 포함 5일 에측값
pred: 상품 예측
pred_ugly: 못난이 예측

배포 할 때, orig_cost.xlsx도 같이 올려줘야함..

'''