import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 저장된 AI 모델 및 정보 불러오기
@st.cache_resource
def load_models():
    model = joblib.load('odor_ai_model.pkl')
    le = joblib.load('label_encoder.pkl')
    features = joblib.load('feature_names.pkl')
    return model, le, features

model, le, features = load_models()

st.title("🧪 1만개 데이터 기반 악취 예측 시스템")
st.sidebar.header("🔬 성분별 농도 입력 (ppm)")

# 학습했던 성분명들을 슬라이더로 자동 생성
input_data = {}
for f in features:
    input_data[f] = st.sidebar.number_input(f"{f}", min_value=0.0, value=0.0, format="%.4f")

if st.button("AI 냄새 분석 결과보기"):
    # 입력 데이터를 모델용 데이터프레임으로 변환
    input_df = pd.DataFrame([input_data])
    
    # AI 예측 수행
    prediction_idx = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # 결과 출력
    predicted_label = le.inverse_transform([prediction_idx])[0]
    
    st.subheader("分析 결과")
    st.success(f"예측된 냄새 종류: **{predicted_label}**")
    
    # 확률 분포 시각화
    proba_df = pd.DataFrame({
        '냄새종류': le.classes_,
        '확률': prediction_proba
    }).sort_values(by='확률', ascending=False)
    
    st.bar_chart(proba_df.set_index('냄새종류'))