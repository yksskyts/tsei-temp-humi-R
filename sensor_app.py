import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 및 모델 정의
a = 494.4331
b = 0.5238

def predict_ppm(del_r):
    """저항 변화값으로 농도(ppm)를 역산"""
    if del_r <= 0: return 0
    return (del_r / a) ** (1 / b)

# 2. Streamlit UI 구성
st.set_page_config(page_title="Toluene 농도 분석기", layout="wide")
st.title("🧪 톨루엔 농도 예측 및 데이터 분석 도구")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("🔍 농도 예측 (Inference)")
    input_dr = st.number_input("저항 변화량(delR, Ohm)을 입력하세요:", min_value=0.0, value=10000.0)
    
    predicted_ppm = predict_ppm(input_dr)
    
    st.metric(label="예측된 톨루엔 농도", value=f"{predicted_ppm:.2f} ppm")
    st.info(f"적용 수식: ppm = (ΔR / {a:.2f})^(1 / {b:.4f})")

with col2:
    st.header("📈 센서 응답 곡선")
    ppm_range = np.linspace(10, 5000, 500)
    dr_range = a * (ppm_range ** b)
    
    fig, ax = plt.subplots()
    ax.plot(ppm_range, dr_range, label="Power Model Fit", color='blue')
    ax.scatter([10, 30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000], 
               [3097, 3234, 3334, 3235, 6721, 7686, 8780, 10027, 11716, 37428, 108259], 
               color='red', label='Actual Data')
    ax.set_xlabel("Concentration (ppm)")
    ax.set_ylabel("Resistance Change (Ohm)")
    ax.legend()
    st.pyplot(fig)

st.divider()
st.subheader("📋 입력 데이터 참조")
data = {
    "ppm": [10, 30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000],
    "delR": [3097, 3234, 3334, 3235, 6721, 7686, 8780, 10027, 11716, 37428, 108259]
}
st.dataframe(pd.DataFrame(data).T)