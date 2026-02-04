import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.metrics import r2_score

# 1. 원본 데이터 정의
ppm_data = np.array([10, 30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000])
dr_data = np.array([3097, 3234, 3334, 3235, 6721, 7686, 8780, 10027, 11716, 37428, 108259])

# 2. 모델 계수 (기 도출된 값 적용)
# [1차] y = p1*x + p0
p1_1, p0_1 = 19.003, -83.379
# [2차] y = p2*x^2 + p1*x + p0
p2_2, p1_2, p0_2 = 0.004106, -0.1064, 5291.92
# [3차] y = p3*x^3 + p2*x^2 + p1*x + p0
p3_3, p2_3, p1_3, p0_3 = 9.438e-07, -0.0027, 10.86, 3505.81

# 3. 예측 함수 정의
def predict_ppm(target_dr, degree):
    if target_dr <= 0: return 0.0
    if degree == 1:
        # 1차 역산: x = (y - p0) / p1
        return (target_dr - p0_1) / p1_1
    elif degree == 2:
        # 2차 역산 (근의 공식): ax^2 + bx + (c-y) = 0
        a, b, c = p2_2, p1_2, p0_2 - target_dr
        return (-b + np.sqrt(max(0, b**2 - 4*a*c))) / (2*a)
    elif degree == 3:
        # 3차 역산 (수치 해석)
        func = lambda x: p3_3*x**3 + p2_3*x**2 + p1_3*x + p0_3 - target_dr
        return float(fsolve(func, x0=1000)[0])

# 4. Streamlit UI
st.set_page_config(page_title="다항식 모델 비교 분석기", layout="wide")
st.title("📊 다항식 차수별 모델 비교 및 ppm 추측 도구")

# 상단: 모델 비교 지표
st.subheader("📌 모델 성능 비교 ($R^2$)")
cols = st.columns(3)
metrics = [("1차 (Linear)", 0.9312), ("2차 (Quadratic)", 0.9935), ("3차 (Cubic)", 0.9991)]
for i, (name, r2) in enumerate(metrics):
    cols[i].metric(name, f"R² = {r2:.4f}")

st.divider()

# 메인 분석 영역
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.header("🔍 농도 추측 실행")
    selected_degree = st.radio("사용할 모델 차수를 선택하세요:", [1, 2, 3], index=2, horizontal=True)
    input_dr = st.number_input("저항 변화량(ΔR, Ohm) 입력:", min_value=0.0, value=15000.0)
    
    res_ppm = predict_ppm(input_dr, selected_degree)
    
    st.success(f"### 예측 농도: {res_ppm:.2f} ppm")
    st.info(f"선택 모델: {selected_degree}차 다항식")

with col_right:
    st.header("📈 전구간 피팅 시각화")
    x_range = np.linspace(0, 5500, 1000)
    y1 = p1_1*x_range + p0_1
    y2 = p2_2*x_range**2 + p1_2*x_range + p0_2
    y3 = p3_3*x_range**3 + p2_3*x_range**2 + p1_3*x_range + p0_3
    
    fig, ax = plt.subplots()
    ax.scatter(ppm_data, dr_data, color='black', label='Actual Data', zorder=5)
    ax.plot(x_range, y1, '--', label='1st Degree', alpha=0.7)
    ax.plot(x_range, y2, '--', label='2nd Degree', alpha=0.7)
    ax.plot(x_range, y3, '-', label='3rd Degree', linewidth=2, color='red')
    ax.set_xlabel("Concentration (ppm)")
    ax.set_ylabel("Resistance Change (Ohm)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)