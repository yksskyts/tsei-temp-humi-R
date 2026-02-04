import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 1. 원본 데이터 (실제 값 찍어주기용)
ppm_data = np.array([10, 30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000])
dr_data = np.array([3097, 3234, 3334, 3235, 6721, 7686, 8780, 10027, 11716, 37428, 108259])

# 2. 모델 계수 설정
p1_1, p0_1 = 19.003, -83.379
p2_2, p1_2, p0_2 = 0.004106, -0.1064, 5291.92
p3_3, p2_3, p1_3, p0_3 = 9.438e-07, -0.0027, 10.86, 3505.81

def predict_ppm(target_dr, degree):
    if target_dr <= 0: return 0.0
    if degree == 1:
        return (target_dr - p0_1) / p1_1
    elif degree == 2:
        a, b, c = p2_2, p1_2, p0_2 - target_dr
        discriminant = b**2 - 4*a*c
        return (-b + np.sqrt(max(0, discriminant))) / (2*a)
    elif degree == 3:
        func = lambda x: p3_3*x**3 + p2_3*x**2 + p1_3*x + p0_3 - target_dr
        return float(fsolve(func, x0=1000)[0])

# --- Streamlit UI ---
st.set_page_config(page_title="실측값 기반 정밀 분석기", layout="wide")
st.title("🧪 실측 데이터 기반 모델 수식 및 ppm 역산")

degree = st.sidebar.selectbox("차수 선택", [1, 2, 3], index=2)
input_dr = st.sidebar.number_input("저항 변화량(ΔR) 입력", value=15000.0)

col1, col2 = st.columns([1, 1.2])

with col1:
    st.header(f"📝 {degree}차 모델 및 수식")
    if degree == 1:
        st.latex(rf"y = {p1_1:.3f}x + ({p0_1:.3f})")
        st.info("**농도(x) 역산 식:**")
        st.latex(rf"x = \frac{{y - ({p0_1:.3f})}}{{{p1_1:.3f}}}")
    elif degree == 2:
        st.latex(rf"y = {p2_2:.6f}x^2 + ({p1_2:.4f})x + {p0_2:.2f}")
        st.info("**농도(x) 역산 식 (근의 공식):**")
        st.latex(r"x = \frac{-b + \sqrt{b^2 - 4a(c-y)}}{2a}")
    elif degree == 3:
        st.latex(rf"y = {p3_3:.4e}x^3 + ({p2_3:.4f})x^2 + {p1_3:.2f}x + {p0_3:.2f}")
    
    st.divider()
    res_ppm = predict_ppm(input_dr, degree)
    st.metric(label="예측 농도", value=f"{res_ppm:.2f} ppm")

with col2:
    st.header("📈 피팅 곡선 및 실측값 (Actual Data)")
    x_range = np.linspace(0, 5200, 1000)
    if degree == 1: y_fit = p1_1*x_range + p0_1
    elif degree == 2: y_fit = p2_2*x_range**2 + p1_2*x_range + p0_2
    else: y_fit = p3_3*x_range**3 + p2_3*x_range**2 + p1_3*x_range + p0_3
    
    fig, ax = plt.subplots()
    # 실제 값 찍기 (가장 중요한 부분 복구)
    ax.scatter(ppm_data, dr_data, color='red', label='Actual Data (Measured)', zorder=5)
    # 피팅 라인
    ax.plot(x_range, y_fit, label=f'{degree}th Degree Fit', color='blue', alpha=0.7)
    # 입력값 위치 표시
    ax.axhline(input_dr, color='green', linestyle='--', label=f'Input ΔR: {input_dr}')
    
    ax.set_xlabel("Concentration (ppm)")
    ax.set_ylabel("Resistance Change (Ohm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)