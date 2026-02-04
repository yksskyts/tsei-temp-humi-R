import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 1. 모델 계수 설정
# [1차] y = ax + b
p1_1, p0_1 = 19.003, -83.379
# [2차] y = ax^2 + bx + c
p2_2, p1_2, p0_2 = 0.004106, -0.1064, 5291.92
# [3차] y = ax^3 + bx^2 + cx + d
p3_3, p2_3, p1_3, p0_3 = 9.438e-07, -0.0027, 10.86, 3505.81

def predict_ppm(target_dr, degree):
    if target_dr <= 0: return 0.0
    if degree == 1:
        return (target_dr - p0_1) / p1_1
    elif degree == 2:
        a, b, c = p2_2, p1_2, p0_2 - target_dr
        return (-b + np.sqrt(max(0, b**2 - 4*a*c))) / (2*a)
    elif degree == 3:
        func = lambda x: p3_3*x**3 + p2_3*x**2 + p1_3*x + p0_3 - target_dr
        return float(fsolve(func, x0=1000)[0])

# --- Streamlit UI ---
st.set_page_config(page_title="정밀 수식 분석기", layout="wide")
st.title("🧪 모델별 정밀 수식 및 ppm 역산 도구")

degree = st.sidebar.selectbox("분석 모델 선택", [1, 2, 3], index=2)
input_dr = st.sidebar.number_input("저항 변화량(ΔR) 입력", value=15000.0)

col1, col2 = st.columns([1, 1])

with col1:
    st.header(f"📝 {degree}차 모델 수식 안내")
    
    if degree == 1:
        st.subheader("1차 선형 모델 (Linear)")
        st.latex(rf"y = {p1_1:.3f}x + ({p0_1:.3f})")
        st.info("**농도(x) 역산 수식:**")
        st.latex(rf"x = \frac{{y - ({p0_1:.3f})}}{{{p1_1:.3f}}}")
        
    elif degree == 2:
        st.subheader("2차 다항식 모델 (Quadratic)")
        st.latex(rf"y = {p2_2:.6f}x^2 + ({p1_2:.4f})x + {p0_2:.2f}")
        st.info("**농도(x) 역산 수식 (근의 공식):**")
        st.latex(r"x = \frac{-b + \sqrt{b^2 - 4a(c-y)}}{2a}")
        
    elif degree == 3:
        st.subheader("3차 다항식 모델 (Cubic)")
        st.latex(rf"y = {p3_3:.4e}x^3 + ({p2_3:.4f})x^2 + {p1_3:.2f}x + {p0_3:.2f}")
        st.warning("⚠️ 3차식은 직접 해를 구하기 복합하여 수치 해석(Newton-Raphson)을 통해 ppm을 산출합니다.")

with col2:
    st.header("🎯 예측 결과")
    res_ppm = predict_ppm(input_dr, degree)
    st.metric(label=f"{degree}차 모델 기준 예측 농도", value=f"{res_ppm:.2f} ppm")
    
    # 그래프 시각화
    x_range = np.linspace(0, 5000, 500)
    if degree == 1: y_plot = p1_1*x_range + p0_1
    elif degree == 2: y_plot = p2_2*x_range**2 + p1_2*x_range + p0_2
    else: y_plot = p3_3*x_range**3 + p2_3*x_range**2 + p1_3*x_range + p0_3
    
    fig, ax = plt.subplots()
    ax.plot(x_range, y_plot, label=f"{degree}th Degree Fit", color='blue')
    ax.axhline(input_dr, color='red', linestyle='--', label='Input ΔR')
    ax.set_xlabel("PPM")
    ax.set_ylabel("Delta R")
    ax.legend()
    st.pyplot(fig)