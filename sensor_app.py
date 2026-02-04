import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 1. 모델 계수 정의 (R^2 = 0.9991)
# y = c3*x^3 + c2*x^2 + c1*x + intercept
c3 = 9.43816e-07
c2 = -0.00270067
c1 = 10.86389
intercept = 3505.8114

# 지수 모델 (초기값 추정용, R^2 = 0.9366)
a_exp = 4527.0943
b_exp = 0.000675

def poly3_func(x, target_y):
    """3차 다항식에서 ppm(x)을 찾기 위한 방정식"""
    return c3*x**3 + c2*x**2 + c1*x + intercept - target_y

def predict_ppm_precise(target_dr):
    """수치 해석을 통한 정밀 ppm 역산"""
    if target_dr <= intercept: return 0.0
    
    # 지수 모델로 대략적인 초기값(Guess) 계산
    initial_guess = np.log(target_dr / a_exp) / b_exp if target_dr > 0 else 0
    if initial_guess < 0: initial_guess = 10.0
    
    # fsolve를 이용해 정밀한 해(ppm) 도출
    solution = fsolve(poly3_func, x0=initial_guess, args=(target_dr))
    return max(0.0, float(solution[0]))

# 2. Streamlit UI 구성
st.set_page_config(page_title="Toluene 정밀 분석기", layout="wide")
st.title("🚀 톨루엔 농도 정밀 분석 도구 (3차 다항식 모델)")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("🎯 정밀 농도 예측")
    st.write("3차 다항식 모델 ($R^2=0.9991$)을 사용하여 ppm을 추측합니다.")
    
    input_dr = st.number_input("저항 변화량(ΔR, Ohm)을 입력하세요:", min_value=0.0, value=15000.0, step=100.0)
    
    precise_ppm = predict_ppm_precise(input_dr)
    
    st.metric(label="예측된 정밀 톨루엔 농도", value=f"{precise_ppm:.2f} ppm")
    st.success(f"모델 신뢰도: R² = 0.9991")

with col2:
    st.header("📉 최적 모델 피팅 곡선")
    ppm_range = np.linspace(10, 5000, 1000)
    dr_poly3 = c3*ppm_range**3 + c2*ppm_range**2 + c1*ppm_range + intercept
    
    fig, ax = plt.subplots()
    ax.plot(ppm_range, dr_poly3, label="3rd Poly Fit (Best R2)", color='green', linewidth=2)
    ax.scatter([10, 30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000], 
               [3097, 3234, 3334, 3235, 6721, 7686, 8780, 10027, 11716, 37428, 108259], 
               color='red', label='Measured Data', zorder=5)
    ax.set_xlabel("Concentration (ppm)")
    ax.set_ylabel("Resistance Change (ΔR, Ohm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.divider()
st.subheader("📋 분석 모델 수식 정보")
st.latex(rf"y = {c3:.4e}x^3 + ({c2:.4f})x^2 + {c1:.2f}x + {intercept:.2f}")