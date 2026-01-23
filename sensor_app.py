import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from datetime import datetime
import time

# 1. 페이지 설정
st.set_page_config(page_title="Sensor Master Expert", layout="wide")

# 2. 사이드바 제어판
st.sidebar.header("🚀 시스템 모드")
app_mode = st.sidebar.radio("작업 선택", [
    "📊 데이터 분석 & 열화 진단", 
    "🧪 물리량 수식 도출 (Polynomial)", # 새 기능 추가
    "📡 실시간 로깅 시뮬레이터"
])

st.sidebar.divider()

# 3. 데이터 로드 및 전처리
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [col.strip() for col in df.columns]
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
        df['Elapsed_Days'] = (df['측정 시간'] - df['측정 시간'].min()).dt.total_seconds() / 86400
    else:
        df['Elapsed_Days'] = np.arange(len(df)) / 1440
    # 저항 단위를 kOhm으로 변환 (기본 '저항' 컬럼 기준)
    if '저항' in df.columns:
        df['Resistance_kOhm'] = df['저항'] / 1000.0
    return df

uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    
    # ---------------------------------------------------------
    # 신규 기능: 🧪 물리량 수식 도출 모드
    # ---------------------------------------------------------
    if app_mode == "🧪 물리량 수식 도출 (Polynomial)":
        st.header("🧪 절대온도(K) 기반 물리 특성 수식 도출")
        st.markdown("특정 변수(저항 또는 농도)와 절대온도 사이의 최적 상관관계식을 산출합니다.")
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            x_var = st.selectbox("독립 변수 (X축) 선택", ["온도"])
            st.caption("※ 선택한 온도는 자동으로 절대온도(Kelvin)로 변환됩니다.")
        with col_sel2:
            y_var = st.selectbox("종속 변수 (Y축) 선택", df.columns.tolist(), index=df.columns.get_loc('Resistance_kOhm') if 'Resistance_kOhm' in df.columns else 0)

        # 데이터 준비
        K = (df[x_var] + 273.15).values.reshape(-1, 1) # 절대온도 변환
        Y = df[y_var].values
        
        st.divider()
        
        # 다항식 차수별 분석
        cols = st.columns(3)
        for i, deg in enumerate([1, 2, 3]):
            with cols[i]:
                poly = PolynomialFeatures(degree=deg)
                K_poly = poly.fit_transform(K)
                model = LinearRegression().fit(K_poly, Y)
                y_fit = model.predict(K_poly)
                r2 = r2_score(Y, y_fit)
                
                st.subheader(f"{deg}차 모델 (Degree {deg})")
                st.metric(f"{deg}차 결정계수 (R²)", f"{r2:.4f}")
                
                # 수식 문자열 생성 (LaTeX 형식)
                coeffs = model.coef_
                intercept = model.intercept_
                if deg == 1:
                    formula = f"y = {coeffs[1]:.4f}K + {intercept:.2f}"
                elif deg == 2:
                    formula = f"y = {coeffs[2]:.6f}K^2 + {coeffs[1]:.4f}K + {intercept:.2f}"
                else:
                    formula = f"y = {coeffs[3]:.8f}K^3 + {coeffs[2]:.6f}K^2 + {coeffs[1]:.4f}K + {intercept:.2f}"
                
                st.latex(formula)

        # 비교 그래프
        st.divider()
        plt.rcdefaults()
        sns.set_theme(style="whitegrid")
        fig_poly, ax_poly = plt.subplots(figsize=(10, 5))
        ax_poly.scatter(K, Y, alpha=0.1, color='gray', s=1, label='Raw Data')
        
        # 정렬된 선으로 그리기 위해 데이터 정렬
        sort_idx = np.argsort(K.flatten())
        K_sorted = K[sort_idx]
        
        for deg, color in zip([1, 2, 3], ['red', 'blue', 'green']):
            p = np.poly1d(np.polyfit(K.flatten(), Y, deg))
            ax_poly.plot(K_sorted, p(K_sorted), label=f'Degree {deg} Fit', color=color, lw=2)
            
        ax_poly.set_xlabel("Absolute Temperature (K)")
        ax_poly.set_ylabel(y_var)
        ax_poly.set_title(f"{y_var} vs Absolute Temperature (K)")
        ax_poly.legend()
        st.pyplot(fig_poly)

    # ---------------------------------------------------------
    # 기존 기능: 📊 데이터 분석 및 📡 시뮬레이터 (구조 유지)
    # ---------------------------------------------------------
    elif app_mode == "📊 데이터 분석 & 열화 진단":
        st.header("📊 센서 정밀 분석 및 열화 리포트")
        # ... (이전 분석 코드와 동일하게 유지) ...
        st.info("기존에 완성한 분석 리포트가 여기에 표시됩니다.")

    elif app_mode == "📡 실시간 로깅 시뮬레이터":
        st.header("📡 실시간 데이터 로깅 시뮬레이션")
        # ... (이전 시뮬레이터 코드와 동일하게 유지) ...
        st.info("1초마다 찍히는 실시간 시뮬레이터가 여기에 표시됩니다.")

else:
    st.info("👋 분석할 CSV 파일을 먼저 업로드해 주세요.")