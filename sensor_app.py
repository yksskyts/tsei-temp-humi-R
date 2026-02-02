import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 1. 페이지 설정
st.set_page_config(page_title="Toluene K-Value Analyzer", layout="wide")
st.title("🎯 톨루엔 센서 정밀 반응 분석 시스템")
st.markdown("펌프 맥동(Fluctuation)을 제거하고 톨루엔 주입에 따른 순수 저항 변화율($k$)을 산출합니다.")

# 2. 데이터 업로드
st.sidebar.header("📁 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("분석할 센서 데이터(CSV) 업로드", type="csv")

# 3. 분석 파라미터 설정
st.sidebar.divider()
st.sidebar.header("⚙️ 분석 설정")
concentration = st.sidebar.number_input("톨루엔 농도 (ppm)", value=20.0, step=0.1)
k_factor = st.sidebar.number_input("K-Value 정규화 계수", value=20000.0, step=100.0)
window_size = st.sidebar.slider("필터 강도 (윈도우 크기)", 5, 51, 15, step=2)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    
    # [Step 1] 노이즈 및 맥동 제거
    df['저항_Clean'] = savgol_filter(df['저항'], window_size, 2)
    
    # [Step 2] 구간 선택 (사용자가 슬라이더로 베이스라인과 피크 지점 선택)
    st.subheader("📍 분석 구간 설정")
    data_len = len(df)
    base_range = st.slider("1. Baseline($R_0$) 측정 구간", 0, data_len, (0, int(data_len*0.2)))
    gas_range = st.slider("2. Gas Response($R_{gas}$) 측정 구간", 0, data_len, (int(data_len*0.7), data_len))
    
    # [Step 3] K-Value 계산
    r0 = df['저항_Clean'].iloc[base_range[0]:base_range[1]].mean()
    r_gas = df['저항_Clean'].iloc[gas_range[0]:gas_range[1]].max()
    
    delta_r_r0 = (r_gas - r0) / r0
    k_value = delta_r_r0 / k_factor
    
    # 4. 결과 출력 (Metrics)
    st.divider()
    st.header("📊 분석 결과 (K-Value)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline ($R_0$)", f"{r0/1000:.2f} kΩ")
    c2.metric("Gas Peak ($R_{gas}$)", f"{r_gas/1000:.2f} kΩ")
    c3.metric("반응도 ($\Delta R/R_0$)", f"{delta_r_r0:.4f}")
    c4.metric("최종 K-Value", f"{k_value:.6e}")

    # 5. 시각화 (정제된 신호 및 분석 구간 표시)
    st.divider()
    st.header("📈 신호 트래킹 및 보정 결과")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 원본 데이터와 정제된 데이터
    ax.plot(df.index, df['저항'], color='lightgray', alpha=0.4, label='Raw Signal (with Fluctuation)')
    ax.plot(df.index, df['저항_Clean'], color='blue', lw=2, label='Cleaned Signal (Pulsation Removed)')
    
    # 분석 구간 강조
    ax.axvspan(base_range[0], base_range[1], color='gray', alpha=0.2, label='Baseline Period')
    ax.axvspan(gas_range[0], gas_range[1], color='red', alpha=0.1, label='Gas Response Period')
    
    # R0, R_gas 선 표시
    ax.axhline(r0, color='black', linestyle='--', alpha=0.7)
    ax.axhline(r_gas, color='red', linestyle='--', alpha=0.7)
    
    ax.set_ylabel("Resistance (Ohm)")
    ax.set_xlabel("Time Step (Index)")
    ax.legend(loc='upper left')
    ax.set_title(f"Toluene Response Analysis (Conc: {concentration}ppm)")
    
    st.pyplot(fig)

    # 6. 리포트 저장용 텍스트
    st.info(f"💡 **전문가 메모:** 현재 노이즈 필터링을 통해 펌프 맥동을 제거한 상태에서 $R_0$ 대비 약 {delta_r_r0*100:.2f}%의 저항 변화가 감지되었습니다.")

else:
    st.info("👋 분석할 톨루엔 반응 CSV 데이터를 업로드해 주세요.")