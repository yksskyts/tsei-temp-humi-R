import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 1. 페이지 설정
st.set_page_config(page_title="Sensor Noise & Baseline Fixer", layout="wide")
st.title("🌊 펌프 맥동 및 수분 오차 자동 보정 시스템")
st.markdown("""
이 도구는 **S-Air(기준)**와 **Lab-Air+Pump(측정)** 데이터를 비교하여 
펌프에 의한 저항 출렁임을 제거하고, 수분으로 인한 베이스라인 상승을 보정합니다.
""")

# 2. 사이드바 - 파일 업로드 및 설정
st.sidebar.header("📁 데이터 업로드")
file_s = st.sidebar.file_uploader("S-Air (펌프 미작동/기준) 파일 업로드", type="csv")
file_l = st.sidebar.file_uploader("Lab-Air+Pump (펌프 작동/측정) 파일 업로드", type="csv")

st.sidebar.divider()
st.sidebar.header("⚙️ 보정 필터 설정")
# Savitzky-Golay 필터 파라미터 조절
window_size = st.sidebar.slider("필터 윈도우 크기 (홀수)", 3, 51, 11, step=2, help="값이 클수록 그래프가 더 매끄러워집니다.")
poly_order = st.sidebar.slider("다항식 차수", 1, 5, 2, help="보통 2차 또는 3차를 사용합니다.")

if file_s and file_l:
    # 데이터 읽기
    df_s = pd.read_csv(file_s)
    df_l = pd.read_csv(file_l)
    
    # 컬럼 정리
    df_s.columns = [c.strip() for c in df_s.columns]
    df_l.columns = [c.strip() for c in df_l.columns]

    # 3. 보정 연산
    # [Step 1] 펌프 맥동 제거 (Savitzky-Golay Filter)
    df_l['저항_Smoothed'] = savgol_filter(df_l['저항'], window_size, poly_order)
    
    # [Step 2] 베이스라인 시프트(수분 오차) 계산
    mean_s = df_s['저항'].mean()
    mean_l_smooth = df_l['저항_Smoothed'].mean()
    moisture_offset = mean_l_smooth - mean_s
    
    # [Step 3] 최종 보정 신호 생성
    df_l['저항_Final'] = df_l['저항_Smoothed'] - moisture_offset

    # 4. 결과 리포트 (Metric)
    st.header("📊 보정 분석 리포트")
    m1, m2, m3 = st.columns(3)
    m1.metric("S-Air 기준 저항", f"{mean_s/1000:.2f} kΩ")
    m2.metric("수분 오차 (Offset)", f"{moisture_offset/1000:+.2f} kΩ", help="Lab-Air의 수분으로 인해 상승한 저항값입니다.")
    m3.metric("맥동 노이즈 강도", f"±{np.std(df_l['저항'] - df_l['저항_Smoothed']):.2f} Ω")

    # 5. 그래프 시각화
    st.divider()
    st.header("📈 보정 시각화 비교")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 원본 노이즈 데이터
    ax.plot(df_l['저항'], color='lightgray', alpha=0.4, label='Raw (Pump Noise)')
    # 매끄럽게 처리된 데이터
    ax.plot(df_l['저항_Smoothed'], color='orange', lw=1.5, label='Pulsation Removed')
    # 수분 보정까지 완료된 데이터
    ax.plot(df_l['저항_Final'], color='green', lw=2, label='Fully Compensated (Target)')
    # 기준선
    ax.axhline(mean_s, color='blue', linestyle='--', label='S-Air Reference')
    
    ax.set_ylabel("Resistance (Ohm)")
    ax.set_xlabel("Data Points")
    ax.legend()
    st.pyplot(fig)

    # 6. 데이터 다운로드
    st.divider()
    st.subheader("📥 보정 결과 데이터 내보내기")
    csv = df_l.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="보정된 데이터 다운로드 (CSV)",
        data=csv,
        file_name="compensated_sensor_data.csv",
        mime="text/csv",
    )
    
    # 보정 전후 비교 상세 표
    st.expander("데이터 상세 보기").write(df_l[['측정 시간', '저항', '저항_Smoothed', '저항_Final']])

else:
    st.info("💡 사이드바에 S-Air 파일과 Lab-Air+Pump 파일을 모두 업로드해 주세요.")