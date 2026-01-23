import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="Real-time Sensor Simulator", layout="wide")

st.title("📡 실시간 센서 인터랙티브 시뮬레이터")
st.markdown("온습도 조절 시 모델이 예측한 저항값이 **실시간 데이터 로깅**처럼 그래프에 추가됩니다.")

# 2. 초기 데이터 및 모델 학습 (파일 업로드 시)
uploaded_file = st.file_uploader("먼저 학습 데이터(CSV)를 업로드하세요", type="csv")

if uploaded_file is not None:
    # 데이터 로드 및 모델 학습 (백그라운드)
    df_raw = pd.read_csv(uploaded_file)
    df_raw.columns = [col.strip() for col in df_raw.columns]
    
    # 모델 학습용 데이터 준비
    if '측정 시간' in df_raw.columns:
        df_raw['측정 시간'] = pd.to_datetime(df_raw['측정 시간'])
        df_raw['Elapsed_Days'] = (df_raw['측정 시간'] - df_raw['측정 시간'].min()).dt.total_seconds() / (24 * 3600)
    else:
        df_raw['Elapsed_Days'] = np.arange(len(df_raw)) / 1440
        
    X = df_raw[['온도', '습도', 'Elapsed_Days']]
    y = df_raw['저항'] / 1000.0
    model = LinearRegression().fit(X, y)
    
    # --- 시뮬레이션 메모리(Session State) 초기화 ---
    if 'sim_data' not in st.session_state:
        # 처음 시작은 원본 데이터의 마지막 50개 포인트로 시작
        last_50 = df_raw.tail(50).copy()
        st.session_state.sim_data = pd.DataFrame({
            'Time': last_50['측정 시간'] if '측정 시간' in last_50.columns else [datetime.now() + timedelta(minutes=i) for i in range(50)],
            'Resistance': last_50['저항'] / 1000.0,
            'Temp': last_50['온도'],
            'Humi': last_50['습도']
        })
        st.session_state.last_day = df_raw['Elapsed_Days'].max()

    # 3. 사이드바 - 실시간 조절 컨트롤러
    st.sidebar.header("🕹️ 실시간 환경 조절")
    curr_temp = st.sidebar.slider("현재 온도 (°C)", 10.0, 50.0, float(df_raw['온도'].mean()), 0.1)
    curr_humi = st.sidebar.slider("현재 습도 (%)", 10.0, 90.0, float(df_raw['습도'].mean()), 0.1)
    
    st.sidebar.divider()
    if st.sidebar.button("🧹 데이터 초기화"):
        st.session_state.sim_data = st.session_state.sim_data.tail(1)
        st.rerun()

    # --- 실시간 포인트 생성 로직 ---
    # 버튼을 누르거나 슬라이더가 변할 때마다 새 포인트 추가
    new_time = st.session_state.sim_data['Time'].iloc[-1] + timedelta(minutes=1)
    st.session_state.last_day += (1 / 1440) # 1분 추가
    
    # 모델로 예측
    new_res = model.predict([[curr_temp, curr_humi, st.session_state.last_day]])[0]
    
    # 새로운 데이터 행 생성
    new_row = pd.DataFrame({
        'Time': [new_time], 
        'Resistance': [new_res],
        'Temp': [curr_temp],
        'Humi': [curr_humi]
    })
    
    # 데이터셋에 추가 (최근 200개만 유지하여 속도 최적화)
    st.session_state.sim_data = pd.concat([st.session_state.sim_data, new_row], ignore_index=True).tail(200)

    # 4. 메인 화면 - 실시간 그래프
    col_chart, col_stat = st.columns([3, 1])
    
    with col_chart:
        # Plotly를 사용한 다이나믹 그래프
        fig = go.Figure()
        
        # 저항 그래프
        fig.add_trace(go.Scatter(
            x=st.session_state.sim_data['Time'], 
            y=st.session_state.sim_state['Resistance'],
            mode='lines+markers',
            name='Resistance (kΩ)',
            line=dict(color='#00FF00', width=3),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title="Real-time Sensor Resistance Monitoring",
            xaxis_title="Time",
            yaxis_title="Resistance (kOhm)",
            template="plotly_dark", # 다크모드로 전문가 포스 강조
            height=500,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stat:
        st.subheader("📊 실시간 상태")
        st.metric("현재 온도", f"{curr_temp} °C")
        st.metric("현재 습도", f"{curr_humi} %")
        st.metric("예측 저항", f"{new_res:.4f} kΩ")
        st.info("슬라이더를 움직이면 그래프 우측에 즉시 반영됩니다.")

    # 5. 하단 보조 그래프 (온습도 변화 추이)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Temperature Trend")
        st.line_chart(st.session_state.sim_data.set_index('Time')['Temp'], height=150)
    with c2:
        st.caption("Humidity Trend")
        st.line_chart(st.session_state.sim_data.set_index('Time')['Humi'], height=150)

    # 실시간 느낌을 위한 자동 리프레시 버튼 (선택 사항)
    if st.button("▶️ 데이터 계속 쌓기"):
        st.rerun()

else:
    st.info("👋 먼저 모델을 학습시키기 위해 CSV 데이터를 업로드해 주세요.")