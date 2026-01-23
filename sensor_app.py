import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import time

# 1. 페이지 설정
st.set_page_config(page_title="Live Sensor Simulator", layout="wide")

st.title("📡 실시간 센서 데이터 로깅 시뮬레이터")
st.markdown("온습도를 조절하면 **1초마다** 모델이 예측한 저항값이 그래프에 실시간으로 기록됩니다.")

# 2. 데이터 로드 및 모델 학습
uploaded_file = st.file_uploader("학습용 CSV 파일을 업로드하세요", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def train_model(file):
        df = pd.read_csv(file)
        df.columns = [col.strip() for col in df.columns]
        if '측정 시간' in df.columns:
            df['측정 시간'] = pd.to_datetime(df['측정 시간'])
            df['Elapsed_Days'] = (df['측정 시간'] - df['측정 시간'].min()).dt.total_seconds() / (24 * 3600)
        else:
            df['Elapsed_Days'] = np.arange(len(df)) / 1440
        X = df[['온도', '습도', 'Elapsed_Days']]
        y = df['저항'] / 1000.0
        model = LinearRegression().fit(X, y)
        return model, df['Elapsed_Days'].max(), df['온도'].mean(), df['습도'].mean(), y.min(), y.max()

    model, last_day_init, avg_temp, avg_humi, y_min, y_max = train_model(uploaded_file)

    # --- 시뮬레이션 상태 관리 (오타 수정됨) ---
    if 'sim_df' not in st.session_state:
        st.session_state.sim_df = pd.DataFrame(columns=['Time', 'Resistance', 'Temp', 'Humi'])
        st.session_state.current_day = last_day_init

    # 3. 사이드바 컨트롤러
    st.sidebar.header("🕹️ 실시간 환경 제어")
    run_sim = st.sidebar.checkbox("▶️ 시뮬레이션 시작", value=False)
    
    st.sidebar.divider()
    curr_temp = st.sidebar.slider("현재 온도 (°C)", 10.0, 50.0, float(avg_temp), 0.1)
    curr_humi = st.sidebar.slider("현재 습도 (%)", 10.0, 95.0, float(avg_humi), 0.1)
    
    if st.sidebar.button("🧹 데이터 초기화"):
        st.session_state.sim_df = pd.DataFrame(columns=['Time', 'Resistance', 'Temp', 'Humi'])
        st.rerun()

    # 4. 실시간 데이터 생성 로직
    if run_sim:
        new_time = datetime.now()
        st.session_state.current_day += (1 / (24 * 3600)) # 1초 추가
        
        # 모델 예측
        pred_res = model.predict([[curr_temp, curr_humi, st.session_state.current_day]])[0]
        
        # 새로운 행 추가
        new_data = pd.DataFrame({
            'Time': [new_time], 
            'Resistance': [pred_res],
            'Temp': [curr_temp],
            'Humi': [curr_humi]
        })
        # 최근 100개 데이터 유지
        st.session_state.sim_df = pd.concat([st.session_state.sim_df, new_data], ignore_index=True).tail(100)

    # 5. 메인 화면 시각화
    col_chart, col_stat = st.columns([3, 1])
    
    with col_chart:
        fig = go.Figure()
        if not st.session_state.sim_df.empty:
            fig.add_trace(go.Scatter(
                x=st.session_state.sim_df['Time'], 
                y=st.session_state.sim_df['Resistance'], # sim_state -> sim_df 로 수정 완료
                mode='lines+markers',
                line=dict(color='#00FF00', width=2),
                marker=dict(size=6, color='#00FF00'),
                name='Predicted Resistance'
            ))
        
        fig.update_layout(
            title="Real-time Sensor Monitoring (Updating every 1s)",
            xaxis_title="System Time",
            yaxis_title="Resistance (kOhm)",
            template="plotly_dark",
            height=550,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(range=[y_min * 0.95, y_max * 1.05]) # 데이터 범위에 맞게 축 고정
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stat:
        st.subheader("📊 Live Status")
        st.metric("Current Temp", f"{curr_temp:.1f} °C")
        st.metric("Current Humi", f"{curr_humi:.1f} %")
        if not st.session_state.sim_df.empty:
            latest_res = st.session_state.sim_df['Resistance'].iloc[-1]
            st.metric("Predicted Res", f"{latest_res:.4f} kΩ")
        
        st.divider()
        st.info("💡 **Tip:** 슬라이더를 움직여보세요! 그래프가 즉시 반응합니다.")

    # --- 1초 대기 후 리프레시 ---
    if run_sim:
        time.sleep(1)
        st.rerun()

else:
    st.info("👋 학습용 센서 데이터(CSV)를 업로드하면 실시간 시뮬레이터가 활성화됩니다.")