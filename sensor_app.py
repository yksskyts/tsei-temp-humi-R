import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time

# 1. 페이지 설정
st.set_page_config(page_title="Stable Real-time Sensor", layout="wide")

# 2. 사이드바 설정
st.sidebar.header("🚀 시스템 모드")
app_mode = st.sidebar.radio("작업 선택", ["📊 데이터 분석 & 열화 진단", "📡 실시간 로깅 시뮬레이터"])

st.sidebar.divider()
model_choice = st.sidebar.selectbox(
    "적용할 모델",
    ["1. Linear Regression", "2. Ridge Regression", "3. Decision Tree", "4. Random Forest", "5. Gradient Boosting"]
)

@st.cache_resource
def get_trained_model(file, model_name):
    df = pd.read_csv(file)
    df.columns = [col.strip() for col in df.columns]
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
        df['Elapsed_Days'] = (df['측정 시간'] - df['측정 시간'].min()).dt.total_seconds() / (24 * 3600)
    else:
        df['Elapsed_Days'] = np.arange(len(df)) / 1440
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    X = df[['온도', '습도', 'Elapsed_Days']]
    y = df['Resistance_kOhm']
    
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    
    if "1." in model_name: model = LinearRegression()
    elif "2." in model_name: model = Ridge(alpha=1.0)
    elif "3." in model_name: model = DecisionTreeRegressor(max_depth=10)
    elif "4." in model_name: model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    else: model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    
    model.fit(X, y)
    return df, model, X, y

uploaded_file = st.sidebar.file_uploader("CSV 업로드", type="csv")

if uploaded_file is not None:
    df, model, X, y = get_trained_model(uploaded_file, model_choice)

    if app_mode == "📊 데이터 분석 & 열화 진단":
        # (기존 분석 코드와 동일하여 생략, 속도를 위해 실시간 모드에 집중)
        st.header("📊 정밀 분석 모드")
        st.info("실시간 시뮬레이션을 원하시면 사이드바에서 모드를 변경하세요.")
        
    else:
        st.header("📡 실시간 데이터 시뮬레이션 (시점 고정 기능 적용)")
        
        if 'sim_df' not in st.session_state:
            st.session_state.sim_df = pd.DataFrame(columns=['Time', 'Resistance', 'Temp', 'Humi'])
            st.session_state.current_day = df['Elapsed_Days'].max()

        c_ctrl, c_view = st.columns([1, 3])
        
        with c_ctrl:
            # 일시정지 기능을 위해 명칭 변경 및 상태 활용
            run_sim = st.checkbox("▶️ 시뮬레이션 활성화", value=False)
            st.write("---")
            curr_t = st.slider("현재 온도", 10.0, 50.0, float(df['온도'].mean()))
            curr_h = st.slider("현재 습도", 10.0, 95.0, float(df['습도'].mean()))
            if st.button("🧹 기록 초기화"):
                st.session_state.sim_df = pd.DataFrame(columns=['Time', 'Resistance', 'Temp', 'Humi'])
                st.rerun()

        with c_view:
            if run_sim:
                st.session_state.current_day += (1 / 86400)
                p_res = model.predict([[curr_t, curr_h, st.session_state.current_day]])[0]
                new_pt = pd.DataFrame({'Time':[datetime.now()], 'Resistance':[p_res], 'Temp':[curr_t], 'Humi':[curr_h]})
                st.session_state.sim_df = pd.concat([st.session_state.sim_df, new_pt], ignore_index=True).tail(200)

            # --- Plotly 시점 유지 설정 ---
            fig_sim = go.Figure()
            if not st.session_state.sim_df.empty:
                fig_sim.add_trace(go.Scatter(
                    x=st.session_state.sim_df['Time'], 
                    y=st.session_state.sim_df['Resistance'], 
                    mode='lines+markers', 
                    line=dict(color='#00FF00', width=2),
                    name='Resistance'
                ))
            
            fig_sim.update_layout(
                template="plotly_dark", 
                height=550,
                # [핵심] uirevision을 True(또는 특정 값)로 설정하면 줌/팬 상태가 유지됩니다.
                uirevision=True, 
                xaxis=dict(title="Time"),
                yaxis=dict(title="Resistance (kOhm)"),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            
            # config 설정으로 그래프 툴바를 항상 표시
            st.plotly_chart(fig_sim, use_container_width=True, config={'displayModeBar': True})

        # 자동 리프레시
        if run_sim:
            time.sleep(1)
            st.rerun()
else:
    st.info("CSV 파일을 먼저 업로드해 주세요.")