import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import time

# 1. 페이지 설정
st.set_page_config(page_title="Sensor Master Pro", layout="wide")

# 2. 사이드바: 핵심 설정
st.sidebar.header("🚀 시스템 제어판")
app_mode = st.sidebar.radio("작업 모드 선택", ["📊 데이터 분석 & 열화 진단", "📡 실시간 로깅 시뮬레이터"])

st.sidebar.divider()
model_choice = st.sidebar.selectbox(
    "알고리즘 선택",
    ["1. Linear Regression", "2. Ridge Regression", "3. Decision Tree", "4. Random Forest", "5. Gradient Boosting"]
)

# 3. 데이터 로드 및 모델 학습 (캐싱으로 속도 극대화)
@st.cache_resource
def train_sensor_model(file, model_name):
    df = pd.read_csv(file)
    df.columns = [col.strip() for col in df.columns]
    
    # 시간 변수 생성
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
        df['Elapsed_Days'] = (df['측정 시간'] - df['측정 시간'].min()).dt.total_seconds() / 86400
    else:
        df['Elapsed_Days'] = np.arange(len(df)) / 1440
        
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    X = df[['온도', '습도', 'Elapsed_Days']]
    y = df['Resistance_kOhm']
    
    # 모델 할당
    if "1." in model_name: model = LinearRegression()
    elif "2." in model_name: model = Ridge(alpha=1.0)
    elif "3." in model_name: model = DecisionTreeRegressor(max_depth=10)
    elif "4." in model_name: model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    else: model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    
    model.fit(X, y)
    return df, model, X, y

uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file:
    df, model, X, y = train_sensor_model(uploaded_file, model_choice)

    # ==========================================================
    # 모드 1: 정밀 데이터 분석 (기존 전문가님 요청 사항 반영)
    # ==========================================================
    if app_mode == "📊 데이터 분석 & 열화 진단":
        st.header("📊 센서 정밀 분석 및 열화 리포트")
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            # 노화율 계산용 선형 회귀
            aging_model = LinearRegression().fit(X, y)
            deg_rate = aging_model.coef_[2]
            if deg_rate > 0:
                st.warning(f"⚠️ **열화 상태:** 저항 증가 중 (+{deg_rate:.4f} kΩ/day)")
            else:
                st.success(f"✅ **안정화 상태:** 저항 감소 중 ({deg_rate:.4f} kΩ/day)")
            
            if hasattr(model, 'coef_'):
                st.info(f"**수식:** $R = {model.intercept_:.2f} + ({model.coef_[0]:.4f} \cdot T) + ({model.coef_[1]:.4f} \cdot H) + ({model.coef_[2]:.4f} \cdot D)$")

        with c2:
            y_pred = model.predict(X)
            st.metric("모델 신뢰도 (R²)", f"{r2_score(y, y_pred):.4f}")
            st.metric("평균 오차 (RMSE)", f"{np.sqrt(mean_squared_error(y, y_pred)):.4f} kΩ")

        # 분석용 무거운 그래프
        st.divider()
        plt.rcdefaults()
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # [1] 온도 영향
        sns.regplot(ax=axes[0,0], x='온도', y='Resistance_kOhm', data=df, scatter_kws={'alpha':0.01}, line_kws={'color':'red'})
        axes[0,0].set_title("Temp Impact")
        
        # [2] 순수 시간 열화 (T/H 보정)
        drift = y - (aging_model.coef_[0]*df['온도'] + aging_model.coef_[1]*df['습도'] + aging_model.intercept_)
        axes[0,1].scatter(df['Elapsed_Days'], drift, alpha=0.05, s=1, color='orange')
        axes[0,1].set_title("Pure Aging Drift")
        
        # [3] 선형성 확인
        axes[1,0].scatter(y, y_pred, alpha=0.05, s=1, color='purple')
        axes[1,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        
        # [4] 시계열 트래킹
        axes[1,1].plot(df['측정 시간'].iloc[::50], y.iloc[::50], color='black', alpha=0.4, label='Actual')
        axes[1,1].plot(df['측정 시간'].iloc[::50], y_pred[::50], color='lime', linestyle='--', label='Pred')
        axes[1,1].legend()
        
        st.pyplot(fig)

    # ==========================================================
    # 모드 2: 실시간 로깅 시뮬레이터 (속도 및 시점 고정 최적화)
    # ==========================================================
    else:
        st.header("📡 실시간 데이터 로깅 시뮬레이션")
        st.caption("그래프를 마우스로 확대/이동해도 시점이 고정됩니다.")
        
        if 'sim_df' not in st.session_state:
            st.session_state.sim_df = pd.DataFrame(columns=['Time', 'Resistance', 'Temp', 'Humi'])
            st.session_state.current_day = df['Elapsed_Days'].max()

        ctrl, view = st.columns([1, 3])
        
        with ctrl:
            is_active = st.checkbox("▶️ 시뮬레이션 활성화", value=False)
            st.write("---")
            in_t = st.slider("현재 온도", 10.0, 50.0, float(df['온도'].mean()))
            in_h = st.slider("현재 습도", 10.0, 95.0, float(df['습도'].mean()))
            if st.button("🧹 데이터 초기화"):
                st.session_state.sim_df = pd.DataFrame(columns=['Time', 'Resistance', 'Temp', 'Humi'])
                st.rerun()

        with view:
            if is_active:
                st.session_state.current_day += (1 / 86400) # 1초씩 증가
                res_val = model.predict([[in_t, in_h, st.session_state.current_day]])[0]
                new_entry = pd.DataFrame({'Time':[datetime.now()], 'Resistance':[res_val], 'Temp':[in_t], 'Humi':[in_h]})
                st.session_state.sim_df = pd.concat([st.session_state.sim_df, new_entry], ignore_index=True).tail(150)

            # Plotly 시점 고정(uirevision) 설정
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
                uirevision='constant', # [핵심] 줌/팬 상태 강제 고정
                xaxis=dict(title="System Time", showgrid=True),
                yaxis=dict(title="Resistance (kOhm)", showgrid=True),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_sim, use_container_width=True, config={'displayModeBar': True})

        if is_active:
            time.sleep(1) # 1초 대기
            st.rerun() # 화면 갱신
else:
    st.info("👋 먼저 CSV 파일을 업로드하여 모델을 학습시켜주세요.")