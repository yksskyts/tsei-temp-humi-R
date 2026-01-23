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
st.set_page_config(page_title="Sensor Expert Fast", layout="wide")

# 2. 사이드바 - 모드 전환 (이것이 속도의 핵심입니다)
st.sidebar.header("🚀 시스템 모드 설정")
app_mode = st.sidebar.radio("작업 선택", ["📊 데이터 분석 & 열화 진단", "📡 실시간 로깅 시뮬레이터"])

st.sidebar.divider()
model_choice = st.sidebar.selectbox(
    "적용할 모델 선택",
    ["1. Linear Regression", "2. Ridge Regression", "3. Decision Tree", "4. Random Forest", "5. Gradient Boosting"]
)

# 3. 데이터 로드 및 모델 학습 (캐싱 적용)
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type="csv")

@st.cache_resource # 모델 학습 결과를 메모리에 고정하여 속도 저하 방지
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
    
    if "1." in model_name: model = LinearRegression()
    elif "2." in model_name: model = Ridge(alpha=1.0)
    elif "3." in model_name: model = DecisionTreeRegressor(max_depth=10)
    elif "4." in model_name: model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    else: model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    
    model.fit(X, y)
    return df, model, X, y

if uploaded_file is not None:
    df, model, X, y = get_trained_model(uploaded_file, model_choice)

    # ==========================================
    # 모드 1: 정밀 데이터 분석 (무거운 그래프 포함)
    # ==========================================
    if app_mode == "📊 데이터 분석 & 열화 진단":
        st.header("🔍 센서 상태 및 열화 정밀 리포트")
        
        col_rep1, col_rep2 = st.columns([1.5, 1])
        with col_rep1:
            aging_model = LinearRegression().fit(X, y)
            deg_rate = aging_model.coef_[2]
            
            if deg_rate > 0:
                st.warning(f"⚠️ **열화 진행 중 (+{deg_rate:.4f} kΩ/day)**")
            else:
                st.success(f"✅ **안정화 중 ({deg_rate:.4f} kΩ/day)**")
            
            if hasattr(model, 'coef_'):
                st.info(f"**Linear Formula:** $R = {model.intercept_:.2f} + ({model.coef_[0]:.4f} \cdot T) + ({model.coef_[1]:.4f} \cdot H) + ({model.coef_[2]:.4f} \cdot D)$")
        
        with col_rep2:
            y_pred_all = model.predict(X)
            st.metric("모델 정확도 (R²)", f"{r2_score(y, y_pred_all):.4f}")
            st.metric("평균 오차 (RMSE)", f"{np.sqrt(mean_squared_error(y, y_pred_all)):.4f} kΩ")

        # 분석용 무거운 그래프들 (이 모드일 때만 실행됨)
        st.divider()
        plt.rcdefaults()
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        sns.regplot(ax=axes[0, 0], x='온도', y='Resistance_kOhm', data=df, scatter_kws={'alpha':0.01}, line_kws={'color':'red'})
        axes[0, 0].set_title("Temperature vs Resistance")
        
        drift = y - (aging_model.coef_[0]*df['온도'] + aging_model.coef_[1]*df['습도'] + aging_model.intercept_)
        axes[0, 1].scatter(df['Elapsed_Days'], drift, alpha=0.05, s=1, color='orange')
        axes[0, 1].set_title("Pure Aging Drift")
        
        axes[1, 0].scatter(y, y_pred_all, alpha=0.05, s=1, color='purple')
        axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        
        axes[1, 1].plot(df['측정 시간'].iloc[::50], y.iloc[::50], color='black', alpha=0.4, label='Actual')
        axes[1, 1].plot(df['측정 시간'].iloc[::50], y_pred_all[::50], color='lime', linestyle='--', label='Pred')
        
        plt.tight_layout()
        st.pyplot(fig)

    # ==========================================
    # 모드 2: 실시간 로깅 시뮬레이터 (가벼운 로직)
    # ==========================================
    else:
        st.header("📡 실시간 데이터 시뮬레이션")
        
        if 'sim_df' not in st.session_state:
            st.session_state.sim_df = pd.DataFrame(columns=['Time', 'Resistance', 'Temp', 'Humi'])
            st.session_state.current_day = df['Elapsed_Days'].max()

        c_ctrl, c_view = st.columns([1, 3])
        
        with c_ctrl:
            run_sim = st.checkbox("▶️ 시뮬레이션 시작", value=False)
            curr_t = st.slider("현재 온도", 10.0, 50.0, float(df['온도'].mean()))
            curr_h = st.slider("현재 습도", 10.0, 95.0, float(df['습도'].mean()))
            if st.button("🧹 기록 초기화"):
                st.session_state.sim_df = pd.DataFrame(columns=['Time', 'Resistance', 'Temp', 'Humi'])
                st.rerun()

        with c_view:
            if run_sim:
                st.session_state.current_day += (1 / 86400) # 1초 추가
                # 예측 시 굳이 DataFrame 안 만들고 배열로 넣어 속도 향상
                p_res = model.predict([[curr_t, curr_h, st.session_state.current_day]])[0]
                new_pt = pd.DataFrame({'Time':[datetime.now()], 'Resistance':[p_res], 'Temp':[curr_t], 'Humi':[curr_h]})
                st.session_state.sim_df = pd.concat([st.session_state.sim_df, new_pt], ignore_index=True).tail(50)

            # Plotly는 웹 최적화 그래프라 매우 빠릅니다
            fig_sim = go.Figure()
            if not st.session_state.sim_df.empty:
                fig_sim.add_trace(go.Scatter(x=st.session_state.sim_df['Time'], y=st.session_state.sim_df['Resistance'], mode='lines+markers', line=dict(color='#00FF00', width=2)))
            
            fig_sim.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_sim, use_container_width=True)

        if run_sim:
            time.sleep(1)
            st.rerun()
else:
    st.info("👋 CSV 파일을 먼저 업로드해 주세요.")