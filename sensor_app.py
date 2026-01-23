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
from datetime import datetime, timedelta
import time

# 1. 페이지 설정
st.set_page_config(page_title="Sensor Expert Master", layout="wide")

st.title("🧪 센서 정밀 분석 및 실시간 시뮬레이션 통합 시스템")
st.markdown("하나의 플랫폼에서 **데이터 분석, 열화 진단, 실시간 시연**을 모두 수행합니다.")

# 2. 사이드바 - 공통 설정
st.sidebar.header("⚙️ 전체 설정")
model_choice = st.sidebar.selectbox(
    "적용할 모델 선택",
    ["1. Linear Regression", "2. Ridge Regression", "3. Decision Tree", "4. Random Forest", "5. Gradient Boosting"]
)

# 3. 데이터 로드 (공통 사용)
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file is not None:
    # 데이터 전처리 함수 (캐싱으로 속도 향상)
    @st.cache_data
    def load_and_train(file, model_name):
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
        
        # 모델 선택
        if "1." in model_name: model = LinearRegression()
        elif "2." in model_name: model = Ridge(alpha=1.0)
        elif "3." in model_name: model = DecisionTreeRegressor(max_depth=10)
        elif "4." in model_name: model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        else: model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        
        model.fit(X, y)
        return df, model, X, y

    df, model, X, y = load_and_train(uploaded_file, model_choice)
    y_pred = model.predict(X)

    # --- 탭 구성 (중요!) ---
    tab_analysis, tab_realtime = st.tabs(["📊 정밀 데이터 분석 & 열화 진단", "📡 실시간 로깅 시뮬레이터"])

    # ==========================================
    # 탭 1: 정밀 데이터 분석 (전문가님 코드 기반)
    # ==========================================
    with tab_analysis:
        st.header("🔍 센서 상태 및 열화 정밀 리포트")
        
        col_rep1, col_rep2 = st.columns([1.5, 1])
        with col_rep1:
            aging_analyzer = LinearRegression().fit(X, y)
            degradation_rate = aging_analyzer.coef_[2]
            
            if degradation_rate > 0:
                st.warning(f"⚠️ **현재 상태: 열화 진행 중 (저항 증가)**")
                st.write(f"하루 평균 **{degradation_rate:.4f} kΩ**씩 상승 중입니다.")
            else:
                st.success(f"✅ **현재 상태: 안정화 중 (저항 감소)**")
                st.write(f"하루 평균 **{abs(degradation_rate):.4f} kΩ**씩 하강 중입니다.")
            
            if hasattr(model, 'coef_'):
                st.info(f"**Formula:** $R = {model.intercept_:.2f} + ({model.coef_[0]:.4f} \cdot T) + ({model.coef_[1]:.4f} \cdot H) + ({model.coef_[2]:.4f} \cdot D)$")
            elif hasattr(model, 'feature_importances_'):
                plt.rcdefaults()
                fig_imp, ax_imp = plt.subplots(figsize=(5, 2.2))
                feat_imp = pd.Series(model.feature_importances_, index=['Temp', 'Humi', 'Aging'])
                feat_imp.sort_values().plot(kind='barh', color='#3498db', ax=ax_imp)
                st.pyplot(fig_imp)

        with col_rep2:
            st.metric("결정계수 (R²)", f"{r2_score(y, y_pred):.4f}")
            st.metric("평균 오차 (RMSE)", f"{np.sqrt(mean_squared_error(y, y_pred)):.4f} kΩ")

        # 시각화 섹션 (4단 그래프)
        st.divider()
        plt.rcdefaults()
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        sns.regplot(ax=axes[0, 0], x='온도', y='Resistance_kOhm', data=df, scatter_kws={'alpha':0.02}, line_kws={'color':'red'})
        
        # 순수 시간 드리프트 계산
        temp_humi_effect = aging_analyzer.coef_[0] * df['온도'] + aging_analyzer.coef_[1] * df['습도'] + aging_analyzer.intercept_
        drift_only = df['Resistance_kOhm'] - temp_humi_effect
        axes[0, 1].scatter(df['Elapsed_Days'], drift_only, alpha=0.05, s=1, color='orange')
        axes[0, 1].set_title("Pure Aging Drift")
        
        axes[1, 0].scatter(y, y_pred, alpha=0.1, s=1, color='purple')
        axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        
        axes[1, 1].plot(df['측정 시간'].iloc[::30], y.iloc[::30], label='Measured', alpha=0.5, color='black')
        axes[1, 1].plot(df['측정 시간'].iloc[::30], y_pred[::30], label='Predicted', color='lime', linestyle='--')
        axes[1, 1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)

    # ==========================================
    # 탭 2: 실시간 시뮬레이터 (1초 로깅)
    # ==========================================
    with tab_realtime:
        st.header("📡 실시간 데이터 시뮬레이션")
        st.write("슬라이더를 움직여 환경을 바꾸면 1초마다 데이터가 실시간으로 생성됩니다.")
        
        # 상태 관리
        if 'sim_df' not in st.session_state:
            st.session_state.sim_df = pd.DataFrame(columns=['Time', 'Resistance', 'Temp', 'Humi'])
            st.session_state.current_day = df['Elapsed_Days'].max()

        col_ctrl, col_view = st.columns([1, 3])
        
        with col_ctrl:
            run_sim = st.checkbox("▶️ 시뮬레이션 시작", value=False)
            curr_temp = st.slider("실시간 온도 (°C)", 10.0, 50.0, float(df['온도'].mean()), 0.1)
            curr_humi = st.slider("실시간 습도 (%)", 10.0, 95.0, float(df['습도'].mean()), 0.1)
            if st.button("🧹 기록 초기화"):
                st.session_state.sim_df = pd.DataFrame(columns=['Time', 'Resistance', 'Temp', 'Humi'])
                st.rerun()

        with col_view:
            if run_sim:
                st.session_state.current_day += (1 / (24 * 3600))
                input_sim = pd.DataFrame([[curr_temp, curr_humi, st.session_state.current_day]], columns=['온도', '습도', 'Elapsed_Days'])
                pred_res = model.predict(input_sim)[0]
                new_entry = pd.DataFrame({'Time':[datetime.now()], 'Resistance':[pred_res], 'Temp':[curr_temp], 'Humi':[curr_humi]})
                st.session_state.sim_df = pd.concat([st.session_state.sim_df, new_entry], ignore_index=True).tail(100)

            # Plotly 그래프
            fig_sim = go.Figure()
            if not st.session_state.sim_df.empty:
                fig_sim.add_trace(go.Scatter(x=st.session_state.sim_df['Time'], y=st.session_state.sim_df['Resistance'], mode='lines+markers', line=dict(color='#00FF00')))
            
            fig_sim.update_layout(template="plotly_dark", height=500, title="Live Sensor Resistance Line", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_sim, use_container_width=True)

        # 무한 리프레시
        if run_sim:
            time.sleep(1)
            st.rerun()

else:
    st.info("👋 먼저 분석할 CSV 파일을 업로드해 주세요.")