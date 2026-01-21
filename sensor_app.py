import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. 페이지 설정
st.set_page_config(page_title="Sensor ML Expert", layout="wide")
st.title("🧪 센서 정밀 분석 및 미래 예측 시스템")

# 2. 사이드바 모델 선택
st.sidebar.header("🤖 알고리즘 선택")
model_choice = st.sidebar.selectbox(
    "알고리즘을 선택하세요 (미래 예측은 1, 2번 권장)",
    ["1. Linear Regression", "2. Ridge Regression", "3. Decision Tree", "4. Random Forest", "5. Gradient Boosting"]
)

# 3. 데이터 업로드 및 전처리
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    
    # [핵심] 경과 일수(Aging) 변수 생성
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
        # 첫 측정 시점으로부터 며칠이 지났는지 계산
        df['Elapsed_Days'] = (df['측정 시간'] - df['측정 시간'].min()).dt.total_seconds() / (24 * 3600)
    else:
        df['Elapsed_Days'] = np.arange(len(df)) / (60 * 24)

    df['Resistance_kOhm'] = df['저항'] / 1000.0
    
    # 학습 변수에 'Elapsed_Days'가 반드시 포함되어야 함
    X_cols = ['온도', '습도', 'Elapsed_Days']
    X = df[X_cols]
    y = df['Resistance_kOhm']
    
    # 모델 정의
    if "1." in model_choice: model = LinearRegression()
    elif "2." in model_choice: model = Ridge(alpha=1.0)
    elif "3." in model_choice: model = DecisionTreeRegressor(max_depth=8)
    elif "4." in model_choice: model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    else: model = GradientBoostingRegressor(n_estimators=50, random_state=42)

    model.fit(X, y)
    y_pred = model.predict(X)

    # 4. 분석 리포트 (중요도 레이블 영문 고정)
    st.divider()
    st.header(f"📊 분석 리포트: {model_choice}")
    c1, c2 = st.columns([1.5, 1])
    
    with c1:
        if hasattr(model, 'feature_importances_'):
            # 중요도 차트에 Temp, Humi, Aging 3개가 모두 나와야 함
            feat_imp = pd.Series(model.feature_importances_, index=['Temp', 'Humi', 'Aging'])
            plt.rcdefaults()
            fig_imp, ax_imp = plt.subplots(figsize=(5, 2.5))
            feat_imp.sort_values().plot(kind='barh', color='#3498db', ax=ax_imp)
            ax_imp.set_title("Feature Importance (Inc. Aging)", fontsize=10)
            st.pyplot(fig_imp)
        elif hasattr(model, 'coef_'):
            st.info(f"**Formula:** $R = {model.intercept_:.2f} + ({model.coef_[0]:.4f} \cdot T) + ({model.coef_[1]:.4f} \cdot H) + ({model.coef_[2]:.4f} \cdot Day)$")

    with c2:
        st.metric("결정계수 (R²)", f"{r2_score(y, y_pred):.4f}")
        st.metric("평균 오차 (RMSE)", f"{np.sqrt(mean_squared_error(y, y_pred)):.4f} kΩ")

    # 5. 미래 예측 시뮬레이터 (수정된 핵심 로직)
    st.divider()
    st.header("🔮 미래 저항 예측 (날짜 반영)")
    sc1, sc2, sc3, sc_res = st.columns([1, 1, 1, 2])
    
    with sc1: f_temp = st.number_input("온도 (°C)", value=float(df['온도'].mean()))
    with sc2: f_humi = st.number_input("습도 (%)", value=float(df['습도'].mean()))
    with sc3: f_days = st.number_input("추가 경과일 (오늘+N일)", value=1, step=1)
    
    # 미래 날짜 = 데이터의 마지막 날짜 + 사용자가 입력한 추가 일수
    target_day = df['Elapsed_Days'].max() + f_days
    # 입력 순서 주의: [온도, 습도, 경과일수]
    f_pred = model.predict([[f_temp, f_humi, target_day]])[0]
    
    with sc_res:
        st.metric(f"{f_days}일 후 예상 저항", f"{f_pred:.4f} kΩ")
        # 변화량 표시
        diff = f_pred - df['Resistance_kOhm'].iloc[-1]
        st.write(f"현재 마지막 측정값 대비: **{diff:+.4f} kΩ**")

    # 6. 시각화
    st.divider()
    plt.rcdefaults()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # [좌] 시간 흐름에 따른 실제 vs 예측
    axes[0].plot(df['측정 시간'].iloc[::30], y.iloc[::30], label='Measured', color='black', alpha=0.5)
    axes[0].plot(df['측정 시간'].iloc[::30], y_pred[::30], label='Predicted', color='lime', linestyle='--')
    axes[0].set_title("Time-series tracking")
    axes[0].legend()

    # [우] 순수 시간 열화 그래프 (T/H 영향 제거)
    base_linear = LinearRegression().fit(X, y)
    drift = y - (base_linear.coef_[0]*df['온도'] + base_linear.coef_[1]*df['습도'] + base_linear.intercept_)
    axes[1].scatter(df['Elapsed_Days'], drift, s=1, alpha=0.1, color='orange')
    axes[1].set_title("Pure Aging Drift (T/H Normalized)")
    
    st.pyplot(fig)

else:
    st.info("CSV 파일을 업로드하면 분석이 시작됩니다.")