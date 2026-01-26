import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. 페이지 설정
st.set_page_config(page_title="Sensor Life & Tracking", layout="wide")

# 모델 설명 데이터베이스 (기존 유지)
model_info = {
    "Linear Regression": {"desc": "직선 관계 가정.", "pros": "해석 최상, 수명 계산용 기울기 추출 용이.", "cons": "비선형 데이터 취약.", "best_for": "기본 보정 및 수명 산출"},
    "Ridge Regression": {"desc": "규제 포함 선형 모델.", "pros": "노이즈에 강함.", "cons": "직선 관계만 학습.", "best_for": "안정적인 수명 예측"},
    "Decision Tree": {"desc": "스무고개 방식.", "pros": "이해 쉬움.", "cons": "과적합 위험.", "best_for": "규칙 파악"},
    "Random Forest": {"desc": "집단지성 나무.", "pros": "정확도와 안정성 매우 높음.", "cons": "연산 무거움.", "best_for": "고정밀 트래킹"},
    "Extra Trees": {"desc": "무작위 앙상블.", "pros": "이상치에 강함.", "cons": "오차 변동성.", "best_for": "노이즈 데이터"},
    "Gradient Boosting": {"desc": "오답 보완 학습.", "pros": "예측 정확도 최상.", "cons": "학습 시간 소요.", "best_for": "최고 성능 추적"}
}

st.title("🧪 센서 수명 추정 및 시계열 트래킹 시스템")

# 2. 사이드바 설정
st.sidebar.header("🤖 ML 모델 설정")
selected_model_name = st.sidebar.selectbox("테스트할 모델을 선택하세요", list(model_info.keys()))

with st.sidebar.expander("💡 모델 특성", expanded=False):
    info = model_info[selected_model_name]
    st.write(info['desc'])
    st.success(f"🎯 추천: {info['best_for']}")

# 수명 진단 기준 설정
st.sidebar.divider()
st.sidebar.header("⏳ 수명 진단 설정")
failure_threshold = st.sidebar.slider("고장 판단 저항 변화율 (%)", 5, 50, 10)

# 3. 데이터 로드 및 물리 변환
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    
    # 시간 변수 처리
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
        df['Elapsed_Days'] = (df['측정 시간'] - df['측정 시간'].min()).dt.total_seconds() / 86400
    else:
        df['Elapsed_Days'] = np.arange(len(df)) / 1440 # 분 단위 가정
    
    # 물리 변환
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    df['Temp_K'] = df['온도'] + 273.15
    p_sat = 6.112 * np.exp((17.62 * df['온도']) / (243.12 + df['온도']))
    df['Humidity_ppm'] = ((df['습도'] / 100) * p_sat / 1013.25) * 1_000_000
    
    # [중요] 학습 변수에 'Elapsed_Days'를 넣어 노화(수명)를 모델링함
    X = df[['Temp_K', 'Humidity_ppm', 'Elapsed_Days']]
    y = df['Resistance_kOhm']
    
    # 모델 선택 및 학습
    model_dict = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=10),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    model = model_dict[selected_model_name]

    with st.spinner('센서 거동 및 노화 패턴 분석 중...'):
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

    # 4. 수명 예측(Life Estimation) 로직
    # 내부적으로 선형 기울기를 추출하여 수명 계산
    linear_trend = LinearRegression().fit(df[['Elapsed_Days']], y)
    daily_drift = linear_trend.coef_[0] # 하루당 저항 변화량
    current_res = y.iloc[-1]
    initial_res = y.iloc[0]
    
    # 고장 지점 계산 (초기값 대비 설정된 % 변화 시)
    fail_limit = initial_res * (1 + (failure_threshold/100 if daily_drift > 0 else -failure_threshold/100))
    remaining_res = fail_limit - current_res
    remaining_days = remaining_res / daily_drift if daily_drift != 0 else float('inf')

    # 5. 대시보드 리포트
    st.divider()
    col_rep1, col_rep2, col_rep3 = st.columns(3)
    
    with col_rep1:
        st.subheader("🎯 모델 분석 성능")
        st.metric("결정계수 ($R^2$)", f"{r2:.4f}")
        st.metric("예측 오차 (RMSE)", f"{rmse:.4f} kΩ")

    with col_rep2:
        st.subheader("📉 노화 진단 (Drift)")
        st.metric("일일 저항 변화율", f"{daily_drift:+.6f} kΩ/day")
        st.write(f"현재 저항: {current_res:.2f} kΩ")

    with col_rep3:
        st.subheader("⏳ 수명 추정 (Life)")
        status_color = "inverse" if remaining_days < 30 else "normal"
        st.metric("예상 잔여 수명", f"{max(0, remaining_days):.1f} 일", delta_color=status_color)
        st.caption(f"기준: 초기값 대비 {failure_threshold}% 변화 시")

    # 6. 시각화 (시간에 따른 저항 경향 + 선형성)
    st.divider()
    st.header("📈 센서 거동 추적 및 예측 (Time-Series Tracking)")
    
    plt.rcdefaults()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # [좌측] 실제 vs 예측 (Measured vs Predicted) - 겹쳐 그려서 트래킹 확인
    # 가독성을 위해 데이터가 많으면 샘플링
    step = max(1, len(df) // 500)
    axes[0].plot(df['측정 시간'].iloc[::step], y.iloc[::step], label='Measured (Actual)', color='black', alpha=0.5, lw=2)
    axes[0].plot(df['측정 시간'].iloc[::step], y_pred[::step], label='ML Predicted', color='limegreen', linestyle='--', lw=2)
    axes[0].set_title(f"Real-time Tracking Performance ($R^2$={r2:.4f})")
    axes[0].set_ylabel("Resistance (kOhm)")
    axes[0].legend()

    # [우측] Measured vs Predicted 산점도 (선형성)
    axes[1].scatter(y, y_pred, alpha=0.3, s=2, color='darkblue')
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[1].set_title("Prediction Linearity")
    axes[1].set_xlabel("Measured (kOhm)")
    axes[1].set_ylabel("Predicted (kOhm)")

    st.pyplot(fig)

    # 7. 수명 예측 그래프 (미래 시뮬레이션)
    st.divider()
    st.subheader("🔮 노화 진행 및 교체 시점 시뮬레이션")
    
    future_days = np.linspace(0, max(remaining_days * 1.5, 30), 100)
    future_res = initial_res + daily_drift * future_days
    
    fig_life, ax_life = plt.subplots(figsize=(10, 4))
    ax_life.plot(future_days, future_res, color='orange', label='Aging Trend')
    ax_life.axhline(fail_limit, color='red', linestyle='--', label='Failure Threshold')
    ax_life.axvline(max(0, remaining_days), color='gray', linestyle=':', label='Estimated End of Life')
    ax_life.set_xlabel("Days from Start")
    ax_life.set_ylabel("Baseline Resistance (kOhm)")
    ax_life.legend()
    st.pyplot(fig_life)

else:
    st.info("👋 센서 데이터를 업로드하여 시계열 트래킹과 수명 진단을 시작하세요.")