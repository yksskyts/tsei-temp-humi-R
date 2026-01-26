import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 사이킷런 모델 임포트
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. 페이지 설정
st.set_page_config(page_title="Sensor ML Tracking", layout="wide")

# 모델 정보 데이터베이스
model_info = {
    "Linear Regression": {"desc": "직선 관계 가정. 수식 해석 용이.", "pros": "계산 속도 최상.", "cons": "비선형 데이터 취약.", "best_for": "기본 보정식"},
    "Ridge Regression": {"desc": "규제 포함 선형 모델.", "pros": "과적합 방지.", "cons": "비선형성 학습 불가.", "best_for": "안정적 선형 모델"},
    "Huber Regressor": {"desc": "이상치에 강한 선형 모델.", "pros": "노이즈 무시 가능.", "cons": "데이터가 깨끗할 땐 일반 모델보다 느림.", "best_for": "노이즈 심한 데이터"},
    "Random Forest": {"desc": "집단지성 나무 모델.", "pros": "정확도와 안정성 매우 높음.", "cons": "수식 추출 불가능.", "best_for": "고정밀 트래킹"},
    "Extra Trees": {"desc": "무작위 앙상블 모델.", "pros": "Random Forest보다 이상치에 강함.", "cons": "결과 변동성 존재.", "best_for": "빠른 앙상블 학습"},
    "Gradient Boosting": {"desc": "오답 보완형 학습.", "pros": "예측 정확도 최상권.", "cons": "학습 시간 소요.", "best_for": "최고 정확도 필요 시"}
}

st.title("🧪 센서 실시간 트래킹 및 ML 모델 벤치마킹")

# 2. 사이드바 모델 선택
st.sidebar.header("🤖 ML 모델 설정")
selected_model_name = st.sidebar.selectbox("테스트할 모델을 선택하세요", list(model_info.keys()))

with st.sidebar.expander("💡 모델 특징", expanded=True):
    info = model_info[selected_model_name]
    st.write(info['desc'])
    st.success(f"🎯 추천: {info['best_for']}")

# 3. 데이터 로드 및 전처리
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    
    # 시간 데이터 처리 (에러 방지를 위한 정렬 및 변환)
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
        df = df.sort_values('측정 시간')
    
    # 물리 변환 (K, ppm)
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    df['Temp_K'] = df['온도'] + 273.15
    p_sat = 6.112 * np.exp((17.62 * df['온도']) / (243.12 + df['온도']))
    df['Humidity_ppm'] = ((df['습도'] / 100) * p_sat / 1013.25) * 1_000_000
    
    X = df[['Temp_K', 'Humidity_ppm']]
    y = df['Resistance_kOhm']
    
    # 모델 학습
    model_dict = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Huber Regressor": HuberRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    model = model_dict[selected_model_name]

    with st.spinner(f'{selected_model_name} 학습 및 트래킹 중...'):
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

    # 4. 성능 지표 리포트
    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("결정계수 ($R^2$)", f"{r2:.4f}")
    with c2:
        st.metric("예측 오차 (RMSE)", f"{rmse:.4f} kΩ")

    # 5. 핵심: 시간에 따른 저항 경향 그래프 (추적 성능)
    st.divider()
    st.header("📈 시간에 따른 저항 변화 및 모델 추적 성능")
    st.caption("실제 측정값(Actual)과 모델이 예측한 값(Predicted)이 얼마나 일치하는지 시계열로 확인합니다.")

    plt.rcdefaults()
    sns.set_theme(style="whitegrid")
    
    fig_time, ax_time = plt.subplots(figsize=(15, 6))
    
    # 데이터가 너무 많을 경우 가독성을 위해 샘플링 (1/5)
    step = max(1, len(df) // 1000)
    
    if '측정 시간' in df.columns:
        x_axis = df['측정 시간']
    else:
        x_axis = df.index

    ax_time.plot(x_axis[::step], y[::step], label='Actual (Measured)', color='black', alpha=0.5, lw=2)
    ax_time.plot(x_axis[::step], y_pred[::step], label='Predicted (ML Model)', color='limegreen', linestyle='--', lw=2)
    
    ax_time.set_ylabel("Resistance (kOhm)")
    ax_time.set_xlabel("Time / Sequence")
    ax_time.legend(loc='upper right')
    ax_time.set_title(f"Model Tracking Performance: {selected_model_name}")
    
    st.pyplot(fig_time)

    # 6. 추가 분석: 선형성 및 오차 분포
    st.divider()
    col_sub1, col_sub2 = st.columns(2)
    
    with col_sub1:
        st.subheader("🎯 예측 선형성 확인")
        fig_scatter, ax_scatter = plt.subplots()
        ax_scatter.scatter(y, y_pred, alpha=0.3, s=2, color='darkblue')
        ax_scatter.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax_scatter.set_xlabel("Measured (kOhm)")
        ax_scatter.set_ylabel("Predicted (kOhm)")
        st.pyplot(fig_scatter)
        
    with col_sub2:
        st.subheader("📊 오차(잔차) 분포")
        fig_res, ax_res = plt.subplots()
        sns.histplot(y - y_pred, kde=True, ax=ax_res, color='purple')
        ax_res.set_xlabel("Error (kOhm)")
        st.pyplot(fig_res)

else:
    st.info("👋 센서 데이터를 업로드하면 시간 경과에 따른 모델 트래킹 분석이 시작됩니다.")