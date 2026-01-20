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

st.title("🧪 센서 정밀 분석: 5대 머신러닝 모델 비교")
st.markdown("다양한 알고리즘을 사용하여 **ECL-S12-173_1** 센서의 온습도 보정 성능을 극대화하세요.")

# 2. 사이드바 - 5가지 모델 추천 및 설정
st.sidebar.header("🤖 모델 알고리즘 선택")
model_choice = st.sidebar.selectbox(
    "적용할 모델을 선택하세요",
    [
        "1. 선형 회귀 (Linear Regression)", 
        "2. 릿지 회귀 (Ridge Regression)", 
        "3. 의사결정 나무 (Decision Tree)", 
        "4. 랜덤 포레스트 (Random Forest)", 
        "5. 그래디언트 부스팅 (Gradient Boosting)"
    ]
)

st.sidebar.divider()
st.sidebar.write("💡 **모델별 특징**")
descriptions = {
    "1. 선형 회귀 (Linear Regression)": "가장 기본적인 모델로, 해석이 명확하고 수식이 단순합니다.",
    "2. 릿지 회귀 (Ridge Regression)": "선형 모델에 규제를 추가하여, 데이터의 노이즈에 강하고 과적합을 방지합니다.",
    "3. 의사결정 나무 (Decision Tree)": "데이터의 경계값을 찾아 분류하는 방식으로, 비선형적 꺾임(Threshold)을 잘 잡습니다.",
    "4. 랜덤 포레스트 (Random Forest)": "여러 개의 나무를 합쳐 예측하는 모델로, 안정성이 높고 성능이 매우 우수합니다.",
    "5. 그래디언트 부스팅 (Gradient Boosting)": "오차를 순차적으로 보정하며 학습하는 최신 기법으로, 가장 정밀한 예측이 가능합니다."
}
st.sidebar.info(descriptions[model_choice])

# 3. 파일 업로더
uploaded_file = st.file_uploader("CSV 파일을 여기에 드래그하여 업로드하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    
    # 모델링 데이터 준비
    X = df[['온도', '습도']]
    y = df['Resistance_kOhm']
    
    # 모델 할당 루틴
    if "1." in model_choice:
        model = LinearRegression()
    elif "2." in model_choice:
        model = Ridge(alpha=1.0)
    elif "3." in model_choice:
        model = DecisionTreeRegressor(max_depth=10)
    elif "4." in model_choice:
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    elif "5." in model_choice:
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)

    # 모델 학습
    with st.spinner(f'{model_choice} 학습 중...'):
        model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # 4. 실시간 시뮬레이터
    st.divider()
    st.header(f"🔍 {model_choice} 저항 예측")
    c_in1, c_in2, c_res = st.columns([1, 1, 2])
    with c_in1:
        input_temp = st.number_input("온도 입력 (°C)", value=float(df['온도'].mean()))
    with c_in2:
        input_humi = st.number_input("습도 입력 (%)", value=float(df['습도'].mean()))
    
    pred_val = model.predict([[input_temp, input_humi]])[0]
    with c_res:
        st.metric("예상 저항값", f"{pred_val:.4f} kΩ")
        st.caption(f"선택된 모델의 평균 오차(RMSE): {rmse:.4f} kΩ")

    # 5. 지표 요약
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("결정계수 (R²)", f"{r2:.4f}")
    m2.metric("예측 오차 (RMSE)", f"{rmse:.4f} kΩ")
    m3.metric("알고리즘", model_choice.split(". ")[1])

    # 6. 시각화 (4단 그래프 - 영문)
    plt.rcdefaults()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # [1] Temp Impact
    sns.regplot(ax=axes[0, 0], x='온도', y='Resistance_kOhm', data=df, 
                scatter_kws={'alpha': 0.03, 's': 1, 'color': 'gray'}, line_kws={'color': 'red'})
    axes[0, 0].set_title("1. Temperature Sensitivity", fontsize=14)

    # [2] Humi Impact
    sns.regplot(ax=axes[0, 1], x='습도', y='Resistance_kOhm', data=df, 
                scatter_kws={'alpha': 0.03, 's': 1, 'color': 'gray'}, line_kws={'color': 'blue'})
    axes[0, 1].set_title("2. Humidity Sensitivity", fontsize=14)

    # [3] Correlation Fit
    axes[1, 0].scatter(y, y_pred, alpha=0.1, s=1, color='purple')
    axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel("Measured (kOhm)")
    axes[1, 0].set_ylabel("Predicted (kOhm)")
    axes[1, 0].set_title(f"3. Model Linearity (R2={r2:.4f})", fontsize=14)

    # [4] Time-series
    sample_df = df.iloc[::25]
    sample_pred = y_pred[::25]
    axes[1, 1].plot(sample_df['측정 시간'], sample_df['Resistance_kOhm'], label='Actual', alpha=0.4, color='black')
    axes[1, 1].plot(sample_df['측정 시간'], sample_pred, label='ML Predicted', color='limegreen', linestyle='--', lw=2)
    axes[1, 1].set_title("4. Time-series Tracking", fontsize=14)
    axes[1, 1].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # 7. 데이터 다운로드
    st.download_button("분석 결과 파일(CSV) 다운로드", df.to_csv(index=False).encode('utf-8'), "sensor_ml_comparison.csv")

else:
    st.info("👋 분석할 센서 데이터 CSV 파일을 업로드해 주세요.")