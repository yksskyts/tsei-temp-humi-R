import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="Sensor ML Expert", layout="wide")
st.title("🧪 센서 정밀 분석 대시보드 (Optimized)")
st.markdown("5가지 머신러닝 모델을 비교 분석하고 실시간 저항을 예측합니다.")

# 2. 사이드바 모델 선택
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

# 3. 데이터 로드
uploaded_file = st.file_uploader("CSV 파일을 여기에 드래그하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    
    X = df[['온도', '습도']]
    y = df['Resistance_kOhm']
    
    # 모델 할당
    if "1." in model_choice: model = LinearRegression()
    elif "2." in model_choice: model = Ridge(alpha=1.0)
    elif "3." in model_choice: model = DecisionTreeRegressor(max_depth=10)
    elif "4." in model_choice: model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    elif "5." in model_choice: model = GradientBoostingRegressor(n_estimators=50, random_state=42)

    # 모델 학습
    with st.spinner(f'{model_choice} 분석 중...'):
        model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # 4. 분석 리포트 (수식/중요도 + 성능 지표)
    st.divider()
    col_rep1, col_rep2 = st.columns([1.5, 1])
    
    with col_rep1:
        # 선형 모델 수식 출력
        if hasattr(model, 'coef_'):
            st.subheader("📝 Regression Formula")
            st.info(f"**$R(k\Omega) = {model.intercept_:.2f} + ({model.coef_[0]:.4f} \\times T) + ({model.coef_[1]:.4f} \\times H)$**")
        # 비선형 모델 변수 중요도 출력
        elif hasattr(model, 'feature_importances_'):
            st.subheader("💡 Feature Importance (Relative Impact)")
            feat_imp = pd.Series(model.feature_importances_, index=['Temp', 'Humi'])
            plt.rcdefaults() # 영문 폰트 강제
            fig_imp, ax_imp = plt.subplots(figsize=(5, 2.2)) # 크기 축소
            feat_imp.sort_values().plot(kind='barh', color=['#3498db', '#e74c3c'], ax=ax_imp)
            ax_imp.set_title("Feature Importance Analysis (Tree-based)", fontsize=9)
            st.pyplot(fig_imp)

    with col_rep2:
        st.subheader("🎯 Model Performance")
        st.metric("결정계수 (R²)", f"{r2:.4f}")
        st.metric("평균 오차 (RMSE)", f"{rmse:.4f} kΩ")

    # 5. 실시간 저항 예측 시뮬레이터
    st.divider()
    st.header("🔍 실시간 저항 예측")
    c_in1, c_in2, c_res = st.columns([1, 1, 2])
    with c_in1:
        input_temp = st.number_input("현재 온도 (°C)", value=float(df['온도'].mean()))
    with c_in2:
        input_humi = st.number_input("현재 습도 (%)", value=float(df['습도'].mean()))
    
    pred_val = model.predict([[input_temp, input_humi]])[0]
    with c_res:
        st.metric(f"예측 저항값 ({model_choice.split('. ')[1]})", f"{pred_val:.4f} kΩ")

    # 6. 영향도 분석 그래프 (전체 영문 레이블 및 크기 최적화)
    st.divider()
    st.header("📈 상세 시각화 분석 (Visual Analysis)")
    
    plt.rcdefaults()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 한 화면에 들어오도록 크기 조정

    # [1] Temp vs Res
    sns.regplot(ax=axes[0, 0], x='온도', y='Resistance_kOhm', data=df, 
                scatter_kws={'alpha': 0.02, 's': 1, 'color': 'gray'}, line_kws={'color': 'red'})
    axes[0, 0].set_title("Temperature vs Resistance", fontsize=12)
    axes[0, 0].set_xlabel("Temp (C)")
    axes[0, 0].set_ylabel("Res (kOhm)")

    # [2] Humi vs Res
    sns.regplot(ax=axes[0, 1], x='습도', y='Resistance_kOhm', data=df, 
                scatter_kws={'alpha': 0.02, 's': 1, 'color': 'gray'}, line_kws={'color': 'blue'})
    axes[0, 1].set_title("Humidity vs Resistance", fontsize=12)
    axes[0, 1].set_xlabel("Humi (%)")
    axes[0, 1].set_ylabel("Res (kOhm)")

    # [3] Correlation
    axes[1, 0].scatter(y, y_pred, alpha=0.1, s=1, color='purple')
    axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5)
    axes[1, 0].set_title(f"Model Linearity (R2={r2:.4f})", fontsize=12)
    axes[1, 0].set_xlabel("Measured (kOhm)")
    axes[1, 0].set_ylabel("Predicted (kOhm)")

    # [4] Time-series
    sample_df = df.iloc[::30]
    axes[1, 1].plot(sample_df['측정 시간'], sample_df['Resistance_kOhm'], label='Measured', alpha=0.5, color='black', lw=1)
    axes[1, 1].plot(sample_df['측정 시간'], y_pred[::30], label='Predicted', color='limegreen', linestyle='--', lw=1.5)
    axes[1, 1].set_title("Model Tracking Performance", fontsize=12)
    axes[1, 1].legend(prop={'size': 9})

    plt.tight_layout()
    st.pyplot(fig)

    # 7. 다운로드
    st.download_button("결과 파일(CSV) 다운로드", df.to_csv(index=False).encode('utf-8'), "sensor_analysis.csv")

else:
    st.info("👋 센서 데이터 CSV 파일을 업로드해 주세요.")