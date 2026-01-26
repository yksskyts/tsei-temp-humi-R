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
st.set_page_config(page_title="Sensor ML Expert (Physical Units)", layout="wide")
st.title("🧪 센서 정밀 분석 대시보드 (K & ppm 버전)")
st.markdown("섭씨와 습도(%)를 **절대온도(K)**와 **수증기 농도(ppm)**로 변환하여 분석합니다.")

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

# 3. 데이터 로드 및 물리 변환
uploaded_file = st.file_uploader("CSV 파일을 여기에 드래그하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
    
    # [물리 변환 단계]
    # 1. 저항 kOhm 변환
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    
    # 2. 절대온도 (K) 변환
    df['Temp_K'] = df['온도'] + 273.15
    
    # 3. 수증기 농도 (ppm) 변환 (Magnus-Tetens 공식 기준)
    p_sat = 6.112 * np.exp((17.62 * df['온도']) / (243.12 + df['온도']))
    p_v = (df['습도'] / 100) * p_sat
    df['Humidity_ppm'] = (p_v / 1013.25) * 1_000_000
    
    # 학습용 데이터셋 구성 (K, ppm 사용)
    X = df[['Temp_K', 'Humidity_ppm']]
    y = df['Resistance_kOhm']
    
    # 모델 할당
    if "1." in model_choice: model = LinearRegression()
    elif "2." in model_choice: model = Ridge(alpha=1.0)
    elif "3." in model_choice: model = DecisionTreeRegressor(max_depth=10)
    elif "4." in model_choice: model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    elif "5." in model_choice: model = GradientBoostingRegressor(n_estimators=50, random_state=42)

    # 모델 학습
    with st.spinner(f'{model_choice} 물리 모델 분석 중...'):
        model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # 4. 분석 리포트 (수식/중요도 + 성능 지표)
    st.divider()
    col_rep1, col_rep2 = st.columns([1.5, 1])
    
    with col_rep1:
        if hasattr(model, 'coef_'):
            st.subheader("📝 Physical Regression Formula")
            # 선형 모델 수식 출력 (K, ppm 기준)
            st.info(f"**$R(k\Omega) = {model.intercept_:.2f} + ({model.coef_[0]:.4f} \\times T_K) + ({model.coef_[1]:.6f} \\times H_{{ppm}})$**")
        elif hasattr(model, 'feature_importances_'):
            st.subheader("💡 Feature Importance (Relative Impact)")
            feat_imp = pd.Series(model.feature_importances_, index=['Temp(K)', 'Humidity(ppm)'])
            plt.rcdefaults()
            fig_imp, ax_imp = plt.subplots(figsize=(5, 2.2))
            feat_imp.sort_values().plot(kind='barh', color=['#3498db', '#e74c3c'], ax=ax_imp)
            ax_imp.set_title("Physical Feature Importance", fontsize=9)
            st.pyplot(fig_imp)

    with col_rep2:
        st.subheader("🎯 Model Performance")
        st.metric("결정계수 (R²)", f"{r2:.4f}")
        st.metric("평균 오차 (RMSE)", f"{rmse:.4f} kΩ")

    # 5. 실시간 저항 예측 시뮬레이터 (사용자 입력은 섭씨/습도 유지하되 내부 변환)
    st.divider()
    st.header("🔍 실시간 저항 예측 (Auto-Conversion)")
    c_in1, c_in2, c_res = st.columns([1, 1, 2])
    with c_in1:
        input_temp_c = st.number_input("입력 온도 (°C)", value=float(df['온도'].mean()))
    with c_in2:
        input_humi_p = st.number_input("입력 습도 (%)", value=float(df['습도'].mean()))
    
    # 입력값 변환
    input_k = input_temp_c + 273.15
    input_p_sat = 6.112 * np.exp((17.62 * input_temp_c) / (243.12 + input_temp_c))
    input_ppm = ((input_humi_p / 100) * input_p_sat / 1013.25) * 1_000_000
    
    pred_val = model.predict([[input_k, input_ppm]])[0]
    
    with c_res:
        st.metric(f"예측 저항값", f"{pred_val:.4f} kΩ")
        st.caption(f"변환된 값: {input_k:.2f} K / {input_ppm:.1f} ppm")

    # 6. 영향도 분석 그래프 (K, ppm 축 사용)
    st.divider()
    st.header("📈 물리 변수 기반 시각화 (Physical Visual Analysis)")
    
    plt.rcdefaults()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # [1] Temp(K) vs Res
    sns.regplot(ax=axes[0, 0], x='Temp_K', y='Resistance_kOhm', data=df, 
                scatter_kws={'alpha': 0.02, 's': 1, 'color': 'gray'}, line_kws={'color': 'red'})
    axes[0, 0].set_title("Absolute Temp (K) vs Resistance", fontsize=12)
    axes[0, 0].set_xlabel("Temp (K)")
    axes[0, 0].set_ylabel("Res (kOhm)")

    # [2] Humidity(ppm) vs Res
    sns.regplot(ax=axes[0, 1], x='Humidity_ppm', y='Resistance_kOhm', data=df, 
                scatter_kws={'alpha': 0.02, 's': 1, 'color': 'gray'}, line_kws={'color': 'blue'})
    axes[0, 1].set_title("Moisture Concentration (ppm) vs Resistance", fontsize=12)
    axes[0, 1].set_xlabel("Humidity (ppm)")
    axes[0, 1].set_ylabel("Res (kOhm)")

    # [3] Accuracy Linearity
    axes[1, 0].scatter(y, y_pred, alpha=0.1, s=1, color='purple')
    axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5)
    axes[1, 0].set_title(f"Model Linearity (R2={r2:.4f})", fontsize=12)
    axes[1, 0].set_xlabel("Measured (kOhm)")
    axes[1, 0].set_ylabel("Predicted (kOhm)")

    # [4] Time-series Tracking
    sample_df = df.iloc[::30]
    axes[1, 1].plot(sample_df['측정 시간'], sample_df['Resistance_kOhm'], label='Measured', alpha=0.5, color='black', lw=1)
    axes[1, 1].plot(sample_df['측정 시간'], y_pred[::30], label='Predicted', color='limegreen', linestyle='--', lw=1.5)
    axes[1, 1].set_title("Time-series Tracking Performance", fontsize=12)
    axes[1, 1].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # 7. 다운로드 (변환된 K, ppm 데이터 포함)
    st.download_button("물리 변환 데이터 다운로드", df.to_csv(index=False).encode('utf-8'), "physical_sensor_analysis.csv")

else:
    st.info("👋 센서 데이터 CSV 파일을 업로드해 주세요 (B열:온도, C열:습도, 저항 열 포함).")