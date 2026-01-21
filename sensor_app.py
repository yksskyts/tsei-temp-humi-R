import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. 페이지 설정
st.set_page_config(page_title="Sensor Aging Analyzer", layout="wide")

st.title("🧪 센서 노화(열화) 및 수명 분석 대시보드")
st.markdown("온습도뿐만 아니라 **'사용 시간'**에 따른 성능 저하를 분석합니다.")

# 2. 파일 업로더
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    df['측정 시간'] = pd.to_datetime(df['측정 시간'])
    
    # [핵심] 노화 분석을 위한 시간 변수 생성 (첫 측정 대비 경과 시간 계산)
    first_time = df['측정 시간'].min()
    df['Elapsed_Days'] = (df['측정 시간'] - first_time).dt.total_seconds() / (24 * 3600)
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    
    # 3. 노화 반영 모델링 (온도, 습도 + 사용일수)
    X = df[['온도', '습도', 'Elapsed_Days']]
    y = df['Resistance_kOhm']
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # 지표 산출
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # 열화율 계산 (1일당 저항 변화량)
    degradation_rate = model.coef_[2] 

    # 4. 노화 분석 리포트
    st.divider()
    st.header("⏳ 센서 노화(Aging) 분석 결과")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("일일 저항 변화율", f"{degradation_rate:.4f} kΩ/day", 
              help="하루에 평균적으로 저항이 얼마나 변하는지 나타냅니다.")
    c2.metric("모델 신뢰도 (R²)", f"{r2:.4f}")
    c3.metric("누적 사용 기간", f"{df['Elapsed_Days'].max():.1f} 일")

    # 수식 기반 미래 예측
    st.info(f"📍 **노화 보정 공식:** $R = {model.intercept_:.2f} + ({model.coef_[0]:.4f} \\times T) + ({model.coef_[1]:.4f} \\times H) + ({degradation_rate:.4f} \\times Days)$")

    # 5. 미래 성능 예측 시뮬레이터
    st.divider()
    st.header("🔮 미래 성능 예측 (Future Prediction)")
    
    f_col1, f_col2, f_col3, f_res = st.columns([1, 1, 1, 2])
    with f_col1:
        f_temp = st.number_input("예상 온도 (°C)", value=25.0)
    with f_col2:
        f_humi = st.number_input("예상 습도 (%)", value=50.0)
    with f_col3:
        f_days = st.number_input("추가 사용 일수 (일 뒤)", value=30)
    
    # 미래 시점 계산 (현재 마지막 데이터 시점 + 추가 일수)
    future_day = df['Elapsed_Days'].max() + f_days
    future_pred = model.predict([[f_temp, f_humi, future_day]])[0]
    
    with f_res:
        st.metric(f"{f_days}일 후 예상 기저 저항", f"{future_pred:.4f} kΩ")
        st.write(f"현재 대비 **{future_pred - df['Resistance_kOhm'].iloc[-1]:.2f} kΩ** 변화가 예상됩니다.")

    # 6. 노화 시각화 그래프
    st.divider()
    st.header("📈 열화 추이 시각화")
    
    plt.rcdefaults()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # [좌] 시간에 따른 저항 드리프트 (온습도 영향 제거 후 순수 시간 영향)
    # 실제값에서 온습도 영향을 뺀 잔차(Residual)를 그리면 노화 패턴이 잘 보입니다.
    temp_humi_effect = model.coef_[0] * df['온도'] + model.coef_[1] * df['습도'] + model.intercept_
    drift_only = df['Resistance_kOhm'] - temp_humi_effect
    
    axes[0].scatter(df['Elapsed_Days'], drift_only, alpha=0.05, s=1, color='orange')
    axes[0].set_title("Pure Aging Drift (T/H Normalized)", fontsize=12)
    axes[0].set_xlabel("Elapsed Days")
    axes[0].set_ylabel("Resistance Drift (kOhm)")

    # [우] 시간에 따른 예측 오차 추이 (오차가 커지면 교체 타이밍)
    residuals = np.abs(y - y_pred)
    axes[1].plot(df['Elapsed_Days'].iloc[::50], residuals.iloc[::50], color='red', alpha=0.3)
    axes[1].set_title("Model Error Over Time (RMSE Trend)", fontsize=12)
    axes[1].set_xlabel("Elapsed Days")
    axes[1].set_ylabel("Error (kOhm)")

    st.pyplot(fig)

else:
    st.info("👋 센서 노화 분석을 위해 CSV 파일을 업로드해 주세요.")