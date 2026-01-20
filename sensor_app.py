import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. 페이지 설정
st.set_page_config(page_title="Sensor Predictor Pro", layout="wide")

st.title("🧪 센서 정밀 분석 및 저항 예측 시뮬레이터")
st.markdown("데이터를 분석하고, 특정 환경(T/H)에서의 **예상 저항값을 실시간으로 확인**하세요.")

# 2. 파일 업로더
uploaded_file = st.file_uploader("CSV 파일을 여기에 드래그하여 업로드하세요", type="csv")

if uploaded_file is not None:
    # 데이터 로드 및 전처리
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    
    st.success(f"✅ 데이터 로드 완료: 총 {len(df):,}행 분석 중")

    # 3. 머신러닝 모델링 (온도 + 습도)
    X = df[['온도', '습도']]
    y = df['Resistance_kOhm']
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    t_coef = model.coef_[0]
    h_coef = model.coef_[1]
    intercept = model.intercept_

    # ---------------------------------------------------------
    # 🌟 [신규] 실시간 저항 예측 시뮬레이터 섹션
    # ---------------------------------------------------------
    st.divider()
    st.header("🔍 실시간 저항 예측 (Prediction Simulator)")
    
    with st.container():
        st.write("측정하고 싶은 온도와 습도 값을 입력하세요.")
        col_input1, col_input2, col_result = st.columns([1, 1, 2])
        
        with col_input1:
            input_temp = st.number_input("온도 입력 (°C)", 
                                         value=float(df['온도'].mean()), 
                                         format="%.2f")
        with col_input2:
            input_humi = st.number_input("습도 입력 (%)", 
                                         value=float(df['습도'].mean()), 
                                         format="%.2f")
            
        # 예측 계산
        predicted_val = model.predict([[input_temp, input_humi]])[0]
        
        with col_result:
            st.metric("예상 저항값 (Predicted Resistance)", f"{predicted_val:.4f} kΩ")
            st.caption(f"오차 범위(RMSE) 고려 시: {predicted_val-rmse:.2f} ~ {predicted_val+rmse:.2f} kΩ")

    # 4. 분석 결과 요약 지표
    st.divider()
    st.header("📊 머신러닝 분석 지표")
    c1, c2, c3 = st.columns(3)
    c1.metric("모델 정확도 (R²)", f"{r2:.4f}")
    c2.metric("평균 예측 오차 (RMSE)", f"{rmse:.4f} kΩ")
    c3.metric("데이터 샘플 수", f"{len(df):,}")

    st.info(f"📍 **센서 보정 공식:** $R(k\Omega) = {intercept:.2f} + ({t_coef:.4f} \\times T) + ({h_coef:.4f} \\times H)$")

    # 5. 시각화 (그래프 내부 영문 유지)
    st.divider()
    st.header("📈 영향도 및 성능 시각화")
    
    plt.rcdefaults()
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 12))

    # [1] Temp vs Resistance
    ax1 = fig.add_subplot(2, 2, 1)
    sns.regplot(ax=ax1, x='온도', y='Resistance_kOhm', data=df, 
                scatter_kws={'alpha': 0.03, 's': 1, 'color': 'gray'}, 
                line_kws={'color': 'red', 'label': 'Temp Trend'})
    ax1.set_title("1. Temperature vs Resistance", fontsize=15)
    ax1.set_xlabel("Temperature (C)")
    ax1.set_ylabel("Resistance (kOhm)")

    # [2] Humi vs Resistance
    ax2 = fig.add_subplot(2, 2, 2)
    sns.regplot(ax=ax2, x='습도', y='Resistance_kOhm', data=df, 
                scatter_kws={'alpha': 0.03, 's': 1, 'color': 'gray'}, 
                line_kws={'color': 'blue', 'label': 'Humi Trend'})
    ax2.set_title("2. Humidity vs Resistance", fontsize=15)
    ax2.set_xlabel("Humidity (%)")
    ax2.set_ylabel("Resistance (kOhm)")

    # [3] Time-series Fit
    ax3 = fig.add_subplot(2, 1, 2)
    sample_df = df.iloc[::25]
    sample_pred = y_pred[::25]
    ax3.plot(sample_df['측정 시간'], sample_df['Resistance_kOhm'], label='Actual', alpha=0.4, color='black')
    ax3.plot(sample_df['측정 시간'], sample_pred, label='Predicted (T+H)', color='limegreen', linestyle='--', lw=2)
    ax3.set_title(f"3. Combined Model Performance (R2={r2:.4f})", fontsize=15)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Resistance (kOhm)")
    ax3.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # 6. 결과 다운로드
    st.download_button("전체 분석 결과 CSV 받기", df.to_csv(index=False).encode('utf-8'), "analysis_result.csv")

else:
    st.info("👋 분석할 센서 데이터 CSV 파일을 업로드해 주세요.")