import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 페이지 설정
st.set_page_config(page_title="Sensor Data Cleaner", layout="wide")
st.title("🧹 센서 데이터 이상치 정제 솔루션")
st.markdown("정제되지 않은 데이터는 모델의 $R^2$를 떨어뜨립니다. 최적의 필터링을 선택하세요.")

# 2. 데이터 로드
uploaded_file = st.sidebar.file_uploader("CSV 업로드", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    
    # 물리 변환 (이전 로직 유지)
    df['Temp_K'] = df['온도'] + 273.15
    p_sat = 6.112 * np.exp((17.62 * df['온도']) / (243.12 + df['온도']))
    df['PPM'] = ((df['습도'] / 100) * p_sat / 1013.25) * 1_000_000
    df['Res_kOhm'] = df['저항'] / 1000.0

    st.sidebar.divider()
    st.sidebar.header("🛡️ 정제 알고리즘 선택")
    clean_method = st.sidebar.selectbox("알고리즘", ["Z-Score", "IQR (통계 기반)", "Isolation Forest (ML 기반)"])
    
    # 정제 강도 설정 (민감도)
    sensitivity = st.sidebar.slider("정제 강도 (높을수록 많이 제거)", 0.01, 0.5, 0.05) if clean_method == "Isolation Forest (ML 기반)" else st.sidebar.slider("임계값 (K-Factor)", 1.0, 5.0, 3.0)

    # 3. 이상치 탐지 로직
    df_clean = df.copy()
    
    if clean_method == "Z-Score":
        z_scores = zscore(df['Res_kOhm'])
        outliers = np.abs(z_scores) > sensitivity
    elif clean_method == "IQR (통계 기반)":
        Q1 = df['Res_kOhm'].quantile(0.25)
        Q3 = df['Res_kOhm'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df['Res_kOhm'] < (Q1 - sensitivity * IQR)) | (df['Res_kOhm'] > (Q3 + sensitivity * IQR))
    else: # Isolation Forest
        iso = IsolationForest(contamination=sensitivity, random_state=42)
        outliers = iso.fit_predict(df[['Temp_K', 'PPM', 'Res_kOhm']]) == -1

    df['is_outlier'] = outliers
    df_clean = df[~df['is_outlier']].copy()

    # 4. 결과 시각화 및 비교
    st.header(f"📊 {clean_method} 정제 결과 비교")
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.subheader("🔴 탐지 결과 (Outliers Highlighted)")
        fig_out = px.scatter(df, x='Temp_K', y='Res_kOhm', color='is_outlier', 
                             color_discrete_map={True: 'red', False: 'blue'},
                             title="Red: Outliers detected")
        st.plotly_chart(fig_out, use_container_width=True)
        st.write(f"총 데이터: {len(df)}개 | 제거된 이상치: {sum(outliers)}개")

    # 모델 성능 비교
    def get_r2(data):
        if len(data) < 2: return 0
        X = data[['Temp_K', 'PPM']]
        y = data['Res_kOhm']
        return LinearRegression().fit(X, y).score(X, y)

    r2_before = get_r2(df)
    r2_after = get_r2(df_clean)

    with col_plot2:
        st.subheader("✅ 정제 성능 향상도")
        st.metric("Before Cleaning $R^2$", f"{r2_before:.4f}")
        st.metric("After Cleaning $R^2$", f"{r2_after:.4f}", f"{r2_after - r2_before:+.4f}")
        
        # 정제 후 데이터 분포
        fig_clean = px.scatter(df_clean, x='Temp_K', y='Res_kOhm', title="Cleaned Data Only")
        st.plotly_chart(fig_clean, use_container_width=True)

    # 5. 다운로드
    st.divider()
    st.download_button("🚿 정제된 데이터 다운로드", df_clean.to_csv(index=False).encode('utf-8'), "cleaned_sensor_data.csv")

else:
    st.info("👋 데이터 정제를 위해 CSV 파일을 업로드해 주세요.")