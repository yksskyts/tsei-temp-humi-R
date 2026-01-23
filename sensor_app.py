import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. 페이지 설정
st.set_page_config(page_title="Sensor ppm Calibration", layout="wide")
st.title("🎯 센서 농도(ppm) & 절대온도(K) 환경 보정식 산출")

# 2. 데이터 업로드
st.sidebar.header("📁 데이터 설정")
uploaded_file = st.sidebar.file_uploader("농도 데이터가 포함된 CSV를 업로드하세요", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    
    # 데이터 변환: 절대온도(K) 및 저항(kOhm)
    if '온도' in df.columns:
        df['Temp_K'] = df['온도'] + 273.15
    if '저항' in df.columns:
        df['Res_kOhm'] = df['저항'] / 1000.0
    
    st.header("1️⃣ 변수 및 모델 설정")
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    
    with col_sel1:
        # 농도(ppm) 컬럼 선택
        ppm_col = st.selectbox("농도(ppm) 데이터 컬럼 선택", df.columns.tolist(), 
                               index=df.columns.get_loc('농도') if '농도' in df.columns else 0)
    with col_sel2:
        # 보정 대상(저항) 선택
        target_y = st.selectbox("보정 대상(Y) 선택", ['Res_kOhm'] + df.columns.tolist())
    with col_sel3:
        # 다항식 차수
        poly_degree = st.slider("보정식 차수 (2차 권장)", 1, 3, 2)

    # 3. 다항 보정식 계산 로직 (X1: 절대온도, X2: 농도)
    X = df[['Temp_K', ppm_col]]
    y = df[target_y]

    # 다항 특성 생성 (K, ppm, K^2, ppm^2, K*ppm 등)
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(['K', 'ppm'])
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # 4. 최종 보정 수식 출력
    st.divider()
    st.header("2️⃣ 최종 보정 공식 (Concentration-Temperature Formula)")
    
    intercept = model.intercept_
    coeffs = model.coef_
    
    # LaTeX 수식 조합
    formula_parts = [f"{intercept:.4f}"]
    for coef, name in zip(coeffs, feature_names):
        # 수식 내 기호 정리
        clean_name = name.replace(" ", " \cdot ")
        formula_parts.append(f"({coef:.6f} \cdot {clean_name})")
    
    full_formula = " + ".join(formula_parts)
    st.latex(f"R_{{predicted}} = {full_formula}")
    
    st.info("💡 **수식 해석:** 위 공식은 주어진 절대온도(K)와 농도(ppm)에서 센서가 나타낼 '예상 기저 저항'을 계산합니다.")

    # 5. 보정 성능 시각화 (Compensation)
    st.divider()
    st.header("3️⃣ 온도 간섭 보정 결과 (Environmental Compensation)")
    
    y_pred = model.predict(X_poly)
    # 보정된 저항값 = 실제 저항 - (온도/농도에 의한 변동분) + 기준값
    df['Corrected_Res'] = y - y_pred + y.mean()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("보정 전 (Raw Resistance vs Temp)")
        fig_raw, ax_raw = plt.subplots()
        ax_raw.scatter(df['Temp_K'], y, alpha=0.3, color='gray', label='Raw Data')
        ax_raw.set_xlabel("Absolute Temperature (K)")
        ax_raw.set_ylabel("Resistance (kOhm)")
        st.pyplot(fig_raw)
        
    with c2:
        st.subheader("보정 후 (Compensated Signal)")
        fig_comp, ax_comp = plt.subplots()
        ax_comp.scatter(df['Temp_K'], df['Corrected_Res'], alpha=0.3, color='red', label='Compensated')
        ax_comp.set_xlabel("Absolute Temperature (K)")
        ax_comp.set_ylabel("Normalized Signal (kOhm)")
        # 보정이 잘 되었다면 온도 변화에도 y축 값이 일정하게 유지됨
        st.pyplot(fig_comp)

    # 데이터 다운로드
    st.download_button("보정된 데이터 다운로드 (CSV)", df.to_csv(index=False).encode('utf-8'), "calibrated_sensor_data.csv")

else:
    st.info("👋 농도(ppm) 데이터가 포함된 CSV 파일을 업로드해 주세요.")