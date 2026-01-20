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

st.title("🧪 센서 정밀 분석 대시보드")
st.markdown("다양한 머신러닝 모델을 통해 **온도와 습도**의 영향력을 정밀하게 분석합니다.")

# 2. 사이드바 - 모델 선택
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

# 3. 파일 업로더
uploaded_file = st.file_uploader("CSV 파일을 여기에 드래그하여 업로드하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    
    X = df[['온도', '습도']]
    y = df['Resistance_kOhm']
    
    # 모델 학습 루틴
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

    with st.spinner(f'{model_choice} 분석 중...'):
        model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # 4. 모델 분석 리포트 섹션
    st.divider()
    st.header(f"📊 분석 리포트: {model_choice.split('. ')[1]}")
    
    col_info1, col_info2 = st.columns([2, 1])
    
    with col_info1:
        # [A] 선형 모델 (Linear, Ridge) -> 수식 출력
        if hasattr(model, 'coef_'):
            st.subheader("📝 Regression Formula")
            st.info(f"**$Resistance(k\Omega) = {model.intercept_:.2f} + ({model.coef_[0]:.4f} \\times Temp) + ({model.coef_[1]:.4f} \\times Humi)$**")
        
        # [B] 비선형 모델 (Tree, RF, GB) -> 변수 중요도 출력 (영문 레이블)
        elif hasattr(model, 'feature_importances_'):
            st.subheader("💡 Feature Importance (Relative Impact)")
            importances = model.feature_importances_
            # 한글 대신 영어 레이블 사용 (Temp, Humi)
            feat_imp = pd.Series(importances, index=['Temp', 'Humi'])
            
            plt.rcdefaults() # 기본 폰트(영문)로 리셋
            fig_imp, ax_imp = plt.subplots(figsize=(7, 3))
            feat_imp.sort_values().plot(kind='barh', color=['#3498db', '#e74c3c'], ax=ax_imp)
            ax_imp.set_title("Feature Importance Analysis", fontsize=12)
            ax_imp.set_xlabel("Importance Score")
            st.pyplot(fig_imp)
            
            st.write(f"이 모델은 **Temp를 {importances[0]*100:.1f}%**, **Humi를 {importances[1]*100:.1f}%** 비중으로 반영했습니다.")

    with col_info2:
        st.subheader("🎯 모델 성능 지표")
        st.metric("결정계수 (R²)", f"{r2:.4f}")
        st.metric("평균 오차 (RMSE)", f"{rmse:.4f} kΩ")

    # 5. 실시간 저항 예측 시뮬레이터
    st.divider()
    st.header("🔍 실시간 저항 예측")
    c_in1, c_in2, c_res = st.columns([1, 1, 2])
    with c_in1:
        input_temp = st.number_input("현재 온도 입력 (°C)", value=float(df['온도'].mean()))
    with c_in2:
        input_humi = st.number_input("현재 습도 입력 (%)", value=float(df['습도'].mean()))
    
    pred_val = model.predict([[input_temp, input_humi]])[0]
    with c_res:
        st.metric(f"예상 저항값 ({model_choice.split('. ')[1]})", f"{pred_val:.4f} kΩ")
        st.caption(f"이 환경에서 센서의 기저 저항은 {pred_val:.2f} kΩ으로 예상됩니다.")

    # 6. 영향도 분석 그래프 (전체 영문 레이블)
    st.divider()
    st.header("📈 상세 영향도 분석 (Visual Analysis)")
    
    plt.rcdefaults()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # [1] Temp Impact
    sns.regplot(ax=axes[0, 0], x='온도', y='Resistance_kOhm', data=df, 
                scatter_kws={'alpha': 0.03, 's': 1, 'color': 'gray'}, line_kws={'color': 'red'})
    axes[0, 0].set_title("Temperature vs Resistance", fontsize=15)
    axes[0, 0].set_xlabel("Temp (C)")
    axes[0, 0].set_ylabel("Resistance (kOhm)")

    # [2] Humi Impact
    sns.regplot(ax=axes[0, 1], x='습도', y='Resistance_kOhm', data=df, 
                scatter_kws={'alpha': 0.03, 's': 1, 'color': 'gray'}, line_kws={'color': 'blue'})
    axes[0, 1].set_title("Humidity vs Resistance", fontsize=15)
    axes[0, 1].set_xlabel("Humi (%)")
    axes[0, 1].set_ylabel("Resistance (kOhm)")

    # [3] Combined Fit (Linearity)
    axes[1, 0].scatter(y, y_pred, alpha=0.1, s=1, color='purple')
    axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[1, 0].set_title(f"Actual vs Predicted (R2={r2:.4f})", fontsize=15)
    axes[1, 0].set_xlabel("Measured (kOhm)")
    axes[1, 0].set_ylabel("Model Predicted (kOhm)")

    # [4] Time-series Tracking
    sample_df = df.iloc[::25]
    axes[1, 1].plot(sample_df['측정 시간'], sample_df['Resistance_kOhm'], label='Measured', alpha=0.4, color='black')
    axes[1, 1].plot(sample_df['측정 시간'], y_pred[::25], label='Predicted', color='limegreen', linestyle='--', lw=2)
    axes[1, 1].set_title("Model Tracking Performance Over Time", fontsize=15)
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # 7. 다운로드 버튼
    st.download_button("최종 분석 데이터(CSV) 다운로드", df.to_csv(index=False).encode('utf-8'), "final_analysis.csv")

else:
    st.info("👋 분석할 센서 데이터 CSV 파일을 업로드해 주세요.")