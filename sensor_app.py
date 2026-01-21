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
st.set_page_config(page_title="Sensor ML Expert Pro", layout="wide")

st.title("🧪 센서 정밀 분석 및 노화 진단 시스템")
st.markdown("온도, 습도뿐만 아니라 **시간 경과에 따른 센서의 열화 상태**를 머신러닝으로 정밀 진단합니다.")

# 2. 사이드바 - 모델 알고리즘 선택
st.sidebar.header("🤖 모델 알고리즘 설정")
model_choice = st.sidebar.selectbox(
    "적용할 모델을 선택하세요",
    [
        "1. Linear Regression (선형)", 
        "2. Ridge Regression (규제 선형)", 
        "3. Decision Tree (의사결정 나무)", 
        "4. Random Forest (랜덤 포레스트)", 
        "5. Gradient Boosting (그래디언트 부스팅)"
    ]
)

# 3. 데이터 로드 및 전처리
uploaded_file = st.file_uploader("CSV 파일을 여기에 드래그하여 업로드하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    
    # 시간 데이터 처리 및 '경과 일수' 변수 생성
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
        first_time = df['측정 시간'].min()
        df['Elapsed_Days'] = (df['측정 시간'] - first_time).dt.total_seconds() / (24 * 3600)
    else:
        # 시간 컬럼 없을 경우 인덱스를 일자로 가정
        df['Elapsed_Days'] = np.arange(len(df)) / (60 * 24) 
        
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    
    # 모델 학습용 데이터 (온도, 습도, 경과시간 3개 변수 사용)
    X = df[['온도', '습도', 'Elapsed_Days']]
    y = df['Resistance_kOhm']
    
    # 4. 모델 학습 루틴
    if "1." in model_choice: model = LinearRegression()
    elif "2." in model_choice: model = Ridge(alpha=1.0)
    elif "3." in model_choice: model = DecisionTreeRegressor(max_depth=10)
    elif "4." in model_choice: model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    elif "5." in model_choice: model = GradientBoostingRegressor(n_estimators=50, random_state=42)

    with st.spinner(f'{model_choice} 학습 및 분석 중...'):
        model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # 5. 분석 리포트 및 노화 진단
    st.divider()
    col_rep1, col_rep2 = st.columns([1.5, 1])
    
    with col_rep1:
        st.subheader("📊 센서 상태 및 열화 진단")
        
        # 선형 계수 추출 (노화율 산출용)
        # 비선형 모델이라도 노화 경향성 파악을 위해 선형 회귀를 별도로 계산
        aging_analyzer = LinearRegression().fit(X, y)
        degradation_rate = aging_analyzer.coef_[2] # Elapsed_Days의 계수
        
        if degradation_rate > 0:
            st.warning(f"⚠️ **현재 상태: 열화 진행 중 (저항 증가)**")
            st.write(f"시간이 지남에 따라 기저 저항이 하루 평균 **{degradation_rate:.4f} kΩ**씩 상승하고 있습니다.")
        else:
            st.success(f"✅ **현재 상태: 안정화/활성화 중 (저항 감소)**")
            st.write(f"시간이 지남에 따라 기저 저항이 하루 평균 **{abs(degradation_rate):.4f} kΩ**씩 낮아지며 안정화되고 있습니다.")
            
        # 수식 또는 중요도 표시
        if hasattr(model, 'coef_'):
            st.info(f"**Regression Formula:** $R = {model.intercept_:.2f} + ({model.coef_[0]:.4f} \cdot T) + ({model.coef_[1]:.4f} \cdot H) + ({model.coef_[2]:.4f} \cdot Day)$")
        elif hasattr(model, 'feature_importances_'):
            plt.rcdefaults()
            fig_imp, ax_imp = plt.subplots(figsize=(5, 2.5))
            feat_imp = pd.Series(model.feature_importances_, index=['Temp', 'Humi', 'Aging'])
            feat_imp.sort_values().plot(kind='barh', color='#3498db', ax=ax_imp)
            ax_imp.set_title("Feature Importance (Relative Impact)", fontsize=10)
            st.pyplot(fig_imp)

    with col_rep2:
        st.subheader("🎯 모델 예측 성능")
        st.metric("결정계수 (R²)", f"{r2:.4f}")
        st.metric("예측 오차 (RMSE)", f"{rmse:.4f} kΩ")
        st.write(f"총 데이터: {len(df):,} Samples")

    # 6. 실시간 미래 예측 시뮬레이터
    st.divider()
    st.header("🔮 미래 저항 예측 시뮬레이터")
    st.write("특정 시점과 온습도 환경을 가정하여 센서의 기저 저항을 예측합니다.")
    
    s_col1, s_col2, s_col3, s_res = st.columns([1, 1, 1, 2])
    with s_col1:
        s_temp = st.number_input("예상 온도 (°C)", value=float(df['온도'].mean()))
    with s_col2:
        s_humi = st.number_input("예상 습도 (%)", value=float(df['습도'].mean()))
    with s_col3:
        s_days = st.number_input("경과 일수 (오늘 기준 +N일)", value=30)
    
    target_day = df['Elapsed_Days'].max() + s_days
    future_val = model.predict([[s_temp, s_humi, target_day]])[0]
    
    with s_res:
        st.metric(f"{s_days}일 후 예상 저항", f"{future_val:.4f} kΩ")
        st.caption(f"예측 범위(RMSE 적용): {future_val-rmse:.2f} ~ {future_val+rmse:.2f} kΩ")

    # 7. 시각화 섹션 (4단 구성, 영문 레이블)
    st.divider()
    st.header("📈 영향도 및 성능 상세 분석 (Visual Analysis)")
    
    plt.rcdefaults()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # [1] Temperature Impact
    sns.regplot(ax=axes[0, 0], x='온도', y='Resistance_kOhm', data=df, 
                scatter_kws={'alpha': 0.02, 's': 1, 'color': 'gray'}, line_kws={'color': 'red'})
    axes[0, 0].set_title("Temp vs Resistance", fontsize=12)

    # [2] Aging Drift (Normalized)
    # 온습도 영향을 뺀 순수 시간 드리프트 시각화
    temp_humi_effect = aging_analyzer.coef_[0] * df['온도'] + aging_analyzer.coef_[1] * df['습도'] + aging_analyzer.intercept_
    drift_only = df['Resistance_kOhm'] - temp_humi_effect
    axes[0, 1].scatter(df['Elapsed_Days'], drift_only, alpha=0.05, s=1, color='orange')
    axes[0, 1].set_title("Pure Aging Drift (T/H Removed)", fontsize=12)
    axes[0, 1].set_xlabel("Elapsed Days")

    # [3] Linearity Fit
    axes[1, 0].scatter(y, y_pred, alpha=0.1, s=1, color='purple')
    axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5)
    axes[1, 0].set_title(f"Actual vs Predicted (R2={r2:.4f})", fontsize=12)

    # [4] Time-series Tracking
    sample_df = df.iloc[::30]
    axes[1, 1].plot(sample_df['측정 시간'], sample_df['Resistance_kOhm'], label='Measured', alpha=0.5, color='black', lw=1)
    axes[1, 1].plot(sample_df['측정 시간'], y_pred[::30], label='ML Predicted', color='limegreen', linestyle='--', lw=1.5)
    axes[1, 1].set_title("Real-time Tracking Performance", fontsize=12)
    axes[1, 1].legend(prop={'size': 8})

    plt.tight_layout()
    st.pyplot(fig)

    # 8. 데이터 다운로드
    st.download_button("최종 분석 데이터 받기", df.to_csv(index=False).encode('utf-8'), "sensor_expert_analysis.csv")

else:
    st.info("👋 분석할 센서 데이터 CSV 파일을 업로드해 주세요.")