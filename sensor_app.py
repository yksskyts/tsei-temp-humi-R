import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 사이킷런 모델 대거 임포트
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. 페이지 설정
st.set_page_config(page_title="Sensor ML Benchmarking", layout="wide")
st.title("🧪 사이킷런 전 모델 비교: 센서 물리 모델링")
st.markdown("모든 주요 ML 모델을 테스트하여 **절대온도(K)**와 **수증기 농도(ppm)**에 대한 최적의 예측 모델을 찾습니다.")

# 2. 사이드바 모델 선택 (10종)
st.sidebar.header("🤖 ML 모델 벤치마킹")
model_dict = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1),
    "Huber Regressor (Robust)": HuberRegressor(),
    "SVR (Support Vector)": SVR(kernel='rbf', C=100, gamma=0.1),
    "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(max_depth=10),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

selected_model_name = st.sidebar.selectbox("테스트할 모델을 선택하세요", list(model_dict.keys()))
model = model_dict[selected_model_name]

# 3. 데이터 로드 및 물리 변환
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
    
    # [물리 변환]
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    df['Temp_K'] = df['온도'] + 273.15
    p_sat = 6.112 * np.exp((17.62 * df['온도']) / (243.12 + df['온도']))
    df['Humidity_ppm'] = ((df['습도'] / 100) * p_sat / 1013.25) * 1_000_000
    
    X = df[['Temp_K', 'Humidity_ppm']]
    y = df['Resistance_kOhm']
    
    # 4. 모델 학습 및 평가
    with st.spinner(f'{selected_model_name} 학습 중...'):
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

    # 5. 리포트 섹션
    st.divider()
    col_rep1, col_rep2 = st.columns([1.5, 1])
    
    with col_rep1:
        st.subheader(f"📝 {selected_model_name} 분석 결과")
        
        # 선형 계수가 있는 모델 (Linear, Ridge, Lasso, Huber 등)
        if hasattr(model, 'coef_'):
            coef = model.coef_.flatten()
            intercept = model.intercept_
            st.info(f"**공식:** $R(k\Omega) = {intercept:.2f} + ({coef[0]:.4f} \\times T_K) + ({coef[1]:.6f} \\times H_{{ppm}})$")
        
        # 중요도 파라미터가 있는 모델 (Tree 기반 앙상블)
        elif hasattr(model, 'feature_importances_'):
            feat_imp = pd.Series(model.feature_importances_, index=['Temp(K)', 'Humidity(ppm)'])
            fig_imp, ax_imp = plt.subplots(figsize=(5, 2))
            feat_imp.sort_values().plot(kind='barh', color=['#3498db', '#e74c3c'], ax=ax_imp)
            ax_imp.set_title("Feature Importance", fontsize=10)
            st.pyplot(fig_imp)
        
        # 그 외 모델 (SVR, KNN 등)
        else:
            st.warning("이 모델은 명시적인 수식이나 피처 중요도를 제공하지 않는 알고리즘입니다.")

    with col_rep2:
        st.subheader("🎯 모델 예측 성능")
        st.metric("결정계수 (R²)", f"{r2:.4f}")
        st.metric("평균 오차 (RMSE)", f"{rmse:.4f} kΩ")

    # 6. 시각화 섹션
    st.divider()
    st.header("📈 예측 성능 시각화")
    
    plt.rcdefaults()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # [좌측] Measured vs Predicted (선형성 확인)
    axes[0].scatter(y, y_pred, alpha=0.2, s=2, color='darkblue')
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0].set_title(f"Measured vs Predicted (R2={r2:.4f})")
    axes[0].set_xlabel("Measured (kOhm)")
    axes[0].set_ylabel("Predicted (kOhm)")

    # [우측] Residuals (오차 분포) - 모델의 건강 상태 확인
    residuals = y - y_pred
    sns.histplot(residuals, kde=True, ax=axes[1], color='purple')
    axes[1].set_title("Residuals Distribution (Error)")
    axes[1].set_xlabel("Error (kOhm)")

    st.pyplot(fig)

    # 7. 전체 모델 성능 비교 (버튼 클릭 시)
    if st.sidebar.button("🏆 전 모델 성능 순위 보기"):
        st.divider()
        st.header("🏆 전 모델 성능 비교 순위")
        results = []
        with st.spinner('전체 모델 벤치마킹 중...'):
            for name, m in model_dict.items():
                m.fit(X, y)
                p = m.predict(X)
                results.append({
                    "Model": name,
                    "R² Score": r2_score(y, p),
                    "RMSE (kΩ)": np.sqrt(mean_squared_error(y, p))
                })
        res_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
        st.table(res_df)

else:
    st.info("👋 센서 데이터 CSV 파일을 업로드하여 벤치마킹을 시작하세요.")