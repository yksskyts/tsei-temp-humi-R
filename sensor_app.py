import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 사이킷런 모델 임포트
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. 페이지 설정
st.set_page_config(page_title="Sensor ML Benchmarking", layout="wide")

# 모델 설명 데이터베이스
model_info = {
    "Linear Regression": {
        "desc": "가장 기본적인 모델로, 온도/농도와 저항이 직선 관계라고 가정합니다.",
        "pros": "계산이 빠르고 수식($y=ax+b$)이 명확해 물리적 해석이 가장 쉽습니다.",
        "cons": "복잡한 비선형 데이터(곡선 등)는 잘 맞추지 못합니다.",
        "best_for": "기본 보정식 산출용"
    },
    "Ridge Regression": {
        "desc": "선형 회귀에 '규제'를 더해 모델이 너무 복잡해지는 것을 막습니다.",
        "pros": "데이터에 노이즈가 많을 때 선형 회귀보다 안정적입니다.",
        "cons": "여전히 직선적인 관계만 학습 가능합니다.",
        "best_for": "데이터가 적고 노이즈가 있을 때"
    },
    "Lasso Regression": {
        "desc": "중요하지 않은 변수의 영향력을 0으로 만들어버리는 규제 모델입니다.",
        "pros": "변수가 많을 때 정말 중요한 변수가 무엇인지 걸러내기 좋습니다.",
        "cons": "변수가 적은 경우 일반 선형 모델과 큰 차이가 없습니다.",
        "best_for": "변수 선택 및 간소화"
    },
    "ElasticNet": {
        "desc": "Ridge와 Lasso의 장점을 합친 하이브리드 모델입니다.",
        "pros": "여러 변수가 서로 얽혀 있을 때(다중공선성) 효과적입니다.",
        "cons": "설정해야 할 파라미터가 많아 다루기 까다롭습니다.",
        "best_for": "복합적인 환경 변수 처리"
    },
    "Huber Regressor (Robust)": {
        "desc": "이상치(튀는 값)에 매우 강한 선형 모델입니다.",
        "pros": "센서 노이즈나 일시적 튀는 값에 휘둘리지 않고 대세 수식을 찾습니다.",
        "cons": "데이터 전체가 깨끗하다면 일반 선형 모델보다 느릴 수 있습니다.",
        "best_for": "전기적 노이즈가 심한 데이터"
    },
    "SVR (Support Vector)": {
        "desc": "데이터를 고차원으로 보내 복잡한 경계선을 찾는 모델입니다.",
        "pros": "비선형적인 센서 반응을 매우 정교하게 잡아냅니다.",
        "cons": "데이터 양이 너무 많으면 학습 속도가 급격히 느려집니다.",
        "best_for": "정밀한 비선형 보정"
    },
    "K-Neighbors Regressor": {
        "desc": "현재 조건과 가장 비슷한 과거 데이터 n개를 찾아 평균을 냅니다.",
        "pros": "데이터 분포를 몰라도 직관적으로 잘 맞춥니다.",
        "cons": "수식이 나오지 않아 하드웨어 이식이 불가능합니다.",
        "best_for": "단순 예측 및 성능 비교"
    },
    "Decision Tree": {
        "desc": "스무고개 방식으로 데이터를 분류하여 예측합니다.",
        "pros": "데이터의 흐름을 시각적으로 이해하기 매우 쉽습니다.",
        "cons": "과적합(Overfitting)되기 쉬워 새로운 데이터에 약할 수 있습니다.",
        "best_for": "데이터 규칙성 파악"
    },
    "Random Forest": {
        "desc": "수많은 결정 나무를 만들어 집단지성으로 결과를 냅니다.",
        "pros": "대부분의 센서 데이터에서 가장 안정적이고 높은 성능을 보입니다.",
        "cons": "모델의 용량이 크고 내부 연산 과정을 이해하기 어렵습니다.",
        "best_for": "범용적인 고성능 분석"
    },
    "Extra Trees": {
        "desc": "Random Forest보다 더 무작위성을 부여해 속도를 높인 모델입니다.",
        "pros": "이상치에 더 강하고 Random Forest보다 계산이 빠릅니다.",
        "cons": "때때로 Random Forest보다 오차가 클 수 있습니다.",
        "best_for": "빠르고 강건한 앙상블 학습"
    },
    "AdaBoost": {
        "desc": "약한 모델들을 순차적으로 학습시켜 이전의 실수를 보완합니다.",
        "pros": "단순한 모델들을 모아 강력한 성능을 끌어냅니다.",
        "cons": "노이즈가 너무 심하면 오히려 성능이 망가질 수 있습니다.",
        "best_for": "단계적 오차 수정"
    },
    "Gradient Boosting": {
        "desc": "현재 가장 널리 쓰이는 강력한 부스팅 알고리즘입니다.",
        "pros": "예측 정확도가 가장 높은 편에 속합니다.",
        "cons": "학습 시간이 길고 파라미터 튜닝이 필수적입니다.",
        "best_for": "정확도 최우선 시"
    }
}

st.title("🧪 사이킷런 전 모델 비교: 센서 물리 모델링")

# 2. 사이드바 모델 선택
st.sidebar.header("🤖 ML 모델 벤치마킹")
selected_model_name = st.sidebar.selectbox("테스트할 모델을 선택하세요", list(model_info.keys()))

# --- 선택된 모델 특징 표시 (전문가님 요청 사항) ---
with st.sidebar.expander("💡 선택된 모델 특성 보기", expanded=True):
    info = model_info[selected_model_name]
    st.markdown(f"**한줄평:** {info['desc']}")
    st.markdown(f"✅ **장점:** {info['pros']}")
    st.markdown(f"❌ **단점:** {info['cons']}")
    st.success(f"🎯 **추천:** {info['best_for']}")

# 모델 객체 생성
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
model = model_dict[selected_model_name]

# (이후 3번~7번 데이터 로드 및 시각화 로직은 기존과 동일)
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    
    # [물리 변환]
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    df['Temp_K'] = df['온도'] + 273.15
    p_sat = 6.112 * np.exp((17.62 * df['온도']) / (243.12 + df['온도']))
    df['Humidity_ppm'] = ((df['습도'] / 100) * p_sat / 1013.25) * 1_000_000
    
    X = df[['Temp_K', 'Humidity_ppm']]
    y = df['Resistance_kOhm']
    
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
        if hasattr(model, 'coef_'):
            coef = model.coef_.flatten()
            intercept = model.intercept_
            st.info(f"**공식:** $R(k\Omega) = {intercept:.2f} + ({coef[0]:.4f} \\times T_K) + ({coef[1]:.6f} \\times H_{{ppm}})$")
        elif hasattr(model, 'feature_importances_'):
            feat_imp = pd.Series(model.feature_importances_, index=['Temp(K)', 'Humidity(ppm)'])
            fig_imp, ax_imp = plt.subplots(figsize=(5, 2))
            feat_imp.sort_values().plot(kind='barh', color=['#3498db', '#e74c3c'], ax=ax_imp)
            ax_imp.set_title("Feature Importance", fontsize=10)
            st.pyplot(fig_imp)
        else:
            st.warning("이 모델은 명시적인 수식이나 피처 중요도를 제공하지 않습니다.")

    with col_rep2:
        st.subheader("🎯 모델 예측 성능")
        st.metric("결정계수 ($R^2$)", f"{r2:.4f}")
        st.metric("평균 오차 (RMSE)", f"{rmse:.4f} kΩ")

    # 6. 시각화
    st.divider()
    st.header("📈 예측 성능 시각화")
    plt.rcdefaults()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(y, y_pred, alpha=0.2, s=2, color='darkblue')
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0].set_title(f"Measured vs Predicted ($R^2$={r2:.4f})")
    axes[0].set_xlabel("Measured (kOhm)")
    axes[0].set_ylabel("Predicted (kOhm)")
    residuals = y - y_pred
    sns.histplot(residuals, kde=True, ax=axes[1], color='purple')
    axes[1].set_title("Residuals Distribution (Error)")
    st.pyplot(fig)

    # 7. 🏆 전 모델 성능 비교 순위 (안내 멘트 추가)
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
                    "RMSE (kΩ)": np.sqrt(mean_squared_error(y, p)),
                    "Best For": model_info[name]['best_for'] # 순위표에도 특징 추가
                })
        res_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
        st.table(res_df)
else:
    st.info("👋 센서 데이터 CSV 파일을 업로드하여 벤치마킹을 시작하세요.")