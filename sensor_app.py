import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 사이킷런 모델 임포트
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. 페이지 설정
st.set_page_config(page_title="Sensor ML Expert Pro", layout="wide")

# 모델 설명 데이터베이스 (벤치마킹용)
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

# --- 사이드바 구성 ---
st.sidebar.title("🛠️ 분석 설정")
app_mode = st.sidebar.radio("분석 기능 선택", ["1. 전 모델 벤치마킹", "2. 노화 진단 및 미래 예측"])

st.sidebar.divider()
st.sidebar.header("📁 데이터 소스")
data_source = st.sidebar.radio("데이터 소스 선택", ["GitHub 기본 데이터 (19.csv)", "사용자 파일 업로드"])

# --- GitHub 데이터 URL 설정 ---
GITHUB_CSV_URL = "https://raw.githubusercontent.com/yksskyts/tsei-temp-humi-R/refs/heads/main/19.csv"

# --- 1. 데이터 로딩 섹션 ---
st.title("🧪 센서 정밀 분석 및 노화 진단 시스템")

df = None

if data_source == "GitHub 기본 데이터 (19.csv)":
    try:
        df = pd.read_csv(GITHUB_CSV_URL)
        st.success(f"✅ GitHub에서 '19.csv' 데이터를 성공적으로 로드했습니다.")
    except Exception as e:
        st.error(f"❌ GitHub 데이터를 불러오지 못했습니다. (사유: {e})")
else:
    uploaded_file = st.file_uploader("분석할 CSV 파일을 업로드하세요", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# 데이터가 존재할 경우에만 이하 로직 실행
if df is not None:
    # 컬럼 공백 제거 및 기본 전처리
    df.columns = [col.strip() for col in df.columns]
    
    # 공통 물리 변환
    df['Resistance_kOhm'] = df['저항'] / 1000.0
    
    # 시간 기반 '경과 일수(Elapsed_Days)' 생성
    if '측정 시간' in df.columns:
        df['측정 시간'] = pd.to_datetime(df['측정 시간'])
        first_time = df['측정 시간'].min()
        df['Elapsed_Days'] = (df['측정 시간'] - first_time).dt.total_seconds() / (24 * 3600)
    else:
        # 시간 컬럼 없을 시 행 인덱스 기반 생성
        df['Elapsed_Days'] = np.arange(len(df)) / (60 * 24)

    # 벤치마킹용 물리 변수 (PPM 변환)
    p_sat = 6.112 * np.exp((17.62 * df['온도']) / (243.12 + df['온도']))
    df['Humidity_ppm'] = ((df['습도'] / 100) * p_sat / 1013.25) * 1_000_000
    df['Temp_K'] = df['온도'] + 273.15

    # ---------------------------------------------------------
    # MODE 1: 전 모델 벤치마킹
    # ---------------------------------------------------------
    if app_mode == "1. 전 모델 벤치마킹":
        st.sidebar.divider()
        st.sidebar.header("🤖 모델 설정")
        selected_model_name = st.sidebar.selectbox("테스트 모델 선택", list(model_info.keys()))

        with st.sidebar.expander("💡 모델 특성", expanded=True):
            info = model_info[selected_model_name]
            st.write(f"**설명:** {info['desc']}")
            st.write(f"✅ **장점:** {info['pros']}")
            st.write(f"❌ **단점:** {info['cons']}")
            st.info(f"🎯 **추천:** {info['best_for']}")

        # 모델 객체 맵핑
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
        X = df[['Temp_K', 'Humidity_ppm']]
        y = df['Resistance_kOhm']
        
        with st.spinner(f'{selected_model_name} 분석 중...'):
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

        st.divider()
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader(f"📝 {selected_model_name} 결과 보고서")
            if hasattr(model, 'coef_'):
                coef = model.coef_.flatten()
                st.info(f"**보정 공식:** $R(k\Omega) = {model.intercept_:.2f} + ({coef[0]:.4f} \\times T_K) + ({coef[1]:.6f} \\times H_{{ppm}})$")
            elif hasattr(model, 'feature_importances_'):
                fig_imp, ax_imp = plt.subplots(figsize=(5, 2))
                pd.Series(model.feature_importances_, index=['Temp(K)', 'Humidity(ppm)']).sort_values().plot(kind='barh', color='#3498db', ax=ax_imp)
                ax_imp.set_title("Feature Importance")
                st.pyplot(fig_imp)
            else:
                st.warning("이 모델은 해석용 수식을 제공하지 않습니다.")

        with col2:
            st.subheader("🎯 성능 지표")
            st.metric("R² Score (정확도)", f"{r2:.4f}")
            st.metric("RMSE (평균 오차)", f"{rmse:.4f} kΩ")

        st.divider()
        st.header("📈 예측 성능 시각화")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.set_theme(style="whitegrid")
        axes[0].scatter(y, y_pred, alpha=0.2, s=2, color='darkblue')
        axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[0].set_title("Measured vs Predicted")
        sns.histplot(y - y_pred, kde=True, ax=axes[1], color='purple')
        axes[1].set_title("Error Distribution (Residuals)")
        st.pyplot(fig)

        if st.sidebar.button("🏆 전체 모델 랭킹 확인"):
            st.divider()
            st.header("🏆 전 모델 성능 비교 순위")
            results = []
            for name, m in model_dict.items():
                m.fit(X, y); p = m.predict(X)
                results.append({"Model": name, "R²": r2_score(y, p), "RMSE": np.sqrt(mean_squared_error(y, p))})
            st.table(pd.DataFrame(results).sort_values(by="R²", ascending=False))

    # ---------------------------------------------------------
    # MODE 2: 노화 진단 및 미래 예측
    # ---------------------------------------------------------
    elif app_mode == "2. 노화 진단 및 미래 예측":
        st.sidebar.divider()
        st.sidebar.header("🤖 알고리즘 설정")
        model_choice = st.sidebar.selectbox("예측 모델 선택", ["1. Linear Regression", "2. Ridge Regression", "3. Decision Tree", "4. Random Forest", "5. Gradient Boosting"])
        st.sidebar.warning("⚠️ 미래 예측 시뮬레이션은 1, 2번 선형 모델을 권장합니다.")

        X_cols = ['온도', '습도', 'Elapsed_Days']
        X = df[X_cols]
        y = df['Resistance_kOhm']
        
        if "1." in model_choice: model = LinearRegression()
        elif "2." in model_choice: model = Ridge(alpha=1.0)
        elif "3." in model_choice: model = DecisionTreeRegressor(max_depth=10)
        elif "4." in model_choice: model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        elif "5." in model_choice: model = GradientBoostingRegressor(n_estimators=50, random_state=42)

        with st.spinner('노화 데이터 분석 중...'):
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

        st.divider()
        col_a, col_b = st.columns([1.5, 1])
        with col_a:
            st.subheader("📊 센서 노화(Aging) 진단")
            # 노화율 계산을 위한 별도 선형 분석
            aging_model = LinearRegression().fit(X, y)
            drift = aging_model.coef_[2]
            if drift > 0:
                st.warning(f"⚠️ **상태: 저항 상승형 열화 발생**\n\n하루 평균 **{drift:.4f} kΩ**씩 저항값이 증가하고 있습니다.")
            else:
                st.success(f"✅ **상태: 안정화 진행 중**\n\n하루 평균 **{abs(drift):.4f} kΩ**씩 저항이 하강하며 안정화되고 있습니다.")
            
            if hasattr(model, 'coef_'):
                st.info(f"**물리 공식:** $R = {model.intercept_:.2f} + ({model.coef_[0]:.4f} \cdot T) + ({model.coef_[1]:.4f} \cdot H) + ({model.coef_[2]:.4f} \cdot Day)$")
            elif hasattr(model, 'feature_importances_'):
                fig_imp2, ax_imp2 = plt.subplots(figsize=(5, 2.2))
                pd.Series(model.feature_importances_, index=['Temp', 'Humi', 'Aging']).sort_values().plot(kind='barh', color='#2ecc71', ax=ax_imp2)
                ax_imp2.set_title("Variable Impact")
                st.pyplot(fig_imp2)

        with col_b:
            st.subheader("🎯 예측 정확도")
            st.metric("R² (결정계수)", f"{r2:.4f}")
            st.metric("RMSE (오차 범위)", f"{rmse:.4f} kΩ")

        st.divider()
        st.header("🔮 미래 기저 저항 예측")
        sc1, sc2, sc3, sr = st.columns([1, 1, 1, 2])
        with sc1: st_temp = st.number_input("설정 온도 (°C)", value=float(df['온도'].mean()))
        with sc2: st_humi = st.number_input("설정 습도 (%)", value=float(df['습도'].mean()))
        with sc3: st_days = st.number_input("추가 사용일수", value=30)
        
        future_day = df['Elapsed_Days'].max() + st_days
        future_pred = model.predict(pd.DataFrame([[st_temp, st_humi, future_day]], columns=X_cols))[0]
        with sr:
            st.metric(f"{st_days}일 후 예상 저항", f"{future_pred:.4f} kΩ")
            st.write(f"현재 마지막 측정값 대비 변화량: **{future_pred - df['Resistance_kOhm'].iloc[-1]:+.4f} kΩ**")

        st.divider()
        st.header("📈 데이터 시각화 분석")
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        sns.regplot(ax=axes2[0, 0], x='온도', y='Resistance_kOhm', data=df, scatter_kws={'alpha': 0.05, 's': 1}, line_kws={'color': 'red'})
        axes2[0, 0].set_title("Temperature Impact")
        
        # 순수 드리프트 시각화
        pure_drift = df['Resistance_kOhm'] - (aging_model.coef_[0]*df['온도'] + aging_model.coef_[1]*df['습도'] + aging_model.intercept_)
        axes2[0, 1].scatter(df['Elapsed_Days'], pure_drift, alpha=0.1, s=1, color='orange')
        axes2[0, 1].set_title("Pure Aging Drift (T/H Removed)")
        
        axes2[1, 0].scatter(y, y_pred, alpha=0.1, s=1, color='green')
        axes2[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        axes2[1, 0].set_title("Model Linearity")
        
        sample = df.iloc[::20]
        axes2[1, 1].plot(sample['측정 시간'], sample['Resistance_kOhm'], label='Actual', alpha=0.6)
        axes2[1, 1].plot(sample['측정 시간'], y_pred[::20], label='ML', linestyle='--', color='lime')
        axes2[1, 1].legend()
        axes2[1, 1].set_title("Time-series Tracking")
        plt.tight_layout()
        st.pyplot(fig2)

    st.divider()
    st.download_button("📂 분석 데이터 다운로드", df.to_csv(index=False).encode('utf-8'), "sensor_analysis.csv")

else:
    st.info("👋 데이터를 불러오는 중입니다. GitHub URL을 확인하거나 파일을 업로드해 주세요.")git push origin main