import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression # 분석 로직은 그대로 유지

# -------------------------------------------------------
# 1. 설정 및 데이터 생성/분석 로직 (기존 코드와 동일)
# -------------------------------------------------------
NUM_ZERNIKE = 30 
np.random.seed(42) 
TARGET_VALUE = 0.0 # 이상적인 목표 수차 값
TARGET_METRICS = ['Average_X', 'Average_Y', '3Sigma_X', '3Sigma_Y', 'Residual_X', 'Residual_Y']
Z_COLS = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]


@st.cache_data # 데이터와 Zernike 측정값은 한 번만 계산하여 성능 최적화
def generate_zernike_data():
    """새로운 측정 웨이퍼의 Zernike 값을 시뮬레이션하고 반환합니다."""
    
    measured_z_values = np.random.normal(0, 0.8, NUM_ZERNIKE)
    measured_z_values[1] = 4.5  # Z2 (Tilt X)에 심각한 편차
    measured_z_values[7] = -3.0 # Z8 (Coma)에 편차
    measured_z_values[19] = 2.5 # Z20 (고차항)에 편차
    
    # 멘티님의 분석에서는 MLR이 필요 없지만, MLR 로직을 가진 동료들을 위해 더미 데이터는 만듭니다.
    df = pd.DataFrame(np.random.rand(10, NUM_ZERNIKE), columns=Z_COLS) 
    
    return pd.Series(measured_z_values, index=Z_COLS)


def get_target_deviation_vector(measured_z_series):
    """Target Z (0) 대비 Actual Z의 편차 벡터를 계산합니다."""
    
    # Reference (기준) 설정: 모든 수차는 0이어야 함 (이상적인 렌즈)
    target_z_series = pd.Series([TARGET_VALUE] * NUM_ZERNIKE, index=Z_COLS)

    # Result: Target - Actual 계산 (수차 편차 벡터)
    return target_z_series - measured_z_series


def plot_zernike_bar(data_series, title, color='tab:blue', ylabel='Value (nm)', ax=None):
    """Zernike 막대 그래프를 그리는 Matplotlib 함수"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    sns.barplot(x=data_series.index, y=data_series.values, ax=ax, 
                palette=np.where(data_series.values > 0, 'Reds_d', 'Blues_d')[0])
    
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis='x', rotation=90, labelsize=7)
    
    return ax.figure, ax

# -------------------------------------------------------
# 2. Streamlit 메인 앱 구동
# -------------------------------------------------------

st.set_page_config(layout="wide")
st.title("🎯 Zernike Target Deviation Dashboard (인터랙티브)")
st.caption("체크박스를 통해 보고 싶은 Overlay 지표의 편차(Deviation)를 동시에 비교합니다.")

# 1. 사이드바에 체크박스 구현 (Metric 선택)
st.sidebar.header("📊 분석 지표 선택")
selected_metrics = st.sidebar.multiselect(
    'Overlay 지표 선택 (최대 4개 추천):', 
    options=TARGET_METRICS, 
    default=['Average_X', '3Sigma_X', 'Residual_Y']
)

# 데이터 준비
measured_z_series = generate_zernike_data()
deviation_vector = get_target_deviation_vector(measured_z_series) # 편차는 한 번만 계산

# 2. Actual Zernike Input 플롯 (모든 Metric의 입력값이 동일하므로 한 번만 표시)
st.subheader("1. 렌즈 입력값 (Actual Zernike Input)")
st.caption("새 웨이퍼에서 측정된 Zernike 값입니다. 이것이 모든 지표의 '입력'이 됩니다.")
fig_actual, _ = plot_zernike_bar(
    measured_z_series, 
    "Actual Zernike Input (Measured Value)", 
    color='tab:green', 
    ylabel='Measured Z Value (nm)'
)
st.pyplot(fig_actual, use_container_width=True) # Streamlit에 플롯 표시


# 3. Deviation (편차) 결과 비교 (선택된 Metric만)
st.subheader("2. 수차 목표 편차 (Target Deviation Result Comparison)")
st.caption("편차 = (이상적인 목표 Zernike 0) - (Actual Zernike Value)")

if selected_metrics:
    # 2x2 그리드 레이아웃 설정 (최대 4개)
    num_plots = min(len(selected_metrics), 4)
    cols = st.columns(num_plots)
    
    for i in range(num_plots):
        metric = selected_metrics[i]
        
        with cols[i]:
            st.markdown(f"**🔴 {metric} 편차 분석**")
            
            # Matplotlib Figure를 Streamlit의 각 컬럼에 할당
            fig, ax = plt.subplots(figsize=(6, 4)) # 개별 그래프 크기 축소
            
            # 편차 벡터를 해당 Metric의 제목으로 다시 플롯
            plot_zernike_bar(
                deviation_vector, 
                f"Deviation for {metric}", 
                ylabel='Deviation (nm)', 
                ax=ax
            )
            ax.tick_params(axis='x', rotation=90, labelsize=5) # 폰트 더 축소
            
            # Streamlit에 플롯 표시
            st.pyplot(fig, use_container_width=True)
else:
    st.info("사이드바에서 분석할 Overlay 지표를 하나 이상 선택하세요.")