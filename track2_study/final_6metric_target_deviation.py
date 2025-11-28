import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 새 창 띄우기 설정 유지
# matplotlib.use('Agg') 

# -------------------------------------------------------
# 1. 환경 설정 및 데이터 생성
# -------------------------------------------------------
NUM_ZERNIKE = 30 
NUM_POINTS = 300 
np.random.seed(42) 

# ★★★★★ 이 리스트를 수정하여 원하는 Metric을 선택하세요! ★★★★★
METRICS_TO_PLOT = ['Average_X', '3Sigma_X', 'Residual_Y', 'Average_Y'] 
# ★★★★★ (최대 4개까지 선택 가능) ★★★★★

def generate_data(n_points):
    """6가지 Overlay 지표를 모두 포함한 데이터를 생성합니다."""
    # (데이터 생성 로직은 동일)
    df = pd.DataFrame()
    z_cols = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]
    for col_name in z_cols:
        df[col_name] = np.random.normal(0, np.random.uniform(1, 3), n_points)
    
    noise = lambda: np.random.normal(0, 0.1, n_points)
    noise_res = np.random.normal(0, 0.05, NUM_POINTS)

    df['Average_X'] = df['Z2'] * 1.5 + df['Z8'] * 0.4 + noise()
    df['Average_Y'] = df['Z3'] * 1.5 + df['Z7'] * 0.4 + noise()
    df['3Sigma_X'] = np.abs(df['Z5'] * 0.6 + df['Z15'] * 0.4) + np.random.normal(0.5, 0.1, n_points)
    df['3Sigma_Y'] = np.abs(df['Z9'] * 0.6 + df['Z10'] * 0.4) + np.random.normal(0.5, 0.1, n_points)
    df['Residual_X'] = df['Z20'] * 0.2 + noise_res
    df['Residual_Y'] = df['Z25'] * 0.2 + noise_res
    
    return df, z_cols

def analyze_and_get_attribution(df, z_cols, target_metric):
    """단일 Metric에 대한 MLR 학습 및 Error Attribution 벡터를 반환합니다."""
    
    # 1. 모델 학습 (Reference Model 생성)
    model = LinearRegression()
    model.fit(df[z_cols], df[target_metric])

    # 2. Reference (기준) 추출: MLR 계수 (Sensitivity)
    ref_coefficients = pd.Series(model.coef_, index=z_cols)
    
    # 3. 새로운 측정값 (Actual) 시뮬레이션
    new_z_vector = np.random.normal(0, 1, NUM_ZERNIKE)
    new_z_vector[1] = 5.0  # Z2 (Tilt X)를 고의로 높게 만듦
    new_z_series = pd.Series(new_z_vector, index=z_cols)

    # 4. Attribution: Contribution = Reference * Actual
    return ref_coefficients * new_z_series


def run_consolidated_dashboard():
    """선택된 Metrics에 대한 최종 Attribution 결과를 2x2 Subplots으로 통합 시각화."""
    
    if not METRICS_TO_PLOT:
        print("경고: METRICS_TO_PLOT 리스트에 분석할 지표를 최소 1개 입력하세요.")
        return

    # 데이터 생성 및 변수 추출
    df, z_cols = generate_data(NUM_POINTS)
    
    # 그래프 크기 축소 (사용자 요청 반영: 24인치 -> 12인치급으로)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10)) 
    axes = axes.flatten()
    
    x_labels = z_cols
    
    print("---------------------------------------------------------------------")
    print(f"🔄 선택된 {len(METRICS_TO_PLOT)}개 지표 통합 분석 시작...")
    print("---------------------------------------------------------------------")

    for i, metric in enumerate(METRICS_TO_PLOT):
        if i >= 4: # 2x2 = 4개이므로, 4개를 초과하면 종료
            break

        print(f"   [Processing] {metric} 분석 및 플롯 중...")
        
        # 1. Attribution 벡터 추출
        attribution_vector = analyze_and_get_attribution(df, z_cols, metric)
        
        # 2. 해당 Subplot에 결과 그리기
        ax = axes[i]
        
        sns.barplot(x=x_labels, y=attribution_vector.values, ax=ax, 
                    palette=np.where(attribution_vector.values > 0, 'Reds_d', 'Blues_d')[0])
        
        ax.axhline(0, color='black', linewidth=0.8) 
        ax.set_title(f"Attribution to: {metric}", fontsize=11, color='red' if 'Residual' in metric else 'black')
        ax.set_xticklabels(x_labels, rotation=90, fontsize=6) # 폰트 크기 대폭 축소
        ax.set_ylabel("Error Contribution (nm)", fontsize=8) 
        
    print("\n🎉 통합 진단 대시보드 새 창으로 표시.")
    
    plt.suptitle("Consolidated Zernike Error Attribution (Selected Metrics)", fontsize=14, fontweight='bold')
    plt.tight_layout() 
    
    plt.show() # 새 창으로 띄웁니다.


if __name__ == "__main__":
    run_consolidated_dashboard()