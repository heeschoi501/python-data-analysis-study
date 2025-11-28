import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Matplotlib Agg 백엔드 사용을 비활성화 (새 창으로 띄우기 위함)
# matplotlib.use('Agg') 

# -------------------------------------------------------
# 1. 환경 설정 및 데이터 생성 (6가지 Target Metric 모두 생성)
# -------------------------------------------------------
NUM_ZERNIKE = 30 
NUM_POINTS = 300 
np.random.seed(42) 

TARGET_METRICS = ['Average_X', 'Average_Y', '3Sigma_X', '3Sigma_Y', 'Residual_X', 'Residual_Y']

def generate_data(n_points):
    """6가지 Overlay 지표를 포함하여 데이터를 생성합니다."""
    
    df = pd.DataFrame()
    z_cols = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]

    for col_name in z_cols:
        df[col_name] = np.random.normal(0, np.random.uniform(1, 3), n_points)
    
    # 6가지 Overlay 지표 계산 (물리적 모델링: Z2/Z3=Avg, Z5/Z9=Sigma, Z20/Z25=Res)
    noise = lambda: np.random.normal(0, 0.1, n_points)
    noise_res = np.random.normal(0, 0.05, NUM_POINTS)

    df['Average_X'] = df['Z2'] * 1.5 + df['Z8'] * 0.4 + noise()
    df['Average_Y'] = df['Z3'] * 1.5 + df['Z7'] * 0.4 + noise()
    df['3Sigma_X'] = np.abs(df['Z5'] * 0.6 + df['Z15'] * 0.4) + np.random.normal(0.5, 0.1, n_points)
    df['3Sigma_Y'] = np.abs(df['Z9'] * 0.6 + df['Z10'] * 0.4) + np.random.normal(0.5, 0.1, n_points)
    df['Residual_X'] = df['Z20'] * 0.2 + noise_res
    df['Residual_Y'] = df['Z25'] * 0.2 + noise_res
    
    return df, z_cols


def run_single_metric_diagnosis(df, z_cols, target_metric):
    """단일 Metric에 대한 3-패널 RCA 분석을 수행합니다."""
    
    # 1. 모델 학습 (Reference Model 생성)
    model = LinearRegression()
    model.fit(df[z_cols], df[target_metric])

    # 2. 데이터 추출
    ref_coefficients = pd.Series(model.coef_, index=z_cols) # Reference (기준: 베타 계수)
    
    # 3. 새로운 측정값 (Actual) 시뮬레이션
    # 가정: Overlay 에러가 심하게 난 '특정 샷'의 Zernike 측정값 (Z2, Z8 높게 설정)
    new_z_vector = np.random.normal(0, 1, NUM_ZERNIKE)
    new_z_vector[1] = 5.0  # Z2 (Tilt X)를 고의로 높게 만듦
    new_z_vector[7] = 3.0  # Z8 (Coma Y)를 고의로 높게 만듦
    new_z_series = pd.Series(new_z_vector, index=z_cols)

    # 4. 결과 분석 (Diagnosis): Contribution = Reference * Actual
    attribution_vector = ref_coefficients * new_z_series

    # -------------------------------------------------------
    # 5. 3-패널 Subplots 시각화
    # -------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)) 
    
    # --- Panel 1 (좌): Reference (민감도/기준) ---
    sns.barplot(x=z_cols, y=ref_coefficients.values, ax=axes[0], 
                palette=np.where(ref_coefficients.values > 0, 'Reds_d', 'Blues_d')[0])
    axes[0].set_title("1. REFERENCE: Sensitivity (MLR Coef.)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Sensitivity (a.u.)")
    axes[0].set_xticklabels(z_cols, rotation=90, fontsize=8) 
    
    # --- Panel 2 (중앙): Actual (측정된 입력값) ---
    sns.barplot(x=z_cols, y=new_z_series.values, ax=axes[1], color='tab:green')
    axes[1].set_title("2. ACTUAL: Measured Zernike Value", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Measured Z Value (nm)")
    axes[1].set_xticklabels(z_cols, rotation=90, fontsize=8)

    # --- Panel 3 (우): Diagnosis (최종 에러 기여도) ---
    sns.barplot(x=z_cols, y=attribution_vector.values, ax=axes[2], 
                palette=np.where(attribution_vector.values > 0, 'Reds_d', 'Blues_d')[0])
    axes[2].set_title("3. DIAGNOSIS: Error Attribution (Contribution)", fontsize=14, color='red', fontweight='bold')
    axes[2].set_ylabel("Attributed Error (nm)")
    axes[2].set_xticklabels(z_cols, rotation=90, fontsize=8)
    
    # 전체 타이틀 및 저장
    plt.suptitle(f"Zernike RCA Dashboard for Overlay Metric: {target_metric}", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    save_path = f'rca_report_{target_metric}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig) # 창 닫고 메모리 해제
    
    print(f"✅ 보고서 생성 완료: {save_path}")
    


def run_all_6_diagnostics():
    """6가지 지표 모두에 대해 3-패널 보고서를 반복 생성합니다."""
    
    df, z_cols = generate_data(NUM_POINTS)
    
    print("---------------------------------------------------------------------")
    print(f"🔄 6가지 Metric에 대한 RCA 보고서 반복 생성 시작...")
    print("---------------------------------------------------------------------")
    
    for metric in TARGET_METRICS:
        run_single_metric_diagnosis(df, z_cols, metric)

    print("\n🎉 모든 진단 완료. 6개의 개별 보고서 파일이 생성되었습니다.")


if __name__ == "__main__":
    run_all_6_diagnostics()