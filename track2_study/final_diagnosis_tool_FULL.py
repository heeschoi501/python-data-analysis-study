import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
#matplotlib.use('Agg') # GUI 없이 파일 저장을 위한 필수 설정 (현업 PC용)
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# 1. 설정 및 데이터 생성
# -------------------------------------------------------
NUM_ZERNIKE = 30 # Z1 부터 Z30 까지
NUM_POINTS = 300 
np.random.seed(42) 

def generate_data(n_points):
    """30개 Zernike 항과 목표 변수(Average_X)를 포함한 가상 데이터를 생성합니다."""
    
    df = pd.DataFrame()
    z_cols = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]

    # Zernike 항 (Z1 ~ Z30) 생성
    for col_name in z_cols:
        df[col_name] = np.random.normal(0, np.random.uniform(1, 3), n_points)
    
    # Target Metric: Average_X (Z2, Z8, Z15가 주도한다고 가정하여 모델 학습)
    df['Average_X'] = df['Z2'] * 1.5 + df['Z8'] * 0.4 - df['Z15'] * 0.2 + np.random.normal(0, 0.1, n_points)
    
    return df, z_cols

def run_full_diagnosis():
    """MLR 모델 학습, 새로운 측정값 시뮬레이션, Zernike 전체 스펙트럼 시각화."""
    
    df, z_cols = generate_data(NUM_POINTS)
    
    # 1. 모델 학습 (Reference Model 생성)
    model = LinearRegression()
    model.fit(df[z_cols], df['Average_X'])

    # 2. Reference (기준) 추출: MLR 계수 (Sensitivity)
    ref_coefficients = pd.Series(model.coef_, index=z_cols)

    # 3. 새로운 측정값 (Actual) 시뮬레이션
    # 가정: Overlay 에러가 심하게 난 '특정 샷'의 Zernike 측정값
    # Z2 (Tilt X)가 Reference보다 튀고, Z20 같은 고차항도 살짝 튀었다고 가정
    new_z_vector = np.random.normal(0, 1, NUM_ZERNIKE)
    new_z_vector[1] = 5.0 # Z2 (Tilt X)를 고의로 높게 만듦
    new_z_vector[19] = -3.0 # Z20 (고차항)도 튀었다고 가정
    new_z_series = pd.Series(new_z_vector, index=z_cols)

    # 4. 결과 분석 (Attribution): 각 수차가 Overlay에 기여한 '실제 에러량' 계산
    # Contribution = Reference Coefficient (Sensitivity) * Actual Input Value
    attribution_vector = ref_coefficients * new_z_series

    # 5. 시각화: Full Spectrum 데이터 준비
    # 전체 Zernike 항을 대상으로 사용 (필터링 없음)
    
    # -------------------------------------------------------
    # 6. 3-패널 Subplots 시각화 (Z1 ~ Z30 전체)
    # -------------------------------------------------------
    # 이미지 크기 조정 (항목이 30개이므로 가로 폭을 늘림)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)) 
    
    x_labels = z_cols # X축 레이블은 Z1~Z30 전체
    
    # --- Panel 1: Reference (민감도/기준) ---
    sns.barplot(x=x_labels, y=ref_coefficients.values, ax=axes[0], 
                palette=np.where(ref_coefficients.values > 0, 'Reds_d', 'Blues_d')[0])
    axes[0].axhline(0, color='gray', linestyle='--')
    axes[0].set_title("1. Reference: Overlay Sensitivity (MLR Coef.)", fontsize=14)
    axes[0].set_ylabel("Sensitivity (a.u.)")
    axes[0].set_xticklabels(x_labels, rotation=90, fontsize=8) # Z30개이므로 글자 회전
    
    # --- Panel 2: Actual (측정된 입력값) ---
    sns.barplot(x=x_labels, y=new_z_series.values, ax=axes[1], color='tab:green')
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].set_title("2. Actual: Measured Zernike Value (Input)", fontsize=14)
    axes[1].set_ylabel("Measured Z Value (nm)")
    axes[1].set_xticklabels(x_labels, rotation=90, fontsize=8)

    # --- Panel 3: Diagnosis (최종 에러 기여도) ---
    # 색상을 Attribution 벡터의 부호에 따라 결정
    sns.barplot(x=x_labels, y=attribution_vector.values, ax=axes[2], 
                palette=np.where(attribution_vector.values > 0, 'Reds_d', 'Blues_d')[0])
    axes[2].axhline(0, color='gray', linestyle='--')
    axes[2].set_title("3. Diagnosis: Error Attribution (Contribution)", fontsize=14, color='red')
    axes[2].set_ylabel("Attributed Overlay Error (nm)")
    axes[2].set_xticklabels(x_labels, rotation=90, fontsize=8)
    
    # 전체 타이틀 및 레이아웃 정리
    plt.suptitle(f"Full-Spectrum Zernike Diagnosis for Overlay ({df.columns[-1]})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 타이틀을 위해 여백 조정

    save_path = 'final_3panel_diagnosis_full.png'
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.close() # 메모리 해제

    # 새로운 코드 (새 창 띄우기)
    plt.show() # <--- 이 코드를 추가하세요.
    # plt.close() # plt.show()가 창을 닫으면 메모리도 해제되므로 필요 없습니다.
    
    print(f"\n🎉 최종 진단 보고서 (Z1~Z30 전체) '{save_path}' 저장 완료. (파일 탐색기에서 확인)")


if __name__ == "__main__":
    run_full_diagnosis()