import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 새 창으로 띄우기 (로컬 GUI 환경에서 실행됩니다)
# matplotlib.use('Agg') 

# -------------------------------------------------------
# 1. 환경 설정 및 데이터 생성
# -------------------------------------------------------
NUM_ZERNIKE = 30 
np.random.seed(42) 
TARGET_VALUE = 0.0 
TARGET_METRICS = ['Average_X', 'Average_Y', '3Sigma_X', '3Sigma_Y', 'Residual_X', 'Residual_Y']
Z_COLS = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]
N_ROWS = len(TARGET_METRICS)
N_COLS = 3 # Reference, Actual, Result

def generate_zernike_data():
    """새로운 측정 웨이퍼의 Zernike 값을 시뮬레이션합니다."""
    measured_z_values = np.random.normal(0, 0.8, NUM_ZERNIKE)
    measured_z_values[1] = 4.5  # Z2 (Tilt X) - 고의 편차 주입
    measured_z_values[7] = -3.0 # Z8 (Coma) - 고의 편차 주입
    measured_z_series = pd.Series(measured_z_values, index=Z_COLS)
    target_z_series = pd.Series([TARGET_VALUE] * NUM_ZERNIKE, index=Z_COLS)
    deviation_vector = target_z_series - measured_z_series
    
    return target_z_series, measured_z_series, deviation_vector


def run_18_panel_diagnosis():
    """6개 Metric의 3가지 구성요소를 6x3 그리드로 통합 시각화."""
    
    # 1. 데이터 준비: Target, Actual, Deviation 벡터는 모든 Metric에 공통
    target_z, actual_z, deviation = generate_zernike_data()
    
    # 2. 6행 3열 Subplots 초기화 (총 18개 플롯)
    # 가독성을 위해 Figure 크기를 대폭 키웁니다.
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(28, 20)) 
    
    print("---------------------------------------------------------------------")
    print(f"🔄 18개 패널 통합 보고서 생성 시작...")
    print("---------------------------------------------------------------------")

    for i, metric in enumerate(TARGET_METRICS):
        # 0. Y축 스케일 통일 (비교 용이성)
        y_max = actual_z.abs().max() * 1.1 
        
        # 1. Panel 1 (좌측): REFERENCE (기준 값)
        sns.barplot(x=Z_COLS, y=target_z.values, ax=axes[i, 0], color='tab:gray')
        axes[i, 0].set_title(f"({metric}) 1. REFERENCE (Target Z)", fontsize=11, fontweight='bold')
        axes[i, 0].set_ylim(-y_max, y_max)
        
        # 2. Panel 2 (중앙): ACTUAL (측정값)
        sns.barplot(x=Z_COLS, y=actual_z.values, ax=axes[i, 1], color='tab:green')
        axes[i, 1].set_title(f"({metric}) 2. ACTUAL (Measured Z)", fontsize=11, fontweight='bold')
        axes[i, 1].set_ylim(-y_max, y_max)

        # 3. Panel 3 (우측): RESULT (편차)
        sns.barplot(x=Z_COLS, y=deviation.values, ax=axes[i, 2], 
                    palette=np.where(deviation.values > 0, 'Reds_d', 'Blues_d')[0])
        axes[i, 2].set_title(f"({metric}) 3. RESULT (Deviation = Target - Actual)", fontsize=11, color='red', fontweight='bold')
        axes[i, 2].set_ylim(-y_max, y_max)
        
        # X축 레이블 설정 (가독성을 위해 맨 아래 행에만 표시)
        if i == N_ROWS - 1:
            axes[i, 0].set_xticklabels(Z_COLS, rotation=90, fontsize=6)
            axes[i, 1].set_xticklabels(Z_COLS, rotation=90, fontsize=6)
            axes[i, 2].set_xticklabels(Z_COLS, rotation=90, fontsize=6)
        else:
            axes[i, 0].set_xticklabels([])
            axes[i, 1].set_xticklabels([])
            axes[i, 2].set_xticklabels([])
        
        # Y축 라벨 설정
        axes[i, 0].set_ylabel(metric, fontsize=10, fontweight='bold')

    
    # 최종 출력
    plt.suptitle("18-Panel Comprehensive Target Deviation Diagnosis", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 

    print("\n🎉 18-패널 통합 보고서 생성 완료. 로컬 창을 확인하세요.")
    plt.show() 


if __name__ == "__main__":
    run_18_panel_diagnosis()