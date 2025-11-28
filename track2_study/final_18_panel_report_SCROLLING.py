import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# -------------------------------------------------------
# 1. 설정 및 데이터 생성
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
    measured_z_values[1] = 4.5
    measured_z_values[7] = -3.0
    measured_z_series = pd.Series(measured_z_values, index=Z_COLS)
    target_z_series = pd.Series([TARGET_VALUE] * NUM_ZERNIKE, index=Z_COLS)
    deviation_vector = target_z_series - measured_z_series
    
    return target_z_series, measured_z_series, deviation_vector


def run_18_panel_diagnosis():
    """6개 Metric의 3가지 구성요소를 6x3 그리드로 통합 시각화."""
    
    target_z, actual_z, deviation = generate_zernike_data()
    
    # 2. 6행 3열 Subplots 초기화 (총 18개 플롯)
    # 가로 크기 2/3 축소 유지 (18, 30)
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(18, 30)) 
    axes = axes.flatten()
    z_labels = Z_COLS 

    print("---------------------------------------------------------------------")
    print(f"✅ 18개 패널 통합 보고서 생성 완료. 로컬 창을 확인하세요.")
    print("---------------------------------------------------------------------")

    for i, metric in enumerate(TARGET_METRICS):
        # 0. Y축 스케일 통일 (비교 용이성)
        y_max = actual_z.abs().max() * 1.1 
        
        # 1. Panel 1 (좌측): Reference (기준 값)
        sns.barplot(x=z_labels, y=target_z.values, ax=axes[i*3 + 0], color='tab:gray')
        axes[i*3 + 0].set_title(f"({metric}) 1. REFERENCE", fontsize=10, fontweight='bold')
        axes[i*3 + 0].set_ylim(-y_max, y_max)
        
        # 2. Panel 2 (중앙): Actual (측정값)
        sns.barplot(x=z_labels, y=actual_z.values, ax=axes[i*3 + 1], color='tab:green')
        axes[i*3 + 1].set_title(f"({metric}) 2. ACTUAL", fontsize=10, fontweight='bold')
        axes[i*3 + 1].set_ylim(-y_max, y_max)

        # 3. Panel 3 (우측): Result (편차)
        sns.barplot(x=z_labels, y=deviation.values, ax=axes[i*3 + 2], 
                    palette=np.where(deviation.values > 0, 'Reds_d', 'Blues_d')[0])
        axes[i*3 + 2].set_title(f"({metric}) 3. RESULT (Deviation)", fontsize=10, color='red', fontweight='bold')
        axes[i*3 + 2].set_ylim(-y_max, y_max)
        
        # X축 레이블 및 Y축 레이블 설정 (모든 패널에 Zernike 항 표시)
        for j in range(N_COLS):
            current_ax = axes[i*3 + j]
            current_ax.set_xticklabels(z_labels, rotation=90, fontsize=6) # Z1~Z30 강제 표시
            current_ax.set_ylabel("", fontsize=0) # Y축 라벨은 제거 (왼쪽 패널에만 이름 붙일 예정)
        
        # 가장 왼쪽 패널에만 Metric 이름 표시
        axes[i*3 + 0].set_ylabel(metric, fontsize=9, fontweight='bold')

    
    # 최종 출력
    plt.suptitle("18-Panel Comprehensive Target Deviation Diagnosis", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
    
    # 파일 저장이 아닌 새 창 띄우기 (사용자 요청)
    plt.show() 


if __name__ == "__main__":
    run_18_panel_diagnosis()