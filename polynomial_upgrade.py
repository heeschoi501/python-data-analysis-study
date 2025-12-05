import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------
# 1. 논문 기반 Zernike 정의 (Fringe Convention)
# -------------------------------------------------------
FRINGE_ZERNIKE_MAP = {
    1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0),
    5: (2, 2), 6: (2, -2), 7: (3, 1), 8: (3, -1), 9: (4, 0),
    10: (3, 3), 11: (3, -3), 12: (4, 2), 13: (4, -2), 14: (5, 1), 15: (5, -1), 16: (6, 0),
    17: (4, 4), 18: (4, -4), 19: (5, 3), 20: (5, -3), 21: (6, 2), 22: (6, -2), 23: (7, 1), 24: (7, -1), 25: (8, 0),
    26: (5, 5), 27: (5, -5), 28: (6, 4), 29: (6, -4), 30: (7, 3), 31: (7, -3), 32: (8, 2), 33: (8, -2), 34: (9, 1), 35: (9, -1), 36: (10, 0), 37: (12, 0)
}

NUM_ZERNIKE = 37 
NUM_POINTS = 1000 
np.random.seed(42) 

Z_COLS = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]

# 현업 수준의 10개 핵심 지표 정의
TARGET_METRICS = [
    'Overlay_Avg_X', 'Overlay_Avg_Y',           
    'Overlay_3Sigma_X', 'Overlay_3Sigma_Y',     
    'Overlay_Res_X', 'Overlay_Res_Y',           
    'Shot_3Sigma_X', 'Shot_3Sigma_Y',           
    'Shot_Res_X', 'Shot_Res_Y'                  
]

def generate_enz_based_data(n_points):
    """
    ENZ 이론과 현업 지표 특성을 반영한 10개 Metric 데이터 생성
    """
    df = pd.DataFrame()
    
    # 1. Zernike 입력값 생성
    for i in range(1, NUM_ZERNIKE + 1):
        if i in FRINGE_ZERNIKE_MAP:
            n, m = FRINGE_ZERNIKE_MAP[i]
            scale = 5.0 / (n + 1) 
        else:
            scale = 0.5
        df[f'Z{i}'] = np.random.normal(0, np.random.uniform(0.1, scale), n_points)
    
    noise = lambda scale=0.1: np.random.normal(0, scale, n_points)

    # -------------------------------------------------------
    # 2. 물리적 모델링 (Overlay Value 생성)
    # -------------------------------------------------------
    # [Group 1] Average X/Y (Mean Shift)
    df['Overlay_Avg_X'] = df['Z2']*2.0 + df['Z7']*1.5 + df['Z14']*0.5 + noise(0.2)
    df['Overlay_Avg_Y'] = df['Z3']*2.0 + df['Z8']*1.5 + df['Z15']*0.5 + noise(0.2)

    # [Group 2] 3Sigma X/Y (Variation)
    df['Overlay_3Sigma_X'] = (df['Z4'] * df['Z5']) * 1.5 + df['Z9']**2 * 0.8 + 2.0 + noise(0.1)
    df['Overlay_3Sigma_Y'] = (df['Z4'] * df['Z6']) * 1.5 + df['Z9']**2 * 0.8 + 2.0 + noise(0.1)

    # [Group 3] Residual X/Y 
    df['Overlay_Res_X'] = df['Z10']**2 * 1.2 + df['Z21'] * 0.8 + noise(0.3)
    df['Overlay_Res_Y'] = df['Z11']**2 * 1.2 + df['Z22'] * 0.8 + noise(0.3)

    # [Group 4] Shot 3Sigma X/Y 
    df['Shot_3Sigma_X'] = df['Z16']**2 * 1.2 + df['Z25']**2 * 0.5 + 1.5 + noise(0.1)
    df['Shot_3Sigma_Y'] = df['Z16']**2 * 1.2 + df['Z25']**2 * 0.5 + 1.5 + noise(0.1)

    # [Group 5] Shot Residual X/Y 
    df['Shot_Res_X'] = df['Z30'] * 0.8 + df['Z37'] * 0.5 + noise(0.5)
    df['Shot_Res_Y'] = df['Z31'] * 0.8 + df['Z36'] * 0.5 + noise(0.5)
    
    return df

def analyze_full_report():
    """
    10개 지표에 대해 [원인 분석(좌)] vs [결과 값 분포(우)]를 동시에 시각화합니다.
    """
    
    print(f"🚀 [Track 2] Full Diagnosis Report 생성 시작 (Cause & Effect)...")
    df = generate_enz_based_data(NUM_POINTS)
    
    # 데이터 통계 요약 출력
    print("\n📊 [Overlay Data Summary]")
    print(df[TARGET_METRICS].describe().T[['mean', 'std', 'min', 'max']])

    # 시각화 준비: 10행 2열 (좌: 원인 / 우: 결과)
    # 세로로 매우 긴 이미지가 생성됩니다 (스크롤 필수)
    fig, axes = plt.subplots(10, 2, figsize=(18, 50))
    
    for i, metric in enumerate(TARGET_METRICS):
        print(f"   Analyzing {metric}...")
        
        X = df[Z_COLS]
        Y = df[metric]

        # --- [Left Panel] 원인 분석 (Feature Importance) ---
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(Z_COLS)

        model = LinearRegression()
        model.fit(X_poly, Y)

        coefficients = pd.DataFrame({
            'Term': feature_names,
            'Importance': np.abs(model.coef_)
        }).sort_values(by='Importance', ascending=False).head(10)

        ax_cause = axes[i, 0]
        sns.barplot(data=coefficients, x='Importance', y='Term', ax=ax_cause, palette='viridis')
        ax_cause.set_title(f"[{metric}] Cause Analysis (Top 10 Zernike)", fontsize=12, fontweight='bold')
        ax_cause.set_xlabel("Impact (Beta Coefficient)")
        ax_cause.grid(axis='x', linestyle='--', alpha=0.5)

        # --- [Right Panel] 결과 분석 (Overlay Value Distribution) ---
        # 실제 시뮬레이션된 Overlay 값들이 어떻게 분포하는지 히스토그램으로 표시
        ax_effect = axes[i, 1]
        sns.histplot(Y, kde=True, ax=ax_effect, color='tab:blue', bins=30)
        
        # 통계치 표시 (Mean, 3Sigma)
        mean_val = Y.mean()
        sigma3_val = Y.std() * 3
        stats_text = f"Mean: {mean_val:.2f} nm\n3Sigma: {sigma3_val:.2f} nm"
        
        ax_effect.axvline(mean_val, color='red', linestyle='--', label='Mean')
        ax_effect.set_title(f"[{metric}] Effect Analysis (Value Distribution)", fontsize=12, fontweight='bold')
        ax_effect.set_xlabel("Overlay Error (nm)")
        ax_effect.legend([stats_text], loc='upper right')
        ax_effect.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("Zernike Aberration Diagnosis: Cause (Left) vs Effect (Right)", fontsize=24, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    save_filename = 'full_diagnosis_report.png'
    print(f"\n💾 전체 분석 리포트 '{save_filename}' 저장 완료.")
    plt.savefig(save_filename, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_full_report()