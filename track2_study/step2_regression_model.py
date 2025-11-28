import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------------------------------
# 1. 환경 설정 및 데이터 생성
# -------------------------------------------------------
NUM_ZERNIKE = 30
NUM_POINTS = 300 
np.random.seed(42) 

def generate_data(n_points):
    """6가지 Overlay 지표를 포함하여 데이터를 생성합니다."""
    
    data = {'Shot_X': np.random.randint(0, 10, n_points), 'Shot_Y': np.random.randint(0, 10, n_points)}
    df = pd.DataFrame(data)

    # Zernike 항 (Z1 ~ Z30) 생성
    for i in range(1, NUM_ZERNIKE + 1):
        col_name = f'Z{i}'
        df[col_name] = np.random.normal(0, np.random.uniform(1, 3), n_points)
    
    # 6가지 Overlay 지표 계산 (물리적 모델링)
    noise_avg = np.random.normal(0, 0.1, n_points)
    noise_sigma = np.random.normal(0.5, 0.1, n_points)
    noise_res = np.random.normal(0, 0.05, NUM_POINTS)

    # Average Shift (Z2/Z3 주도)
    df['Average_X'] = df['Z2'] * 1.5 + df['Z8'] * 0.4 + noise_avg
    df['Average_Y'] = df['Z3'] * 1.5 + df['Z7'] * 0.4 + noise_avg

    # 3Sigma Spread (Z5/Z9 주도)
    df['3Sigma_X'] = np.abs(df['Z5'] * 0.6 + df['Z15'] * 0.4) + noise_sigma
    df['3Sigma_Y'] = np.abs(df['Z9'] * 0.6 + df['Z10'] * 0.4) + noise_sigma

    # Residual (Z20/Z25 주도)
    df['Residual_X'] = df['Z20'] * 0.2 + noise_res
    df['Residual_Y'] = df['Z25'] * 0.2 + noise_res
    
    return df

def run_mlr_diagnostic(df, target_metric):
    """단일 목표 변수에 대해 MLR 모델을 학습하고 결과를 저장합니다."""
    
    # 1. 데이터 분리
    z_cols = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]
    X = df[z_cols]          
    Y = df[target_metric]

    # 2. 모델 학습
    model = LinearRegression()
    model.fit(X, Y)

    # 3. 모델 결과 해석: '수차 기여도' (회귀 계수) 추출
    coefficients = pd.Series(model.coef_, index=z_cols).sort_values(ascending=False)

    # 4. 시각화: 기여도 막대 그래프 (Top 15개만 표시)
    plt.figure(figsize=(12, 6))
    
    # 상위 15개와 하위 15개를 명확히 구분하여 표시 (전체 30개를 다 표시하면 그래프가 복잡해지므로)
    top_15 = pd.concat([coefficients.head(8), coefficients.tail(7)]).sort_values(ascending=False)
    
    top_15.plot(kind='bar', color=np.where(top_15 > 0, 'tab:red', 'tab:blue'))
    
    plt.title(f'MLR Zernike Contribution to: {target_metric}', fontsize=14)
    plt.ylabel('Regression Coefficient (Physical Contribution)')
    plt.xlabel('Zernike Term')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    save_path = f'mlr_diagnosis_{target_metric}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close() # 메모리 확보

    print(f"✅ 진단 보고서 생성 완료: {save_path}")


def run_all_diagnostics(df):
    """6가지 모든 지표에 대해 반복 진단합니다."""
    
    TARGET_METRICS = ['Average_X', 'Average_Y', '3Sigma_X', '3Sigma_Y', 'Residual_X', 'Residual_Y']
    
    print("---------------------------------------------------------------------")
    print(f"🔄 6개 지표에 대한 반복 MLR 진단 시작...")
    print("---------------------------------------------------------------------")
    
    for metric in TARGET_METRICS:
        run_mlr_diagnostic(df, metric)

    print("\n🎉 모든 진단 완료. 6개 이미지 파일을 확인하십시오.")


if __name__ == "__main__":
    df_sim = generate_data(NUM_POINTS)
    run_all_diagnostics(df_sim)