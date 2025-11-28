import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 설정 및 데이터 생성 (Z1 ~ Z30)
NUM_ZERNIKE = 30
NUM_POINTS = 200  
np.random.seed(42) 

print(f"🚀 다변량 분석 데이터 생성 중... (Z1 ~ Z{NUM_ZERNIKE})")

data = {
    'Shot_X': np.random.randint(0, 10, NUM_POINTS),
    'Shot_Y': np.random.randint(0, 10, NUM_POINTS)
}
df = pd.DataFrame(data)

# Zernike 항 생성 및 가중치 설정 (모두 랜덤 가중치를 가지나, 핵심 Z항은 큰 영향을 갖도록 설정)
for i in range(1, NUM_ZERNIKE + 1):
    col_name = f'Z{i}'
    df[col_name] = np.random.normal(0, np.random.uniform(1, 3), NUM_POINTS)
    
# -------------------------------------------------------
# 2. 4가지 공정 지표 계산 (물리적 모델링)
# -------------------------------------------------------
# [Model 1: Overlay X] - Z2(Tilt X)가 주도하고 Z8(Coma X)이 보조
df['Overlay_X'] = df['Z2'] * 1.5 + df['Z8'] * 0.5 + np.random.normal(0, 0.2, NUM_POINTS)

# [Model 2: Overlay Y] - Z3(Tilt Y)가 주도하고 Z7(Coma Y)이 보조
df['Overlay_Y'] = df['Z3'] * 1.5 + df['Z7'] * 0.5 + np.random.normal(0, 0.2, NUM_POINTS)

# [Model 3: CD Mean] - Z9(Spherical)가 주도하고 Z5(Astigmatism)가 보조 (Focus 관련)
df['CD_Mean'] = 100 + df['Z9'] * 1.0 + df['Z5'] * 0.8 + np.random.normal(0, 0.1, NUM_POINTS)

# [Model 4: CD Uniformity] - Z5(Astigmatism)가 주도하고 Z15(HOZ)가 보조
df['CD_Uniformity'] = 1 + df['Z5'] * 1.2 + df['Z15'] * 0.7 + np.random.normal(0, 0.05, NUM_POINTS)

# -------------------------------------------------------
# 3. 분석 시각화: 다변량 상관관계 매트릭스
# -------------------------------------------------------
print(f"✅ 데이터 생성 완료! (Shape: {df.shape})")

# 분석 대상 컬럼 설정
z_cols = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]
metric_cols = ['Overlay_X', 'Overlay_Y', 'CD_Mean', 'CD_Uniformity']

# Zernike 항과 4가지 지표 간의 상관관계만 추출
corr_subset = df[z_cols + metric_cols].corr().loc[z_cols, metric_cols]

# -------------------------------------------------------
# 3. 분석 시각화: Zernike 스펙트럼 Line Plot
# -------------------------------------------------------
print("📊 수차 스펙트럼 분석 및 Line Plot 최적화 중...")

z_cols = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]
metric_cols = ['Overlay_X', 'Overlay_Y', 'CD_Mean', 'CD_Uniformity']

# Zernike 항과 4가지 지표 간의 상관관계만 추출
corr_subset = df[z_cols + metric_cols].corr().loc[z_cols, metric_cols]

# 1. 전체 상관관계 데이터 준비 (Wide to Long Format)
full_plot_data = corr_subset.reset_index().rename(columns={'index': 'Zernike_Term'})

plot_data_melted = full_plot_data.melt(
    id_vars='Zernike_Term', 
    value_vars=metric_cols, 
    var_name='Metric', 
    value_name='Correlation'
)

# 2. X축 순서를 강제 지정 (Z1, Z2, Z3... 순서로 정렬하기 위함)
plot_data_melted['Z_Index'] = plot_data_melted['Zernike_Term'].str.replace('Z', '').astype(int)
plot_data_melted = plot_data_melted.sort_values(by='Z_Index')


# 3. Line Plot 그리기 (Graph Visualization)
plt.figure(figsize=(16, 8))
sns.lineplot(
    data=plot_data_melted,
    x='Zernike_Term',       # X축: Z1~Z30 (순서 정렬됨)
    y='Correlation',        # Y축: 상관계수 강도
    hue='Metric',           # 4가지 지표를 색깔로 구분
    marker='o',             # 꺾은선 형태를 명확히 하기 위한 점 추가
    palette='Spectral'
)

# X축 레이블을 45도 회전 (글씨 겹침 방지)
plt.ylabel('Correlation Coefficient (r-value)')
plt.xlabel('Zernike Term Index (Z1 ~ Z30)')
plt.title(f'Zernike Spectrum of Influence on Process Metrics')
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='gray', linestyle='--') # 기준선 추가 (0이 기준)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(title='Process Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

# 이미지 저장
save_path = 'z30_spectrum_analysis.png'
plt.savefig(save_path, bbox_inches='tight')
print(f"\n💾 분석 결과가 '{save_path}'에 저장되었습니다.")