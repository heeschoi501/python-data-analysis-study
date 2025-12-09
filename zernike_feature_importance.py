import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
import os
import random

# -------------------------------------------------------
# 1. 설정 및 상수 정의
# -------------------------------------------------------
FRINGE_ZERNIKE_MAP = {
    1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0),
    5: (2, 2), 6: (2, -2), 7: (3, 1), 8: (3, -1), 9: (4, 0),
    10: (3, 3), 11: (3, -3), 12: (4, 2), 13: (4, -2), 14: (5, 1), 15: (5, -1), 16: (6, 0),
    17: (4, 4), 18: (4, -4), 19: (5, 3), 20: (5, -3), 21: (6, 2), 22: (6, -2), 23: (7, 1), 24: (7, -1), 
    25: (8, 0),
    26: (5, 5), 27: (5, -5), 28: (6, 4), 29: (6, -4), 30: (7, 3), 31: (7, -3), 32: (8, 2), 33: (8, -2), 
    34: (9, 1), 35: (9, -1), 36: (10, 0), 37: (12, 0)
}

NUM_ZERNIKE = 37 
Z_COLS = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]
TARGET_VALUE = 0.0

# [기본 조명계 설정]
ILLUM_SETTINGS = {
    "pupil": "Annular",
    "na": 1.35,
    "sigma_out": 0.9,
    "sigma_in": 0.7
}

# [역추적 분석 시나리오 정의]
SCENARIOS = {
    "1. OVERLAY ONLY": ["Overlay"],
    "2. NORMAL CD ONLY (Wafer + 4 Corners)": ["CD_Normal"],
    "3. FULL SHOT CD (Dense)": ["CD_Full"],
    "4. OVERLAY + FULL SHOT CD": ["Overlay", "CD_Full"],
    "5. INTRA CD ONLY": ["CD_Intra"],
    "6. INTRA CD + OVERLAY": ["Overlay", "CD_Intra"],
    "7. TOTAL (All Combined)": ["Overlay", "CD_Normal", "CD_Full", "CD_Intra"]
}

# [Metric 그룹 정의]
METRIC_GROUPS = {
    "Overlay": [
        'Overlay_Avg_X', 'Overlay_Avg_Y', 'Overlay_3Sigma_X', 'Overlay_3Sigma_Y', 
        'Overlay_Res_X', 'Overlay_Res_Y', 'Shot_3Sigma_X', 'Shot_3Sigma_Y', 'Shot_Res_X', 'Shot_Res_Y'
    ],
    "CD_Normal": [
        'CD_Normal_Wafer_Mean', 
        'CD_Normal_MAT_LT', 'CD_Normal_MAT_RT', 'CD_Normal_MAT_LB', 'CD_Normal_MAT_RB'
    ],
    "CD_Full": ['CD_Full_Mean', 'CD_Full_HV_Bias'],       
    "CD_Intra": ['CD_Intra_Uniformity', 'CD_Intra_Curvature'] 
}

# [Physics Mapping] : 측정 데이터 -> 물리적 원형
PHYSICS_MAPPING = {
    'CD_Normal_Wafer_Mean': 'CD_Mean_Bias',
    'CD_Normal_MAT_LT': 'CD_Mean_Bias',
    'CD_Normal_MAT_RT': 'CD_Mean_Bias',
    'CD_Normal_MAT_LB': 'CD_Mean_Bias',
    'CD_Normal_MAT_RB': 'CD_Mean_Bias',
    'CD_Full_Mean': 'CD_Mean_Bias',
    'CD_Full_HV_Bias': 'CD_HV_Bias',
}

TARGET_METRICS = []
for m_list in METRIC_GROUPS.values():
    for m in m_list:
        if m not in TARGET_METRICS:
            TARGET_METRICS.append(m)

# -------------------------------------------------------
# 2. 데이터 생성 (Machine Constant + Drift + Parameter Split)
# -------------------------------------------------------
def get_measurement_points_template():
    hx, hy = 26 / 2, 33 / 2
    # 0: Center
    # 1: LT, 2: RT, 3: LB, 4: RB
    base_points = [
        (0, 0),         # Center (Idx 0)
        (-hx, hy),      # LT (Idx 1)
        (hx, hy),       # RT (Idx 2)
        (-hx, -hy),     # LB (Idx 3)
        (hx, -hy),      # RB (Idx 4)
        (0, hy), (0, -hy), (-hx, 0), (hx, 0)
    ]
    mid_points = [(x/2, y/2) for x, y in base_points[1:]]
    boundary_mid_points = [(-hx/2, hy), (hx/2, hy), (hx, hy/2), (hx, -hy/2), (hx/2, -hy), (-hx/2, -hy), (-hx, -hy/2), (-hx, hy/2)]
    return base_points + mid_points + boundary_mid_points

def get_illumination_multipliers():
    w_tilt_x = 1.0; w_tilt_y = 1.0
    w_coma_x = 1.0; w_coma_y = 1.0
    w_astig  = 1.0; w_spherical = 1.0
    w_high_order = 1.0 
    
    pupil = ILLUM_SETTINGS["pupil"]
    na = ILLUM_SETTINGS["na"]
    
    if pupil == "Dipole X":
        w_tilt_x *= 2.0; w_coma_x *= 2.0  
        w_tilt_y *= 0.5; w_coma_y *= 0.5
    elif pupil == "Dipole Y":
        w_tilt_x *= 0.5; w_coma_x *= 0.5
        w_tilt_y *= 2.0; w_coma_y *= 2.0
    elif pupil == "Crosspole" or pupil == "Quasar":
        w_astig *= 1.5                    
    elif pupil == "Conventional":
        w_spherical *= 1.2; w_astig *= 0.8
    
    if na < 0.8: w_high_order *= 0.5 
    elif na > 1.2: w_high_order *= 1.5 
        
    return {
        "tilt_x": w_tilt_x, "tilt_y": w_tilt_y,
        "coma_x": w_coma_x, "coma_y": w_coma_y,
        "astig": w_astig, "sph": w_spherical,
        "ho": w_high_order
    }

def get_machine_constant_zernike():
    rng = np.random.RandomState(999) 
    mc = rng.normal(0, 0.3, NUM_ZERNIKE) 
    return pd.Series(mc, index=Z_COLS)

def generate_realistic_wafer_data():
    print(f"⚙️ Generating Data with Illumination: {ILLUM_SETTINGS}...")
    
    shots = []
    grid_range = range(-6, 7)
    FIELD_SIZE_X, FIELD_SIZE_Y = 26, 33
    WAFER_RADIUS = 150
    
    for i in grid_range:
        for j in grid_range:
            center_x = i * FIELD_SIZE_X
            center_y = j * FIELD_SIZE_Y
            if np.sqrt(center_x**2 + center_y**2) < WAFER_RADIUS - 5:
                shots.append((center_x, center_y))
    
    # 5 Normal Shots Selection
    center_shot_idx = np.argmin([x**2 + y**2 for x, y in shots])
    top_shot_idx = np.argmax([y if abs(x) < FIELD_SIZE_X else -999 for x, y in shots])
    bottom_shot_idx = np.argmin([y if abs(x) < FIELD_SIZE_X else 999 for x, y in shots])
    right_shot_idx = np.argmax([x if abs(y) < FIELD_SIZE_Y else -999 for x, y in shots])
    left_shot_idx = np.argmin([x if abs(y) < FIELD_SIZE_Y else 999 for x, y in shots])
    
    normal_shot_indices = {center_shot_idx, top_shot_idx, bottom_shot_idx, left_shot_idx, right_shot_idx}

    point_template = get_measurement_points_template()
    total_points = len(shots) * len(point_template)
    
    df = pd.DataFrame()
    wafer_x_list, wafer_y_list, dx_list, dy_list = [], [], [], []
    shot_id_list, point_idx_list = [], []
    
    high_order_candidates = list(range(10, 38))
    c_list = random.sample(high_order_candidates, 3) 
    print(f"🎲 [Scenario] System Drift Detected at: Z{c_list[0]}, Z{c_list[1]}, Z{c_list[2]}")

    w = get_illumination_multipliers()
    mc_z = get_machine_constant_zernike() # Reference (Machine Constant)

    # 1. Zernike Input 생성
    for i in range(1, NUM_ZERNIKE + 1):
        col = f'Z{i}'
        base_val = mc_z[col] # 기준값
        drift = 0.0
        
        if 2 <= i <= 9: drift = np.random.uniform(-0.05, 0.05) 
        if i in c_list: drift = np.random.choice([1, -1]) * np.random.uniform(4.0, 6.0)
        
        df[col] = np.random.normal(base_val + drift, 0.15, total_points)
        
    noise = lambda s=0.05: np.random.normal(0, s, total_points)

    # 2. Modeling
    df['Overlay_Avg_X'] = (df['Z2'] * 2.5 * w['tilt_x']) + (df[f'Z{c_list[0]}'] * 0.8 * w['ho']) + noise(0.2)
    df['Overlay_Avg_Y'] = (df['Z3'] * 2.5 * w['tilt_y']) + (df[f'Z{c_list[1]}'] * 0.8 * w['ho']) + noise(0.2)
    df['Overlay_3Sigma_X'] = np.abs(df['Z4']) * 1.5 + np.abs(df[f'Z{c_list[0]}']) * 0.7 * w['ho'] + 1.0 + noise()
    df['Overlay_3Sigma_Y'] = np.abs(df['Z4']) * 1.5 + np.abs(df[f'Z{c_list[1]}']) * 0.7 * w['ho'] + 1.0 + noise()
    df['Overlay_Res_X'] = df[f'Z{c_list[2]}']**2 * 0.6 * w['ho'] + noise(0.3)
    df['Overlay_Res_Y'] = df[f'Z{c_list[2]}']**2 * 0.6 * w['ho'] + noise(0.3)
    df['Shot_3Sigma_X'] = df['Z16']**2 * 1.2 * w['ho'] + 1.5 + noise()
    df['Shot_3Sigma_Y'] = df['Z16']**2 * 1.2 * w['ho'] + 1.5 + noise()
    df['Shot_Res_X'] = df['Z30'] * 0.8 * w['ho'] + noise(0.5)
    df['Shot_Res_Y'] = df['Z31'] * 0.8 * w['ho'] + noise(0.5)
    
    df['CD_Mean_Bias'] = df['Z9'] * 2.5 * w['sph'] + df['Z4']**2 * 1.0 + df[f'Z{c_list[0]}'] * 0.6 * w['ho'] + noise(0.2)
    df['CD_HV_Bias'] = df['Z5'] * 3.0 * w['astig'] + df[f'Z{c_list[1]}'] * 0.6 * w['ho'] + noise()
    df['CD_Uniformity'] = np.abs(df['Z4']) * 2.0 + np.abs(df[f'Z{c_list[2]}']) * 0.5 * w['ho'] + 1.0 + noise(0.1)
    df['CD_Intra_Curvature'] = df['Z4']**2 * 1.5 + noise(0.1)
    
    df['CD_Raw_Mean_Bias'] = df['CD_Mean_Bias']
    df['CD_Raw_HV_Bias'] = df['CD_HV_Bias']

    idx = 0
    for shot_id, (sx, sy) in enumerate(shots):
        for p_idx, (local_x, local_y) in enumerate(point_template):
            wx = sx + local_x
            wy = sy + local_y
            val_dx = df.iloc[idx]['Overlay_Avg_X']
            val_dy = df.iloc[idx]['Overlay_Avg_Y']
            wafer_x_list.append(wx)
            wafer_y_list.append(wy)
            dx_list.append(val_dx)
            dy_list.append(val_dy)
            shot_id_list.append(shot_id)
            point_idx_list.append(p_idx)
            idx += 1
            
    df['Wafer_X'] = wafer_x_list
    df['Wafer_Y'] = wafer_y_list
    df['dX'] = dx_list
    df['dY'] = dy_list
    df['Shot_ID'] = shot_id_list
    df['Point_Index'] = point_idx_list
    
    # 3. CD Sampling Logic (Normal CD Breakdown)
    normal_mask = df['Shot_ID'].isin(normal_shot_indices)
    
    if normal_mask.sum() > 0:
        # 1. Wafer Mean
        df['CD_Normal_Wafer_Mean'] = df.loc[normal_mask, 'CD_Raw_Mean_Bias'].mean() + noise(0.0)
        
        # 2. 4-Corner Parameters (LT, RT, LB, RB)
        corner_map = {1: 'CD_Normal_MAT_LT', 2: 'CD_Normal_MAT_RT', 3: 'CD_Normal_MAT_LB', 4: 'CD_Normal_MAT_RB'}
        for p_idx, col_name in corner_map.items():
            # 해당하는 포인트 인덱스만 추출하여 평균
            corner_val = df.loc[normal_mask & (df['Point_Index'] == p_idx), 'CD_Raw_Mean_Bias'].mean()
            df[col_name] = corner_val + noise(0.0)
    else:
        df['CD_Normal_Wafer_Mean'] = df['CD_Raw_Mean_Bias'].mean()
        for col_name in ['CD_Normal_MAT_LT', 'CD_Normal_MAT_RT', 'CD_Normal_MAT_LB', 'CD_Normal_MAT_RB']:
            df[col_name] = df['CD_Raw_Mean_Bias'].mean()

    df['CD_Full_Mean'] = df['CD_Raw_Mean_Bias'].mean() + noise(0.0)
    df['CD_Full_HV_Bias'] = df['CD_Raw_HV_Bias'].mean()
    df['CD_Intra_Uniformity'] = df['CD_Uniformity']
    
    generate_realistic_wafer_data.cached_df = df
    generate_realistic_wafer_data.cached_shots = shots
    generate_realistic_wafer_data.cached_normal_indices = normal_shot_indices

    return df, shots

# -------------------------------------------------------
# 3. 시각화 함수 (Map & Forward)
# -------------------------------------------------------
def plot_wafer_vector_map(df, shot_centers, metric_to_plot='Overlay_Vector'):
    if hasattr(generate_realistic_wafer_data, "cached_normal_indices"):
        normal_indices = generate_realistic_wafer_data.cached_normal_indices
    else:
        normal_indices = set()

    plt.figure(figsize=(9, 9))
    ax = plt.gca()
    wafer_circle = plt.Circle((0, 0), 150, color='gray', fill=False, linewidth=2, linestyle='--')
    ax.add_patch(wafer_circle)
    
    for shot_id, (sx, sy) in enumerate(shot_centers):
        edge_color = 'red' if shot_id in normal_indices else 'lightgray'
        line_width = 2.0 if shot_id in normal_indices else 0.5
        rect = patches.Rectangle((sx - 13, sy - 16.5), 26, 33, linewidth=line_width, edgecolor=edge_color, facecolor='none')
        ax.add_patch(rect)
        
    if metric_to_plot == 'Overlay_Vector':
        magnitudes = np.sqrt(df['dX']**2 + df['dY']**2)
        q = ax.quiver(df['Wafer_X'], df['Wafer_Y'], df['dX'], df['dY'], magnitudes, scale=50, cmap='jet', width=0.002)
        plt.colorbar(q, label='Overlay Error (nm)')
        title_str = "Overlay Vector Map"
    elif metric_to_plot in df.columns:
        sc = ax.scatter(df['Wafer_X'], df['Wafer_Y'], c=df[metric_to_plot], s=10, cmap='coolwarm', alpha=0.7)
        plt.colorbar(sc, label=f'{metric_to_plot} (nm)')
        title_str = f"CD Heatmap: {metric_to_plot}"
    else:
        magnitudes = np.sqrt(df['dX']**2 + df['dY']**2)
        q = ax.quiver(df['Wafer_X'], df['Wafer_Y'], df['dX'], df['dY'], magnitudes, scale=50, cmap='jet', width=0.002)
        title_str = "Overlay Vector Map"

    plt.title(f"{title_str} (NA={ILLUM_SETTINGS['na']})", fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.show()

def _embed_matplotlib_in_tkinter(fig, window_title):
    result_window = tk.Toplevel()
    result_window.title(window_title)
    result_window.geometry("1400x800")
    main_frame = ttk.Frame(result_window)
    main_frame.pack(fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(main_frame, bg='white')
    v_scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    h_scroll = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
    canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
    v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollable_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    def on_frame_configure(event): canvas.configure(scrollregion=canvas.bbox("all"))
    scrollable_frame.bind("<Configure>", on_frame_configure)
    canvas_agg = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_agg.draw()
    width_px = int(fig.get_size_inches()[0] * fig.dpi)
    height_px = int(fig.get_size_inches()[1] * fig.dpi)
    canvas_widget = canvas_agg.get_tk_widget()
    canvas_widget.config(width=width_px, height=height_px)
    canvas_widget.pack(side=tk.TOP, anchor='nw')
    toolbar_frame = ttk.Frame(result_window)
    toolbar_frame.pack(side=tk.TOP, fill=tk.X)
    toolbar = NavigationToolbar2Tk(canvas_agg, toolbar_frame)
    toolbar.update()
    def _on_mousewheel(event):
        if event.delta: canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif event.num == 4: canvas.yview_scroll(-1, "units")
        elif event.num == 5: canvas.yview_scroll(1, "units")
    canvas.bind("<Enter>", lambda _: canvas.bind_all("<MouseWheel>", _on_mousewheel))
    canvas.bind("<Leave>", lambda _: canvas.unbind_all("<MouseWheel>"))
    canvas.bind("<Button-4>", _on_mousewheel)
    canvas.bind("<Button-5>", _on_mousewheel)
    def on_close():
        plt.close(fig)
        result_window.destroy()
    result_window.protocol("WM_DELETE_WINDOW", on_close)

# -------------------------------------------------------
# 4. 정방향 분석 (Forward)
# -------------------------------------------------------
def analyze_feature_importance(df, selected_metrics):
    if df is None: messagebox.showwarning("Error", "No data loaded."); return
    if not selected_metrics: messagebox.showwarning("Warning", "Select metrics!"); return

    print(f"🚀 Forward Analyzing {len(selected_metrics)} metrics...")
    n = len(selected_metrics)
    fig, axes = plt.subplots(n, 3, figsize=(24, 6 * n))
    if n == 1: axes = np.array([axes]) 
    
    actual_z_mean = df[Z_COLS].mean()
    target_z_mean = get_machine_constant_zernike()
    
    for i, metric in enumerate(selected_metrics):
        X = df[Z_COLS]
        Y = df[metric]
        model = LinearRegression()
        model.fit(X, Y)
        sensitivities = model.coef_ 
        
        raw_target = target_z_mean.values
        raw_actual = actual_z_mean.values
        raw_deviation = raw_target - raw_actual 
        weighted_result = raw_deviation * sensitivities 
        terms = Z_COLS 
        
        contributions = pd.Series(weighted_result, index=terms)
        top_5 = contributions.abs().sort_values(ascending=False).head(5)
        title_color = 'blue' if 'CD' in metric else 'black'
        
        ax1 = axes[i, 0]
        sns.barplot(x=contributions[top_5.index].values, y=top_5.index, ax=ax1, palette='viridis')
        ax1.set_title(f"[{metric}] 1. Top 5 Contributors", fontsize=11, fontweight='bold', color=title_color)
        ax1.grid(axis='x', linestyle='--', alpha=0.5)
        
        ax2 = axes[i, 1]
        ax2.plot(terms, raw_actual, label='Actual', marker='o', linestyle='-', color='tab:blue', markersize=4)
        ax2.plot(terms, raw_target, label='Ref (MC)', linestyle='--', color='gray', alpha=0.5)
        ax2.set_title(f"[{metric}] 2. Raw Zernike", fontsize=11, fontweight='bold', color=title_color)
        ax2.set_xticklabels(terms, rotation=90, fontsize=7)
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        ax3 = axes[i, 2]
        ax3.plot(terms, sensitivities, label='Sensitivity', marker='D', linestyle='-', color='purple', markersize=4)
        ax3.set_title(f"[{metric}] 3. Sensitivity", fontsize=11, fontweight='bold', color='red')
        ax3.set_xticklabels(terms, rotation=90, fontsize=7)
        ax3.legend()
        ax3.grid(True, linestyle=':', alpha=0.6)
        
    plt.suptitle(f"Forward Analysis (NA={ILLUM_SETTINGS['na']})", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    _embed_matplotlib_in_tkinter(fig, "Forward Analysis")

# -------------------------------------------------------
# 5. 역추적 분석 (Inverse - Lasso)
# -------------------------------------------------------
def estimate_zernike_state_with_sensitivity(df, selected_metrics, lambda_reg=0.05):
    X = df[Z_COLS]
    Y_list = []
    S_rows = []
    
    if hasattr(generate_realistic_wafer_data, "cached_normal_indices"):
        normal_indices = generate_realistic_wafer_data.cached_normal_indices
    else:
        normal_indices = set()
    
    for metric in selected_metrics:
        physics_metric = PHYSICS_MAPPING.get(metric, metric)
        model = LinearRegression()
        model.fit(X, df[physics_metric])
        S_rows.append(model.coef_)
        Y_list.append(df[metric].mean())
        
        # [수정] CD_Normal 세부 파라미터가 선택된 경우에도 올바른 민감도를 찾아감
        if "Normal" in metric:
             # CD_Normal_MAT_LT 같은 세부 파라미터는 이미 df에 컬럼으로 존재하므로
             # 위의 Y_list.append(df[metric].mean())에서 처리됨.
             # 추가적인 Shot별 Loop가 필요한 구조가 아니라, 컬럼 자체가 세분화되어 있음.
             pass

    S = np.vstack(S_rows)
    Y_measured = np.array(Y_list)
    lasso = Lasso(alpha=lambda_reg, fit_intercept=False, max_iter=50000)
    lasso.fit(S, Y_measured)
    return lasso.coef_, df[Z_COLS].mean().values, np.mean(np.abs(S), axis=0)

def show_inverse_estimation_result(df, scenario_name, selected_metrics):
    if df is None: messagebox.showwarning("Error", "No data loaded."); return

    try:
        Z_est, Z_actual, Z_sense = estimate_zernike_state_with_sensitivity(df, selected_metrics)
    except Exception as e:
        messagebox.showerror("Error", f"Inverse failed: {e}"); return

    terms = Z_COLS
    diff = Z_est - Z_actual
    accuracy = 1 - (np.sum((Z_est - Z_actual)**2) / np.sum(Z_actual**2))
    accuracy = max(0, accuracy) * 100 
    
    Z_ref = get_machine_constant_zernike().values

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    ax1 = axes[0]
    top_idx = np.argsort(np.abs(Z_est))[::-1][:10]
    ax1.bar(np.arange(10)-0.175, Z_actual[top_idx], 0.35, label='Actual (Drift)', color='tab:blue')
    ax1.bar(np.arange(10)+0.175, Z_est[top_idx], 0.35, label='Estimated (AI)', color='tab:orange')
    ax1.set_xticks(np.arange(10))
    ax1.set_xticklabels([terms[i] for i in top_idx], rotation=45)
    ax1.set_title(f"1. Diagnosis Result (Top 10)", fontsize=12, fontweight='bold', color='red')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(terms, Z_sense, marker='s', color='purple', alpha=0.7)
    ax2.fill_between(terms, 0, Z_sense, color='purple', alpha=0.1)
    ax2.set_title("2. Sensitivity Profile", fontsize=12, fontweight='bold')
    ax2.set_xticklabels(terms, rotation=90, fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(terms, Z_ref, label='Ref (MC)', linestyle='--', color='gray', alpha=0.6)
    ax3.plot(terms, Z_actual, label='Actual', marker='o', alpha=0.6)
    ax3.plot(terms, Z_est, label='Estimated', marker='x', alpha=0.8)
    ax3.set_title(f"3. Full Validation (Acc: {accuracy:.1f}%)", fontsize=12, fontweight='bold')
    ax3.axhline(0, color='black')
    ax3.set_xticklabels(terms, rotation=90, fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f"Inverse Retrieval: {scenario_name}", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _embed_matplotlib_in_tkinter(fig, f"Inverse Result - {scenario_name}")

# -------------------------------------------------------
# 6. 메인 GUI 메뉴
# -------------------------------------------------------
def main_menu_gui():
    root = tk.Tk()
    root.title("Zernike Analysis Dashboard (Pro)")
    root.geometry("600x900")
    tk.Label(root, text="Zernike Process Diagnosis Tool", font=("Arial", 14, "bold")).pack(pady=10)
    illum_frame = ttk.LabelFrame(root, text="0. Illumination & Data Source")
    illum_frame.pack(fill="x", padx=15, pady=5)
    tk.Label(illum_frame, text="NA:").grid(row=0, column=0, padx=5, pady=5)
    na_var = tk.DoubleVar(value=ILLUM_SETTINGS["na"])
    ttk.Entry(illum_frame, textvariable=na_var, width=8).grid(row=0, column=1)
    tk.Label(illum_frame, text="Pupil:").grid(row=0, column=2, padx=5)
    pupil_var = tk.StringVar(value=ILLUM_SETTINGS["pupil"])
    ttk.Combobox(illum_frame, textvariable=pupil_var, values=["Conventional", "Annular", "Dipole X", "Dipole Y", "Crosspole"], width=12).grid(row=0, column=3)
    tk.Label(illum_frame, text="Sigma Out:").grid(row=1, column=0, padx=5, pady=5)
    s_out_var = tk.DoubleVar(value=ILLUM_SETTINGS["sigma_out"])
    ttk.Entry(illum_frame, textvariable=s_out_var, width=8).grid(row=1, column=1)
    tk.Label(illum_frame, text="Sigma In:").grid(row=1, column=2, padx=5, pady=5)
    s_in_var = tk.DoubleVar(value=ILLUM_SETTINGS["sigma_in"])
    ttk.Entry(illum_frame, textvariable=s_in_var, width=12).grid(row=1, column=3)
    data_store = {"df": None, "shots": None} 

    def regen_sim_data(show_message=True):
        ILLUM_SETTINGS["na"] = na_var.get()
        ILLUM_SETTINGS["pupil"] = pupil_var.get()
        ILLUM_SETTINGS["sigma_out"] = s_out_var.get()
        ILLUM_SETTINGS["sigma_in"] = s_in_var.get()
        if hasattr(generate_realistic_wafer_data, "cached_df"): del generate_realistic_wafer_data.cached_df
        df, shots = generate_realistic_wafer_data()
        data_store["df"] = df; data_store["shots"] = shots
        if show_message: messagebox.showinfo("Sim", "Simulation Data Regenerated!")

    def load_csv_data():
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            try:
                df_real = pd.read_csv(path)
                for z in Z_COLS:
                    if z not in df_real.columns: df_real[z] = 0.0
                data_store["df"] = df_real
                data_store["shots"] = [] 
                messagebox.showinfo("Load", f"Loaded {len(df_real)} rows.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    ttk.Button(illum_frame, text="Generate SIM Data (Apply Settings)", command=lambda: regen_sim_data(True)).grid(row=2, column=0, columnspan=2, pady=5)
    ttk.Button(illum_frame, text="Load CSV File", command=load_csv_data).grid(row=2, column=2, columnspan=2, pady=5)
    regen_sim_data(show_message=False)
    
    frame_map = ttk.LabelFrame(root, text="1. Pre-Analysis")
    frame_map.pack(fill="x", padx=15, pady=5)
    map_metric_var = tk.StringVar(value="Overlay_Vector")
    map_metric_combo = ttk.Combobox(frame_map, textvariable=map_metric_var, values=["Overlay_Vector"] + TARGET_METRICS, state="readonly")
    map_metric_combo.pack(side="left", padx=10, pady=5)
    ttk.Button(frame_map, text="🗺️ Show Map", command=lambda: plot_wafer_vector_map(data_store["df"], data_store["shots"], map_metric_var.get())).pack(side="left", padx=10, pady=5)

    frame_fwd = ttk.LabelFrame(root, text="2. Forward Analysis")
    frame_fwd.pack(fill="both", expand=True, padx=15, pady=5)
    canvas = tk.Canvas(frame_fwd, height=150)
    scrollbar = ttk.Scrollbar(frame_fwd, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True, padx=5)
    scrollbar.pack(side="right", fill="y")
    var_dict = {m: tk.BooleanVar(value=False) for m in TARGET_METRICS}
    for m in TARGET_METRICS: ttk.Checkbutton(scroll_frame, text=m, variable=var_dict[m]).pack(anchor="w")
    def run_fwd():
        sel = [m for m, v in var_dict.items() if v.get()]
        analyze_feature_importance(data_store["df"], sel)
    ttk.Button(root, text="📊 Run Forward Analysis", command=run_fwd).pack(fill="x", padx=20, pady=5)

    lbl_inv = ttk.LabelFrame(root, text="🔁 Inverse Retrieval")
    lbl_inv.pack(fill="x", padx=20, pady=10)
    sc_var = tk.StringVar()
    combo = ttk.Combobox(lbl_inv, textvariable=sc_var, state="readonly", values=list(SCENARIOS.keys()))
    combo.current(2)
    combo.pack(fill="x", padx=10, pady=5)
    def run_inv():
        key = sc_var.get()
        mets = []
        for g in SCENARIOS[key]: mets.extend(METRIC_GROUPS[g])
        show_inverse_estimation_result(data_store["df"], key, mets)

    ttk.Button(lbl_inv, text="Run Inverse Solver", command=run_inv).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main_menu_gui()