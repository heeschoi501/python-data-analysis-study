import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os
import random

# -------------------------------------------------------
# 1. 설정 및 데이터 생성
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
Z_COLS = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]

# 13개 핵심 지표
TARGET_METRICS = [
    'Overlay_Avg_X', 'Overlay_Avg_Y',           
    'Overlay_3Sigma_X', 'Overlay_3Sigma_Y',     
    'Overlay_Res_X', 'Overlay_Res_Y',           
    'Shot_3Sigma_X', 'Shot_3Sigma_Y',           
    'Shot_Res_X', 'Shot_Res_Y',
    'CD_Mean_Bias', 'CD_HV_Bias', 'CD_Uniformity'
]

def get_measurement_points_template():
    hx, hy = 26 / 2, 33 / 2
    base_points = [(0, 0), (-hx, hy), (hx, hy), (-hx, -hy), (hx, -hy), (0, hy), (0, -hy), (-hx, 0), (hx, 0)]
    mid_points = [(x/2, y/2) for x, y in base_points[1:]]
    boundary_mid_points = [(-hx/2, hy), (hx/2, hy), (hx, hy/2), (hx, -hy/2), (hx/2, -hy), (-hx/2, -hy), (-hx, -hy/2), (-hx, hy/2)]
    return base_points + mid_points + boundary_mid_points

def generate_realistic_wafer_data():
    print(f"⚙️ Generating Wafer Map Data (APC Logic Applied)...")
    
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
    
    point_template = get_measurement_points_template()
    total_points = len(shots) * len(point_template)
    
    df = pd.DataFrame()
    wafer_x_list = []
    wafer_y_list = []
    dx_list = []
    dy_list = []
    
    # [Dynamic Scenario] 숨겨진 범인(Culprit) 선정
    high_order_candidates = list(range(10, 38))
    c_list = random.sample(high_order_candidates, 3) 
    print(f"🎲 [Scenario] APC Missed Culprits: Z{c_list[0]}, Z{c_list[1]}, Z{c_list[2]}")

    # 1. Zernike Input 생성
    for i in range(1, NUM_ZERNIKE + 1):
        bias = 0.0
        # 저차 수차 (APC 보정됨 -> 작음)
        if 2 <= i <= 9: 
            bias = np.random.uniform(-0.05, 0.05) 
        # 고차 수차 (범인 -> 큼)
        if i in c_list:
            bias = np.random.choice([1, -1]) * np.random.uniform(4.0, 6.0)
        
        df[f'Z{i}'] = np.random.normal(bias, 0.15, total_points)
        
    noise = lambda s=0.05: np.random.normal(0, s, total_points)

    # 2. Modeling (물리 법칙 - 민감도 고정)
    df['Overlay_Avg_X'] = (df['Z2'] * 2.5) + (df[f'Z{c_list[0]}'] * 0.8) + noise(0.2)
    df['Overlay_Avg_Y'] = (df['Z3'] * 2.5) + (df[f'Z{c_list[1]}'] * 0.8) + noise(0.2)
    
    df['Overlay_3Sigma_X'] = np.abs(df['Z4']) * 1.5 + np.abs(df[f'Z{c_list[0]}']) * 0.7 + 1.0 + noise()
    df['Overlay_3Sigma_Y'] = np.abs(df['Z4']) * 1.5 + np.abs(df[f'Z{c_list[1]}']) * 0.7 + 1.0 + noise()
    
    df['Overlay_Res_X'] = df[f'Z{c_list[2]}']**2 * 0.6 + noise(0.3)
    df['Overlay_Res_Y'] = df[f'Z{c_list[2]}']**2 * 0.6 + noise(0.3)
    
    df['Shot_3Sigma_X'] = df['Z16']**2 * 1.2 + 1.5 + noise()
    df['Shot_3Sigma_Y'] = df['Z16']**2 * 1.2 + 1.5 + noise()
    df['Shot_Res_X'] = df['Z30'] * 0.8 + noise(0.5)
    df['Shot_Res_Y'] = df['Z31'] * 0.8 + noise(0.5)
    
    df['CD_Mean_Bias'] = df['Z9'] * 2.5 + df[f'Z{c_list[0]}'] * 0.6 + noise()
    df['CD_HV_Bias'] = df['Z5'] * 3.0 + df[f'Z{c_list[1]}'] * 0.6 + noise()
    df['CD_Uniformity'] = np.abs(df['Z4']) * 2.0 + np.abs(df[f'Z{c_list[2]}']) * 0.5 + 1.0 + noise()
    
    idx = 0
    for shot_id, (sx, sy) in enumerate(shots):
        for local_x, local_y in point_template:
            wx = sx + local_x
            wy = sy + local_y
            
            z15, z24 = df.iloc[idx]['Z15'], df.iloc[idx]['Z24']
            val_dx = z15 * 0.2 * (local_x/26) + z24 * 0.1
            val_dy = z15 * 0.2 * (local_y/33) + z24 * 0.1
            
            wafer_x_list.append(wx)
            wafer_y_list.append(wy)
            dx_list.append(val_dx)
            dy_list.append(val_dy)
            idx += 1
            
    df['Wafer_X'] = wafer_x_list
    df['Wafer_Y'] = wafer_y_list
    df['dX'] = dx_list
    df['dY'] = dy_list
    
    return df, shots

# -------------------------------------------------------
# 2. 시각화 함수
# -------------------------------------------------------
def plot_wafer_vector_map(df, shot_centers):
    """Wafer Map 시각화"""
    plt.figure(figsize=(9, 9))
    ax = plt.gca()
    wafer_circle = plt.Circle((0, 0), 150, color='gray', fill=False, linewidth=2, linestyle='--')
    ax.add_patch(wafer_circle)
    
    for (sx, sy) in shot_centers:
        rect = patches.Rectangle((sx - 13, sy - 16.5), 26, 33, linewidth=0.5, edgecolor='lightgray', facecolor='none')
        ax.add_patch(rect)
        
    magnitudes = np.sqrt(df['dX']**2 + df['dY']**2)
    q = ax.quiver(df['Wafer_X'], df['Wafer_Y'], df['dX'], df['dY'], magnitudes, scale=50, cmap='jet', width=0.002)
    plt.colorbar(q, label='Overlay Error Magnitude (nm)', fraction=0.046, pad=0.04)
    plt.title("Full Wafer Overlay Vector Map (APC Residuals)", fontsize=14, fontweight='bold')
    plt.xlabel("Wafer X (mm)")
    plt.ylabel("Wafer Y (mm)")
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.show()

def analyze_feature_importance(df, selected_metrics):
    """3단계 논리적 분석 시각화 (Top 5 -> Raw Deviation -> Weighted Impact)"""
    if not selected_metrics:
        messagebox.showwarning("Warning", "분석할 지표를 선택하세요!")
        return

    print(f"🚀 3-Panel Logic Analyzing for {len(selected_metrics)} metrics...")
    
    n = len(selected_metrics)
    fig, axes = plt.subplots(n, 3, figsize=(24, 6 * n))
    if n == 1: axes = np.array([axes]) 

    actual_z_mean = df[Z_COLS].mean()
    target_z_mean = pd.Series(0.0, index=Z_COLS)

    for i, metric in enumerate(selected_metrics):
        X = df[Z_COLS]
        Y = df[metric]

        # 1. 모델 학습 (민감도 추출)
        model = LinearRegression()
        model.fit(X, Y)
        sensitivities = model.coef_ 
        
        # 2. 데이터 계산
        raw_target = target_z_mean.values
        raw_actual = actual_z_mean.values
        raw_deviation = raw_target - raw_actual 
        
        # 3. Weighted Result (Impact) = Deviation * Sensitivity
        weighted_result = raw_deviation * sensitivities 
        
        terms = Z_COLS 
        
        # Top 5 Contribution (Weighted Result 기준)
        contributions = pd.Series(weighted_result, index=terms)
        top_5_contributors = contributions.abs().sort_values(ascending=False).head(5)
        top_5_names = top_5_contributors.index
        top_5_values = contributions[top_5_names]

        title_color = 'blue' if 'CD' in metric else 'black'
        
        # [Panel 1] Left: Top 5 Contributors (Bar)
        ax1 = axes[i, 0]
        sns.barplot(x=top_5_values.values, y=top_5_names, ax=ax1, palette='viridis')
        ax1.set_title(f"[{metric}] 1. Top 5 Main Culprits", fontsize=11, fontweight='bold', color=title_color)
        ax1.set_xlabel("Weighted Impact (nm)")
        ax1.grid(axis='x', linestyle='--', alpha=0.5)

        # [Panel 2] Center: Raw Zernike Spectrum (Line)
        # Ref, Actual, Result 3가지를 보여줌
        ax2 = axes[i, 1]
        ax2.plot(terms, raw_target, label='Reference (0)', linestyle='--', color='gray', alpha=0.5)
        ax2.plot(terms, raw_actual, label='Actual (Measured)', marker='o', linestyle='-', color='tab:blue', markersize=4)
        ax2.plot(terms, raw_deviation, label='Result (Ref - Actual)', marker='x', linestyle='-', color='tab:red', markersize=4)
        
        ax2.set_title(f"[{metric}] 2. Raw Zernike Status (Input)", fontsize=11, fontweight='bold', color=title_color)
        ax2.set_ylabel("Zernike Value (nm)")
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xticklabels(terms, rotation=90, fontsize=7)
        ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(True, linestyle=':', alpha=0.6)

        # [Panel 3] Right: Weighted Result Spectrum (Line) - Only Result
        ax3 = axes[i, 2]
        # [수정] Result 그래프만 단독 표시 (Weighted Ref/Actual 제거)
        ax3.plot(terms, weighted_result, label='Weighted Result (Attributed Error)', marker='D', linestyle='-', color='tab:red', markersize=4)
        ax3.fill_between(terms, 0, weighted_result, alpha=0.1, color='tab:red') # 시각적 강조를 위해 영역 채우기

        ax3.set_title(f"[{metric}] 3. Weighted Result (Attributed Error)", fontsize=11, fontweight='bold', color='red')
        ax3.set_ylabel("Attributed Error (nm)")
        ax3.axhline(0, color='black', linewidth=0.8)
        ax3.set_xticklabels(terms, rotation=90, fontsize=7)
        ax3.legend(fontsize=8, loc='upper right')
        ax3.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("APC-Aware Diagnosis: High Sensitivity vs High Deviation", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Scrollable Window Code (생략 - 기존과 동일)
    result_window = tk.Toplevel()
    result_window.title("Analysis Results (Scrollable View)")
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
    canvas_widget = canvas_agg.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
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
# 3. Tkinter GUI
# -------------------------------------------------------
def main_menu_gui():
    df, shot_centers = generate_realistic_wafer_data()

    root = tk.Tk()
    root.title("Zernike Analysis Dashboard")
    root.geometry("450x650")

    tk.Label(root, text="Zernike Process Diagnosis Tool", font=("Arial", 14, "bold")).pack(pady=10)

    frame_map = ttk.LabelFrame(root, text="1. Pre-Analysis")
    frame_map.pack(fill="x", padx=15, pady=5)
    ttk.Button(frame_map, text="🗺️ Show Wafer Vector Map", command=lambda: plot_wafer_vector_map(df, shot_centers)).pack(fill="x", padx=10, pady=10)

    frame_analysis = ttk.LabelFrame(root, text="2. Root Cause Analysis (Select Metrics)")
    frame_analysis.pack(fill="both", expand=True, padx=15, pady=5)

    canvas = tk.Canvas(frame_analysis)
    scrollbar = ttk.Scrollbar(frame_analysis, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    scrollbar.pack(side="right", fill="y")

    var_dict = {m: tk.BooleanVar(value=False) for m in TARGET_METRICS}
    groups = {
        "Overlay Global": ['Overlay_Avg_X', 'Overlay_Avg_Y'],
        "Overlay Variation": ['Overlay_3Sigma_X', 'Overlay_3Sigma_Y', 'Overlay_Res_X', 'Overlay_Res_Y'],
        "Overlay Shot": ['Shot_3Sigma_X', 'Shot_3Sigma_Y', 'Shot_Res_X', 'Shot_Res_Y'],
        "CD Metrics": ['CD_Mean_Bias', 'CD_HV_Bias', 'CD_Uniformity']
    }

    for group, metrics in groups.items():
        tk.Label(scroll_frame, text=f"--- {group} ---", font=("Arial", 9, "bold"), fg="gray").pack(anchor="w", pady=(5, 0))
        for m in metrics:
            ttk.Checkbutton(scroll_frame, text=m, variable=var_dict[m]).pack(anchor="w", padx=10)

    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=10)

    def select_all():
        for v in var_dict.values(): v.set(True)
    def clear_all():
        for v in var_dict.values(): v.set(False)
    
    def run_analysis():
        selected = [m for m, v in var_dict.items() if v.get()]
        analyze_feature_importance(df, selected)

    ttk.Button(btn_frame, text="Select All", command=select_all).grid(row=0, column=0, padx=5)
    ttk.Button(btn_frame, text="Clear All", command=clear_all).grid(row=0, column=1, padx=5)
    
    style = ttk.Style()
    style.configure("Accent.TButton", font=("Arial", 10, "bold"))
    ttk.Button(root, text="📊 Analyze Selected Metrics", style="Accent.TButton", command=run_analysis).pack(fill="x", padx=20, pady=15)

    root.mainloop()

if __name__ == "__main__":
    main_menu_gui()