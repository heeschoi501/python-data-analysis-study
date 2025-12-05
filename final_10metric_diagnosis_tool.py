import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib

# -------------------------------------------------------
# 1. Environment and Data Initialization
# -------------------------------------------------------
NUM_ZERNIKE = 37 
np.random.seed(42) 
TARGET_VALUE = 0.0 

TARGET_METRICS = [
    'Overlay_Avg_X', 'Overlay_Avg_Y',           
    'Overlay_3Sigma_X', 'Overlay_3Sigma_Y',     
    'Overlay_Res_X', 'Overlay_Res_Y',           
    'Shot_3Sigma_X', 'Shot_3Sigma_Y',           
    'Shot_Res_X', 'Shot_Res_Y'                  
]

Z_COLS = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]

# Wafer & Field Constants
WAFER_RADIUS = 150 
FIELD_SIZE_X = 26  
FIELD_SIZE_Y = 33  

# --- Wafer Map Data Generation ---
def generate_wafer_overlay_data():
    """Generates raw overlay vector data for a full wafer map."""
    shots = []
    
    # Generate Grid of Shots (Center positions)
    grid_range = range(-6, 7) # 범위를 조금 더 넓혀서 외곽 샷까지 확실히 커버
    for i in grid_range:
        for j in grid_range:
            center_x = i * FIELD_SIZE_X
            center_y = j * FIELD_SIZE_Y
            
            # Check if shot is within wafer (Partial coverage check)
            # 샷의 중심이 웨이퍼 반경 내에 있으면 유효 샷으로 인정
            if np.sqrt(center_x**2 + center_y**2) < WAFER_RADIUS - 5:
                shots.append((center_x, center_y))
    
    # Define 25 fixed measurement points relative to shot center (High Density)
    # 1. Center (1)
    # 2. 4 Corners + 4 Edge Mids (8)
    # 3. Intermediate points between Center and Outer 8 (8)
    # 4. [NEW] Intermediate points along the boundary (8)
    hx, hy = FIELD_SIZE_X / 2, FIELD_SIZE_Y / 2
    
    # Base points (Center + Outer 8)
    base_points = [
        (0, 0),
        (-hx, hy), (hx, hy), (-hx, -hy), (hx, -hy),
        (0, hy), (0, -hy), (-hx, 0), (hx, 0)
    ]
    
    # Intermediate points (Center to Outer)
    mid_points = [(x/2, y/2) for x, y in base_points[1:]]
    
    # [NEW] Boundary Intermediate points (Between Corners and Edge Mids)
    # 테두리를 따라 코너와 중점 사이를 채웁니다.
    boundary_mid_points = [
        (-hx/2, hy), (hx/2, hy),    # Top edge intermediates
        (hx, hy/2), (hx, -hy/2),    # Right edge intermediates
        (hx/2, -hy), (-hx/2, -hy),  # Bottom edge intermediates
        (-hx, -hy/2), (-hx, hy/2)   # Left edge intermediates
    ]
    
    # Total 25 points template
    local_points_template = base_points + mid_points + boundary_mid_points
                
    data = []
    # [수정] 인위적인 60개 제한(break)을 제거하여 모든 샷 생성
    for shot_id, (sx, sy) in enumerate(shots):
        
        # Shot마다 고유한 에러 패턴 부여
        # 특정 샷(예: ID 10번)에 큰 에러 주입
        shot_error_scale = 5.0 if shot_id == 10 else 1.0
        
        # [수정] 25개의 고밀도 측정 포인트에 대해 데이터 생성
        for local_x, local_y in local_points_template:
            # Global coordinate
            px = sx + local_x
            py = sy + local_y
            
            # Error Model
            dx = (-0.0001 * py + 0.00005 * px) * shot_error_scale + np.random.normal(0, 1)
            dy = (0.0001 * px + 0.00005 * py) * shot_error_scale + np.random.normal(0, 1)
            
            # Worst Shot (ID 10) Distortion
            if shot_id == 10:
                dx += -0.05 * local_y 
                dy += 0.05 * local_x

            data.append([shot_id, px, py, dx, dy, local_x, local_y])
            
    return pd.DataFrame(data, columns=['Shot_ID', 'Wafer_X', 'Wafer_Y', 'dX', 'dY', 'Local_X', 'Local_Y']), shots

def generate_zernike_data():
    """Simulates Zernike data."""
    measured_z_values = np.random.normal(0, 0.5, NUM_ZERNIKE)
    measured_z_values[1] = 4.5   
    measured_z_values[7] = -3.0  
    measured_z_values[4] = 2.0   
    measured_z_values[15] = 1.5  
    
    measured_z_series = pd.Series(measured_z_values, index=Z_COLS)
    target_z_series = pd.Series([TARGET_VALUE] * NUM_ZERNIKE, index=Z_COLS)
    deviation_vector = target_z_series - measured_z_series
    
    return target_z_series, measured_z_series, deviation_vector


# -------------------------------------------------------
# 2. Tkinter GUI 
# -------------------------------------------------------
def select_metrics_gui():
    """Tkinter window for selecting metrics."""
    root = tk.Tk()
    root.title("Metric Selection")
    root.geometry("400x600") 

    tk.Label(root, text="Select Analysis Metrics:", font=("Arial", 12, "bold")).pack(pady=10)

    container = ttk.Frame(root)
    canvas = tk.Canvas(container, height=300)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    container.pack(fill="both", expand=True, padx=10)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    var_dict = {m: tk.BooleanVar(value=True) for m in TARGET_METRICS}
    groups = {
        "Global Trends": ['Overlay_Avg_X', 'Overlay_Avg_Y'],
        "Wafer Variation": ['Overlay_3Sigma_X', 'Overlay_3Sigma_Y', 'Overlay_Res_X', 'Overlay_Res_Y'],
        "Intra-Shot Analysis": ['Shot_3Sigma_X', 'Shot_3Sigma_Y', 'Shot_Res_X', 'Shot_Res_Y']
    }

    for group_name, metrics in groups.items():
        tk.Label(scroll_frame, text=f"--- {group_name} ---", font=("Arial", 9, "bold"), fg="gray").pack(anchor="w", pady=(10, 2))
        for m in metrics:
            ttk.Checkbutton(scroll_frame, text=m, variable=var_dict[m]).pack(anchor="w", padx=10)

    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=15)

    def select_all():
        for v in var_dict.values(): v.set(True)
    def clear_all():
        for v in var_dict.values(): v.set(False)

    ttk.Button(btn_frame, text="Select All", command=select_all).grid(row=0, column=0, padx=5)
    ttk.Button(btn_frame, text="Clear All", command=clear_all).grid(row=0, column=1, padx=5)

    selected = []
    show_map_flag = False

    def confirm():
        nonlocal selected
        selected = [m for m, v in var_dict.items() if v.get()]
        if len(selected) == 0:
            messagebox.showwarning("Warning", "Please select at least one metric.")
            return
        root.destroy()
        
    def show_wafer_map_click():
        nonlocal show_map_flag
        show_map_flag = True
        root.destroy()

    action_frame = ttk.Frame(root)
    action_frame.pack(pady=10, fill='x', padx=20)
    ttk.Button(action_frame, text="🗺️ Show Wafer Vector Map", command=show_wafer_map_click, width=25).pack(pady=5)
    ttk.Button(action_frame, text="📊 Analyze & Visualize (Zernike)", command=confirm, width=25).pack(pady=5)

    root.mainloop()
    return selected, show_map_flag


# -------------------------------------------------------
# 3. Visualization Functions (Updated)
# -------------------------------------------------------
def plot_wafer_vector_map():
    """Displays Full Wafer Map + Statistics + Worst Shot Zoom-in."""
    df, shot_centers = generate_wafer_overlay_data()
    
    # 레이아웃: 좌측(Full Map), 우측 상단(Stats), 우측 하단(Worst Shot Zoom)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 2])
    
    ax_map = fig.add_subplot(gs[:, 0])      # 좌측 전체
    ax_stat = fig.add_subplot(gs[0, 1])     # 우측 상단 (통계)
    ax_zoom = fig.add_subplot(gs[1, 1])     # 우측 하단 (Worst Shot Zoom)
    
    # --- [Left] Full Wafer Map ---
    # 1. Wafer Outline
    wafer_circle = plt.Circle((0, 0), WAFER_RADIUS, color='gray', fill=False, linewidth=2, linestyle='--')
    ax_map.add_patch(wafer_circle)
    
    # 2. Clipping Mask (For clean wafer boundary)
    # 이 원은 보이지 않지만, 그리드와 벡터를 잘라내는(Clipping) 역할만 합니다.
    clip_circle = plt.Circle((0, 0), WAFER_RADIUS, transform=ax_map.transData)
    
    # 3. Shot Boundaries (Grid)
    for (sx, sy) in shot_centers:
        rect = patches.Rectangle(
            (sx - FIELD_SIZE_X/2, sy - FIELD_SIZE_Y/2),
            FIELD_SIZE_X, FIELD_SIZE_Y,
            linewidth=0.5, edgecolor='lightgray', facecolor='none'
        )
        # [수정] 그리드가 웨이퍼 밖으로 나가지 않도록 Clipping 적용
        rect.set_clip_path(clip_circle)
        ax_map.add_patch(rect)

    df['Mag'] = np.sqrt(df['dX']**2 + df['dY']**2)
    q = ax_map.quiver(df['Wafer_X'], df['Wafer_Y'], df['dX'], df['dY'], 
                  df['Mag'], scale=150, cmap='jet', width=0.002)
    
    # [수정] 벡터 화살표도 웨이퍼 밖으로 나가지 않도록 Clipping 적용
    q.set_clip_path(clip_circle)
    
    # Find Worst Shot (Max Mean Error)
    shot_means = df.groupby('Shot_ID')['Mag'].mean()
    worst_shot_id = shot_means.idxmax()
    worst_shot_val = shot_means.max()
    
    # Highlight Worst Shot on Map
    worst_center = shot_centers[worst_shot_id]
    rect_worst = patches.Rectangle(
        (worst_center[0] - FIELD_SIZE_X/2, worst_center[1] - FIELD_SIZE_Y/2),
        FIELD_SIZE_X, FIELD_SIZE_Y,
        linewidth=2, edgecolor='red', facecolor='none', label='Worst Shot'
    )
    # [수정] Worst Shot 테두리도 Clipping 적용
    rect_worst.set_clip_path(clip_circle)
    
    ax_map.add_patch(rect_worst)
    ax_map.legend(loc='upper right')

    plt.colorbar(q, ax=ax_map, label='Error Magnitude (nm)', fraction=0.046, pad=0.04)
    ax_map.set_title(f"Full Wafer Overlay Map (Total {len(shot_centers)} Shots)", fontsize=16, fontweight='bold')
    ax_map.set_xlabel("Wafer X (mm)")
    ax_map.set_ylabel("Wafer Y (mm)")
    ax_map.axis('equal')
    ax_map.grid(True, alpha=0.3)

    # --- [Right Top] Statistics Panel ---
    ax_stat.axis('off')
    stats_text = f"""
    [ Wafer Summary ]
    -----------------------
    Total Shots : {len(shot_centers)}
    Total Points: {len(df)}
    
    [ Global Statistics ]
    Mean X : {df['dX'].mean():.2f} nm
    Mean Y : {df['dY'].mean():.2f} nm
    3Sigma X: {df['dX'].std()*3:.2f} nm
    3Sigma Y: {df['dY'].std()*3:.2f} nm
    
    [ Worst Shot Info ]
    Shot ID : #{worst_shot_id}
    Mean Err: {worst_shot_val:.2f} nm
    Location: ({worst_center[0]:.1f}, {worst_center[1]:.1f})
    """
    ax_stat.text(0.05, 0.95, stats_text, transform=ax_stat.transAxes, 
                 fontsize=13, family='monospace', verticalalignment='top')

    # --- [Right Bottom] Worst Shot Zoom-in Map ---
    df_worst = df[df['Shot_ID'] == worst_shot_id]
    
    rect_zoom = patches.Rectangle(
        (-FIELD_SIZE_X/2, -FIELD_SIZE_Y/2),
        FIELD_SIZE_X, FIELD_SIZE_Y,
        linewidth=2, edgecolor='black', facecolor='whitesmoke'
    )
    ax_zoom.add_patch(rect_zoom)
    
    q_zoom = ax_zoom.quiver(df_worst['Local_X'], df_worst['Local_Y'], 
                            df_worst['dX'], df_worst['dY'], 
                            df_worst['Mag'],
                            scale=50, cmap='jet', width=0.015, headwidth=4)
    
    ax_zoom.set_title(f"Worst Shot Zoom-in (ID: {worst_shot_id})", fontsize=14, fontweight='bold', color='red')
    ax_zoom.set_xlabel("Field X (mm)")
    ax_zoom.set_ylabel("Field Y (mm)")
    ax_zoom.set_xlim(-FIELD_SIZE_X/2 - 2, FIELD_SIZE_X/2 + 2)
    ax_zoom.set_ylim(-FIELD_SIZE_Y/2 - 2, FIELD_SIZE_Y/2 + 2)
    ax_zoom.grid(True, linestyle='--')
    ax_zoom.set_aspect('equal')
    
    # [수정] 25개 포인트 측정 위치 강조 (Scatter Plot)
    ax_zoom.scatter(df_worst['Local_X'], df_worst['Local_Y'], c='red', s=20, marker='x', label='Meas. Points')
    ax_zoom.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


def run_10metric_dashboard(selected_metrics):
    """Generates the dashboard for selected metrics."""
    target_z, actual_z, deviation = generate_zernike_data()
    n = len(selected_metrics)
    cols = 2 
    rows = int(np.ceil(n / cols))

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)
    
    for i, metric in enumerate(selected_metrics):
        ax = axes[i]
        ax.plot(Z_COLS, target_z, label="Reference (Target=0)", linestyle='--', color='black', alpha=0.5)
        ax.plot(Z_COLS, actual_z, label="Actual (Measured)", marker='o', markersize=4, linestyle='-', color='tab:blue', alpha=0.8)
        ax.plot(Z_COLS, deviation, label="Result (Deviation)", marker='s', markersize=4, linestyle='-', color='tab:red', linewidth=2)

        ax.set_title(f"Diagnostic Report: {metric}", fontsize=11, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticklabels(Z_COLS, rotation=90, fontsize=6)
        ax.set_ylabel("Zernike Value (nm)", fontsize=8)
        
        max_val = max(actual_z.abs().max(), deviation.abs().max()) + 1
        ax.set_ylim(-max_val, max_val)
        ax.legend(fontsize=7, loc='upper right')

    for ax in axes[n:]:
        ax.remove()

    plt.suptitle("10-Metric Overlay Process Diagnosis Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show() 


if __name__ == "__main__":
    selected_metrics, show_map = select_metrics_gui()
    
    if show_map:
        plot_wafer_vector_map()
    elif selected_metrics:
        run_10metric_dashboard(selected_metrics)