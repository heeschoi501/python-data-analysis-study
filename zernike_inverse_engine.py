import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.linear_model import LinearRegression

# ==========================
# Global Config
# ==========================

NUM_ZERNIKE = 37
Z_COLS = [f"Z{i}" for i in range(1, NUM_ZERNIKE + 1)]
PATTERNS = ["LINE", "HOLE"]

# 7개 모드 정의 (프리셋)
MODE_DEFS = {
    "1_OVERLAY_ONLY": [
        "Overlay_Avg_X", "Overlay_Avg_Y",
        "Overlay_3Sigma_X", "Overlay_3Sigma_Y",
        "dX", "dY"
    ],
    "2_CD_INTER_NORMAL_ONLY": [
        "CD_INTER_NORMAL_LINE", "CD_INTER_NORMAL_HOLE"
    ],
    "3_CD_INTER_FULL_ONLY": [
        "CD_INTER_FULL_LINE", "CD_INTER_FULL_HOLE",
        "CD_MACD", "CD_MEEF"
    ],
    "4_OVERLAY_PLUS_INTER_FULL": [
        "Overlay_Avg_X", "Overlay_Avg_Y",
        "dX", "dY",
        "CD_INTER_FULL_LINE", "CD_INTER_FULL_HOLE",
        "CD_MACD"
    ],
    "5_CD_INTRA_ONLY": [
        "CD_INTRA_LINE", "CD_INTRA_HOLE",
        "CD_LER",
        "CD_LINE_H", "CD_LINE_V", "CD_HOLE"
    ],
    "6_INTRA_PLUS_OVERLAY": [
        "Overlay_Avg_X", "Overlay_Avg_Y",
        "dX", "dY",
        "CD_INTRA_LINE", "CD_INTRA_HOLE",
        "CD_LER",
        "CD_LINE_H", "CD_LINE_V", "CD_HOLE"
    ],
    "7_TOTAL_ALL": [
        "Overlay_Avg_X", "Overlay_Avg_Y",
        "Overlay_3Sigma_X", "Overlay_3Sigma_Y",
        "dX", "dY",
        "CD_INTER_NORMAL_LINE", "CD_INTER_NORMAL_HOLE",
        "CD_INTER_FULL_LINE", "CD_INTER_FULL_HOLE",
        "CD_INTRA_LINE", "CD_INTRA_HOLE",
        "CD_MACD", "CD_LER", "CD_MEEF",
        "CD_LINE_H", "CD_LINE_V", "CD_HOLE",
    ],
}

# RCA / Inverse에서 선택 가능한 Metric 전체 리스트
RCA_METRICS = [
    # Overlay 통계 + 맵
    "Overlay_Avg_X", "Overlay_Avg_Y",
    "Overlay_3Sigma_X", "Overlay_3Sigma_Y",
    "dX", "dY",
    # Inter CD
    "CD_INTER_NORMAL_LINE", "CD_INTER_NORMAL_HOLE",
    "CD_INTER_FULL_LINE", "CD_INTER_FULL_HOLE",
    # Intra / Pattern
    "CD_INTRA_LINE", "CD_INTRA_HOLE",
    "CD_LINE_H", "CD_LINE_V", "CD_HOLE",
    # 기타 파생 파라미터
    "CD_MACD", "CD_LER", "CD_MEEF",
]

# ==========================
# Measurement Pattern
# ==========================

def get_measurement_points_template():
    """하나의 필드 안에서의 로컬 좌표 + 패턴(Line/Hole) 템플릿."""
    hx, hy = 26 / 2, 33 / 2
    base_points = [
        (0, 0),
        (-hx, hy), (hx, hy),
        (-hx, -hy), (hx, -hy),
        (0, hy), (0, -hy),
        (-hx, 0), (hx, 0),
    ]
    mid_points = [(x/2, y/2) for x, y in base_points[1:]]
    boundary_mid_points = [
        (-hx/2, hy), (hx/2, hy),
        (hx, hy/2), (hx, -hy/2),
        (hx/2, -hy), (-hx/2, -hy),
        (-hx, -hy/2), (-hx, hy/2),
    ]
    pts = base_points + mid_points + boundary_mid_points
    patterns = [PATTERNS[i % len(PATTERNS)] for i in range(len(pts))]
    return pts, patterns

# ==========================
# Data Generation (Z_ref + Z_true)
# ==========================

def generate_wafer_data(random_seed: int = 123):
    """
    - Z_ref: 장비에 세팅된 Reference Zernike
    - Z_true = Z_ref + ΔZ (실제 렌즈 상태, culprit 수차 포함)
    - df['Z1'..'Z37'] = Z_true + noise (wafer 위치별 변동)
    - Overlay / CD / MACD / LER / MEEF / Pattern CD 생성
    """
    np.random.seed(random_seed)
    FIELD_X, FIELD_Y = 26, 33
    WAFER_R = 150

    # Shot grid 생성
    shots = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            cx, cy = i * FIELD_X, j * FIELD_Y
            if np.hypot(cx, cy) < WAFER_R - 5:
                shots.append((len(shots), cx, cy))

    # Shot 내 measurement point 생성
    pts, pat_for_pt = get_measurement_points_template()
    rows = []
    for shot_id, sx, sy in shots:
        for (lx, ly), pat in zip(pts, pat_for_pt):
            wx, wy = sx + lx, sy + ly
            rows.append((shot_id, wx, wy, lx, ly, pat))

    df = pd.DataFrame(rows, columns=["ShotID", "Wafer_X", "Wafer_Y", "Local_X", "Local_Y", "Pattern"])
    N = len(df)

    # --- Reference Zernike (장비 설정값 Z_ref) ---
    Z_ref = np.zeros(NUM_ZERNIKE)
    for i in range(1, NUM_ZERNIKE + 1):
        if i == 4:  # 예: Spherical baseline
            Z_ref[i-1] = np.random.uniform(-0.5, 0.5)
        elif 2 <= i <= 3:  # Tilt/Coma 계열
            Z_ref[i-1] = np.random.uniform(-0.3, 0.3)
        else:
            Z_ref[i-1] = np.random.uniform(-0.1, 0.1)

    # --- Culprit ΔZ (Reference에서 벗어난 추가 에러) ---
    high_order_candidates = np.arange(10, NUM_ZERNIKE + 1)
    culprits = np.random.choice(high_order_candidates, size=3, replace=False)

    dZ = np.zeros(NUM_ZERNIKE)
    for i in culprits:
        dZ[i-1] = np.random.choice([1, -1]) * np.random.uniform(3.0, 6.0)

    Z_true = Z_ref + dZ  # 실제 렌즈 상태

    # --- 각 포인트별 수차 = Z_true + noise ---
    for i in range(1, NUM_ZERNIKE + 1):
        if i in culprits:
            sigma = 0.3
        elif 2 <= i <= 9:
            sigma = 0.1
        else:
            sigma = 0.05
        noise = np.random.normal(0.0, sigma, N)
        df[f"Z{i}"] = Z_true[i-1] + noise

    # --- Overlay metrics (통계용) + dX/dY (맵용) ---
    noise = lambda s: np.random.normal(0, s, N)
    df["Overlay_Avg_X"] = df["Z2"] * 2.5 + df["Z10"] * 0.5 + noise(0.1)
    df["Overlay_Avg_Y"] = df["Z3"] * 2.5 + df["Z11"] * 0.5 + noise(0.1)
    df["Overlay_3Sigma_X"] = np.abs(df["Z4"]) * 1.5 + noise(0.1)
    df["Overlay_3Sigma_Y"] = np.abs(df["Z4"]) * 1.5 + noise(0.1)

    # full map vector
    df["dX"] = df["Overlay_Avg_X"] * 0.1
    df["dY"] = df["Overlay_Avg_Y"] * 0.1

    # --- CD metrics (Pattern C: Line-H / Line-V / Hole) ---
    mask_line = (df["Pattern"] == "LINE").astype(float)
    mask_hole = (df["Pattern"] == "HOLE").astype(float)
    base_cd = 50.0

    # Full / Normal / Intra (기존)
    df["CD_INTER_FULL_LINE"] = base_cd + (df["Z4"] * 2.0 + df["Z9"] * 0.5) * mask_line + noise(0.3)
    df["CD_INTER_FULL_HOLE"] = base_cd + (df["Z9"] * 2.0 + df["Z4"] * 0.5) * mask_hole + noise(0.3)

    df["CD_INTER_NORMAL_LINE"] = base_cd + (df["Z4"] * 1.2 + df["Z9"] * 0.3) * mask_line + noise(0.6)
    df["CD_INTER_NORMAL_HOLE"] = base_cd + (df["Z9"] * 1.2 + df["Z4"] * 0.3) * mask_hole + noise(0.6)

    df["CD_INTRA_LINE"] = (np.abs(df["Z15"]) * (1.0 + 0.7 * np.abs(df["Local_X"])) + noise(0.2)) * mask_line
    df["CD_INTRA_HOLE"] = (np.abs(df["Z16"]) * (1.0 + 0.7 * np.abs(df["Local_Y"])) + noise(0.2)) * mask_hole

    # Pattern C: 라인 방향성(H/V) + Hole 분리
    is_H = (np.abs(df["Local_X"]) >= np.abs(df["Local_Y"])).astype(float)
    is_V = 1.0 - is_H

    mask_line_h = mask_line * is_H
    mask_line_v = mask_line * is_V
    mask_hole_only = mask_hole

    # CD_HOLE은 Z9, Z13, Z24 쪽에 더 강하게 민감하도록 설정
    df["CD_LINE_H"] = base_cd + (df["Z4"] * 2.2 + df["Z5"] * 0.8) * mask_line_h + noise(0.3)
    df["CD_LINE_V"] = base_cd + (df["Z4"] * 2.0 - df["Z5"] * 0.8) * mask_line_v + noise(0.3)
    df["CD_HOLE"]   = base_cd + (df["Z9"] * 2.0 + df["Z13"] * 1.5 + df["Z24"] * 1.2) * mask_hole_only + noise(0.3)

    # --- MACD / LER / MEEF 추가 ---
    df["CD_MACD"] = (df["CD_INTER_FULL_LINE"] - df["CD_INTER_FULL_HOLE"]) + noise(0.1)
    df["CD_LER"] = np.abs(df["Z27"]) * 0.8 + np.abs(df["Z28"]) * 0.5 + noise(0.3)
    df["CD_MEEF"] = 1.5 + 0.05 * df["Z4"] + 0.03 * df["Z9"] + noise(0.05)

    return df, shots, Z_ref, Z_true, culprits

# ==========================
# Pupil / NA Sensitivity Scaling
# ==========================

def pupil_sensitivity_scale(pupil_shape: str, NA: float, sigma_inner: float, sigma_outer: float):
    """
    노광 조건에 따라 각 Zernike 항의 민감도를 스케일링.
    pupil_shape는 다음과 같은 일반적인 이름을 받습니다.
      - CONVENTIONAL  (일반 원형 조명)
      - ANNULAR       (환형)
      - X-POLE, Y-POLE  (디폴, 한 방향 극 조명)
      - X&Y-POLE        (쿼드폴, double dipole)

    내부적으로는 이름을 정규화해서 매핑합니다.
    """
    scale = np.ones(NUM_ZERNIKE)

    # NA 의존 (0.85 기준)
    scale *= (1.0 + 0.4 * (NA - 0.85))

    # 이름 정규화
    shape_raw = pupil_shape.upper().replace(" ", "")
    shape_raw = shape_raw.replace("&", "").replace("-", "")

    # 매핑
    if shape_raw in ("CONVENTIONAL", "CIRCULAR"):
        base_type = "CONVENTIONAL"
    elif shape_raw == "ANNULAR":
        base_type = "ANNULAR"
    elif shape_raw in ("XPOLE", "YPOLE", "DIPOLE"):
        base_type = "DIPOLE"
    elif shape_raw in ("XYPOLE", "XYPOLES", "XYPOL", "QUADRUPOLE", "DOUBLEPOLE"):
        base_type = "QUADRUPOLE"
    else:
        base_type = "CONVENTIONAL"  # 알 수 없으면 기본값

    # 각 타입별 대략적인 민감도 스케일링
    if base_type == "CONVENTIONAL":
        # 구면 + 저차 Astig 쪽이 상대적으로 강한 상황
        scale[3] *= 1.1      # 대략 Z4
        scale[4:6] *= 1.1    # Astig 계열
    elif base_type == "ANNULAR":
        # 고차 성분에 더 민감
        scale[1:3] *= 1.1            # Coma-ish
        scale[4:10] *= 1.15          # mid-order (Trefoil, Secondary astig 등)
        scale[10:20] *= 1.1
    elif base_type == "DIPOLE":
        # 한 방향 극 조명 → 비대칭 수차(Coma, Trefoil) 쪽을 강조
        scale[1:3] *= 1.4            # Tilt/Coma
        scale[6:12] *= 1.3           # Trefoil, secondary coma
    elif base_type == "QUADRUPOLE":
        # X/Y-POLE 같이 양 방향 dipole → 4-fold 계열 강화
        scale[4:8] *= 1.25           # Astig/Trefoil-ish
        scale[8:16] *= 1.2           # higher-order 4-fold

    # 링 폭 (sigma_outer - sigma_inner) 영향
    ring_width = max(0.0, sigma_outer - sigma_inner)
    scale *= (1.0 + 0.2 * ring_width)

    return scale

# ==========================
# RCA (3-panel)
# ==========================

def rca_analyze(df: pd.DataFrame, selected_metrics):
    if not selected_metrics:
        messagebox.showwarning("경고", "분석할 Metric을 선택해 주세요.")
        return

    X = df[Z_COLS]
    actual_z_mean = df[Z_COLS].mean()
    target_z_mean = pd.Series(0.0, index=Z_COLS)

    n = len(selected_metrics)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = np.array([axes])

    for i, metric in enumerate(selected_metrics):
        if metric not in df.columns:
            continue
        y = df[metric]
        if y.nunique() < 2:
            continue

        model = LinearRegression()
        model.fit(X, y)
        sens = pd.Series(model.coef_, index=Z_COLS)

        raw_target = target_z_mean.values
        raw_actual = actual_z_mean.values
        raw_deviation = raw_target - raw_actual

        weighted_target = raw_target * sens.values
        weighted_actual = raw_actual * sens.values
        weighted_result = raw_deviation * sens.values

        contributions = pd.Series(weighted_result, index=Z_COLS)
        top5 = contributions.abs().sort_values(ascending=False).head(5)
        top5_names = top5.index
        top5_vals = contributions[top5_names].values

        ax1, ax2, ax3 = axes[i]

        # Panel 1: Top5
        ax1.barh(top5_names, top5_vals)
        ax1.set_title(f"[{metric}] 1. Top-5 Contributors")
        ax1.set_xlabel("Weighted Impact")
        ax1.grid(axis="x", linestyle=":", alpha=0.5)

        # Panel 2: Raw Z
        terms = Z_COLS
        ax2.plot(terms, raw_target, label="Ref(0)", linestyle="--", color="gray")
        ax2.plot(terms, raw_actual, label="Actual Zmean", marker="o", markersize=3)
        ax2.plot(terms, raw_deviation, label="Ref-Actual", marker="x", markersize=3)
        ax2.set_title(f"[{metric}] 2. Raw Zernike Status")
        ax2.set_xticks(range(len(terms)))
        ax2.set_xticklabels(terms, rotation=90, fontsize=7)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(fontsize=8)

        # Panel 3: Weighted
        ax3.plot(terms, weighted_target, label="W.Ref", linestyle="--", color="gray")
        ax3.plot(terms, weighted_actual, label="W.Actual", marker="o", markersize=3)
        ax3.plot(terms, weighted_result, label="W.Result", marker="D", markersize=3)
        ax3.set_title(f"[{metric}] 3. Weighted Impact")
        ax3.set_xticks(range(len(terms)))
        ax3.set_xticklabels(terms, rotation=90, fontsize=7)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.grid(True, linestyle=":", alpha=0.5)
        ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

# ==========================
# Inverse Solver
# ==========================

def build_sensitivity_matrix(df: pd.DataFrame, metrics):
    X = df[Z_COLS]
    S_rows = []
    b_list = []
    y_mean_list = []
    used_metrics = []

    for m in metrics:
        if m not in df.columns:
            continue
        y = df[m]
        if y.nunique() < 2:
            continue
        model = LinearRegression()
        model.fit(X, y)
        S_rows.append(model.coef_)
        b_list.append(model.intercept_)
        y_mean_list.append(y.mean())
        used_metrics.append(m)

    if not S_rows:
        raise ValueError("선택된 Metric으로 민감도 행렬을 만들 수 없습니다.")

    S = np.vstack(S_rows)
    b = np.array(b_list)
    y_mean = np.array(y_mean_list)
    return S, b, y_mean, used_metrics

def inverse_estimate(
    df: pd.DataFrame,
    metrics,
    Z_ref,
    Z_true,
    pupil_shape: str,
    NA: float,
    sigma_inner: float,
    sigma_outer: float,
    lambda_reg: float = 0.1,
):
    """
    Metric 개수에 따라 두 가지 경로:
    - 1개: 단일-파라미터 최소노름 해 (그 Metric이 민감한 Z 조합만 추정)
    - 2개 이상: 정규화된 최소제곱 + Tikhonov regularization
    """
    S, b, y_mean, used_metrics = build_sensitivity_matrix(df, metrics)

    # Pupil / NA scaling
    scale = pupil_sensitivity_scale(pupil_shape, NA, sigma_inner, sigma_outer)
    S_scaled = S * scale[np.newaxis, :]

    # ----- (A) Metric이 1개뿐인 경우: 단일-파라미터 모드 -----
    if len(used_metrics) == 1:
        s = S_scaled[0].astype(float)   # (37,)
        y_eff = float(y_mean[0] - b[0])

        norm_s2 = float(np.dot(s, s))
        if norm_s2 < 1e-8:
            raise ValueError(
                f"{used_metrics[0]} 는 Zernike에 대한 민감도가 너무 작아서 "
                "단일 파라미터 역추적이 불안정합니다."
            )

        # 최소 노름 해: ΔZ_est = (y_eff / (||s||^2 + λ)) * s
        alpha = y_eff / (norm_s2 + lambda_reg)
        dZ_est = alpha * s            # (37,)

        Z_reference = np.asarray(Z_ref, dtype=float)
        Z_actual    = np.asarray(Z_true, dtype=float)   # 시뮬 검증용
        Z_est       = Z_reference + dZ_est

        return Z_est, Z_actual, Z_reference, used_metrics

    # ----- (B) Metric이 2개 이상인 경우: 기존 regularized LS -----
    # 정규화(스케일링)로 수치 안정성 개선
    row_std = np.linalg.norm(S_scaled, axis=1, keepdims=True) + 1e-8
    S_norm = S_scaled / row_std

    y_eff = (y_mean - b)
    y_norm = y_eff / (np.std(y_eff) + 1e-8)

    STS = S_norm.T @ S_norm
    num_z = STS.shape[0]
    STS_reg = STS + lambda_reg * np.eye(num_z)
    STy = S_norm.T @ y_norm

    Z_est = np.linalg.solve(STS_reg, STy)

    Z_actual = np.asarray(Z_true, dtype=float)
    Z_reference = np.asarray(Z_ref, dtype=float)
    return Z_est, Z_actual, Z_reference, used_metrics

def evaluate_estimation(Z_est, Z_actual, top_k=10):
    diff = Z_est - Z_actual
    rmse = float(np.sqrt(np.mean(diff**2)))
    corr = float(np.corrcoef(Z_est, Z_actual)[0, 1])
    idx_actual = np.argsort(np.abs(Z_actual))[::-1][:top_k]
    idx_est = np.argsort(np.abs(Z_est))[::-1][:top_k]
    hit = len(set(idx_actual) & set(idx_est)) / top_k
    return {
        "rmse": rmse,
        "corr": corr,
        "hit_topk": hit,
        "idx_actual": idx_actual,
        "idx_est": idx_est,
        "diff": diff,
    }

def show_inverse_popup(Z_est, Z_actual, Z_ref, eval_info, title_suffix, culprits, used_metrics):
    terms = Z_COLS
    diff = eval_info["diff"]
    idx_est = eval_info["idx_est"]
    top_terms = [terms[i] for i in idx_est]
    top_est = Z_est[idx_est]
    top_act = Z_actual[idx_est]
    top_ref = Z_ref[idx_est]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax1, ax2 = axes

    # Panel 1: Full spectrum
    ax1.plot(terms, Z_ref,   label="Z_ref (Reference)", linestyle="--", marker=".", markersize=3)
    ax1.plot(terms, Z_actual, label="Z_true (Actual)", marker="o", markersize=3)
    ax1.plot(terms, Z_est,    label="Z_est (Estimated)", marker="x", markersize=3)
    ax1.plot(terms, diff,     label="Z_est - Z_true", marker="d", markersize=3)
    ax1.set_xticks(range(len(terms)))
    ax1.set_xticklabels(terms, rotation=90, fontsize=7)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.set_title(
        f"[{title_suffix}] Full Z Spectrum\n"
        f"RMSE={eval_info['rmse']:.3f}, "
        f"Corr={eval_info['corr']:.3f}, Hit@10={eval_info['hit_topk']*100:.1f}%\n"
        f"Culprit(ΔZ) ~ Z{culprits[0]}, Z{culprits[1]}, Z{culprits[2]}"
    )
    ax1.legend(fontsize=8)

    # Panel 2: Top-k
    x = np.arange(len(top_terms))
    width = 0.25
    ax2.bar(x - width, top_ref, width, label="Z_ref")
    ax2.bar(x,         top_act, width, label="Z_true")
    ax2.bar(x + width, top_est, width, label="Z_est")
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_terms, rotation=45)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2.set_title(f"[{title_suffix}] Top-{len(top_terms)} Zernike Comparison (Ref vs True vs Est)\n"
                  f"Metrics used: {', '.join(used_metrics)}")
    ax2.set_ylabel("Z value")
    ax2.legend(fontsize=8)

    plt.tight_layout()

    # Tkinter 팝업
    win = tk.Toplevel()
    win.title("Inverse Zernike Estimation Result")
    win.geometry("1400x800")

    main_frame = ttk.Frame(win)
    main_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_frame, bg="white")
    vscroll = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    hscroll = ttk.Scrollbar(main_frame, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
    vscroll.pack(side=tk.RIGHT, fill=tk.Y)
    hscroll.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    inner = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")

    def on_config(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    inner.bind("<Configure>", on_config)

    fig_canvas = FigureCanvasTkAgg(fig, master=inner)
    fig_canvas.draw()
    widget = fig_canvas.get_tk_widget()
    widget.config(width=int(fig.get_size_inches()[0] * fig.dpi),
                  height=int(fig.get_size_inches()[1] * fig.dpi))
    widget.pack(fill=tk.BOTH, expand=True)

    toolbar_frame = ttk.Frame(win)
    toolbar_frame.pack(side=tk.TOP, fill=tk.X)
    toolbar = NavigationToolbar2Tk(fig_canvas, toolbar_frame)
    toolbar.update()

    def on_mousewheel(event):
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta/120)), "units")
        elif event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")

    canvas.bind("<Enter>", lambda _: canvas.bind_all("<MouseWheel>", on_mousewheel))
    canvas.bind("<Leave>", lambda _: canvas.unbind_all("<MouseWheel>"))
    canvas.bind("<Button-4>", on_mousewheel)
    canvas.bind("<Button-5>", on_mousewheel)

    def on_close():
        plt.close(fig)
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)

# ==========================
# Wafer Overlay Plot
# ==========================

def plot_wafer_overlay(df: pd.DataFrame, shots):
    fig, ax = plt.subplots(figsize=(8, 8))
    wafer_circle = Circle((0, 0), 150, fill=False, linestyle="--", linewidth=2)
    ax.add_patch(wafer_circle)

    FIELD_X, FIELD_Y = 26, 33
    for shot_id, cx, cy in shots:
        rect = Rectangle(
            (cx - FIELD_X/2, cy - FIELD_Y/2),
            FIELD_X, FIELD_Y,
            linewidth=0.5, edgecolor="lightgray", facecolor="none"
        )
        ax.add_patch(rect)

    mags = np.sqrt(df["dX"]**2 + df["dY"]**2)
    q = ax.quiver(df["Wafer_X"], df["Wafer_Y"], df["dX"], df["dY"], mags, cmap="jet", scale=10)
    fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04, label="Overlay mag (arb.)")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Wafer X (mm)")
    ax.set_ylabel("Wafer Y (mm)")
    ax.set_title("Simulated Wafer Overlay Vector Map")
    ax.grid(True, linestyle=":", alpha=0.3)
    plt.show()

# ==========================
# GUI
# ==========================

def launch_gui():
    df, shots, Z_ref, Z_true, culprits = generate_wafer_data(random_seed=123)

    root = tk.Tk()
    root.title("Zernike Lens Diagnosis & Inverse RCA Tool (Metric-select)")
    # 세로 길이를 넉넉하게 (이렇게 하면 Sigma/λ/Run 버튼까지 한 화면에 들어옵니다)
    root.geometry("900x900")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # --- Tab 1: Pre-Analysis ---
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Pre-Analysis")

    ttk.Label(tab1, text="Wafer / Overlay View", font=("Arial", 12, "bold")).pack(pady=10)
    ttk.Button(
        tab1,
        text="🗺️ Show Wafer Overlay Vector Map",
        command=lambda: plot_wafer_overlay(df, shots)
    ).pack(padx=20, pady=20, fill="x")

    # --- Tab 2: RCA ---
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="RCA (3-Panel)")

    ttk.Label(tab2, text="Root Cause Analysis (Metric 선택 후 실행)", font=("Arial", 12, "bold")).pack(pady=10)

    frame_metrics = ttk.Frame(tab2)
    frame_metrics.pack(fill="both", expand=True, padx=10, pady=5)

    canvas = tk.Canvas(frame_metrics)
    scrollbar = ttk.Scrollbar(frame_metrics, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    metric_vars_rca = {}
    for m in RCA_METRICS:
        var = tk.BooleanVar(value=False)
        metric_vars_rca[m] = var
        ttk.Checkbutton(scroll_frame, text=m, variable=var).pack(anchor="w")

    def select_all_metrics_rca():
        for v in metric_vars_rca.values():
            v.set(True)

    def clear_all_metrics_rca():
        for v in metric_vars_rca.values():
            v.set(False)

    btn_frame_rca = ttk.Frame(tab2)
    btn_frame_rca.pack(fill="x", pady=10)

    ttk.Button(btn_frame_rca, text="Select All", command=select_all_metrics_rca).pack(side="left", padx=10)
    ttk.Button(btn_frame_rca, text="Clear All", command=clear_all_metrics_rca).pack(side="left", padx=10)

    def run_rca():
        selected = [m for m, v in metric_vars_rca.items() if v.get()]
        try:
            rca_analyze(df, selected)
        except Exception as e:
            messagebox.showerror("오류", f"RCA 분석 중 오류 발생:\n{e}")

    ttk.Button(
        tab2,
        text="📊 Run RCA Analysis (3-Panel)",
        command=run_rca
    ).pack(fill="x", padx=20, pady=10)

    # --- Tab 3: Inverse Solver ---
    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text="Inverse Solver")

    ttk.Label(tab3, text="Inverse Zernike Estimation (Metric 선택 + Pupil/NA + Ref)", font=("Arial", 12, "bold")).pack(pady=10)

    # 3-1. Mode (프리셋)
    mode_var = tk.StringVar(value=list(MODE_DEFS.keys())[0])
    ttk.Label(tab3, text="역추정 모드 (프리셋):").pack(anchor="w", padx=20, pady=(5, 2))
    mode_combo = ttk.Combobox(tab3, textvariable=mode_var, values=list(MODE_DEFS.keys()), state="readonly")
    mode_combo.pack(fill="x", padx=20, pady=5)

    ttk.Label(tab3, text="이 모드에서 추천하는 Metric 목록:").pack(anchor="w", padx=20, pady=(5, 2))
    metrics_listbox = tk.Listbox(tab3, height=6)
    metrics_listbox.pack(fill="x", expand=False, padx=20, pady=5)

    # 3-2. 실제로 사용할 Metric 선택 UI
    ttk.Label(tab3, text="실제 역추적에 사용할 Metric 선택:").pack(anchor="w", padx=20, pady=(10, 2))
    frame_inv_metrics = ttk.Frame(tab3)
    frame_inv_metrics.pack(fill="both", expand=True, padx=20, pady=5)

    canvas_inv = tk.Canvas(frame_inv_metrics)
    scrollbar_inv = ttk.Scrollbar(frame_inv_metrics, orient="vertical", command=canvas_inv.yview)
    scroll_frame_inv = ttk.Frame(canvas_inv)
    scroll_frame_inv.bind(
        "<Configure>",
        lambda e: canvas_inv.configure(scrollregion=canvas_inv.bbox("all"))
    )
    canvas_inv.create_window((0, 0), window=scroll_frame_inv, anchor="nw")
    canvas_inv.configure(yscrollcommand=scrollbar_inv.set)

    canvas_inv.pack(side="left", fill="both", expand=True)
    scrollbar_inv.pack(side="right", fill="y")

    metric_vars_inv = {}
    for m in RCA_METRICS:
        var = tk.BooleanVar(value=False)
        metric_vars_inv[m] = var
        ttk.Checkbutton(scroll_frame_inv, text=m, variable=var).pack(anchor="w")

    def refresh_metric_list(*args):
        metrics_listbox.delete(0, tk.END)
        preset = MODE_DEFS.get(mode_var.get(), [])
        for m in preset:
            metrics_listbox.insert(tk.END, m)

        # 프리셋 선택 시, 역추적 Metric 체크 상태도 맞춰 줌
        for v in metric_vars_inv.values():
            v.set(False)
        for m in preset:
            if m in metric_vars_inv:
                metric_vars_inv[m].set(True)

    mode_combo.bind("<<ComboboxSelected>>", refresh_metric_list)
    refresh_metric_list()

    def select_all_metrics_inv():
        for v in metric_vars_inv.values():
            v.set(True)

    def clear_all_metrics_inv():
        for v in metric_vars_inv.values():
            v.set(False)

    btn_frame_inv_sel = ttk.Frame(tab3)
    btn_frame_inv_sel.pack(fill="x", pady=5)
    ttk.Button(btn_frame_inv_sel, text="Select All", command=select_all_metrics_inv).pack(side="left", padx=10)
    ttk.Button(btn_frame_inv_sel, text="Clear All", command=clear_all_metrics_inv).pack(side="left", padx=10)

    # 3-3. 노광 / Pupil / Regularization 입력
    pupil_var = tk.StringVar(value="ANNULAR")   # 기본값: ANNULAR
    na_var = tk.DoubleVar(value=0.85)
    sig_in_var = tk.DoubleVar(value=0.4)
    sig_out_var = tk.DoubleVar(value=0.9)
    lambda_var = tk.DoubleVar(value=0.1)

    param_frame = ttk.LabelFrame(tab3, text="Exposure / Pupil Parameters")
    param_frame.pack(fill="x", padx=20, pady=10)

    ttk.Label(param_frame, text="Pupil Shape:").grid(row=0, column=0, padx=5, pady=2, sticky="e")
    pupil_cb = ttk.Combobox(
        param_frame,
        textvariable=pupil_var,
        # UI에 보여지는 이름들을 일반 조명계 이름으로 변경
        values=["CONVENTIONAL", "ANNULAR", "X-POLE", "Y-POLE", "X&Y-POLE"],
        state="readonly",
        width=14
    )
    pupil_cb.grid(row=0, column=1, padx=5, pady=2, sticky="w")

    ttk.Label(param_frame, text="NA:").grid(row=1, column=0, padx=5, pady=2, sticky="e")
    ttk.Entry(param_frame, textvariable=na_var, width=8).grid(row=1, column=1, padx=5, pady=2, sticky="w")

    ttk.Label(param_frame, text="Sigma Inner:").grid(row=2, column=0, padx=5, pady=2, sticky="e")
    ttk.Entry(param_frame, textvariable=sig_in_var, width=8).grid(row=2, column=1, padx=5, pady=2, sticky="w")

    ttk.Label(param_frame, text="Sigma Outer:").grid(row=3, column=0, padx=5, pady=2, sticky="e")
    ttk.Entry(param_frame, textvariable=sig_out_var, width=8).grid(row=3, column=1, padx=5, pady=2, sticky="w")

    ttk.Label(param_frame, text="Lambda (Reg.):").grid(row=4, column=0, padx=5, pady=2, sticky="e")
    ttk.Entry(param_frame, textvariable=lambda_var, width=8).grid(row=4, column=1, padx=5, pady=2, sticky="w")

    def run_inverse():
        try:
            selected_metrics = [m for m, v in metric_vars_inv.items() if v.get()]
            if not selected_metrics:
                selected_metrics = MODE_DEFS.get(mode_var.get(), [])
                if not selected_metrics:
                    messagebox.showwarning("경고", "역추적에 사용할 Metric이 없습니다.")
                    return

            pupil = pupil_var.get()
            NA = float(na_var.get())
            sig_in = float(sig_in_var.get())
            sig_out = float(sig_out_var.get())
            lam = float(lambda_var.get())

            if sig_in >= sig_out:
                messagebox.showwarning("경고", "Sigma Inner는 Sigma Outer보다 작아야 합니다.")
                return
            if not (0.6 <= NA <= 1.0):
                messagebox.showwarning("경고", "NA 값이 비정상적입니다. (0.6~1.0 권장)")
                return

            Z_est, Z_act, Z_reference, used_metrics = inverse_estimate(
                df, selected_metrics, Z_ref, Z_true,
                pupil, NA, sig_in, sig_out,
                lambda_reg=lam
            )
            eval_info = evaluate_estimation(Z_est, Z_act, top_k=10)
            mode_name = mode_var.get()
            title_suffix = f"{mode_name} / Metrics: {', '.join(used_metrics)}"
            show_inverse_popup(Z_est, Z_act, Z_reference, eval_info, title_suffix, culprits, used_metrics)
        except Exception as e:
            messagebox.showerror("오류", f"역추적 실행 중 오류 발생:\n{e}")

    ttk.Button(
        tab3,
        text="🔁 Run Inverse Estimation",
        command=run_inverse
    ).pack(fill="x", padx=20, pady=10)

    root.mainloop()

if __name__ == "__main__":
    launch_gui()
