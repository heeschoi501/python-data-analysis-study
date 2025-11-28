import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -------------------------------------------------------
# 초기 설정
# -------------------------------------------------------
NUM_ZERNIKE = 30
np.random.seed(42)
TARGET_VALUE = 0.0

TARGET_METRICS = ['Average_X', 'Average_Y', '3Sigma_X', '3Sigma_Y', 'Residual_X', 'Residual_Y']
Z_COLS = [f'Z{i}' for i in range(1, NUM_ZERNIKE + 1)]


# -------------------------------------------------------
# 데이터 생성
# -------------------------------------------------------
def generate_zernike_data():
    measured = np.random.normal(0, 0.8, NUM_ZERNIKE)
    measured[1] = 4.5
    measured[7] = -3.0
    measured[19] = 2.5
    return pd.Series(measured, index=Z_COLS)


def get_deviation_results(measured_z):
    target_z = pd.Series([TARGET_VALUE] * NUM_ZERNIKE, index=Z_COLS)
    deviation = target_z - measured_z
    return {metric: deviation for metric in TARGET_METRICS}


# -------------------------------------------------------
# Tkinter GUI — Metric 선택창 (스크롤 + Select All/Clear All)
# -------------------------------------------------------
def select_metrics_gui():

    root = tk.Tk()
    root.title("Metric 선택")
    root.geometry("350x400")

    tk.Label(root, text="표시할 Metric 선택:", font=("Arial", 12)).pack(pady=5)

    # 프레임 + 캔버스 + 스크롤바 구성
    container = ttk.Frame(root)
    canvas = tk.Canvas(container, height=220)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    container.pack(fill="both", expand=True, padx=10)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # 체크박스 변수 생성
    var_dict = {m: tk.BooleanVar(value=True) for m in TARGET_METRICS}

    for m in TARGET_METRICS:
        ttk.Checkbutton(scroll_frame, text=m, variable=var_dict[m]).pack(anchor="w")

    # 버튼 영역
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=15)

    def select_all():
        for v in var_dict.values():
            v.set(True)

    def clear_all():
        for v in var_dict.values():
            v.set(False)

    ttk.Button(btn_frame, text="Select All", command=select_all).grid(row=0, column=0, padx=5)
    ttk.Button(btn_frame, text="Clear All", command=clear_all).grid(row=0, column=1, padx=5)

    selected = []

    def confirm():
        nonlocal selected
        selected = [m for m, v in var_dict.items() if v.get()]
        if len(selected) == 0:
            messagebox.showwarning("Warning", "하나 이상 선택해야 합니다.")
            return
        root.destroy()

    ttk.Button(root, text="확인", command=confirm).pack(pady=10)

    root.mainloop()
    return selected


# -------------------------------------------------------
# Dashboard 생성
# -------------------------------------------------------
def run_consolidated_dashboard():
    measured_z = generate_zernike_data()
    deviation_dict = get_deviation_results(measured_z)
    target_z = pd.Series([TARGET_VALUE] * NUM_ZERNIKE, index=Z_COLS)

    # 🔥 Metric 선택 GUI 실행
    selected_metrics = select_metrics_gui()

    n = len(selected_metrics)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    plt.style.use("seaborn-v0_8-darkgrid")  # 🔥 테마 적용

    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = np.array(axes).reshape(-1)

    # 🔥 사용 안 하는 subplot 삭제
    for ax in axes[n:]:
        ax.remove()

    # 그래프 그리기
    for i, metric in enumerate(selected_metrics):
        ax = axes[i]
        deviation = deviation_dict[metric]

        ax.plot(Z_COLS, target_z, label="ref (Target)", linestyle="--")
        ax.plot(Z_COLS, measured_z, label="actual", marker="o")
        ax.plot(Z_COLS, deviation, label="result", marker="s")

        ax.set_title(metric, fontsize=12)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticklabels(Z_COLS, rotation=90, fontsize=7)
        ax.set_ylim(-6, 6)
        ax.legend(fontsize=8)

    plt.tight_layout()

    # -------------------------------------------------------
    # 🔥 Export 기능 추가 (PNG/PDF 저장)
    # -------------------------------------------------------
    def save_png():
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG Files", "*.png")]
        )
        if path:
            fig.savefig(path)
            messagebox.showinfo("Saved", "PNG 저장 완료!")

    def save_pdf():
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")]
        )
        if path:
            fig.savefig(path)
            messagebox.showinfo("Saved", "PDF 저장 완료!")

    # Tkinter 작은 메뉴창 생성
    save_win = tk.Tk()
    save_win.title("Export Options")
    save_win.geometry("250x120")

    ttk.Label(save_win, text="그래프 저장:", font=("Arial", 12)).pack(pady=10)
    ttk.Button(save_win, text="Export as PNG", command=save_png).pack(pady=5)
    ttk.Button(save_win, text="Export as PDF", command=save_pdf).pack(pady=5)

    plt.show()


# -------------------------------------------------------
if __name__ == "__main__":
    run_consolidated_dashboard()
