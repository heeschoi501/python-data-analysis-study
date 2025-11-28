import pandas as pd
import tkinter as tk
import os


print("✅ Python: 준비 완료!")

# OpenROAD 실행 여부 테스트
exit_code = os.system("openroad -version")
if exit_code == 0:
    print("✅ OpenROAD: Python 안에서 호출 성공!")
else:
    print("❌ OpenROAD: 호출 실패 (PATH 확인 필요)")

import subprocess

def show_text():
    # 1. OpenROAD 실행해서 버전 가져오기
    try:
        result = subprocess.run(['openroad', '-version'], capture_output=True, text=True)
        version_info = result.stdout.strip()
    except Exception:
        version_info = "OpenROAD를 찾을 수 없음"

    # 2. Tkinter 창 만들기
    root = tk.Tk()
    root.title("OpenROAD Report Viewer")
    root.geometry("600x400")

    # 3. 텍스트 박스 만들고 내용 넣기
    # font=("NanumGothic", 12) 추가
    text_box = tk.Text(root, height=20, width=70, font=("NanumGothic", 12))
    text_box.pack(pady=20)

    report = f"""
    ========================================
       [OpenROAD Analysis Report]
    ========================================
    ✅ 상태: 실행 성공
    ℹ️ 버전: {version_info}
    ========================================
    """
    text_box.insert(tk.END, report)

    # 4. 창 실행
    print("🖥️ GUI 창을 띄웁니다...")
    root.mainloop()

if __name__ == "__main__":
    show_text()