🔬 Zernike Target Deviation Analyzer (6-Metric Overlay RCA)

This project serves as the final deliverable for the Track 2: Data Analysis & Automation phase, demonstrating the ability to perform precise Root Cause Analysis (RCA) on lithography process failures using Python.

The tool provides an interactive, local GUI dashboard for comparative diagnosis between ideal and measured lens aberration states.

🔬 Zernike Target Deviation Analyzer (6-Metric Overlay RCA)

This project serves as the final deliverable for the Track 2: Data Analysis & Automation phase, demonstrating the ability to perform precise Root Cause Analysis (RCA) on lithography process failures using Python.

The tool provides an interactive, local GUI dashboard for comparative diagnosis between ideal and measured lens aberration states.

1. 🔑 Core Diagnostic Logic: Target Deviation

This tool adheres to the critical diagnostic principle of Deviation Analysis, analyzing the pure difference between the desired state and the measured state.

$$\text{Deviation Vector} = \text{Target Zernike (Reference)} - \text{Actual Zernike (Measured)}$$

Interpretation: A large deviation indicates a significant physical flaw in the lens system (Target Z is 0, so the Deviation equals the Measured Zernike).

2. 📊 Analysis Scope and Metrics

The tool analyzes the Zernike spectrum across $\mathbf{6}$ distinct $\text{Overlay}$ components.

Metric

Physical Significance

Analysis Type

Average X/Y

Overall image placement error

Position Error

3Sigma X/Y

Intra-field uniformity and statistical spread

Uniformity Error

Residual X/Y

Unexplained errors remaining after systematic correction

Unexplained Noise

3. 🖥️ Tool Features

A. Local GUI Control (Tkinter)

The application uses a Tkinter GUI for metric selection, allowing the user to select multiple metrics (up to 6) for display without modifying the source code.

B. Consolidated Visualization (Matplotlib)

The final report displays selected metrics simultaneously on a single figure, with each plot showing three critical lines for comparison:

Reference (Target Z)

Actual (Measured Z)

Result (Deviation)

C. Execution Command

Run the application using the local Python interpreter:

python3 track2_study/final_6metric_diagnosis_tool.py


1. 🔑 Core Diagnostic Logic: Target Deviation

This tool adheres to the critical diagnostic principle of Deviation Analysis, analyzing the pure difference between the desired state and the measured state.

$$\text{Deviation Vector} = \text{Target Zernike (Reference)} - \text{Actual Zernike (Measured)}$$

Interpretation: A large deviation indicates a significant physical flaw in the lens system (Target Z is 0, so the Deviation equals the Measured Zernike).

2. 📊 Analysis Scope and Metrics

The tool analyzes the Zernike spectrum across $\mathbf{6}$ distinct $\text{Overlay}$ components.

Metric

Physical Significance

Analysis Type

Average X/Y

Overall image placement error

Position Error

3Sigma X/Y

Intra-field uniformity and statistical spread

Uniformity Error

Residual X/Y

Unexplained errors remaining after systematic correction

Unexplained Noise

3. 🖥️ Tool Features

A. Local GUI Control (Tkinter)

The application uses a Tkinter GUI for metric selection, allowing the user to select multiple metrics (up to 6) for display without modifying the source code.

B. Consolidated Visualization (Matplotlib)

The final report displays selected metrics simultaneously on a single figure, with each plot showing three critical lines for comparison:

Reference (Target Z)

Actual (Measured Z)

Result (Deviation)

C. Execution Command

Run the application using the local Python interpreter:

python3 track2_study/final_6metric_diagnosis_tool.py
