🔬 Zernike Target Deviation Analyzer (6-Metric Overlay RCA)

This project serves as the final deliverable for the Track 2: Data Analysis & Automation phase. It demonstrates the ability to perform Zernike deviation analysis on lithography process parameters, focusing on simulation for research purposes.

The tool provides an interactive, local GUI dashboard for comparative diagnosis between ideal and measured lens aberration states.

1. 🔑 Core Diagnostic Logic: Target Deviation

This tool utilizes the principle of Deviation Analysis, calculating the pure difference between the desired state and the measured state.

$$\text{Deviation Vector} = \text{Target Zernike (Reference)} - \text{Actual Zernike (Measured)}$$

Interpretation: A calculated deviation indicates a discrepancy between the expected performance of the lens and the measured input. (Target Z is 0, so the Deviation equals the Measured Zernike).

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

The application uses a Tkinter GUI for metric selection, allowing the user to dynamically select the metrics (up to 6) for simultaneous display.

B. Consolidated Visualization (Matplotlib)

The final report displays selected deviation results simultaneously on a single figure, with each plot showing three critical lines for comparison:

Reference (Target Z)

Actual (Measured Z)

Result (Deviation)

C. Execution Command

Run the application using the local Python interpreter:

python3 track2_study/final_6metric_diagnosis_tool.py
