🔬 Zernike Target Deviation Analyzer (6-Metric Overlay RCA)
Purpose
This project is a research prototype designed for simulation and academic analysis of Zernike deviation in lithography processes.
It is not intended for production or manufacturing environments.

1. Core Concept: Target Deviation
The tool applies Deviation Analysis, calculating the difference between the desired and measured states:
Deviation Vector=Target Zernike (Reference)−Actual Zernike (Measured)\text{Deviation Vector} = \text{Target Zernike (Reference)} - \text{Actual Zernike (Measured)}Deviation Vector=Target Zernike (Reference)−Actual Zernike (Measured)
Since Target Z = 0, the deviation equals the measured Zernike.
Interpretation: Large deviations indicate discrepancies for study purposes, not real-time correction.

2. 📊 Analysis Scope and Metrics

The tool analyzes the Zernike spectrum across $\mathbf{6}$ distinct $\text{Overlay}$ components.


| Metric        | Physical Significance                  | Analysis Type       |
|--------------|----------------------------------------|----------------------|
| Average X/Y  | Overall image placement error          | Position Error       |
| 3Sigma X/Y   | Field Uniformity and Statistical Spread| Uniformity Error     |
| Residual X/Y | Unexplained errors after correction    | Unexplained Noise    |

3. Features (Prototype)

GUI Control (Tkinter): Select up to 6 metrics dynamically for visualization.
Visualization (Matplotlib): Combined plots showing:

Target (Reference)
Measured
Deviation


Execution:

Run the application using the local Python interpreter:

python3 track2_study/final_6metric_diagnosis_tool.py
