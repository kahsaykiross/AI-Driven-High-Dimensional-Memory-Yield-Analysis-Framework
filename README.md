# AI-Driven-High-Dimensional-Memory-Yield-Analysis-Framework
high-dimensional yield analysis, meta-learning, low-rank tensor approximation, and adaptive importance sampling, we can create a Python-based simulation skeleton to illustrate your methodology. This can later be extended for actual SRAM datasets and AI accelerator memory benchmarks.
AI-Driven High-Dimensional Memory Yield Analysis

Author: Kahsay Kiross Meresa
Email: kahsaykirossmtc@gmail.com

Overview

This repository provides a comprehensive Python framework for AI-driven high-dimensional yield analysis and reliability modeling of SRAM and AI accelerator memory systems. It combines meta-learning, tensor decomposition, adaptive importance sampling, tail-distribution modeling, and LVF2 Gaussian Mixture Models to analyze rare-event failures, timing-speculation, and system-level memory reliability.

The framework includes:

Automated report generation – produces multi-failure region heatmaps, rare-event heatmaps, LVF2 speed bin plots, and statistical summaries.

Interactive 3D dashboard – explore high-dimensional failure regions and rare events in real-time using Plotly Dash.

System-level benchmark evaluation – supports FPGA and AI accelerator memory datasets for industrial validation.

This project is ideal for academics, circuit designers, and engineers interested in next-generation memory yield analysis and high-dimensional reliability modeling.

Features

Real Memory Dataset Integration – load SRAM or AI accelerator memory measurements.

High-Dimensional Tensor Decomposition – CP or Tucker decomposition for scalable modeling.

Meta-Learning Neural Networks – fast, accurate yield prediction across process corners.

Adaptive Importance Sampling – focuses on multi-failure regions for efficiency.

Tail Distribution Analysis – captures rare but critical memory failures.

LVF2 Gaussian Mixture Timing Analysis – speed binning and timing-speculation modeling.

Automated Reporting – generates publication-ready figures and CSV tables.

3D Interactive Dashboard – visualize multi-failure regions and rare events interactively.

System-Level Benchmarking – evaluate SRAM and AI memory on FPGA/AI accelerator designs.

Installation

Clone the repository:

git clone https://github.com/kahsaykiross/ai-memory-yield.git
cd ai-memory-yield

Install dependencies (recommended via pip or conda):

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow tensorly plotly dash
Usage
1. Automated Report Generation
from report_generator import load_memory_dataset, build_meta_model, generate_report
from sklearn.model_selection import train_test_split

# Load dataset (CSV: features + yield column)
X, Y = load_memory_dataset("real_memory_dataset.csv")

# Train meta-learning model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
meta_model = build_meta_model(input_dim=X_train.shape[1])
meta_model.fit(X_train, Y_train, epochs=100, batch_size=128, validation_split=0.1)

# Generate report (figures + rare-event statistics)
X_reduced, Y_sampled, stats = generate_report(X_test, Y_test, meta_model, save_dir="memory_yield_report")

The report directory includes:

multi_failure_heatmap.png

rare_event_heatmap.png

lvf2_speed_bins.png

rare_event_statistics.csv

2. Interactive 3D Dashboard
from dashboard_3d import load_memory_dataset, tensor_decomposition, build_meta_model, adaptive_sampling, prepare_3d_data, build_dash_app
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
X, Y = load_memory_dataset("real_memory_dataset.csv")

# Tensor decomposition
X_reduced = tensor_decomposition(X, rank=10)

# Train meta-learning model
X_train, X_test, Y_train, Y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=42)
meta_model = build_meta_model(input_dim=X_train.shape[1])
meta_model.fit(X_train, Y_train, epochs=100, batch_size=128, validation_split=0.1)

# Adaptive importance sampling
X_sampled, Y_sampled = adaptive_sampling(meta_model, X_test, num_samples=1000)

# Prepare 3D data
X_3d, Y_vis, rare_mask, threshold = prepare_3d_data(X_sampled, Y_sampled)

# Launch interactive dashboard
app = build_dash_app(X_3d, Y_vis, pd.Series(rare_mask), threshold)
app.run_server(debug=True)

Open the dashboard in your browser to explore multi-failure regions and rare events interactively.

File Structure
ai-memory-yield/
│
├─ report_generator.py        # Automated report generation scripts
├─ dashboard_3d.py            # Interactive 3D dashboard scripts
├─ real_memory_dataset.csv    # Example SRAM/AI memory dataset (replace with your data)
├─ fpga_ai_benchmark.csv      # Optional system-level benchmark dataset
├─ README.md
└─ requirements.txt           # Python dependencies
References

Shi, X., Yan, H., Huang, Q., Xuan, C., Shi, L., He, L. (2021). A Compact High-Dimensional Yield Analysis Method Using Low-Rank Tensor Approximation. TODAES.

Wang, Z., Pang, L., Shi, X., Shi, L. (2024). Efficient Memory Circuits Yield Analysis and Optimization Framework via Meta-Learning. TCASII.

Pang, L., Yao, M., Shi, X., Yan, H., Shi, L. (2023). CharTM: Dynamic Stability Characterization for Memory. Microelectronics Journal.

Zhou, J., Huang, L., Xia, H., Cai, Y., Jin, L., Shi, X., et al. (2024). LVF2: Statistical Timing Model based on Gaussian Mixture. DAC.
