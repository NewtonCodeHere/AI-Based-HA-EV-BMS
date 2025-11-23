AI-Based FPGA Accelerator for Electric Vehicles’ Battery Management System

Project Status: Ongoing

Team Members:

    MANI BARATHI P

    VEL GNANA SATISH M

    RAJIV PRASAAD

📘 Project Overview

This project focuses on developing a high-accuracy, low-latency State-of-Charge (SoC) prediction engine for Lithium-ion batteries used in Electric Vehicles (EVs).
The work integrates AI-based SoC estimation (LSTM) with FPGA-based hardware acceleration, enabling real-time deployment inside Battery Management Systems (BMS).

The project replicates and extends the methodology used in the research paper:
➡️ “An FPGA-Based LSTM Accelerator for SoC Prediction in Lithium-Ion Batteries” (reference paper used for Phase-1).

🚀 Motivation

Traditional SoC estimation techniques—Coulomb counting, Kalman filters, and equivalent circuit modeling—are:

Model-dependent

Limited by linearity assumptions

Sensitive to noise, aging, and temperature variation

AI approaches overcome these limitations by learning nonlinear electrochemical dynamics directly from data.
However, deploying neural networks in embedded BMS systems requires low power, low latency, and real-time inference—which leads to FPGA acceleration.

🧠 Phase 1: AI Model Development (Completed)
📌 Dataset Used

Panasonic 18650PF Li-ion Battery Cycling Dataset
Provided by Dr. Phillip Kollmeyer (University of Wisconsin–Madison).

Cycles Used: 1–4 discharge cycles at 25°C

📊 Features (Inputs)

Voltage

Current

Battery Temperature

Timestamp

🎯 Target

State of Charge (SoC)

🛠 Preprocessing Steps

Loaded .mat files

Removed NaN & outliers

MinMax normalization

80/20 train–test split

Sliding window generation (look-back 60)

📈 Phase 1: LSTM Model Training (Completed)
🧩 LSTM Architecture

Look-back window: 60

Hidden units: 5

Optimizer: Adam, learning rate 0.1

Epochs: 100

Batch size: 60

📉 Achieved Accuracy

Training RMSE: 0.3438

Validation RMSE: 0.3681

Model training implemented and validated using Google Colab.
Colab notebook link:
🔗 https://colab.research.google.com/drive/1y1bPJLSouUYYb7cqpMXGproDoee-XFyw?usp=sharing

⚙️ Phase 1: Hardware Translation (Completed)
📥 Weight Extraction

Performed using:

model.get_weights()


Weights & biases exported into C++ header files as constant arrays.

🔧 C++ LSTM Inference Engine

Implements LSTM forward pass

Includes matrix multiplications, activations, cell updates

Written using HLS-synthesizable constructs

🏗 High-Level Synthesis

Using Xilinx Vitis HLS, the C++ model was:

Simulated

Optimized with HLS pragmas

Synthesized into RTL (Verilog/VHDL)

This RTL is ready for integration into:

Xilinx Zynq SoC

PYNQ-Z2 board

Any FPGA-based BMS prototype

🧩 Current Phase (Ongoing Work)

RTL verification

FPGA resource utilization analysis

Latency & power benchmarking

Preparing system-level integration for real BMS deployment

🎯 Final Goal

To build a fully functional FPGA-based SoC estimation subsystem capable of:

Real-time inference

Low latency

Low power consumption

High SoC prediction accuracy

Compatibility with real EV Battery Management Systems

📂 Repository Structure 
├── dataset/                 # Panasonic data files (.mat)
├── notebooks/               # Colab notebooks
├── src/
│   ├── python_model/        # LSTM training code
│   ├── hls_cpp/             # C++ inference code for Vitis HLS
│   ├── headers/             # Exported weight files (.h)
│   └── rtl/                 # Generated RTL (Verilog/VHDL)
├── docs/                    # Reports, diagrams, explanations
└── README.md                # Project documentation