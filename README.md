# TinyML Noise-Reduction on the STM32 NUCLEO-F446RE
This project demonstrates the implementation and evaluation of real-time noise reduction on a STM32 Nucleo microcontroller, using a quantized Artificial Neural-Network.
The Network handles three different types of synthetic noise (as well as their combination) from a 5 Hz sine wave:
1. White noise (Gaussian White Noise)
2. Echo (Phase-shifted signal superposition)
3. Quantization noise (ADC simulation)

## Project structure
- network.py: Python pipeline for signal generation, PyTorch model training, post-training quantization (8-bit fixed-point), data export, and C-code header generation.
- validate_fixedpoint_data.py: Validate the exported data from network.py by simulating the exact bit-true integer arithmetic designed for the microcontroller implementation. Ensures that file format, parsed data, and fixed-point scaling match the expected mathematical results before the hardware implementation.
- generate_header.py: Automation script that converts exported weights, biases, and scaling factors into a C-code header.
- nn_data.h: Generated C-code header file, containing the network configuration/data for all evaluation scenarios.
- main.c: STM32 firmware. Receives noisy signal windows via UART, executes the fixed-point inference, and measures CPU execution cycles.
- inference.c: Contains the fixed-point logic for the hardware neural network inference.
- run_evaluation.py: PC-side evaluation script. Handles serial hardware communication, verifies bit-accuracy (hardware prediction vs. software prediction), and calculates the final signal metrics (Mean-Squarred-Error, Signal-to-Noise ratio, energy efficiency metrics)

## Evaluated metrics
After hardware inference, the system prints an evaluation-report for each noise scenario:
- Mean-Squarred-Error: MSE for "clean signal vs. noisy signal", and "clean signal vs. hardware prediction", as well as the overall MSE improvement achieved by the neural network (noise eliminated by the network).
- Signal-to-Noise ratio: SNR for "clean signal vs. noisy signal", and "clean signal vs. hardware prediction", as well as the overall SNR improvement achieved by the neural network (gain in signal quality).
- Hardware validation: MSE for "software-calculated prediction vs. hardware prediction" (compares the fixed-point calculation of the hardware against the Python simulation (Target value = 0)).
- Hardware Power and Energy consumption: Data-sheet-based estimates for the active power draw (mW) and the energy consumed per inference window (μJ) based on the hardware cycle ccounts.
