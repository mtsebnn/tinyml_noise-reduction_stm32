import numpy as np
from pathlib import Path
from textwrap import dedent
import subprocess
import serial
import time
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
COM_PORT = "COM4"
BAUD_RATE = 115200
NUM_TEST_SAMPLES = 200 # One complete sin-wave needs 20 samples (samp per wave = T * samples rate = (1 / freq) * sample rate = (1/5Hz) * 100 Hz = 20)

scenarios = ["only_whitenoise", "only_echo", "only_quantization", "whitenoise_echo", 
             "whitenoise_quantization", "echo_quantization", "combined"]


def load_test_data(path, scenario):
    try:
        clean = np.load(path / f"clean_signal_{scenario}.npy") # clean signal for evaluation
        noisy = np.load(path / f"noisy_signal_{scenario}.npy")
        s_output = np.load(BASE_DIR / "network_data" / f"output_scalingfactor.npy") # scaling factor for dequantization of hardware predictions (for clean vs. hardw. pred.)
        s_input = np.load(BASE_DIR / "network_data" / f"input_scalingfactor.npy")
    except Exception:
        print(f"Could not load all .npy files for {scenario}: {e}")
        clean = None
        noisy = None
        s_output = 1
        s_input = 1

    tv = path / f"neuronTV.dat"
    inputs = []
    targets = []
    try:
        with open(tv, "r") as f:
            for line in f:
                if not line.strip():
                    continue # skip empty lines
                
                # split line into input and target
                values = [int(v) for v in line.split()]
                inputs.append(values[:16])
                targets.append(values[16])
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Testvector '{tv.name}' is missing.")
    except ValueError as e:
        raise ValueError(f"Corrupted data in '{tv.name}': {e}")

    return np.array(inputs, dtype=np.int8), np.array(targets, dtype=np.int32), np.array(clean), np.array(noisy), s_output, s_input


def get_memory_usage():
    elf_dir = BASE_DIR / "network.elf"
    if not elf_dir.exists():
        return {}
    
    result = subprocess.run(["arm-none-eabi-nm", "-S", "--size-sort" , elf_dir], capture_output=True, text=True)
    
    # memory
    usage = {
        "nn": {"flash": 0, "ram": 0},
        "kalman": {"flash": 0, "ram": 0},
        "wiener": {"flash": 0, "ram": 0},
        "iir": {"flash": 0, "ram": 0},
    }

    # parsing nm-output lines
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue

        # nm -S format: [address] [size] [type] [name]
        # i.e. 08000118 000000c0 T predict_nn
        usage_bytes = int(parts[1], 16)
        sym_type = parts[2].upper()
        sym_name = parts[3].lower()

        # T/W/V = code, R = read only (flash), D/B = ram (data/bss)
        if "kalman" in sym_name:
            if sym_type in ["T", "W", "V", "R"]: usage["kalman"]["flash"] += usage_bytes
            else: usage["kalman"]["ram"] += usage_bytes
        elif "wiener" in sym_name:
            if sym_type in ["T", "W", "V", "R"]: usage["wiener"]["flash"] += usage_bytes
            else: usage["wiener"]["ram"] += usage_bytes
        elif "iir" in sym_name:
            if sym_type in ["T", "W", "V", "R"]: usage["iir"]["flash"] += usage_bytes
            else: usage["iir"]["ram"] += usage_bytes
        elif sym_name in ["predict_nn", "data", "w1", "b1", "w2", "b2"]:
            if sym_type in ["T", "W", "V", "R"]: usage["nn"]["flash"] += usage_bytes
            else: usage["nn"]["ram"] += usage_bytes

    return usage


def evaluate(expected, prediction):
    expected = np.array(expected)
    prediction = np.array(prediction)

    # MSE (^2 to prevent pos. and neg. errors from canceling out)
    mse = np.mean((expected - prediction)**2) 

    # signal to noise ratio
    signal_power = np.mean(expected**2)
    noise_power = mse if mse > 0 else 1e-6
    snr_db = 10 * np.log10(signal_power / noise_power)

    return mse, snr_db

def plot_signals(path, samples, scenario, filter_name, clean, noisy, nn, filter):
    plt.figure(figsize=(12, 5))

    plt.plot(samples, clean[:80], label="Clean signal (target)", color="black", alpha=1, linewidth=2)
    plt.plot(samples, noisy[:80], label="Noisy signal (input)", color="red", alpha=0.5, linewidth=2)
    plt.plot(samples, nn[:80], label="Filtered signal (Neural Network)", color="lightgreen", alpha=1, linewidth=1.5)
    plt.plot(samples, filter[:80], label=f"Filtered signal ({filter_name})", color="lightblue", alpha=1, linewidth=1.5)

    plt.grid(True, linestyle="--", alpha=0.5, linewidth=1)
    plt.title(f"Compared signals - scenario: {scenario.upper()}")
    plt.xlabel("sample index")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()

    plt.savefig(path / f"plot_signals_{scenario}_{filter_name}.png", dpi=150)
    plt.close()

def plot_efficiency(path, scenario, data):
    plt.figure(figsize=(12, 5))

    colors = {"nn": "lightgreen", "kalman": "blue", "wiener": "red", "iir": "purple"}
    labels = {"nn": "Neural Network", "kalman": "Kalman", "wiener": "Wiener", "iir": "IIR"}

    for algorithm, (time_us, imp_pct) in data.items():
        plt.scatter(time_us, imp_pct, color=colors[algorithm], marker=".", s=120, label=labels[algorithm], zorder=3)
        plt.text(time_us * 1.1, imp_pct, f" {labels[algorithm]}\n ({time_us:.1f} μs, {imp_pct:.1f}%)", fontsize=9, verticalalignment="center", zorder=4)

    plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.5, linewidth=1, which="both")
    plt.title(f"Efficiency vs. Quality - scenario: {scenario.upper()}")
    plt.xlabel("Average inference time in μs (logarithmic scale)")
    plt.ylabel("MSE Improvement in %")

    all_pct = [val[1] for val in data.values()]
    plt.ylim(min(all_pct) - 20 if min(all_pct) < 0 else -10, max(all_pct) + 20)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="-")
    plt.tight_layout()

    plt.savefig(path / f"plot_efficiency_vs_quality_{scenario}.png", dpi=150)
    plt.close()

def plot_error(path, data):
    scenarios = list(data.keys())
    if not scenarios:
        return
    
    algorithms = ["nn", "kalman", "wiener", "iir"]
    colors = {"nn": "lightgreen", "kalman": "blue", "wiener": "red", "iir": "purple"}
    labels = {"nn": "Neural Network", "kalman": "Kalman", "wiener": "Wiener", "iir": "IIR"}

    x = np.arange(len(scenarios))
    width = 0.2

    fig, p = plt.subplots(figsize=(12, 5))

    for i, algorithm in enumerate(algorithms):
        offset = x + (i - 1.5) * width
        percent = [data[s][algorithm] for s in scenarios]
        p.bar(offset, percent, width, color=colors[algorithm], label=labels[algorithm])

    p.set_ylabel("MSE Improvement in %")
    p.set_title("Comparison of filter performance (MSE Improvement) for every scenario")
    p.set_xticks(x)
    p.set_xticklabels([s.upper() for s in scenarios], rotation=10, ha="right")
    p.legend()
    p.grid(True, axis="y", linestyle="--", alpha=0.5, linewidth=1)
    p.axhline(0, color="black", linewidth=1)

    plt.tight_layout()
    plt.savefig(path / f"overall_mse_improvement.png", dpi=150)
    plt.close()


def run_hardware_evaluation():
    report = dedent (f"""\
    -----------------------------------------------------------
        Noise-Reduction Network Hardware Evaluation Results
    -----------------------------------------------------------                
    """)

    error_data = {}

    try:
        with serial.Serial(COM_PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(2) # short break for bootloader

            for scenario in scenarios:
                print(f"Processing evaluation for scenario '{scenario}'. Please wait...")

                # load test data
                scenario_dir = BASE_DIR / "scenario_data" / f"scenario_{scenario}"
                if not scenario_dir.exists():
                    print(f"Skip '{scenario}': Folder '{scenario_dir.name}' could not be found.")
                    continue
                try:
                    all_inputs, all_targets, all_clean_signals, all_noisy_signals, S_output, S_input = load_test_data(scenario_dir, scenario)
                except FileNotFoundError as e:
                    print(f"Skip '{scenario}': {e}")
                    continue
                except ValueError as e:
                    print(f"Skip '{scenario}': Data corruption detected. Details: {e}")
                    continue

                num_tests = min(NUM_TEST_SAMPLES, len(all_inputs)) # avoid error for NUM_TEST_SAMPLES > total_num_samples

                hardw_results = {
                    "nn": {"pred": [], "cycles": []},
                    "kalman": {"pred": [], "cycles": []},
                    "wiener": {"pred": [], "cycles": []},
                    "iir": {"pred": [], "cycles": []},
                }

                expected_targets = []
                clean_targets = []

                ser.write('R'.encode("utf-8")) # reset filters before running scenario
                time.sleep(0.02)

                # inference and evaluation
                for i in range(num_tests):
                    input_vector = all_inputs[i]
                    
                    ser.write('S'.encode("utf-8")) # send command
                    ser.write(input_vector.tobytes()) # send 16 byte input-vector (in binary)

                    # process response 
                    response = ser.readline().decode("utf-8").strip()
                    if response:
                        parts = response.split(",")
                        if len(parts) == 8:
                            # extract predictions and cycle counts
                            hardw_results["nn"]["cycles"].append(int(parts[0]))
                            hardw_results["nn"]["pred"].append(int(parts[1]))
                            
                            hardw_results["kalman"]["cycles"].append(int(parts[2]))
                            hardw_results["kalman"]["pred"].append(int(parts[3]))

                            hardw_results["wiener"]["cycles"].append(int(parts[4]))
                            hardw_results["wiener"]["pred"].append(int(parts[5]))

                            hardw_results["iir"]["cycles"].append(int(parts[6]))
                            hardw_results["iir"]["pred"].append(int(parts[7]))
                            
                            expected_targets.append(all_targets[i])

                            # only append clean signal if the hardware inference was successfull
                            # -> else clean_targets would get its value even if inference crashes which would lead to a np.array dimension error
                            if all_clean_signals is not None:
                                clean_targets.append(all_clean_signals[i]) 
                    
                    time.sleep(0.05) # short stability break

                    
                if len(hardw_results["nn"]["pred"]) == num_tests:
                    report += dedent(f"""\n\n--- SCENARIO: {scenario.upper()} ({num_tests} TEST SAMPLES) ---""")

                    # overall noise
                    noisy_signal_float = all_inputs[:num_tests, 8] * S_input
                    mse_noise, snr_noise = evaluate(clean_targets, noisy_signal_float)
                    report += dedent(f"""
                    Initial Noise MSE (clean signal vs. noisy signal): {mse_noise:.4f}
                    Initial Signal-to-Noise Ratio SNR (clean signal vs. noisy signal): {snr_noise:.2f} dB
                    """)

                    nn_predictions = None
                    efficiency_data = {}
                    error_data[scenario] = {}

                    memory_usage = get_memory_usage()

                    for algorithm in ["nn", "kalman", "wiener", "iir"]:
                        avg_cycles = np.mean(hardw_results[algorithm]["cycles"])
                        avg_inference_time = (avg_cycles / 16000000) * 1000000 # avg. inference time in μs 

                        if algorithm == "nn":
                            hardw_predictions_float = nn_predictions = np.array(hardw_results[algorithm]["pred"]) * S_output # dequantization to int -> float
                            # validation hardware prediction vs. expected prediction calculated by python (int vs. int) (should be the same -> MSE = 0)
                            mse_hardware_validation, snr_hardware_validation = evaluate(expected_targets, hardw_results[algorithm]["pred"])

                            report += dedent(f"MSE (software-calculated prediction vs. hardware prediction => should be near 0): {mse_hardware_validation:.4f}\n")

                        else:
                            hardw_predictions_float = np.array(hardw_results[algorithm]["pred"]) * S_input

                        # evaluation hardware predictions vs. clean sin-signal (float vs. float)
                        mse_reduct, snr_reduct = evaluate(clean_targets, hardw_predictions_float)
                        mse_imp = mse_noise - mse_reduct
                        snr_imp = snr_reduct - snr_noise

                        # plot data
                        mse_imp_percent = (mse_imp / mse_noise) * 100
                        efficiency_data[algorithm] = (avg_inference_time, mse_imp_percent)
                        error_data[scenario][algorithm] = mse_imp_percent

                        # get memory usage for each algorithm
                        flash_usage = memory_usage.get(algorithm, {}).get("flash", 0)
                        ram_usage = memory_usage.get(algorithm, {}).get("ram", 0)
                    
                        algo_name = None
                        match algorithm:
                            case "nn":
                                algo_name = "Neural Network"
                            case "kalman":
                                algo_name = "Kalman algorithm"
                            case "wiener":
                                algo_name = "Wiener algorithm (FIR-filter)"
                            case "iir":
                                algo_name = "IIR (low-pass) filter"

                        report += dedent(f"""
                        >>> ALGORITHM: {algo_name}                                         
        
                            Average inference time: {avg_inference_time:.2f} μs ({avg_cycles:.0f} CPU-cycles)
                            Inference rate: {1/ (avg_inference_time / 1000000):.0f} Hz (samples/s)

                            Memory usage (static):
                                - Flash-Usage: {flash_usage} Bytes / 512KB ({(flash_usage/(512*1024))*100:.2f}%)
                                - RAM-Usage: {ram_usage} Bytes / 128KB ({(ram_usage/(128*1024))*100:.2f}%)

                            Noise-Reduction MSE (clean signal vs. hardware prediction): {mse_reduct:.4f}
                            MSE Improvement: {(mse_imp / mse_noise) * 100:.2f}% ({(-1)*mse_imp:.4f})

                            Noise-Reduction Signal-to-Noise Ratio SNR (clean signal vs. hardware prediction): {snr_reduct:.2f} dB
                            SNR Improvement: {snr_imp:+.2f} dB
                        \n""")

                        if algorithm != "nn":
                            # plot scenario specific algorithm signals
                            plot_signals(scenario_dir, np.arange(80), scenario, algo_name , clean_targets, noisy_signal_float, nn_predictions, hardw_predictions_float)

                    plot_efficiency(scenario_dir, scenario, efficiency_data)

                else:
                    report += dedent(f"""\
                    --- Scenario: {scenario.upper()} evaluation error. ---
                    """)
    except serial.SerialException as e:
        print(f"Error at COM-Port: {e}")
        return
    
    plot_error(BASE_DIR, error_data)
    
    # console and file ouput
    print("\n" + report)
    with open(BASE_DIR / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    run_hardware_evaluation()