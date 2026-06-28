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
NUM_TEST_SAMPLES = 100 # One complete sin-wave needs 20 samples (samp per wave = T * samples rate = (1 / freq) * sample rate = (1/5Hz) * 100 Hz = 20)

scenarios = ["only_whitenoise", "only_echo", "only_quantization", 
             "combined"]


def load_test_data(path, scenario):
    try:
        clean = np.load(path / f"clean_signal.npy") # clean signal for evaluation
        s_output = np.load(path / f"output_scalingfactor.npy") # scaling factor for dequantization of hardware predictions (for clean vs. hardw. pred.)
        s_input = np.load(path / f"input_scalingfactor.npy")
    except Exception:
        clean = None
        s_output = 1
        s_input = 1

    tv = path / f"neuronTV_{scenario}.dat"
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

    return np.array(inputs, dtype=np.int8), np.array(targets, dtype=np.int32), np.array(clean), s_output, s_input


def get_memory_usage():
    usage = subprocess.run(["arm-none-eabi-size", BASE_DIR / "neural_network.elf"], capture_output=True, text=True)
    lines = usage.stdout.strip().split("\n")
    if len(lines) > 1:
        values = lines[1].split()
        return (int(values[0]) + int(values[1])), (int(values[1]) + int(values[2]))
    return 0, 0


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

def plot_signals(path, scenario, clean, noisy, filtered):
    plt.figure(figsize=(12, 5))

    plt.plot(clean, label="Clean signal (target)", color="black", alpha=0.7, linewidth=2)
    plt.plot(noisy, label="Noisy signal (input)", color="red", alpha=0.9, linewidth=2)
    plt.plot(filtered, label="Filtered signal (hardware)", color="green", alpha=1, linewidth=2)

    plt.grid(True, linestyle="--", alpha=0.5, linewidth=1)
    plt.title(f"Compared signals - scenario: {scenario.upper()}")
    plt.xlabel("sample index")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()

    plt.savefig(path / f"plot_signals_{scenario}.png", dpi=150)
    plt.close()

def run_hardware_evaluation():
    flash_usage, ram_usage = get_memory_usage()
    report = dedent (f"""\
    -----------------------------------------------------------
        Noise-Reduction Network Hardware Evaluation Results
    -----------------------------------------------------------
                     
    Flash-Usage: {flash_usage} Bytes / 512KB ({(flash_usage/(512*1024))*100:.2f}%)
    RAM-Usage: {ram_usage} Bytes / 128KB ({(ram_usage/(128*1024))*100:.2f}%)
    \n""")

    try:
        with serial.Serial(COM_PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(2) # short break for bootloader

            for scenario_i, scenario_name in enumerate(scenarios):
                # load test data
                data_dir = BASE_DIR / f"data_{scenario_name}"
                if not data_dir.exists():
                    print(f"Skip '{scenario_name}': Folder '{data_dir.name}' could not be found.")
                    continue
                try:
                    all_inputs, all_targets, all_clean_signals, S_output, S_input = load_test_data(data_dir, scenario_name)
                except FileNotFoundError as e:
                    print(f"Skip '{scenario_name}': {e}")
                    continue
                except ValueError as e:
                    print(f"Skip '{scenario_name}': Data corruption detected. Details: {e}")
                    continue

                print(f"Process data for scenario '{scenario_name}'. Please wait...")

                total_num_samples = len(all_inputs)
                num_tests = min(NUM_TEST_SAMPLES, total_num_samples) # avoid error for NUM_TEST_SAMPLES > total_num_samples

                hardw_predictions = []
                expected_targets = []
                clean_targets = []
                cycles_list = []

                # inference and evaluation
                for i in range(num_tests):
                    input_vector = all_inputs[i]
                    target_value = all_targets[i]
                    
                    ser.write(f'S{scenario_i}'.encode("utf-8")) # send command
                    ser.write(input_vector.tobytes()) # send 16 byte input-vector (in binary)

                    # process response 
                    response = ser.readline().decode("utf-8").strip()
                    if response:
                        parts = response.split(",")
                        if len(parts) == 2:
                            cycles, predictions = parts
                            cycles_list.append(int(cycles))
                            hardw_predictions.append(int(predictions))
                            expected_targets.append(target_value)

                            # only append clean signal if the hardware inference was successfull
                            # -> else clean_targets would get its value even if inference crashes which would lead to a np.array dimension error
                            if all_clean_signals is not None:
                                clean_targets.append(all_clean_signals[i]) 
                    
                    time.sleep(0.05) # short stability break

                    
                if len(hardw_predictions) == num_tests:
                    avg_cycles = np.mean(cycles_list)
                    avg_inference_time = (avg_cycles / 16000000) * 1000000 # avg. inference time in μs 

                    # validation hardware prediction vs. expected prediction calculated by python (int vs. int) (should be the same -> MSE = 0)
                    mse_hardware_validation, snr_hardware_validation = evaluate(expected_targets, hardw_predictions)
                    
                    # evaluation hardware predictions vs. clean sin-signal (float vs. float)
                    if all_clean_signals is not None:
                        hardw_predictions_float = np.array(hardw_predictions) * S_output # dequantization to int -> float
                        noisy_signal_float = all_inputs[:num_tests, 8] * S_input

                        mse_noise, snr_noise = evaluate(clean_targets, noisy_signal_float)
                        mse_reduct, snr_reduct = evaluate(clean_targets, hardw_predictions_float)

                        # plot scenario
                        plot_signals(data_dir, scenario_name, clean_targets, noisy_signal_float, hardw_predictions_float)

                        mse_imp = mse_noise - mse_reduct
                        snr_imp = snr_reduct - snr_noise


                    else :
                        mse_reduct = float('NaN') # float value: 'nan' (not a number)
                        snr_reduct = float('NaN')

                    report += dedent(f"""\
                    --- Scenario {scenario_i}: {scenario_name.upper()} ({num_tests} test samples) ---

                    > Average inference time: {avg_inference_time:.2f} μs ({avg_cycles:.0f} CPU-cycles)
                    > Inference rate: {1/ (avg_inference_time / 1000000):.0f} Hz (samples/s)
                    > MSE (software-calculated prediction vs. hardware prediction => should be near 0): {mse_hardware_validation:.4f}
                    
                    > Initial Noise MSE (clean signal vs. noisy signal): {mse_noise:.4f}
                    > Noise-Reduction MSE (clean signal vs. hardware prediction): {mse_reduct:.4f}
                    > MSE Improvement: {(mse_imp / mse_noise) * 100:.2f}% (-{mse_imp:.4f})

                    > Initial Signal-to-Noise Ratio SNR (clean signal vs. noisy signal): {snr_noise:.2f} dB
                    > Noise-Reduction Signal-to-Noise Ratio SNR (clean signal vs. hardware prediction): {snr_reduct:.2f} dB
                    > SNR Improvement: {snr_imp:+.2f} dB
                    \n\n""")
                else:
                    report += dedent(f"""\
                    --- Scenario {scenario_i}: {scenario_name.upper()} evaluation error. ---
                    """)

    except serial.SerialException as e:
        print(f"Error at COM-Port: {e}")
        return
    
    # console and file ouput
    print("\n" + report)
    with open(BASE_DIR / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)



if __name__ == "__main__":
    run_hardware_evaluation()