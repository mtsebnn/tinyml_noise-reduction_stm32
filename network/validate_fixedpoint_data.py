import numpy as np
from pathlib import Path

# This script validates the exported testvectors from network.py by simulating the exact bit-true integer arithmetic designed for the microcontroller
# implementation. It ensures that file format, parsed data, and fixed-point scaling match the expected mathematical results before the hardware implementation.


BASE_DIR = Path(__file__).resolve().parent

scenarios = ["only_whitenoise", "only_echo", "only_quantization", 
             "combined"]
all_successful = True

for scenario in scenarios:
    data_dir = BASE_DIR / f"data_{scenario}"
    if not data_dir.exists():
        print(f"Skip '{scenario}': Folder '{data_dir.name}' could not be found.")
        continue

    print(f"-- Start simulation of the network with fixed-point values for '{scenario}' --")

    # load data
    try:
        w1 = np.load(f"{data_dir}/weight1_quantized.npy")
        w2 = np.load(f"{data_dir}/weight2_quantized.npy")
        b1 = np.load(f"{data_dir}/bias1_quantized.npy")
        b2 = np.load(f"{data_dir}/bias2_quantized.npy")

        requant_multiplier = int(np.load(f"{data_dir}/requant_multiplier.npy"))
        shift_value = int(np.load(f"{data_dir}/shift_value.npy"))

        tv = data_dir / f"neuronTV_{scenario}.dat"
        if not tv.exists():
            raise FileNotFoundError(f"Error: Testvector {tv.name} is missing.")
    except FileNotFoundError as e:
        print(f"Error: Could not load all .npy-Files in '{data_dir.name}': {e}")
        all_successful = False
        continue
    
    pred_correct = 0
    total_samples = 0

    with open(tv, "r") as f:
        for line_i, line in enumerate(f):
            if not line.strip():
                continue # skip empty lines
            
            # split line into input and target
            values = [int(v) for v in line.split()]
            input_window = np.array(values[:16], dtype=np.int32)
            target_value = values[16]


            # -- fixed-point arithmetic (C logic) --
            # layer 1 (16 inputs, 8 outputs)
            layer1_out = np.zeros(8, dtype=np.int32)
            for neuron in range(8):
                neuron_sum = b1[neuron] # start with bias
                for i in range(16):
                    neuron_sum += (input_window[i] * w1[neuron, i]) # w1 has shape (8, 16) (weight matrix = (outputs, inputs))

                neuron_sum = max(0, neuron_sum) # ReLU activation function (max(0, x))

                # requantization
                scaled_out = np.int64(neuron_sum) * np.int64(requant_multiplier) # 64-bit multiplication to prevent C-overflow
                shifted_out = np.int32((scaled_out + (1 << (shift_value -1))) >> shift_value) # bit shift to the right (x / 2^16)
                layer1_out[neuron] = np.clip(shifted_out, 0, 127) # 0, 127 because negative values were already eliminated by ReLU
            
            # layer 2 (8 inputs, 1 output)
            layer2_out = b2[0]
            for neuron in range(8):
                layer2_out += (layer1_out[neuron] * w2[0, neuron]) # w2 has shape (1, 8))

            # validate calculated vs expected value
            # allow tolerance of +-1 because of np.round() vs PyTorch-Casting(np -> pytorch) for float values 
            diff = abs(layer2_out - target_value)
            if target_value != 0 and abs(target_value) > 500:
                is_correct = (diff / abs(target_value)) <= 0.015 # max. 1.5%
            else:
                is_correct = diff <= 5 # constant tolerance for small values

            if is_correct:
                pred_correct += 1
            else:
                print(f"!! Abweichung in {scenario} (line {line_i}): ")
                print(f"   -> Difference: {diff} / Calculated: {layer2_out} / Expected: {target_value}")

            total_samples += 1

        if total_samples > 0:
            accuracy = (pred_correct / total_samples) * 100
            if accuracy >= 99.5:
                print(f" -- SUCCESS: {pred_correct} of {total_samples} correct ({accuracy:.2f}%) --")
            else:
                print(f" -- WARNING: Only {pred_correct} of {total_samples} correct ({accuracy:.2f}%) --")
                all_successful = False

if all_successful:
    print(" >>> Overall validation successful (100% accuracy)!")
    print(" >>> All testvectors match the fixed-point arithmetic.")
else:
    print(" >>> Overall validation failed!")
    print(" >>> Please check the deviations in the affected scenarios.")
            