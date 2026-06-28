import numpy as np
from pathlib import Path
from textwrap import dedent

# This script automatically generates a C-code header file using the exported network-data from network.py.

def generate_header():
    BASE_DIR = Path(__file__).resolve().parent
    HEADER_PATH = BASE_DIR / "nn_data.h"

    scenarios = ["only_whitenoise", "only_echo", "only_quantization", 
                "combined"]

    # C code array
    def c_array_1d(array):
        return ", ".join(str(int(x)) for x in array)

    # C code matrix
    def c_array_2d(matrix):
        rows = []
        for row in matrix:
            row_string = "  {" + ", ".join(str(int(x)) for x in row) + "}"
            rows.append(row_string)
        return ",\n".join(rows)


    # start of header structure
    # Using dedent() to avoid indentation caused by """..."""
    header = dedent("""
    #ifndef NN_DATA_H_
    #define NN_DATA_H_

    #include <stdint.h> // declares sets of integer types with specific widths e.g. int8_t

    #define WINDOW_SIZE 16
    #define HIDDEN_NEURONS 8
    
    typedef struct {
        const char *name;
        // w1 is a pointer to an array of WINDOW_SIZE int8_t elements
        // C array decay: In most expressions array "decays" into a pointer to its first element (i.e. 2D-array becomes a pointer to its first row)
        // i.e. w1_only_whitenoise has type int8_t [HIDDEN_NEURONS][WINDOW_SIZE] -> decays to -> int8_t (*) [WINDOW_SIZE]
        // i.e. we are going to use: scenario_whitenoise.w1 = w1_only_whitenoise = &w1_only_whitenoise[0] 
        // (means that w1 is a pointer(stores the address) to the first row of the matrix w1_only_whitenoise)
        // w1[2][1] = *(*(w1 + 2) + 1) -> value at row 3, column 2
        const int8_t (*w1)[WINDOW_SIZE]; 
        const int32_t *b1;
        const int8_t *w2; // w2 is a pointer to a single int8_t element (ptr[0] = *(ptr + 0) = ptr*)
        const int32_t b2;
        const int32_t requant_multiplier;
        const uint8_t shift_value;
    } nn_scenario_t;
    """)

    # scenario specific data
    for scenario in scenarios:
        data_dir = BASE_DIR / f"data_{scenario}"
        if not data_dir.exists():
            print(f"Skip '{scenario}': Folder '{data_dir.name}' could not be found.")
            continue

        try:
            w1 = np.load(f"{data_dir}/weight1_quantized.npy")
            w2 = np.load(f"{data_dir}/weight2_quantized.npy")
            b1 = np.load(f"{data_dir}/bias1_quantized.npy")
            b2 = np.load(f"{data_dir}/bias2_quantized.npy")

            requant_multiplier = int(np.load(f"{data_dir}/requant_multiplier.npy"))
            shift_value = int(np.load(f"{data_dir}/shift_value.npy"))

        except FileNotFoundError as e:
            print(f"Error: Could not load all .npy-Files in '{data_dir.name}': {e}")
            print(f"Skip '{scenario} for the header generation.")
            continue

        header += f"\n// scenario: {scenario}\n"
        header += f"static const int8_t w1_{scenario}" + "[HIDDEN_NEURONS][WINDOW_SIZE] = {\n" + c_array_2d(w1) + "\n};\n"
        header += f"static const int32_t b1_{scenario}" + "[HIDDEN_NEURONS] = {" + f"{c_array_1d(b1)}" + "};\n"
        header += f"static const int8_t w2_{scenario}" + "[HIDDEN_NEURONS] = {" + f"{c_array_1d(w2[0])}" + "};\n"
        header += f"#define B2_{scenario.upper()} {int(b2[0])}\n"
        header += f"#define REQUANT_MULT_{scenario.upper()} {requant_multiplier}\n"
        header += f"#define SHIFT_VALUE_{scenario.upper()} {shift_value}\n\n"

    header += dedent("""
    // global scenario config for main.c
    extern const nn_scenario_t scenario_whitenoise;
    extern const nn_scenario_t scenario_echo;
    extern const nn_scenario_t scenario_quantization;
    extern const nn_scenario_t scenario_combined;

    int32_t predict(const int8_t *input_window, const nn_scenario_t *config);

    #endif /* NN_DATA_H_ */
    """)

    # write header
    with open(HEADER_PATH, "w", encoding="utf-8") as f:
        f.write(header)

    print("-- Succesfully created/updated C-code header 'nn_data.h'. --")


if __name__ == "__main__":
    generate_header()