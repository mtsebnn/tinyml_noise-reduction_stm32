import numpy as np

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

from sklearn.model_selection import train_test_split

from pathlib import Path

import generate_header

torch.manual_seed(42)

# generate signal data
def generate_signals(num_samples = 10000, freq = 5, sample_rate = 100, config = None):
    t = np.linspace(0, num_samples / sample_rate, num_samples)
    signal_clean = np.sin(2 * np.pi * freq * t)
    signal_noise = signal_clean.copy()

    # white noise
    if(config.get("white_noise", False)):
        signal_noise += np.random.normal(0, 0.3, num_samples)

    # echo
    if(config.get("echo", False)):
        delay = int(sample_rate * 0.1) # 100ms / 10 samples delay (with a sample rate of 100(samples/s), a single sample takes 10ms)
        signal_echo = np.zeros_like(signal_clean)
        signal_echo[delay:] = signal_clean[:-delay] * 0.5 # create echo (take 0.5*clean_value and shift it {delay} samples to the right)
        signal_noise += signal_echo

    # quantization noise (analog-to-digital conversion)
    if(config.get("quantization", False)):
        quantization_steps = 8
        signal_noise = np.round(signal_noise * quantization_steps) / quantization_steps

    return signal_clean, signal_noise

# sliding window / local receptive field
def create_windows(sig_clean, sig_noise, window_size):
    x, y = [], []
    for i in range(len(sig_noise) - window_size):
        x.append(sig_noise[i:i + window_size]) # input window
        y.append(sig_clean[i + window_size // 2]) # target value
    return np.array(x), np.array(y)

# dataloader
class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        # .tensor(): If input is a np-array the data is copied to a new tesnsor-object (inefficient in terms of memory usage and time)
        # => Use .as_tensor() to avoid copying data (if input already is a np-array, the tensor and the np-array share memory)
        self.x = torch.as_tensor(x, dtype=torch.float32) 
        # unsqueeze(): Inserts a new dimension to the target tensor to match shape of the output [..., 1]
        # (if dataloader builds a batch of size e.g. 32 the target tensor is [32], but the output layer nn.Linear(8,1) always gives a matrix [32, 1].
        # unsqueeze(1) adds a dimension to the target tensor [32, 1] so it matches the shape of the output => MSE can be calculated without causing any problems)
        self.y = torch.as_tensor(y, dtype=torch.float32).unsqueeze(1) 
    def __len__(self):
        return self.x.size(dim=0)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

# network
class NoiseReductionNetwork(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.l1 = nn.Linear(window_size, 8)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(8,1) # one output because we use a regression network for noise reduction
    
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

# training loop
def train_model(name, model, dataloader, epochs, learningrate = 0.01):
    lossFunc = nn.MSELoss() # Mean Squared Error
    optimizer = torch.optim.Adam(model.parameters(), lr = learningrate) # Adam optimizer
    
    # put model into training mode -> turn on gradient calculation, BatchNorm (normalizes input data based on the statistics (mean and variance)
    # of the current batch) and dropout(randomly zeroe some elements of the input tensor during training to prevent overfitting)
    model.train()
    for e in range(epochs):
        loss_total = 0
        for batch_i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad() # reset gradient information of the optimizer
            outputs = model(inputs) # make prediction using the model
            loss = lossFunc(outputs, targets) # pass predictions and actual targets to the loss function
            loss.backward()  # loss function performs calculation backward through the network
            optimizer.step() # optimizer updates weights
            loss_total += loss.item()
        if (e % 5 == 0):
            print(f"Scenario: {name} / Epoch {e} / MSE-Loss: {loss_total/len(dataloader):.5f}")



# --- different noising scenarios ---
WINDOW_SIZE = 16

scenarios = {
    "only_whitenoise": {"white_noise": True, "echo": False, "quantization": False},
    "only_echo": {"white_noise": False, "echo": True, "quantization": False},
    "only_quantization": {"white_noise": False, "echo": False, "quantization": True},
    "combined": {"white_noise": True, "echo": True, "quantization": True}
}

for name, config in scenarios.items():
    print(f"\n- {name} -")
    # data
    data_clean, data_noise = generate_signals(config=config)
    i, t = create_windows(data_clean, data_noise, WINDOW_SIZE)
    i_train, i_test, t_train, t_test = train_test_split(i, t, test_size = 0.3, random_state = 42) # spit data into test and training set
    print(f"Training samples: {len(i_train)}")
    print(f"Test samples: {len(i_test)}")
    train = torch.utils.data.DataLoader(SignalDataset(i_train, t_train), batch_size = 32, shuffle = True)

    # initalize and train model
    model = NoiseReductionNetwork(WINDOW_SIZE)
    train_model(name, model, train, 11)

    # symmetric quantization
    # put model into evaluation mode (disable dropout and ensure Batch-
    # normalization layers use learned running instead of batch-specific statistics)
    model.eval()
    with torch.no_grad(): # turn off gradient calculation
        w1, b1 = model.l1.weight.numpy(), model.l1.bias.numpy()
        w2, b2 = model.l2.weight.numpy(), model.l2.bias.numpy()
        q_max = 127
        S_input = np.max(np.abs(i_train)) / q_max
        # y = weight*input+bias which leads to S_y*q_y = (S_w*q_w)(S_in*q_in)+(S_b*q_b)
        # (weight*input) and (bias) need same scaling for the addition to be correct, else adding values with different magnitudes
        # -> (weight*input) is (q_w*q_in) scaled by (S_w*S_in) so it follows that Sb = S_w*S_in
        # first layer
        S_w1 = np.max(np.abs(w1)) / q_max # determine scaling factor for weights of the first layer
        S_b1 = S_w1 * S_input
        q_w1 = np.clip(np.round(w1/S_w1), -127, 127).astype(np.int8) # limit quantized weights to range -127, 127 and convert to 8-bit-int
        q_b1 = np.round(b1 / S_b1).astype(np.int32)

        # requantization of the 32-bit output value from layer 1 to a 8-bit input for layer 2
        inp_train_tensor = torch.as_tensor(i_train, dtype=torch.float32)
        out_l1_float = model.relu(model.l1(inp_train_tensor)).numpy() # outputs of layer 1 after ReLU
        S_out1 = np.max(np.abs(out_l1_float)) / q_max # scaling factor for outputs of the first layer
        M = S_b1 / S_out1 # scaling factor ratio
        SHIFT_VALUE = 16
        requant_multiplier = int(np.round(M * (1 << SHIFT_VALUE))) # convert into int using a 16-bit shift (1 << 16 = 2^16)

        # second layer
        S_w2 = np.max(np.abs(w2)) / q_max # determine scaling factor for weights of the second layer
        S_b2 = S_w2 * S_out1
        q_w2 = np.clip(np.round(w2/S_w2), -127, 127).astype(np.int8) # limit quantized weights to range -127, 127 and convert to 8-bit-int
        q_b2 = np.round(b2 / S_b2).astype(np.int32)


    # save values
    BASE_DIR = Path(__file__).resolve().parent
    data_dir = BASE_DIR / f"data_{name}"
    data_dir.mkdir(exist_ok=True)
    np.save(f"{data_dir}/clean_signal.npy", t)
    np.save(f"{data_dir}/input_scalingfactor.npy", S_input)
    np.save(f"{data_dir}/output_scalingfactor.npy", S_b2)
    np.save(f"{data_dir}/weight1_quantized.npy", q_w1)
    np.save(f"{data_dir}/weight2_quantized.npy", q_w2)
    np.save(f"{data_dir}/bias1_quantized.npy", q_b1)
    np.save(f"{data_dir}/bias2_quantized.npy", q_b2)
    np.save(f"{data_dir}/shift_value.npy", SHIFT_VALUE)
    np.save(f"{data_dir}/requant_multiplier.npy", requant_multiplier)


    # generate testvector
    # use unshuffled inputs for correct sine-wave order of samples (for plots in run_evaluation.py)
    q_i_test = np.clip(np.round(i/S_input), -128, 127).astype(np.int8) # quantize test data
    # calculate target values using fixed-point arithmetic (C logic)
    target_values = []
    for sample in q_i_test:

        # layer 1 (16 inputs, 8 outputs)
        layer1_out = np.zeros(8, dtype=np.int32)
        for neuron in range(8):
            neuron_sum = int(q_b1[neuron]) # start with bias
            for i in range(16):
                neuron_sum += (int(sample[i]) * int(q_w1[neuron, i])) # w1 has shape (8, 16) (weight matrix = (outputs, inputs))
            neuron_sum = max(0, neuron_sum) # ReLU activation function (max(0, x))

            # requantization
            scaled_out = np.int64(neuron_sum) * np.int64(requant_multiplier) # 64-bit multiplication to prevent overflow
            shifted_out = np.int32((scaled_out + (1 << (SHIFT_VALUE -1))) >> SHIFT_VALUE) # bit shift to the right (x / 2^16)
            layer1_out[neuron] = np.clip(shifted_out, 0, 127) # 0, 127 because negative values were already eliminated by ReLU

        # layer 2 (8 inputs, 1 output)
        layer2_out = int(q_b2[0])
        for neuron in range(8):
                layer2_out += (int(layer1_out[neuron]) * int(q_w2[0, neuron])) # w2 has shape (1, 8))

        target_values.append(int(layer2_out))
    target_values = np.array(target_values, dtype=np.int32)

    with open(f"{data_dir}/neuronTV_{name}.dat", "w") as f:
        for idx in range(min(len(q_i_test), 200)): # first 200 test samples
            inp = q_i_test[idx].tolist()
            inp.append(target_values[idx]) # add scaled target/clean value to validate in C code later
            line = " ".join(str(v) for v in inp)
            f.write(line + "\n")
    
    print("M = ", M)
    print("Bit shift =", SHIFT_VALUE)
    print("requant =", requant_multiplier)
    print(f"Data succesfully exported to '{data_dir}'.\n")
print(f"-- Succesfully processed all scenarios. --")

generate_header.generate_header()
