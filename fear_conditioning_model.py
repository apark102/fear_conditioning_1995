import numpy as np
import matplotlib.pyplot as plt

# Auditory Input (Conditioned Stimulus)
CS1 = [
    [0.5, 0.5, 0.0, 0.0] + [0]*12,
    [0.0, 0.5, 0.5, 0.0] + [0]*12,
    [0.0, 0.0, 0.5, 0.5] + [0]*12,
    [0.0, 0.0, 0.0, 0.5, 0.5] + [0]*11,
    [0.0]*4 + [0.5, 0.5] + [0]*10,
    [0.0]*5 + [0.5, 0.5] + [0]*9,
    [0.0]*6 + [0.5, 0.5] + [0]*8,
    [0.0]*7 + [0.5, 0.5] + [0]*7,
    [0.0]*8 + [0.5, 0.5] + [0]*6,
    [0.0]*9 + [0.5, 0.5] + [0]*5,
    [0.0]*10 + [0.5, 0.5] + [0]*4,
    [0.0]*11 + [0.5, 0.5] + [0]*3,
    [0.0]*12 + [0.5, 0.5] + [0]*2,
    [0.0]*13 + [0.5, 0.5] + [0],
    [0.0]*14 + [0.5, 0.5]
]

# Weight Initialization
# Conditioned stimulus to Thalamus
CS_W_MGv = np.random.rand(16, 8)
CS_W_MGm = np.random.rand(16, 3)

# Thalamus to Auditory Cortex
MGv_W = np.random.rand(8, 8)
MGm_W = np.random.rand(3, 8)

# Thalamus to Amygdala
MGm_W_Amy = np.random.rand(3, 3)

# Auditory Cortex to Amygdala
AC_W = np.random.rand(8, 3)

# Weights for second model (CS+US)
US_CS_W_MGv = np.random.rand(16, 8)
US_CS_W_MGm = np.random.rand(16, 3)
US_MGv_W = np.random.rand(8, 8)
US_MGm_W = np.random.rand(3, 8)
US_MGm_W_Amy = np.random.rand(3, 3)
US_AC_W = np.random.rand(8, 3)


# Unconditioned Stimulus Weights
US_W = np.full(3, 0.4)

# Core Functions

# Net Input
def netr(module, weight, column):
    return sum(module[i] * weight[i][column] for i in range(len(module)))

# Activation Function
def activation_function(module, weight):
    return [netr(module, weight, j) for j in range(weight.shape[1])]

# Ramp function
def squash(x):
    return max(0, min(x, 1))

# Lateral Inhibition function
def inhibition(activations):
    a_win = squash(np.max(activations))
    u = 0.2
    return [squash(a - u * a_win) if a < a_win else a_win for a in activations]

# Learning function
def learning(input_vec, output_vec, weights, learning_rate=0.1):
    input_avg = np.average(input_vec)
    for i in range(len(input_vec)):
        if input_vec[i] > input_avg:
            for j in range(len(output_vec)):
                weights[i][j] += learning_rate * input_vec[i] * output_vec[j]
    weights /= weights.sum(axis=0, keepdims=True)

# Feedforward propagation function for CS only
def forward_pass(CS_input, CS_MGv_weight, CS_MGm_weight, MGv_weight, MGm_weight, AC_weight, MGm_Amy_weight):
    MGv = inhibition(activation_function(CS_input, CS_MGv_weight))
    MGm = inhibition(activation_function(CS_input, CS_MGm_weight))
    AC = inhibition([a + b for a, b in zip(
        activation_function(MGv, MGv_weight),
        activation_function(MGm, MGm_weight)
    )])
    Amy = inhibition([a + b for a, b in zip(
        activation_function(AC, AC_weight),
        activation_function(MGm, MGm_Amy_weight)
    )])
    return MGv, MGm, AC, Amy

# Feedforward propagation for CS+US
def US_pass(CS_input, CS_MGv_weight, CS_MGm_weight, MGv_weight, MGm_weight, AC_weight, MGm_Amy_weight, US_weight):
    MGv = inhibition(activation_function(CS_input, CS_MGv_weight))
    MGm_in = [a + b for a, b in zip(activation_function(CS_input, CS_MGm_weight), US_weight)]
    MGm = inhibition(MGm_in)
    AC = inhibition([a + b for a, b in zip(
        activation_function(MGv, MGv_weight),
        activation_function(MGm, MGm_weight)
    )])
    Amy = inhibition([a + b + c for a, b, c in zip(
        activation_function(AC, AC_weight),
        activation_function(MGm, MGm_Amy_weight),
        US_weight
    )])
    return MGv, MGm, AC, Amy

# Preconditioning training to establish receptive fields
pretrain_epochs = 50
for _ in range(pretrain_epochs):
    for cs_input in CS1:
        MGv, MGm, AC, Amy = forward_pass(cs_input, CS_W_MGv, CS_W_MGm, MGv_W, MGm_W, AC_W, MGm_W_Amy)
        learning(cs_input, MGv, CS_W_MGv)
        learning(cs_input, MGm, CS_W_MGm)
        learning(MGv, activation_function(MGv, MGv_W), MGv_W)
        learning(MGm, activation_function(MGm, MGm_W), MGm_W)
        learning(AC, activation_function(AC, AC_W), AC_W)
        learning(MGm, activation_function(MGm, MGm_W_Amy), MGm_W_Amy)
    for cs_input in CS1:
        MGv_us, MGm_us, AC_us, Amy_us = forward_pass(cs_input, US_CS_W_MGv, US_CS_W_MGm, US_MGv_W, US_MGm_W, US_AC_W, US_MGm_W_Amy)
        learning(cs_input, MGv_us, US_CS_W_MGv)
        learning(cs_input, MGm_us, US_CS_W_MGm)
        learning(MGv_us, activation_function(MGv_us, US_MGv_W), US_MGv_W)
        learning(MGm_us, activation_function(MGm_us, US_MGm_W), US_MGm_W)
        learning(AC_us, activation_function(AC_us, US_AC_W), US_AC_W)
        learning(MGm_us, activation_function(MGm_us, US_MGm_W_Amy), US_MGm_W_Amy)

# Record preconditioning for graphs
mgv_pre_RF, mgm_pre_RF, ac_pre_RF, amy_pre_RF = [], [], [], []
for cs_input in CS1:
    MGv, MGm, AC, Amy = forward_pass(cs_input, CS_W_MGv, CS_W_MGm, MGv_W, MGm_W, AC_W, MGm_W_Amy)
    mgv_pre_RF.append(MGv)
    mgm_pre_RF.append(MGm)
    ac_pre_RF.append(AC)
    amy_pre_RF.append(Amy)

# Conditioning training

# Choose Frequency
print("Available inputs: indices 0 to", len(CS1) - 1)
while True:
    try:
        us_index = int(input("Which frequency(0–14) is being paired with the US?"))
        if 0 <= us_index < len(CS1):
            break
        else:
            print("Please enter a valid number.")
    except ValueError:
        print("Invalid input.")

cond_epochs = 50
for _ in range(cond_epochs):
    # Goes through every frequency and checks if its paired
    for i, cs_input in enumerate(CS1):
        if i == us_index:
            # Implements Hebbian learning if paired
            MGv_us, MGm_us, AC_us, Amy_us = US_pass(cs_input, US_CS_W_MGv, US_CS_W_MGm, US_MGv_W, US_MGm_W, US_AC_W, US_MGm_W_Amy, US_W)
            learning(cs_input, MGm_us, US_CS_W_MGm)
            learning(MGm_us, activation_function(MGm_us, US_MGm_W), US_MGm_W)
            learning(AC_us, activation_function(AC_us, US_AC_W), US_AC_W)
            learning(MGm_us, activation_function(MGm_us, US_MGm_W_Amy), US_MGm_W_Amy)

# Post Conditioning
mgv_post_RF, mgm_post_RF, ac_post_RF, amy_post_RF = [], [], [], []
mgv_post_RF_us, mgm_post_RF_us, ac_post_RF_us, amy_post_RF_us = [], [], [], []

# Goes through each frequency and measures final activation output for each 
for i, cs_input in enumerate(CS1):
    MGv_u, MGm_u, AC_u, Amy_u = forward_pass(cs_input, US_CS_W_MGv, US_CS_W_MGm, US_MGv_W, US_MGm_W, US_AC_W, US_MGm_W_Amy)
    mgv_post_RF_us.append(MGv_u)
    mgm_post_RF_us.append(MGm_u)
    ac_post_RF_us.append(AC_u)
    amy_post_RF_us.append(Amy_u)

# Line Plots
def plot_module_response(module_name, pre_RF, post_RF, post_US_RF):
    pre = [np.mean(x) for x in pre_RF]
    post = [np.mean(x) for x in post_RF]
    post_us = [np.mean(x) for x in post_US_RF]
    x = list(range(len(CS1)))
    plt.figure(figsize=(10, 5))
    plt.plot(x, pre, label="Pre-conditioning", linestyle='--', marker='o')
    plt.plot(x, post_us, label="Post-conditioning (CS + US)", linestyle='-', marker='^')
    plt.axvline(x=us_index, color='red', linestyle=':', label='US-paired CS')
    plt.title(f"{module_name} Mean Activation")
    plt.xlabel("CS1 Input Index")
    plt.ylabel("Mean Activation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# Plot for each module
plot_module_response("MGv", mgv_pre_RF, mgv_post_RF, mgv_post_RF_us)
plot_module_response("MGm", mgm_pre_RF, mgm_post_RF, mgm_post_RF_us)
plot_module_response("Auditory Cortex", ac_pre_RF, ac_post_RF, ac_post_RF_us)
plot_module_response("Amygdala", amy_pre_RF, amy_post_RF, amy_post_RF_us)

