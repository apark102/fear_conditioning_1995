import numpy as np
import matplotlib.pyplot as plt

# === STIMULI DEFINITION ===
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

# === WEIGHT INITIALIZATION ===
CS_W_MGv = np.random.rand(16, 8)
CS_W_MGm = np.random.rand(16, 3)
MGv_W = np.random.rand(8, 8)
MGm_W = np.random.rand(3, 8)
MGm_W_Amy = np.random.rand(3, 3)
AC_W = np.random.rand(8, 3)
US_W = np.full(3, 0.4)

# === CORE FUNCTIONS ===
def netr(module, weight, column):
    return sum(module[i] * weight[i][column] for i in range(len(module)))

def activation_function(module, weight):
    return [netr(module, weight, j) for j in range(weight.shape[1])]

def squash(x):
    return max(0, min(x, 1))

def inhibition(activations):
    a_win = squash(np.max(activations))
    u = 0.2
    return [squash(a - u * a_win) if a < a_win else a_win for a in activations]

def learning(input_vec, output_vec, weights, learning_rate=0.1):
    input_avg = np.average(input_vec)
    for i in range(len(input_vec)):
        if input_vec[i] > input_avg:
            for j in range(len(output_vec)):
                weights[i][j] += learning_rate * input_vec[i] * output_vec[j]
    weights /= weights.sum(axis=1, keepdims=True)

# === FORWARD PROPAGATION FUNCTIONS ===
def forward_pass(CS_input):
    MGv = inhibition(activation_function(CS_input, CS_W_MGv))
    MGm = inhibition(activation_function(CS_input, CS_W_MGm))
    AC = inhibition([a + b for a, b in zip(
        activation_function(MGv, MGv_W),
        activation_function(MGm, MGm_W)
    )])
    Amy = inhibition([a + b for a, b in zip(
        activation_function(AC, AC_W),
        activation_function(MGm, MGm_W_Amy)
    )])
    return MGv, MGm, AC, Amy

def US_pass(CS_input):
    MGv = inhibition(activation_function(CS_input, CS_W_MGv))
    MGm_in = [a + b for a, b in zip(activation_function(CS_input, CS_W_MGm), US_W)]
    MGm = inhibition(MGm_in)
    AC = inhibition([a + b for a, b in zip(
        activation_function(MGv, MGv_W),
        activation_function(MGm, MGm_W)
    )])
    Amy = inhibition([a + b + c for a, b, c in zip(
        activation_function(AC, AC_W),
        activation_function(MGm, MGm_W_Amy),
        US_W
    )])
    return MGv, MGm, AC, Amy

# === PHASE 1: PRE-CONDITIONING TRAINING ===
pretrain_epochs = 50
for _ in range(pretrain_epochs):
    for cs_input in CS1:
        MGv, MGm, AC, Amy = forward_pass(cs_input)
        learning(cs_input, MGv, CS_W_MGv)
        learning(cs_input, MGm, CS_W_MGm)
        learning(MGv, activation_function(MGv, MGv_W), MGv_W)
        learning(MGm, activation_function(MGm, MGm_W), MGm_W)
        learning(AC, activation_function(AC, AC_W), AC_W)
        learning(MGm, activation_function(MGm, MGm_W_Amy), MGm_W_Amy)

# === RECORD PRECONDITIONING RFs ===
mgv_pre_RF, mgm_pre_RF, ac_pre_RF, amy_pre_RF = [], [], [], []
for cs_input in CS1:
    MGv, MGm, AC, Amy = forward_pass(cs_input)
    mgv_pre_RF.append(MGv)
    mgm_pre_RF.append(MGm)
    ac_pre_RF.append(AC)
    amy_pre_RF.append(Amy)

# === PHASE 2: CONDITIONING TRAINING ===
print("Available CS1 inputs: indices 0 to", len(CS1) - 1)
while True:
    try:
        us_index = int(input("Which CS1 index (0–14) should be paired with the US? "))
        if 0 <= us_index < len(CS1):
            break
        else:
            print("❌ Please enter a valid number.")
    except ValueError:
        print("❌ Invalid input.")

cond_epochs = 50
for _ in range(cond_epochs):
    for i, cs_input in enumerate(CS1):
        if i == us_index:
            MGv, MGm, AC, Amy = US_pass(cs_input)
        else:
            MGv, MGm, AC, Amy = forward_pass(cs_input)
        learning(cs_input, MGv, CS_W_MGv)
        learning(cs_input, MGm, CS_W_MGm)
        learning(MGv, activation_function(MGv, MGv_W), MGv_W)
        learning(MGm, activation_function(MGm, MGm_W), MGm_W)
        learning(AC, activation_function(AC, AC_W), AC_W)
        learning(MGm, activation_function(MGm, MGm_W_Amy), MGm_W_Amy)

# === POST-CONDITIONING RFs ===
mgv_post_RF, mgm_post_RF, ac_post_RF, amy_post_RF = [], [], [], []
mgv_post_RF_us, mgm_post_RF_us, ac_post_RF_us, amy_post_RF_us = [], [], [], []

for i, cs_input in enumerate(CS1):
    MGv, MGm, AC, Amy = forward_pass(cs_input)
    mgv_post_RF.append(MGv)
    mgm_post_RF.append(MGm)
    ac_post_RF.append(AC)
    amy_post_RF.append(Amy)

    if i == us_index:
        MGv_u, MGm_u, AC_u, Amy_u = US_pass(cs_input)
    else:
        MGv_u, MGm_u, AC_u, Amy_u = forward_pass(cs_input)

    mgv_post_RF_us.append(MGv_u)
    mgm_post_RF_us.append(MGm_u)
    ac_post_RF_us.append(AC_u)
    amy_post_RF_us.append(Amy_u)

# === MODULE PLOT ===
def plot_module_response(module_name, pre_RF, post_RF, post_US_RF):
    pre = [np.mean(x) for x in pre_RF]
    post = [np.mean(x) for x in post_RF]
    post_us = [np.mean(x) for x in post_US_RF]
    x = list(range(len(CS1)))
    plt.figure(figsize=(10, 5))
    plt.plot(x, pre, label="Pre-conditioning", linestyle='--', marker='o')
    plt.plot(x, post, label="Post-conditioning (CS only)", linestyle='-', marker='s')
    plt.plot(x, post_us, label="Post-conditioning (CS + US)", linestyle='-', marker='^')
    plt.axvline(x=us_index, color='red', linestyle=':', label='US-paired CS')
    plt.title(f"{module_name} Mean Activation")
    plt.xlabel("CS1 Input Index")
    plt.ylabel("Mean Activation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === PLOT FOR ALL MODULES ===
plot_module_response("MGv", mgv_pre_RF, mgv_post_RF, mgv_post_RF_us)
plot_module_response("MGm", mgm_pre_RF, mgm_post_RF, mgm_post_RF_us)
plot_module_response("Auditory Cortex", ac_pre_RF, ac_post_RF, ac_post_RF_us)
plot_module_response("Amygdala", amy_pre_RF, amy_post_RF, amy_post_RF_us)
