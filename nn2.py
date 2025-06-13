import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === STIMULI DEFINITION ===
CS1 = [
    [0.5, 0.5, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.5, 0.5, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
]

# === NETWORK INITIALIZATION ===
CS_W_MGv = np.random.rand(16, 8)
CS_W_MGm = np.random.rand(16, 3)
MGv_W = np.random.rand(8, 8)
MGm_W = np.random.rand(3, 8)
MGm_W_Amy = np.random.rand(3, 3)
AC_W = np.random.rand(8, 3)
US_W = np.full(3, 0.4)  # constant US input to MGm and Amygdala

# === BASIC FUNCTIONS ===
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
def forward_pass(CS_input):  # CS only
    MGv_out = inhibition(activation_function(CS_input, CS_W_MGv))
    MGm_out = inhibition(activation_function(CS_input, CS_W_MGm))
    AC_input = [a + b for a, b in zip(
        activation_function(MGv_out, MGv_W),
        activation_function(MGm_out, MGm_W)
    )]
    AC_out = inhibition(AC_input)
    Amy_input = [a + b for a, b in zip(
        activation_function(AC_out, AC_W),
        activation_function(MGm_out, MGm_W_Amy)
    )]
    Amy_out = inhibition(Amy_input)
    return MGv_out, MGm_out, AC_out, Amy_out

def US_pass(CS_input):  # CS + US
    MGv_out = inhibition(activation_function(CS_input, CS_W_MGv))
    MGm_input = [a + b for a, b in zip(
        activation_function(CS_input, CS_W_MGm),
        US_W
    )]
    MGm_out = inhibition(MGm_input)
    AC_input = [a + b for a, b in zip(
        activation_function(MGv_out, MGv_W),
        activation_function(MGm_out, MGm_W)
    )]
    AC_out = inhibition(AC_input)
    Amy_input = [a + b + c for a, b, c in zip(
        activation_function(AC_out, AC_W),
        activation_function(MGm_out, MGm_W_Amy),
        US_W
    )]
    Amy_out = inhibition(Amy_input)
    return MGv_out, MGm_out, AC_out, Amy_out

# === PRECONDITIONING RECEPTIVE FIELDS ===
mgv_pre_RF, mgm_pre_RF, ac_pre_RF, amy_pre_RF = [], [], [], []
for cs_input in CS1:
    MGv, MGm, AC, Amy = forward_pass(cs_input)
    mgv_pre_RF.append(MGv)
    mgm_pre_RF.append(MGm)
    ac_pre_RF.append(AC)
    amy_pre_RF.append(Amy)

# === TRAINING: CS + US ===
epochs = 50
for epoch in range(epochs):
    for cs_input in CS1:
        MGv, MGm, AC, Amy = US_pass(cs_input)
        learning(cs_input, MGv, CS_W_MGv)
        learning(cs_input, MGm, CS_W_MGm)
        learning(MGv, activation_function(MGv, MGv_W), MGv_W)
        learning(MGm, activation_function(MGm, MGm_W), MGm_W)
        learning(AC, activation_function(AC, AC_W), AC_W)
        learning(MGm, activation_function(MGm, MGm_W_Amy), MGm_W_Amy)

# === POSTCONDITIONING RFs ===
mgv_post_RF, mgm_post_RF, ac_post_RF, amy_post_RF = [], [], [], []
for cs_input in CS1:
    MGv, MGm, AC, Amy = forward_pass(cs_input)
    mgv_post_RF.append(MGv)
    mgm_post_RF.append(MGm)
    ac_post_RF.append(AC)
    amy_post_RF.append(Amy)

# === AMYGDALA OUTPUT COMPARISON (Figure 4 equivalent) ===
amygdala_pre = [sum(forward_pass(cs)[-1]) for cs in CS1]
amygdala_post = [sum(forward_pass(cs)[-1]) for cs in CS1]
amygdala_us = [sum(US_pass(cs)[-1]) for cs in CS1]

plt.figure(figsize=(10, 5))
plt.plot(amygdala_pre, label="Pre-conditioning", linestyle='--', marker='o')
plt.plot(amygdala_post, label="Post-conditioning (CS only)", linestyle='-', marker='s')
plt.plot(amygdala_us, label="Post-conditioning (CS + US)", linestyle='-', marker='^')
plt.title("Amygdala Output Across Frequencies")
plt.xlabel("CS1 Input Index (Frequency)")
plt.ylabel("Total Amygdala Activation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def plot_module_response(module_name, pre_RF, post_RF, post_US_RF):
    pre = [np.mean(x) for x in pre_RF]
    post = [np.mean(x) for x in post_RF]
    post_us = [np.mean(x) for x in post_US_RF]

    plt.figure(figsize=(10, 5))
    plt.plot(pre, label="Pre-conditioning", linestyle='--', marker='o')
    plt.plot(post, label="Post-conditioning (CS only)", linestyle='-', marker='s')
    plt.plot(post_us, label="Post-conditioning (CS + US)", linestyle='-', marker='^')
    plt.title(f"{module_name} Mean Activation Across Frequencies")
    plt.xlabel("CS1 Input Index (Frequency)")
    plt.ylabel("Mean Activation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Post-conditioning (CS + US) RFs ===
mgv_post_RF_us, mgm_post_RF_us, ac_post_RF_us, amy_post_RF_us = [], [], [], []
for cs_input in CS1:
    MGv, MGm, AC, Amy = US_pass(cs_input)
    mgv_post_RF_us.append(MGv)
    mgm_post_RF_us.append(MGm)
    ac_post_RF_us.append(AC)
    amy_post_RF_us.append(Amy)

# === PLOT COMPARISONS FOR ALL MODULES ===
plot_module_response("MGv", mgv_pre_RF, mgv_post_RF, mgv_post_RF_us)
plot_module_response("MGm", mgm_pre_RF, mgm_post_RF, mgm_post_RF_us)
plot_module_response("Auditory Cortex (AC)", ac_pre_RF, ac_post_RF, ac_post_RF_us)
plot_module_response("Amygdala", amy_pre_RF, amy_post_RF, amy_post_RF_us)
