import numpy as np
import matplotlib.pyplot as plt

CS = [0.0, 0.0, 0.1, 0.1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
CS1 = [
    [0.5, 0.5, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.5, 0.5, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 
     
     
     0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
]

MGv = np.zeros(8)

MGm = np.zeros(3)

AC = np.zeros(8)

AMYG = np.zeros(3)

CS_W_MGv = np.random.rand(16, 8)

CS_W_MGm = np.random.rand(16, 3)

MGv_W = np.random.rand(8, 8)
#MG_W_combined = np.random.rand(11, 8)

MGm_W = np.random.rand(3, 8)
MGm_W_Amy = np.random.rand(3, 3)

AC_W = np.random.rand(8,3)
# Activation function
def netr(module, weight, column):
    output = 0
    for i in range(len(module)):
        output += module[i] * weight[i][column]
    
    #Net input to above layer, sum of (total outputs times relative weights)
    return output

def activation_function(module, weight):
    activations = []
    num_outputs = weight.shape[1]  # number of columns = output neurons
    #For each unit in the lower module, sends an input to every unit in the above module
    for j in range(num_outputs):
        #print(j)
        activation = netr(module, weight, j)
        activations.append(activation)
    return activations

#Simple ramp squashing function
def squash (x):
    if x < 0:
        return 0
    if x > 0 and x < 1:
        return x
    if x > 1:
        return 1

#Competitive inhibition when activated
def inhibition(activations):
    a_win = squash(np.max(activations))
    inhibited_a = 0
    inhibited = []
    u = 0.2
    for i in activations:
        if i < a_win:
            inhibited_a = squash(i - (u*a_win)) 
            inhibited.append(inhibited_a)
        else:
            inhibited.append(a_win)
    return(inhibited)

def learning(input_vec, output_vec, weights, learning_rate=0.1):
    input_avg = np.average(input_vec)
    
    for i in range(len(input_vec)):
        if input_vec[i] > input_avg:
            for j in range(len(output_vec)):
                weights[i][j] += learning_rate * input_vec[i] * output_vec[j]

        weights[i] /= np.sum(weights[i])

def plot_activations(activations, title="Activations"):
    plt.figure(figsize=(8,4))
    plt.bar(range(len(activations)), activations, color='skyblue')
    plt.xlabel("Neuron index")
    plt.ylabel("Activation level")
    plt.title(title)
    plt.ylim(0,1)
    plt.show()

def plot_weights(weights, title="Weight Matrix"):
    plt.figure(figsize=(8,6))
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Weight value')
    plt.xlabel("Output neurons")
    plt.ylabel("Input neurons")
    plt.title(title)
    plt.show()

def forward_pass(CS_input):
    CS_MGv_outputs = activation_function(CS_input, CS_W_MGv)
    inhibited_outputs_CS_MGv = inhibition(CS_MGv_outputs)
    #print(CS_MGv_outputs)
    CS_MGm_outputs = activation_function(CS_input, CS_W_MGm)
    inhibited_outputs_CS_MGm = inhibition(CS_MGm_outputs)
    #print(CS_MGm_outputs)
    #print(CS_W_MGv, inhibited_outputs_CS_MGv)

    MGv_to_AC = activation_function(inhibited_outputs_CS_MGv, MGv_W)
    MGm_to_AC = activation_function(inhibited_outputs_CS_MGm, MGm_W)
    MGm_to_Amy = activation_function(inhibited_outputs_CS_MGm, MGm_W_Amy)
    #print(MGv_to_AC, len(MGv_to_AC))
    #print(MGm_to_AC, len(MGm_to_AC))
    AC_inputs = [x + y for x, y in zip(MGv_to_AC, MGm_to_AC)]
    #print(AC_inputs)
    inhibited_AC_outputs = inhibition(AC_inputs)
    #print(inhibited_AC_outputs)
    AC_to_Amy = activation_function(inhibited_AC_outputs, AC_W)
    #print(AC_W)
    #print(AC_to_Amy)
    Amy_inputs = [a + b for a, b in zip(AC_to_Amy, MGm_to_Amy)]
#AC_outputs = activation_function(combined_input[0], MGv_W)
#AC_outputs += activation_function(combined_input[1], MGm_W)
    

    inhibited_outputs_Amy = inhibition(Amy_inputs)

    print(inhibited_outputs_Amy)
    return (CS_input, inhibited_outputs_CS_MGv, inhibited_outputs_CS_MGm,
            MGv_to_AC, MGm_to_AC,
            inhibited_AC_outputs,
            AC_to_Amy, MGm_to_Amy,
            inhibited_outputs_Amy)

import seaborn as sns

# === TRAINING: Repeated exposure to all CS1 inputs ===
epochs = 50  # or however many you want
mgv_activity_over_time = []

for epoch in range(epochs):
    for cs_input in CS1:
        (
            _, inhibited_CS_MGv, inhibited_CS_MGm,
            MGv_to_AC, MGm_to_AC,
            inhibited_AC_outputs,
            AC_to_Amy, MGm_to_Amy,
            _
        ) = forward_pass(cs_input)

        # Hebbian learning
        learning(cs_input, inhibited_CS_MGv, CS_W_MGv)
        learning(cs_input, inhibited_CS_MGm, CS_W_MGm)
        learning(inhibited_CS_MGv, MGv_to_AC, MGv_W)
        learning(inhibited_CS_MGm, MGm_to_AC, MGm_W)
        learning(inhibited_AC_outputs, AC_to_Amy, AC_W)
        learning(inhibited_CS_MGm, MGm_to_Amy, MGm_W_Amy)

    # === Record and optionally plot every 10 epochs ===
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        mgv_activations = []
        for input_vec in CS1:
            mgv_raw = activation_function(input_vec, CS_W_MGv)
            mgv_inhibited = inhibition(mgv_raw)
            mgv_activations.append(mgv_inhibited)

        mgv_activity_over_time.append(mgv_activations)

        # Plot MGv heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(mgv_activations, cmap='viridis', cbar=True,
                    xticklabels=[f'MGv {i}' for i in range(8)],
                    yticklabels=[f'CS{i}' for i in range(len(CS1))])
        plt.xlabel("MGv Neuron Index")
        plt.ylabel("Input Pattern")
        plt.title(f"MGv Activation After Epoch {epoch + 1}")
        plt.tight_layout()
        plt.show()
