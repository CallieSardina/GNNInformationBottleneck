# import re
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse

import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Argument Parser
parser = argparse.ArgumentParser(description="Plot Info Plane from MI data")
parser.add_argument("--file_name", type=str, required=True, help="Name of the input file without extension")
parser.add_argument("--model", type=str, required=True, help="Model architecture")
args = parser.parse_args()
file_name = args.file_name
model = args.model

# Read the file and determine number of layers
file_path = f"{file_name}.txt"

num_layers = 0
layer_mi_data = {}  # Dictionary to store MI values per layer

with open(file_path, "r") as file:
    for line in file:
        match = re.match(r"Layer (\d+): Avg MI_XZ: ([\d.]+), Avg MI_ZY: ([\d.]+)", line)
        if match:
            layer_idx = int(match.group(1))
            mi_xz_value = float(match.group(2))
            mi_zy_value = float(match.group(3))

            if layer_idx not in layer_mi_data:
                layer_mi_data[layer_idx] = {"mi_xz": [], "mi_zy": []}

            layer_mi_data[layer_idx]["mi_xz"].append(mi_xz_value)
            layer_mi_data[layer_idx]["mi_zy"].append(mi_zy_value)

            num_layers = max(num_layers, layer_idx + 1)  # Track total number of layers

# Convert MI data to NumPy arrays
epochs = np.arange(len(next(iter(layer_mi_data.values()))["mi_xz"]))  # Number of epochs
mi_xz = [np.array(layer_mi_data[i]["mi_xz"]) for i in range(num_layers)]
mi_zy = [np.array(layer_mi_data[i]["mi_zy"]) for i in range(num_layers)]

# Set up color mapping
COLORBAR_MAX_EPOCHS = len(epochs)
sm = plt.cm.ScalarMappable(cmap="gnuplot", norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
sm._A = []

# **Main Info Plane Plot**
fig, ax = plt.subplots(figsize=(8, 6))

for epoch in epochs:
    c = sm.to_rgba(epoch)
    avg_mi_xz = [mi_xz[i][epoch] for i in range(num_layers)]
    avg_mi_zy = [mi_zy[i][epoch] for i in range(num_layers)]
    ax.plot(avg_mi_xz, avg_mi_zy, c=c, alpha=0.3, zorder=1)
    ax.scatter(avg_mi_xz, avg_mi_zy, s=30, facecolors=[c] * num_layers, edgecolor="none", alpha=1, zorder=2)

ax.set_xlabel("I(X; Z)")
ax.set_ylabel("I(Z; Y)")
ax.set_title("Info Plane Across Layers")
cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
plt.colorbar(sm, label="Epoch", cax=cbaxes)
plt.tight_layout()

# Ensure output directory exists
os.makedirs(f"plots/{model}", exist_ok=True)
plt.savefig(f"plots/{model}/infoplane_{os.path.basename(file_name)}.png", bbox_inches="tight")
plt.show()

# **Subplots for Each Layer**
fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 6))
if num_layers == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    for epoch in epochs:
        c = sm.to_rgba(epoch)
        ax.scatter(mi_xz[i][epoch], mi_zy[i][epoch], color=c, alpha=0.7)

    ax.set_xlabel("I(X; Z)")
    ax.set_ylabel("I(Z; Y)")
    ax.set_title(f"Layer {i}")

plt.tight_layout()
plt.savefig(f"plots/{model}/infoplane_{os.path.basename(file_name)}_layers.png", bbox_inches="tight")
plt.show()

print("Done")


# parser = argparse.ArgumentParser(description='Plot Info Plane from MI data')
# parser.add_argument('--file_name', type=str, required=True, help='Name of the input file without extension')
# parser.add_argument('--model', type=str, help='Model architecture')
# args = parser.parse_args()
# file_name = args.file_name
# model = args.model

# # Read the file to determine the number of embeddings
# file_path = f'{file_name}.txt'
# with open(file_path, 'r') as file:
#     first_line = next(file).strip()
#     # Layer 0: Avg MI_XZ: 2.1929, Avg MI_ZY: 0.0010
#     match = re.match(r'Layer (\d+): Avg MI_XZ: ([\d.]+), Avg MI_ZY: ([\d.]+)', first_line)
#     if match:
#         num_embeddings = len(match.group(1).split(','))
#     else:
#         raise ValueError("Invalid file format. Could not determine the number of embeddings.")

# # Initialize lists
# epochs = []
# mi_xz = [[] for _ in range(num_embeddings)]
# mi_zy = [[] for _ in range(num_embeddings)]

# # Read data from file
# with open(file_path, 'r') as file:
#     for line in file:
#         match = re.match(r'Layer (\d+): Avg MI_XZ: ([\d.]+), Avg MI_ZY: ([\d.]+)', line)
#         if match:
#             # epoch = int(match.group(1))
#             epochs.append(len(epochs) + 1)
#             xz_values = list(map(float, match.group(3).split(',')))
#             zy_values = list(map(float, match.group(4).split(',')))
#             for i in range(num_embeddings):
#                 mi_xz[i].append(xz_values[i])
#                 mi_zy[i].append(zy_values[i])

# # Convert to numpy arrays
# epochs = np.array(epochs)
# mi_xz = [np.array(layer) for layer in mi_xz]
# mi_zy = [np.array(layer) for layer in mi_zy]

# # Set maximum epoch for colorbar range
# COLORBAR_MAX_EPOCHS = max(epochs)
# sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
# sm._A = []

# # Create the main figure
# fig, ax = plt.subplots(figsize=(8, 6))
# for epoch in sorted(set(epochs)):
#     c = sm.to_rgba(epoch)
#     avg_mi_xz = [np.mean(layer[epochs == epoch]) for layer in mi_xz]
#     avg_mi_zy = [np.mean(layer[epochs == epoch]) for layer in mi_zy]
#     ax.plot(avg_mi_xz, avg_mi_zy, c=c, alpha=0.1, zorder=1)
#     ax.scatter(avg_mi_xz, avg_mi_zy, s=30, facecolors=[c]*num_embeddings, edgecolor='none', alpha=1, zorder=2)

# ax.set_xlabel('I(X; Z)')
# ax.set_ylabel('I(Z; Y)')
# ax.set_title('Info Plane Across Layers')
# cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
# plt.colorbar(sm, label='Epoch', cax=cbaxes)
# plt.tight_layout()
# plt.savefig(f'plots/{model}/infoplane_{file_name.split("/")[-1]}.png', bbox_inches='tight')
# plt.show()

# # Create subplots for each layer
# fig, axes = plt.subplots(1, num_embeddings, figsize=(6 * num_embeddings, 6))
# if num_embeddings == 1:
#     axes = [axes]
# for i, ax in enumerate(axes):
#     for epoch in sorted(set(epochs)):
#         c = sm.to_rgba(epoch)
#         avg_mi_xz = np.mean(mi_xz[i][epochs == epoch])
#         avg_mi_zy = np.mean(mi_zy[i][epochs == epoch])
#         ax.scatter(avg_mi_xz, avg_mi_zy, color=c, alpha=0.7)
#     ax.set_xlabel('I(X; Z)')
#     ax.set_ylabel('I(Z; Y)')
#     ax.set_title(f'Layer {i + 1}')
# plt.tight_layout()
# plt.savefig(f'plots/{model}/infoplane_{file_name.split("/")[-1]}_layers.png', bbox_inches='tight')
# plt.show()

# print("Done")