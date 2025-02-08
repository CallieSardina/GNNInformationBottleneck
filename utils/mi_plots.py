import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot Info Plane from MI data')
parser.add_argument('--file_name', type=str, required=True, help='Name of the input file without extension')
parser.add_argument('--model', type=str, help='Model architecture')
args = parser.parse_args()
file_name = args.file_name
model = args.model

# Read the file to determine the number of embeddings
file_path = f'{file_name}.txt'
with open(file_path, 'r') as file:
    first_line = next(file).strip()
    match = re.match(r'Epoch \d+, Batch \d+, MI_XZ: ([\d\.]+(?:, [\d\.]+)*), MI_ZY: ([\d\.]+(?:, [\d\.]+)*)', first_line)
    if match:
        num_embeddings = len(match.group(1).split(','))
    else:
        raise ValueError("Invalid file format. Could not determine the number of embeddings.")

# Initialize lists
epochs = []
mi_xz = [[] for _ in range(num_embeddings)]
mi_zy = [[] for _ in range(num_embeddings)]

# Read data from file
with open(file_path, 'r') as file:
    for line in file:
        match = re.match(r'Epoch (\d+), Batch (\d+), MI_XZ: ([\d\.]+(?:, [\d\.]+)*), MI_ZY: ([\d\.]+(?:, [\d\.]+)*)', line)
        if match:
            epoch = int(match.group(1))
            epochs.append(epoch)
            xz_values = list(map(float, match.group(3).split(',')))
            zy_values = list(map(float, match.group(4).split(',')))
            for i in range(num_embeddings):
                mi_xz[i].append(xz_values[i])
                mi_zy[i].append(zy_values[i])

# Convert to numpy arrays
epochs = np.array(epochs)
mi_xz = [np.array(layer) for layer in mi_xz]
mi_zy = [np.array(layer) for layer in mi_zy]

# Set maximum epoch for colorbar range
COLORBAR_MAX_EPOCHS = max(epochs)
sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
sm._A = []

# Create the main figure
fig, ax = plt.subplots(figsize=(8, 6))
for epoch in sorted(set(epochs)):
    c = sm.to_rgba(epoch)
    avg_mi_xz = [np.mean(layer[epochs == epoch]) for layer in mi_xz]
    avg_mi_zy = [np.mean(layer[epochs == epoch]) for layer in mi_zy]
    ax.plot(avg_mi_xz, avg_mi_zy, c=c, alpha=0.1, zorder=1)
    ax.scatter(avg_mi_xz, avg_mi_zy, s=30, facecolors=[c]*num_embeddings, edgecolor='none', alpha=1, zorder=2)

ax.set_xlabel('I(X; Z)')
ax.set_ylabel('I(Z; Y)')
ax.set_title('Info Plane Across Layers')
cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
plt.colorbar(sm, label='Epoch', cax=cbaxes)
plt.tight_layout()
plt.savefig(f'plots/{model}/infoplane_{file_name.split("/")[-1]}.png', bbox_inches='tight')
plt.show()

# Create subplots for each layer
fig, axes = plt.subplots(1, num_embeddings, figsize=(6 * num_embeddings, 6))
if num_embeddings == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    for epoch in sorted(set(epochs)):
        c = sm.to_rgba(epoch)
        avg_mi_xz = np.mean(mi_xz[i][epochs == epoch])
        avg_mi_zy = np.mean(mi_zy[i][epochs == epoch])
        ax.scatter(avg_mi_xz, avg_mi_zy, color=c, alpha=0.7)
    ax.set_xlabel('I(X; Z)')
    ax.set_ylabel('I(Z; Y)')
    ax.set_title(f'Layer {i + 1}')
plt.tight_layout()
plt.savefig(f'plots/{model}/infoplane_{file_name.split("/")[-1]}_layers.png', bbox_inches='tight')
plt.show()