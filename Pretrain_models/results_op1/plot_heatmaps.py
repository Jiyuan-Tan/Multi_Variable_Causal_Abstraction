import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON files
with open('candidates_boundless_op1.json', 'r') as f:
    candidates = json.load(f)

with open('feature_counts_partial_op1.json', 'r') as f:
    feature_counts = json.load(f)

# Extract unique layers and positions
layers = sorted(set(int(k.split('_')[0][1:]) for k in candidates.keys()))
positions = sorted(set(int(k.split('_')[1][1:]) for k in candidates.keys()))

n_layers = len(layers)
n_positions = len(positions)

print(f"Layers: {layers[0]} to {layers[-1]} ({n_layers} total)")
print(f"Positions: {positions[0]} to {positions[-1]} ({n_positions} total)")

# Create 2D matrices
candidates_matrix = np.zeros((n_layers, n_positions))
counts_matrix = np.zeros((n_layers, n_positions))

for i, layer in enumerate(layers):
    for j, pos in enumerate(positions):
        key = f"L{layer}_P{pos}"
        if key in candidates:
            candidates_matrix[i, j] = candidates[key]
        if key in feature_counts:
            counts_matrix[i, j] = feature_counts[key]

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14))

# Heatmap 1: Candidates Boundless
im1 = ax1.imshow(candidates_matrix, aspect='auto', cmap='viridis', origin='lower')
ax1.set_xlabel('Position', fontsize=12)
ax1.set_ylabel('Layer', fontsize=12)
ax1.set_title('Candidates Boundless (Accuracy)', fontsize=14)
ax1.set_xticks(range(n_positions))
ax1.set_xticklabels(positions, rotation=45)
ax1.set_yticks(range(n_layers))
ax1.set_yticklabels(layers)
cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
cbar1.set_label('Value', fontsize=10)

# Add text annotations for candidates heatmap
for i in range(n_layers):
    for j in range(n_positions):
        text = ax1.text(j, i, f'{candidates_matrix[i, j]:.2f}',
                       ha='center', va='center', color='white', fontsize=5)

# Heatmap 2: Feature Counts
im2 = ax2.imshow(counts_matrix, aspect='auto', cmap='hot', origin='lower')
ax2.set_xlabel('Position', fontsize=12)
ax2.set_ylabel('Layer', fontsize=12)
ax2.set_title('Feature Counts (Partial)', fontsize=14)
ax2.set_xticks(range(n_positions))
ax2.set_xticklabels(positions, rotation=45)
ax2.set_yticks(range(n_layers))
ax2.set_yticklabels(layers)
cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
cbar2.set_label('Count', fontsize=10)

# Add text annotations for feature counts heatmap
for i in range(n_layers):
    for j in range(n_positions):
        text = ax2.text(j, i, f'{int(counts_matrix[i, j])}',
                       ha='center', va='center', color='white', fontsize=5)

plt.tight_layout()
plt.savefig('heatmaps_op1.png', dpi=150, bbox_inches='tight')
plt.savefig('heatmaps_op1.pdf', bbox_inches='tight')
print("Saved heatmaps to heatmaps_op1.png and heatmaps_op1.pdf")
plt.show()

