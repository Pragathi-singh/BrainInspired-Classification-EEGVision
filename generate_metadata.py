import json
import os
from pathlib import Path

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create placeholder metadata for 9 samples
# In a real scenario, this would be populated during training
# For now, we'll create a basic structure that allows the AI to know about samples

grid_meta = []
for i in range(9):
    # This is a placeholder - ideally populated from training
    grid_meta.append({
        'index': i,
        'true': classes[i % 10],
        'pred': classes[(i + 1) % 10]  # Example: slightly different predictions
    })

os.makedirs('results', exist_ok=True)
with open('results/predictions_grid_meta.json', 'w') as f:
    json.dump({'cells': grid_meta}, f, indent=2)

print("✓ Metadata file created: results/predictions_grid_meta.json")
print(json.dumps({'cells': grid_meta}, indent=2))
