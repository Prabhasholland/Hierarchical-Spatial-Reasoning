import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from our experiments
labels = ['Baseline v1', 'Object-Centric', '+ Topology', 'Full Spatial-Object', 'Hierarchical (Depth-4)']
train_acc = [6.00, 7.75, 6.50, 8.25, 8.25]
eval_acc = [0.50, 0.50, 0.50, 0.50, 0.75]  # Adjusted based on the final reports

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, train_acc, width, label='Train Accuracy (%)', color='#2c3e50')
rects2 = ax.bar(x + width/2, eval_acc, width, label='Eval Accuracy (%)', color='#e74c3c')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('ARC-AGI Solver Ablation Results', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
ax.legend(fontsize=11)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
out_path = Path('C:/Users/DELL/.gemini/antigravity/scratch/arc-reasoning-agent/results/ablation_chart.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to {out_path}")
