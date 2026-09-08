import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from pathlib import Path

# Standard ARC colors
ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
cmap = ListedColormap(ARC_COLORS)
norm = plt.Normalize(vmin=0, vmax=9)

def draw_grid(ax, grid, title):
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_title(title, color='#c9d1d9', pad=10, fontsize=14, fontweight='bold')
    
    # Draw gridlines
    h, w = grid.shape
    for i in range(h + 1):
        ax.axhline(i - 0.5, color='#30363d', linewidth=1)
    for j in range(w + 1):
        ax.axvline(j - 0.5, color='#30363d', linewidth=1)
        
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
        spine.set_linewidth(2)

def generate_reasoning_example():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='#0d1117')
    
    # Simulated Task 6150a2bd (or similar concept)
    # Input: scattered pixels
    grid_in = np.zeros((10, 10), dtype=int)
    grid_in[2, 2] = 1
    grid_in[2, 7] = 2
    grid_in[7, 2] = 3
    grid_in[7, 7] = 4
    
    # Intermediate: Raycasting / Topology (represented by bounding boxes or lines)
    grid_mid = np.copy(grid_in)
    grid_mid[2, 3:7] = 8 # connecting line
    grid_mid[3:7, 2] = 8
    grid_mid[7, 3:7] = 8
    grid_mid[3:7, 7] = 8
    
    # Output: Filled enclosed region
    grid_out = np.copy(grid_mid)
    grid_out[3:7, 3:7] = 2 # filled inside
    
    draw_grid(axes[0], grid_in, "1. Input Grid")
    draw_grid(axes[1], grid_mid, "2. Topology & Raycasting")
    draw_grid(axes[2], grid_out, "3. Sub-Goal Executed")
    
    # Add arrows between plots
    fig.text(0.35, 0.5, "➔", color='#58a6ff', fontsize=30, ha='center', va='center', fontweight='bold')
    fig.text(0.66, 0.5, "➔", color='#58a6ff', fontsize=30, ha='center', va='center', fontweight='bold')
    
    plt.tight_layout(pad=3.0)
    out_path = Path('C:/Users/DELL/.gemini/antigravity/scratch/arc-reasoning-agent/results/reasoning_example.png')
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200, bbox_inches='tight')
    print(f"Reasoning example saved to {out_path}")

if __name__ == '__main__':
    generate_reasoning_example()
