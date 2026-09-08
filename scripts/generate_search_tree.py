import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def generate_search_tree():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 7)
    ax.axis('off')
    
    ax.text(5, 6.2, "Hierarchical Sub-Goal A* Search Space", 
            color='#c9d1d9', fontsize=16, fontweight='bold', ha='center', va='center')
            
    # Nodes (x, y, label, type)
    nodes = {
        'root': (5, 5, 'Input State\n(Depth 0)', 'active'),
        
        # Depth 1
        'd1_1': (1, 3, 'Filter Color\n(d=45)', 'pruned'),
        'd1_2': (5, 3, 'Raycast Lines\n(d=12)', 'active'),
        'd1_3': (9, 3, 'Rotate 90\n(d=38)', 'pruned'),
        
        # Depth 2 (from d1_2)
        'd2_1': (3, 1, 'Crop Area\n(d=8)', 'active'),
        'd2_2': (7, 1, 'Fill Holes\n(d=18)', 'pruned'),
        
        # Depth 3 (from d2_1)
        'd3_1': (3, -0.5, 'Exact Match!\n(d=0)', 'solution')
    }
    
    edges = [
        ('root', 'd1_1'), ('root', 'd1_2'), ('root', 'd1_3'),
        ('d1_2', 'd2_1'), ('d1_2', 'd2_2'),
        ('d2_1', 'd3_1')
    ]
    
    colors = {
        'active': ('#1f6feb', '#0d1117'),   # Blue
        'pruned': ('#8b949e', '#0d1117'),   # Gray
        'solution': ('#238636', '#0d1117')  # Green
    }
    
    # Draw edges
    for n1, n2 in edges:
        x1, y1 = nodes[n1][0], nodes[n1][1]
        x2, y2 = nodes[n2][0], nodes[n2][1]
        style = '--' if nodes[n2][3] == 'pruned' else '-'
        alpha = 0.4 if nodes[n2][3] == 'pruned' else 1.0
        color = '#238636' if nodes[n2][3] == 'solution' else '#c9d1d9'
        
        ax.plot([x1, x2], [y1, y2], color=color, linestyle=style, alpha=alpha, linewidth=2, zorder=1)
        
    # Draw nodes
    for k, (x, y, label, ntype) in nodes.items():
        edgecolor, facecolor = colors[ntype]
        alpha = 0.5 if ntype == 'pruned' else 1.0
        
        rect = patches.FancyBboxPatch((x - 1, y - 0.5), 2, 1,
                                      boxstyle="round,pad=0.1,rounding_size=0.1", 
                                      linewidth=2, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, label, color='#c9d1d9', fontsize=10, fontweight='bold', ha='center', va='center', alpha=alpha, zorder=3)
        
    plt.tight_layout()
    out_path = Path('C:/Users/DELL/.gemini/antigravity/scratch/arc-reasoning-agent/results/search_tree_vis.png')
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    print(f"Search tree saved to {out_path}")

if __name__ == '__main__':
    generate_search_tree()
