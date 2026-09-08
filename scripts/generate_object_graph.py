import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def generate_object_graph():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 7)
    ax.axis('off')
    
    ax.text(5, 6.2, "Relational Object Graph G=(V,E)", 
            color='#c9d1d9', fontsize=16, fontweight='bold', ha='center', va='center')
            
    # Nodes (x, y, label, color)
    nodes = {
        'o1': (5, 4, 'Obj_1\n(Blue, A=12)', '#0074D9'),
        'o2': (2, 1.5, 'Obj_2\n(Red, A=4)', '#FF4136'),
        'o3': (8, 1.5, 'Obj_3\n(Red, A=4)', '#FF4136'),
        'o4': (5, 0, 'Obj_4\n(Green, A=8)', '#2ECC40')
    }
    
    edges = [
        ('o1', 'o2', 'Encloses', 'left', -0.5),
        ('o1', 'o3', 'Encloses', 'right', 0.5),
        ('o2', 'o4', 'Above', 'left', -0.3),
        ('o3', 'o4', 'Above', 'right', 0.3),
        ('o2', 'o3', 'Collinear (H)', 'center', 0)
    ]
    
    # Draw edges
    for n1, n2, label, align, offset in edges:
        x1, y1 = nodes[n1][0], nodes[n1][1]
        x2, y2 = nodes[n2][0], nodes[n2][1]
        
        # Simple line
        ax.plot([x1, x2], [y1, y2], color='#8b949e', linestyle='-', linewidth=2, zorder=1)
        
        # Midpoint for text
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + offset, my + 0.2, label, color='#8b949e', fontsize=9, fontweight='bold', 
                ha='center', va='center', bbox=dict(facecolor='#0d1117', edgecolor='none', pad=2))
        
    # Draw nodes
    for k, (x, y, label, color) in nodes.items():
        circle = patches.Circle((x, y), radius=0.8, linewidth=3, edgecolor=color, facecolor='#161b22', zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, label, color='#c9d1d9', fontsize=9, fontweight='bold', ha='center', va='center', zorder=3)
        
    plt.tight_layout()
    out_path = Path('C:/Users/DELL/.gemini/antigravity/scratch/arc-reasoning-agent/results/object_graph_vis.png')
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    print(f"Object graph saved to {out_path}")

if __name__ == '__main__':
    generate_object_graph()
