import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_cover_image():
    # 560x280 pixels at 100 DPI
    fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=100)
    fig.patch.set_facecolor('#0d1117') # GitHub dark background
    ax.set_facecolor('#0d1117')
    
    # Remove axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Title
    ax.text(5, 4.2, "Hierarchical Spatial Reasoning", 
            color='#c9d1d9', fontsize=18, fontweight='bold', ha='center', va='center', fontfamily='sans-serif')
    ax.text(5, 3.6, "ARC Prize 2026 Paper Track", 
            color='#58a6ff', fontsize=10, fontweight='bold', ha='center', va='center', fontfamily='sans-serif')
    
    # Pipeline Boxes
    boxes = [
        (0.5, "Input\nGrid", '#238636'),
        (2.5, "Topology &\nObjects", '#8957e5'),
        (4.5, "Spatial\nRelations", '#d29922'),
        (6.5, "Sub-goal\nA* Search", '#f85149'),
        (8.5, "Output\nProgram", '#238636')
    ]
    
    y_pos = 1.5
    box_width = 1.4
    box_height = 1.0
    
    for i, (x, text, color) in enumerate(boxes):
        # Draw box
        rect = patches.FancyBboxPatch((x - box_width/2, y_pos - box_height/2), 
                                      box_width, box_height, 
                                      boxstyle="round,pad=0.1,rounding_size=0.1", 
                                      linewidth=1.5, edgecolor=color, facecolor='#161b22')
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y_pos, text, color='#c9d1d9', fontsize=9, fontweight='bold', ha='center', va='center')
        
        # Draw arrow to next box
        if i < len(boxes) - 1:
            next_x = boxes[i+1][0]
            arrow = patches.FancyArrowPatch((x + box_width/2 + 0.05, y_pos), 
                                            (next_x - box_width/2 - 0.05, y_pos),
                                            arrowstyle='-|>', mutation_scale=15, 
                                            color='#8b949e', linewidth=1.5)
            ax.add_patch(arrow)

    plt.tight_layout(pad=0)
    
    out_path = Path('C:/Users/DELL/.gemini/antigravity/scratch/arc-reasoning-agent/results/cover_image_560x280.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches=None)
    print(f"Cover image saved to {out_path}")

if __name__ == '__main__':
    create_cover_image()
