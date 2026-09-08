import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_cover_image():
    # 1280x720 pixels (16:9 aspect ratio, well over the 640x360 minimum)
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor('#0d1117') 
    ax.set_facecolor('#0d1117')
    
    # Remove axes
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 7.2)
    ax.axis('off')
    
    # Title
    ax.text(6.4, 6.0, "Hierarchical Spatial Reasoning", 
            color='#c9d1d9', fontsize=32, fontweight='bold', ha='center', va='center', fontfamily='sans-serif')
    ax.text(6.4, 5.2, "ARC Prize 2026 Paper Track", 
            color='#58a6ff', fontsize=18, fontweight='bold', ha='center', va='center', fontfamily='sans-serif')
    
    # Pipeline Boxes
    boxes = [
        (1.5, "Input\nGrid", '#238636'),
        (3.9, "Topology &\nObjects", '#8957e5'),
        (6.4, "Spatial\nRelations", '#d29922'),
        (8.9, "Sub-goal\nA* Search", '#f85149'),
        (11.3, "Output\nProgram", '#238636')
    ]
    
    y_pos = 2.5
    box_width = 1.8
    box_height = 1.4
    
    for i, (x, text, color) in enumerate(boxes):
        # Draw box
        rect = patches.FancyBboxPatch((x - box_width/2, y_pos - box_height/2), 
                                      box_width, box_height, 
                                      boxstyle="round,pad=0.2,rounding_size=0.15", 
                                      linewidth=2.5, edgecolor=color, facecolor='#161b22')
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y_pos, text, color='#c9d1d9', fontsize=14, fontweight='bold', ha='center', va='center')
        
        # Draw arrow to next box
        if i < len(boxes) - 1:
            next_x = boxes[i+1][0]
            arrow = patches.FancyArrowPatch((x + box_width/2 + 0.1, y_pos), 
                                            (next_x - box_width/2 - 0.1, y_pos),
                                            arrowstyle='-|>', mutation_scale=25, 
                                            color='#8b949e', linewidth=2.5)
            ax.add_patch(arrow)

    plt.tight_layout(pad=0)
    
    out_path = Path('C:/Users/DELL/.gemini/antigravity/scratch/arc-reasoning-agent/results/cover_image_1280x720.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches=None)
    print(f"Cover image saved to {out_path}")

if __name__ == '__main__':
    create_cover_image()
