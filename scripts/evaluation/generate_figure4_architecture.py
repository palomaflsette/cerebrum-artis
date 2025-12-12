"""
Generate Figure 4: Cerebrum-Artis Architecture Diagram for Paper
----------------------------------------------------------------
Creates a comprehensive architecture flowchart showing:
- Visual pathway: Image → ResNet50 → [2048]
- Textual pathway: Text → RoBERTa → [768]
- Symbolic pathway: HSV → Fuzzy System → [7]
- Fusion: Concat → MLP → [9 emotions]
- Annotations for V1, V2, V3, V4 variants

Output: 300 DPI PDF (vector graphics) for LaTeX inclusion
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

OUTPUT_DIR = Path('/home/paloma/cerebrum-artis/paper-factory/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'visual': '#3498db',      # Blue
    'textual': '#2ecc71',     # Green
    'fuzzy': '#e67e22',       # Orange
    'fusion': '#e74c3c',      # Red
    'background': '#ecf0f1'   # Light gray
}


def draw_box(ax, x, y, width, height, text, color, title=None, fontsize=9):
    """Draw a component box"""
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='black',
                          linewidth=2, alpha=0.85)
    ax.add_patch(box)
    
    if title:
        ax.text(x + width/2, y + height - 0.08, title,
                ha='center', va='top', fontsize=fontsize, fontweight='bold')
        ax.text(x + width/2, y + height/2 - 0.1, text,
                ha='center', va='center', fontsize=fontsize-1)
    else:
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center', fontsize=fontsize, fontweight='bold')


def draw_arrow(ax, x1, y1, x2, y2, color='black', width=2):
    """Draw connection arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=width, color=color, alpha=0.8)
    ax.add_patch(arrow)


def draw_input(ax, x, y, size, text, color):
    """Draw input node"""
    circle = Circle((x, y), size, facecolor=color, edgecolor='black',
                   linewidth=2, alpha=0.85)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')


def create_architecture_diagram():
    """
    Create complete Figure 4: Architecture Diagram
    """
    print("\n🎯 Creating Figure 4: Architecture Diagram")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Title
    fig.suptitle('Cerebrum-Artis: Hybrid Neuro-Symbolic Architecture',
                 fontsize=15, fontweight='bold', y=0.96)
    
    # ==================== INPUT LAYER ====================
    print("📊 Drawing input layer...")
    
    # Visual input
    draw_input(ax, 1.5, 8.5, 0.4, 'Image', COLORS['visual'])
    ax.text(1.5, 9.3, 'Visual Pathway', ha='center', fontsize=9, 
            fontweight='bold', color=COLORS['visual'])
    
    # Textual input
    draw_input(ax, 1.5, 6, 0.4, 'Text', COLORS['textual'])
    ax.text(1.5, 6.8, 'Textual Pathway', ha='center', fontsize=9,
            fontweight='bold', color=COLORS['textual'])
    
    # Fuzzy input
    draw_input(ax, 1.5, 3.5, 0.4, 'HSV', COLORS['fuzzy'])
    ax.text(1.5, 4.3, 'Symbolic Pathway', ha='center', fontsize=9,
            fontweight='bold', color=COLORS['fuzzy'])
    
    # ==================== FEATURE EXTRACTION ====================
    print("🔍 Drawing feature extractors...")
    
    # ResNet50
    draw_box(ax, 3, 8, 2, 1, 'ResNet50\n(pretrained)', COLORS['visual'], fontsize=10)
    draw_arrow(ax, 1.9, 8.5, 3, 8.5, COLORS['visual'])
    
    # Visual features
    draw_box(ax, 5.5, 8, 1.2, 1, '[2048]', COLORS['visual'])
    draw_arrow(ax, 5, 8.5, 5.5, 8.5, COLORS['visual'])
    
    # RoBERTa
    draw_box(ax, 3, 5.5, 2, 1, 'RoBERTa-base\n(pretrained)', COLORS['textual'], fontsize=10)
    draw_arrow(ax, 1.9, 6, 3, 6, COLORS['textual'])
    
    # Text features
    draw_box(ax, 5.5, 5.5, 1.2, 1, '[768]', COLORS['textual'])
    draw_arrow(ax, 5, 6, 5.5, 6, COLORS['textual'])
    
    # Fuzzy System
    draw_box(ax, 3, 3, 2, 1, 'Fuzzy\nInference', COLORS['fuzzy'], fontsize=10)
    draw_arrow(ax, 1.9, 3.5, 3, 3.5, COLORS['fuzzy'])
    
    # Fuzzy features
    draw_box(ax, 5.5, 3, 1.2, 1, '[7]', COLORS['fuzzy'])
    draw_arrow(ax, 5, 3.5, 5.5, 3.5, COLORS['fuzzy'])
    
    # ==================== FUSION LAYER ====================
    print("🔗 Drawing fusion layer...")
    
    # Concatenation
    draw_box(ax, 7.5, 5, 1.5, 3, 'Concat',
             COLORS['fusion'], fontsize=10)
    
    # Arrows to concat
    draw_arrow(ax, 6.7, 8.5, 7.5, 7.5, COLORS['visual'], width=3)
    draw_arrow(ax, 6.7, 6, 7.5, 6.5, COLORS['textual'], width=3)
    draw_arrow(ax, 6.7, 3.5, 7.5, 5.5, COLORS['fuzzy'], width=2)
    
    # MLP
    draw_box(ax, 9.5, 5.5, 2, 2, 'MLP\nClassifier\n\n3 Layers\nDropout 0.3',
             COLORS['fusion'], fontsize=9)
    draw_arrow(ax, 9, 6.5, 9.5, 6.5, COLORS['fusion'], width=3)
    
    # ==================== OUTPUT ====================
    print("📤 Drawing output...")
    
    # Output layer
    draw_box(ax, 12, 5.5, 2, 2, 'Softmax\n\n[9 Emotions]',
             COLORS['fusion'], fontsize=10)
    draw_arrow(ax, 11.5, 6.5, 12, 6.5, COLORS['fusion'], width=3)
    
    # Emotion list
    emotions_text = ('amusement, awe,\ncontentment, excitement,\n' +
                    'anger, disgust, fear,\nsadness, other')
    ax.text(13, 4.5, emotions_text, ha='center', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))
    
    # ==================== VERSION ANNOTATIONS ====================
    print("📝 Adding version annotations...")
    
    # Create version comparison table on the right side
    table_x = 10.5
    table_y = 9.5
    
    # Table header
    ax.text(table_x + 2, table_y, 'Model Variants', ha='center', 
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black', linewidth=2))
    
    # V1
    y_offset = table_y - 0.8
    ax.add_patch(FancyBboxPatch((table_x, y_offset - 0.35), 4, 0.6,
                                boxstyle="round,pad=0.05", 
                                facecolor='lavender', edgecolor='purple', linewidth=2))
    ax.text(table_x + 0.5, y_offset, 'V1', fontsize=10, fontweight='bold', color='purple', va='center')
    ax.text(table_x + 1.2, y_offset, 'Baseline: Visual + Text (2816-dim)', 
            fontsize=8, va='center')
    
    # V2
    y_offset -= 0.9
    ax.add_patch(FancyBboxPatch((table_x, y_offset - 0.35), 4, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor='peachpuff', edgecolor='darkorange', linewidth=2))
    ax.text(table_x + 0.5, y_offset, 'V2', fontsize=10, fontweight='bold', color='darkorange', va='center')
    ax.text(table_x + 1.2, y_offset, '+ Fuzzy: All features (2823-dim)', 
            fontsize=8, va='center')
    
    # V3
    y_offset -= 0.9
    ax.add_patch(FancyBboxPatch((table_x, y_offset - 0.35), 4, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor='lightgreen', edgecolor='darkgreen', linewidth=2))
    ax.text(table_x + 0.5, y_offset, 'V3', fontsize=10, fontweight='bold', color='darkgreen', va='center')
    ax.text(table_x + 1.2, y_offset, 'Adaptive Gating: Learned fusion', 
            fontsize=8, va='center')
    
    # V4
    y_offset -= 0.9
    ax.add_patch(FancyBboxPatch((table_x, y_offset - 0.35), 4, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor='mistyrose', edgecolor='darkred', linewidth=2))
    ax.text(table_x + 0.5, y_offset, 'V4', fontsize=10, fontweight='bold', color='darkred', va='center')
    ax.text(table_x + 1.2, y_offset, 'Ensemble: Avg(V1, V2, V3)', 
            fontsize=8, va='center')
    
    # ==================== LEGEND ====================
    print("🎨 Adding legend...")
    
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['visual'], edgecolor='black', label='Visual Pathway (CNN)'),
        mpatches.Patch(facecolor=COLORS['textual'], edgecolor='black', label='Textual Pathway (Transformer)'),
        mpatches.Patch(facecolor=COLORS['fuzzy'], edgecolor='black', label='Symbolic Pathway (Fuzzy Logic)'),
        mpatches.Patch(facecolor=COLORS['fusion'], edgecolor='black', label='Fusion & Classification'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.02, 0.05),
             fontsize=9, framealpha=0.95, fancybox=True, shadow=True)
    
    # ==================== NOTES ====================
    
    # Dimension note
    ax.text(8.2, 1.2, 'Feature Dimensions: Visual [2048] + Text [768] + Fuzzy [7] = 2823 total',
            ha='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=0.4))
    
    # Save figure
    output_pdf = OUTPUT_DIR / 'fig4_architecture.pdf'
    output_png = OUTPUT_DIR / 'fig4_architecture.png'
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white', dpi=300)
    fig.savefig(output_png, bbox_inches='tight', facecolor='white', dpi=300)
    
    print("\n✅ Figure 4 saved:")
    print(f"   PDF: {output_pdf}")
    print(f"   PNG: {output_png}")
    
    plt.close()
    
    print("\n📝 LaTeX usage:")
    print("   \\includegraphics[width=\\textwidth]{figures/fig4_architecture.pdf}")
    
    print("\n📊 Architecture Summary:")
    print("   Visual:   ResNet50 → [2048]")
    print("   Textual:  RoBERTa  → [768]")
    print("   Symbolic: Fuzzy    → [7]")
    print("   --------------------------------")
    print("   V1: 2048 + 768 = 2816-dim (no fuzzy)")
    print("   V2: 2048 + 768 + 7 = 2823-dim (concat)")
    print("   V3: Adaptive gating mechanism")
    print("   V4: Ensemble of V1+V2+V3")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    create_architecture_diagram()
    print("\n🎉 Done! Figure 4 ready for paper insertion.")
