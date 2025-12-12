"""
Generate Figure 3: Fuzzy Inference Process Example for Paper
-------------------------------------------------------------
Creates a flowchart visualization showing the 4 steps of Mamdani fuzzy inference:
1. Input features (brightness, color_temp, saturation, etc.)
2. Fuzzification (membership degrees)
3. Rule firing (which rules activate)
4. Defuzzification (output emotion vector)

Output: 300 DPI PDF for LaTeX inclusion
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

OUTPUT_DIR = Path('/home/paloma/cerebrum-artis/paper-factory/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Example input features (representing a dark, cold, desaturated painting → sadness)
EXAMPLE_FEATURES = {
    'brightness': 0.2,      # Dark
    'color_temp': 0.3,      # Cold
    'saturation': 0.25,     # Desaturated
    'color_harmony': 0.4,
    'complexity': 0.5,
    'symmetry': 0.45,
    'texture': 0.6
}

# Fuzzification results (membership degrees for each fuzzy term)
FUZZIFICATION = {
    'brightness': {'Very Low': 0.8, 'Low': 0.2, 'Medium': 0.0, 'High': 0.0, 'Very High': 0.0},
    'color_temp': {'Very Low': 0.6, 'Low': 0.4, 'Medium': 0.0, 'High': 0.0, 'Very High': 0.0},
    'saturation': {'Very Low': 0.75, 'Low': 0.25, 'Medium': 0.0, 'High': 0.0, 'Very High': 0.0}
}

# Example rules that fire
ACTIVE_RULES = [
    {'rule': 'IF brightness=Dark AND color_temp=Cold THEN sadness', 'strength': 0.6},
    {'rule': 'IF brightness=Dark AND saturation=Low THEN fear', 'strength': 0.75},
    {'rule': 'IF color_temp=Cold AND saturation=Low THEN sadness', 'strength': 0.4},
]

# Final output (emotion membership vector)
OUTPUT_EMOTIONS = {
    'amusement': 0.05,
    'awe': 0.08,
    'contentment': 0.03,
    'excitement': 0.02,
    'anger': 0.12,
    'disgust': 0.15,
    'fear': 0.21,
    'sadness': 0.68,
    'other': 0.10
}


def draw_box(ax, x, y, width, height, text, color='lightblue', title=None):
    """Draw a fancy box with text"""
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='black',
                          linewidth=2, alpha=0.8)
    ax.add_patch(box)
    
    # Title
    if title:
        ax.text(x + width/2, y + height - 0.05, title,
                ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Content
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.3))


def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw arrow between boxes"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='darkblue')
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.1, mid_y, label, fontsize=7,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))


def create_fuzzy_inference_figure():
    """
    Create complete Figure 3: Fuzzy Inference Process
    """
    print("\n🎯 Creating Figure 3: Fuzzy Inference Example")
    print("="*60)
    
    fig = plt.figure(figsize=(14, 11))
    
    # Create two sections: flowchart (top) and output bar chart (bottom)
    ax = fig.add_axes([0.05, 0.35, 0.9, 0.6])  # Flowchart area
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    fig.suptitle('Fuzzy Inference Process: From Visual Features to Emotion',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Step 1: Input Features (top)
    print("📊 Step 1: Input Features...")
    input_text = '\n'.join([f'{k}: {v:.2f}' for k, v in list(EXAMPLE_FEATURES.items())[:4]])
    input_text += '\n...'
    draw_box(ax, 0.3, 6.2, 2.2, 1.3, input_text, color='#ffe6e6',
             title='1. INPUT FEATURES')
    
    # Step 2: Fuzzification (upper middle)
    print("🔍 Step 2: Fuzzification...")
    fuzz_text = 'brightness:\n  Very Low: 0.8\n  Low: 0.2\n\ncolor_temp:\n  Very Low: 0.6\n  Low: 0.4'
    draw_box(ax, 3.3, 6.2, 3.4, 1.3, fuzz_text, color='#e6f2ff',
             title='2. FUZZIFICATION')
    
    # Arrow 1→2
    draw_arrow(ax, 2.5, 6.85, 3.3, 6.85)
    
    # Step 3: Rule Firing (middle)
    print("⚡ Step 3: Rule Evaluation...")
    rule_box_y = 3.8
    
    # Title for rule section
    ax.text(5, 5.3, '3. RULE EVALUATION', ha='center', va='center',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#fff4e6', alpha=0.9, pad=0.3))
    
    for i, rule_info in enumerate(ACTIVE_RULES):
        # Simplify rule text properly
        rule_text = rule_info['rule']
        
        # Extract emotion from THEN clause
        if 'THEN sadness' in rule_text:
            emotion = 'sadness'
        elif 'THEN fear' in rule_text:
            emotion = 'fear'
        else:
            emotion = 'emotion'
        
        # Create readable short form
        if i == 0:
            display_text = f"IF Dark AND Cold\nTHEN {emotion}"
        elif i == 1:
            display_text = f"IF Dark AND Low Sat\nTHEN {emotion}"
        else:
            display_text = f"IF Cold AND Low Sat\nTHEN {emotion}"
        
        strength = rule_info['strength']
        color = '#fff9e6'
        draw_box(ax, 0.3 + i * 3.2, rule_box_y, 2.9, 0.9,
                 f"{display_text}\nStrength: {strength:.2f}",
                 color=color)
    
    # Arrow 2→3 (aproximar do bloco fuzzification)
    draw_arrow(ax, 5, 6.2, 5, 5.4)
    
    # Step 4: Aggregation & Defuzzification (lower middle)
    print("📐 Step 4: Defuzzification...")
    agg_text = 'MAX-MIN Composition\n↓\nCentroid Method'
    draw_box(ax, 2.5, 1.8, 5, 1.1, agg_text, color='#e6ffe6',
             title='4. AGGREGATION & DEFUZZIFICATION')
    
    # Arrows 3→4 (from each rule box)
    draw_arrow(ax, 1.8, 3.8, 4, 2.9)
    draw_arrow(ax, 5, 3.8, 5, 2.9)
    draw_arrow(ax, 8.2, 3.8, 6, 2.9)
    
    # Step 5: Output (bottom) - Bar chart of emotions
    print("📊 Step 5: Output Emotion Vector...")
    ax_bar = fig.add_axes([0.12, 0.05, 0.76, 0.25])
    
    emotions = list(OUTPUT_EMOTIONS.keys())
    values = list(OUTPUT_EMOTIONS.values())
    colors = ['#ff9999' if e in ['amusement', 'awe', 'contentment', 'excitement']
              else '#9999ff' if e in ['anger', 'disgust', 'fear', 'sadness']
              else '#999999' for e in emotions]
    
    bars = ax_bar.bar(emotions, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Highlight sadness
    bars[emotions.index('sadness')].set_linewidth(3)
    bars[emotions.index('sadness')].set_edgecolor('red')
    
    ax_bar.set_ylabel('Membership Degree', fontsize=10, fontweight='bold')
    ax_bar.set_xlabel('Emotion Category', fontsize=10, fontweight='bold')
    ax_bar.set_title('5. OUTPUT: Emotion Membership Vector', 
                     fontsize=11, fontweight='bold', pad=10,
                     bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    ax_bar.set_ylim(0, 0.8)  # Adjusted for better visualization
    ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
    ax_bar.set_axisbelow(True)
    
    # Rotate x labels
    plt.setp(ax_bar.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    # Add values on top of bars
    for bar, val in zip(bars, values):
        if val > 0.15:  # Only show significant values
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    # Arrow pointing to output (seta menor e mais próxima do gráfico)
    ax.annotate('', xy=(5, 0.5), xytext=(5, 1.6),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='darkblue'))
    ax.text(5.3, 1.0, 'Output\nVector', fontsize=7, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))
    
    # Add example painting reference at top
    ax.text(5, 7.6, 'Example: Dark, cold, desaturated painting → Sadness', 
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, pad=0.4))
    
    # Save figure
    output_pdf = OUTPUT_DIR / 'fig3_fuzzy_inference.pdf'
    output_png = OUTPUT_DIR / 'fig3_fuzzy_inference.png'
    
    # No tight_layout since we're using manual positioning
    
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white', dpi=300)
    fig.savefig(output_png, bbox_inches='tight', facecolor='white', dpi=300)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white', dpi=300)
    fig.savefig(output_png, bbox_inches='tight', facecolor='white', dpi=300)
    
    print("\n✅ Figure 3 saved:")
    print(f"   PDF: {output_pdf}")
    print(f"   PNG: {output_png}")
    
    plt.close()
    
    print("\n📝 LaTeX usage:")
    print("   \\includegraphics[width=\\textwidth]{figures/fig3_fuzzy_inference.pdf}")
    
    # Print example values for caption verification
    print("\n📊 Values for caption:")
    print(f"   Input: dark={1-EXAMPLE_FEATURES['brightness']:.1f}, "
          f"cold={1-EXAMPLE_FEATURES['color_temp']:.1f}, "
          f"desaturated={1-EXAMPLE_FEATURES['saturation']:.1f}")
    print(f"   Output: sadness={OUTPUT_EMOTIONS['sadness']:.2f}, "
          f"fear={OUTPUT_EMOTIONS['fear']:.2f}")
    print("\n" + "="*60)


if __name__ == "__main__":
    create_fuzzy_inference_figure()
    print("\n🎉 Done! Figure 3 ready for paper insertion.")
