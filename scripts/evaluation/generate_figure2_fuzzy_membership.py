"""
Generate Figure 2: Fuzzy Membership Functions for Paper
--------------------------------------------------------
Creates a 3x3 grid showing triangular membership functions for 7 visual features:
- Brightness, Color Temperature, Saturation, Color Harmony
- Complexity, Symmetry, Texture Roughness

Each subplot shows 5 fuzzy terms: Very Low, Low, Medium, High, Very High

Output: 300 DPI PDF (vector graphics) for LaTeX inclusion
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

OUTPUT_DIR = Path('/home/paloma/cerebrum-artis/paper-factory/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fuzzy features (7 features)
FEATURES = [
    'Brightness',
    'Color Temperature',
    'Saturation',
    'Color Harmony',
    'Complexity',
    'Symmetry',
    'Texture Roughness'
]

# Fuzzy terms and their triangular parameters [a, b, c]
FUZZY_TERMS = {
    'Very Low': ([0.0, 0.0, 0.25], '#3498db'),    # Blue
    'Low':      ([0.0, 0.25, 0.5], '#1abc9c'),    # Cyan
    'Medium':   ([0.25, 0.5, 0.75], '#2ecc71'),   # Green
    'High':     ([0.5, 0.75, 1.0], '#f39c12'),    # Yellow/Orange
    'Very High':([0.75, 1.0, 1.0], '#e74c3c')     # Red
}


def trimf(x, abc):
    """Triangular membership function"""
    a, b, c = abc
    y = np.zeros_like(x)
    
    # Left slope
    if a != b:
        idx = np.logical_and(a <= x, x < b)
        y[idx] = (x[idx] - a) / (b - a)
    
    # Right slope
    if b != c:
        idx = np.logical_and(b <= x, x <= c)
        y[idx] = (c - x[idx]) / (c - b)
    
    # Peak
    y[x == b] = 1.0
    
    return y


def plot_membership_function(ax, feature_name):
    """Plot membership functions for a single feature"""
    universe = np.linspace(0, 1, 1000)
    
    # Plot each fuzzy term
    for term_name, (params, color) in FUZZY_TERMS.items():
        y = trimf(universe, params)
        ax.plot(universe, y, label=term_name, linewidth=2.5, color=color, alpha=0.85)
        ax.fill_between(universe, y, alpha=0.15, color=color)
    
    # Formatting
    ax.set_xlabel('Feature Value', fontsize=9, fontweight='bold')
    ax.set_ylabel('Membership ($\mu$)', fontsize=9, fontweight='bold')
    ax.set_title(feature_name, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Ticks
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=8)


def create_fuzzy_membership_figure():
    """
    Create complete Figure 2: Fuzzy Membership Functions
    Single subplot + table showing all features use same partitioning
    """
    print("\n🎯 Creating Figure 2: Fuzzy Membership Functions")
    print("="*60)
    
    fig = plt.figure(figsize=(12, 6))
    
    # Create main subplot for membership functions (left side)
    ax_plot = plt.subplot2grid((1, 2), (0, 0))
    
    print("📊 Plotting generic membership functions...")
    
    # Plot membership functions
    universe = np.linspace(0, 1, 1000)
    
    for term_name, (params, color) in FUZZY_TERMS.items():
        y = trimf(universe, params)
        ax_plot.plot(universe, y, label=term_name, linewidth=3, color=color, alpha=0.9)
        ax_plot.fill_between(universe, y, alpha=0.2, color=color)
    
    # Formatting
    ax_plot.set_xlabel('Normalized Feature Value', fontsize=11, fontweight='bold')
    ax_plot.set_ylabel('Membership Degree ($\mu$)', fontsize=11, fontweight='bold')
    ax_plot.set_title('(a) Triangular Membership Functions', fontsize=12, fontweight='bold', pad=12)
    ax_plot.set_xlim(0, 1)
    ax_plot.set_ylim(0, 1.05)
    ax_plot.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax_plot.set_axisbelow(True)
    ax_plot.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_plot.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_plot.tick_params(labelsize=10)
    ax_plot.legend(loc='upper right', fontsize=9, framealpha=0.95, fancybox=True, shadow=True)
    
    # Create table (right side)
    ax_table = plt.subplot2grid((1, 2), (0, 1))
    ax_table.axis('off')
    
    # Table title
    ax_table.text(0.5, 0.95, '(b) Applied to Seven Visual Features', 
                  ha='center', va='top', fontsize=12, fontweight='bold',
                  transform=ax_table.transAxes)
    
    # Feature list with descriptions
    features_text = [
        ('1. Brightness', 'Overall luminosity'),
        ('2. Color Temperature', 'Warm ↔ Cold palette'),
        ('3. Saturation', 'Color intensity/vividness'),
        ('4. Color Harmony', 'Palette coherence'),
        ('5. Complexity', 'Visual detail density'),
        ('6. Symmetry', 'Compositional balance'),
        ('7. Texture Roughness', 'Surface smoothness'),
    ]
    
    y_start = 0.82
    for i, (feat, desc) in enumerate(features_text):
        y_pos = y_start - i * 0.11
        
        # Feature name
        ax_table.text(0.05, y_pos, feat, 
                     ha='left', va='top', fontsize=10, fontweight='bold',
                     transform=ax_table.transAxes)
        
        # Description
        ax_table.text(0.05, y_pos - 0.04, desc, 
                     ha='left', va='top', fontsize=8, style='italic', color='gray',
                     transform=ax_table.transAxes)
    
    # Add note at bottom
    note_text = ('All features normalized to [0,1] domain\n' +
                'Same triangular partitioning ensures\n' +
                'consistent fuzzy reasoning across features')
    ax_table.text(0.5, 0.08, note_text,
                 ha='center', va='center', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, pad=0.5),
                 transform=ax_table.transAxes, style='italic')
    
    # Overall title
    fig.suptitle('Fuzzy Linguistic Variable Definition for Visual Features',
                 fontsize=13, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_pdf = OUTPUT_DIR / 'fig2_fuzzy_membership.pdf'
    output_png = OUTPUT_DIR / 'fig2_fuzzy_membership.png'
    
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white', dpi=300)
    fig.savefig(output_png, bbox_inches='tight', facecolor='white', dpi=300)
    
    print("\n✅ Figure 2 saved:")
    print(f"   PDF: {output_pdf}")
    print(f"   PNG: {output_png}")
    
    plt.close()
    
    print("\n📝 LaTeX usage:")
    print("   \\includegraphics[width=\\textwidth]{figures/fig2_fuzzy_membership.pdf}")
    
    print("\n📊 Fuzzy Terms:")
    for term, (params, _) in FUZZY_TERMS.items():
        print(f"   {term:12s}: [{params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f}]")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    create_fuzzy_membership_figure()
    print("\n🎉 Done! Figure 2 ready for paper insertion.")
