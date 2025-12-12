"""
Generate Figure 1: ArtEmis Dataset Overview for Paper
------------------------------------------------------
Creates a 2-column publication-quality figure with:
(a) Horizontal bar chart of emotion distribution across train/val/test splits
(b) 3x3 grid of representative paintings per emotion category

Output: 300 DPI PNG/PDF for LaTeX inclusion
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import os
from PIL import Image
from pathlib import Path

# Publication-quality settings
plt.style.use('seaborn-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# Paths
DATA_DIR = Path('/home/paloma/cerebrum-artis/data/artemis/dataset/official_data')
CSV_PATH = DATA_DIR / 'combined_artemis_with_splits.csv'
IMAGES_BASE = Path('/data/paloma/data/paintings/wikiart')  # WikiArt images
OUTPUT_DIR = Path('/home/paloma/cerebrum-artis/paper-factory/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Emotion categories and colors
EMOTIONS = [
    'amusement', 'awe', 'contentment', 'excitement',  # Positive
    'anger', 'disgust', 'fear', 'sadness',            # Negative
    'something else'                                   # Neutral
]

# Color palette: positive (warm), negative (cool), neutral (gray)
EMOTION_COLORS = {
    'amusement': '#ff9999',      # light red
    'awe': '#ffcc66',            # gold
    'contentment': '#99cc99',    # light green
    'excitement': '#ff6666',     # bright red
    'anger': '#cc6666',          # dark red
    'disgust': '#669999',        # teal
    'fear': '#6666cc',           # blue
    'sadness': '#9999cc',        # purple
    'something else': '#999999'  # gray
}

SPLIT_COLORS = {
    'train': '#2ecc71',
    'val': '#3498db',
    'test': '#e74c3c'
}


def load_dataset():
    """Load ArtEmis dataset with splits"""
    print("📊 Loading ArtEmis dataset...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    
    # Create unique painting ID (art_style + painting name)
    df['painting_id'] = df['art_style'].astype(str) + '/' + df['painting'].astype(str)
    
    print(f"✓ Loaded {len(df):,} annotations")
    print(f"✓ Unique paintings: {df['painting_id'].nunique():,}")
    return df


def plot_emotion_distribution(df, ax):
    """
    Plot (a): Horizontal bar chart showing split distribution
    Shows UNIQUE PAINTINGS per split (each painting counted once only)
    """
    print("📈 Plotting split distribution...")
    
    # First, get unique paintings per split (each painting counted ONCE)
    # Group by painting_id and split, take the FIRST emotion annotation
    unique_paintings = df.groupby(['painting_id', 'split']).first().reset_index()
    
    # Now count emotions from this deduplicated set
    split_emotion_counts = unique_paintings.groupby(['split', 'emotion']).size().unstack(fill_value=0)
    
    # Reorder emotions
    split_emotion_counts = split_emotion_counts[EMOTIONS]
    
    # Create stacked horizontal bar chart
    splits = ['train', 'val', 'test']
    split_emotion_counts = split_emotion_counts.loc[splits]
    
    # Plot
    left = np.zeros(len(splits))
    for emotion in EMOTIONS:
        counts = split_emotion_counts[emotion].values
        ax.barh(splits, counts, left=left, 
                color=EMOTION_COLORS[emotion], 
                label=emotion.replace('something else', 'other'),
                edgecolor='white', linewidth=0.5)
        left += counts
    
    # Formatting
    ax.set_xlabel('Number of Unique Paintings', fontsize=11, fontweight='bold')
    ax.set_ylabel('Split', fontsize=11, fontweight='bold')
    ax.set_title('(a) Distribution Across Splits', 
                 fontsize=12, fontweight='bold', pad=10)
    
    # Add total counts as text
    for i, split in enumerate(splits):
        total = split_emotion_counts.loc[split].sum()
        ax.text(total + 500, i, f'{total:,}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], 
              loc='upper right', ncol=3, 
              fontsize=8, framealpha=0.9)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    print("✓ Distribution plot complete")


def find_representative_paintings(df):
    """
    Find 9 representative paintings (1 per emotion, excluding 'something else')
    Prioritize high-quality, diverse art styles
    """
    print("🖼️  Selecting representative paintings...")
    
    representative = {}
    emotions_to_show = [e for e in EMOTIONS if e != 'something else'][:9]
    
    for emotion in emotions_to_show:
        emotion_df = df[df['emotion'] == emotion]
        
        # Sample from validation set for consistency
        val_samples = emotion_df[emotion_df['split'] == 'val']
        if len(val_samples) > 0:
            # Pick a random sample (can be refined to pick best examples)
            sample = val_samples.sample(n=1, random_state=42).iloc[0]
            representative[emotion] = {
                'painting': sample['painting'],
                'art_style': sample['art_style'],
                'utterance': sample['utterance'][:60] + '...' if len(sample['utterance']) > 60 else sample['utterance']
            }
    
    print(f"✓ Selected {len(representative)} paintings")
    return representative


def plot_painting_grid(representative, ax, fig):
    """
    Plot (b): 3x3 grid of representative paintings
    """
    print("🎨 Creating painting grid...")
    
    # Hide main axes
    ax.axis('off')
    
    emotions = list(representative.keys())[:9]
    
    # Create 3x3 grid manually using fig.add_axes
    grid_left = 0.55  # Start after distribution plot
    grid_bottom = 0.15
    grid_width = 0.40
    grid_height = 0.70
    
    cell_w = grid_width / 3
    cell_h = grid_height / 3
    padding = 0.015  # Increased padding to prevent label overlap
    
    loaded_count = 0
    for idx, emotion in enumerate(emotions):
        row = idx // 3
        col = idx % 3
        
        # Calculate position
        left = grid_left + col * cell_w + padding
        bottom = grid_bottom + (2 - row) * cell_h + padding  # Reverse row for top-to-bottom
        
        # Create axis
        inner_ax = fig.add_axes([left, bottom, cell_w - 2*padding, cell_h - 2*padding])
        inner_ax.axis('off')
        
        # Try to load image
        painting_info = representative[emotion]
        art_style = painting_info['art_style'].replace(' ', '_')
        painting_name = painting_info['painting']
        
        # Try multiple path variations
        possible_paths = [
            IMAGES_BASE / art_style / f"{painting_name}.jpg",
            IMAGES_BASE / art_style.replace('_', ' ') / f"{painting_name}.jpg",
            IMAGES_BASE / painting_info['art_style'] / f"{painting_name}.jpg",
        ]
        
        img_loaded = False
        for img_path in possible_paths:
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    inner_ax.imshow(img)
                    img_loaded = True
                    loaded_count += 1
                    break
                except Exception as e:
                    continue
        
        if not img_loaded:
            # Placeholder if image not found
            inner_ax.text(0.5, 0.5, f'{emotion}\n(not found)', 
                         ha='center', va='center', fontsize=7,
                         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            inner_ax.set_xlim(0, 1)
            inner_ax.set_ylim(0, 1)
        
        # Add emotion label below image with better spacing
        inner_ax.text(0.5, -0.01, emotion.replace('_', ' ').title(), 
                     ha='center', va='top', fontsize=7, fontweight='bold',
                     transform=inner_ax.transAxes)
    
    # Add section title
    ax.text(0.5, 0.98, '(b) Representative Artwork Examples', 
            ha='center', va='top', fontsize=12, fontweight='bold',
            transform=ax.transAxes)
    
    print(f"✓ Grid complete ({loaded_count}/9 images loaded)")
    
    if loaded_count < 9:
        print(f"⚠️  {9-loaded_count} images not found at {IMAGES_BASE}")


def create_figure1(df):
    """
    Create complete Figure 1 with both subplots
    """
    print("\n🎯 Creating Figure 1: Dataset Overview")
    print("="*60)
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(14, 5))
    
    # GridSpec for layout control
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Plot (a): Distribution
    plot_emotion_distribution(df, ax1)
    
    # Plot (b): Representative paintings
    representative = find_representative_paintings(df)
    plot_painting_grid(representative, ax2, fig)
    
    # Overall title
    fig.suptitle('ArtEmis v2.0 Dataset Overview', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save in multiple formats
    output_png = OUTPUT_DIR / 'fig1_dataset_overview.png'
    output_pdf = OUTPUT_DIR / 'fig1_dataset_overview.pdf'
    
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    
    print("\n✅ Figure 1 saved:")
    print(f"   PNG: {output_png}")
    print(f"   PDF: {output_pdf}")
    
    plt.close()
    
    return representative


def print_dataset_stats(df):
    """
    Print statistics for caption
    NOTE: Reports UNIQUE PAINTINGS, not total annotations
    """
    print("\n📊 Dataset Statistics for Caption:")
    print("="*60)
    
    # Count unique paintings (not annotations!)
    # Use painting_id (art_style + painting) as unique identifier
    total_paintings = df['painting_id'].nunique()
    total_annotations = len(df)
    
    print(f"Unique paintings: {total_paintings:,}")
    print(f"Total annotations: {total_annotations:,}")
    print(f"Avg annotations/painting: {total_annotations/total_paintings:.1f}")
    
    # Emotion categories
    positive = ['amusement', 'awe', 'contentment', 'excitement']
    negative = ['anger', 'disgust', 'fear', 'sadness']
    neutral = ['something else']
    
    # Count unique paintings per emotion category
    pos_count = df[df['emotion'].isin(positive)]['painting_id'].nunique()
    neg_count = df[df['emotion'].isin(negative)]['painting_id'].nunique()
    neu_count = df[df['emotion'].isin(neutral)]['painting_id'].nunique()
    
    print(f"\nEmotion Breakdown (unique paintings):")
    print(f"  Positive: {pos_count:,} ({pos_count/total_paintings*100:.1f}%)")
    print(f"  Negative: {neg_count:,} ({neg_count/total_paintings*100:.1f}%)")
    print(f"  Neutral:  {neu_count:,} ({neu_count/total_paintings*100:.1f}%)")
    
    print(f"\nSplit Breakdown (unique paintings):")
    for split in ['train', 'val', 'test']:
        count = df[df['split'] == split]['painting_id'].nunique()
        print(f"  {split.capitalize()}: {count:,} ({count/total_paintings*100:.1f}%)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Load data
    df = load_dataset()
    
    # Print statistics
    print_dataset_stats(df)
    
    # Generate figure
    representative = create_figure1(df)
    
    print("\n🎉 Done! Figure 1 ready for paper insertion.")
    print("\n📝 LaTeX usage:")
    print("   \\includegraphics[width=\\textwidth]{figures/fig1_dataset_overview.pdf}")
