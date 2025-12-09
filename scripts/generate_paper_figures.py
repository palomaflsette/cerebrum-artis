import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Configuração de estilo para paper
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'

OUTPUT_DIR = '/home/paloma/cerebrum-artis/outputs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_emotion_distribution():
    print("Gerando Figure 2: Distribuição de Emoções...")
    csv_path = '/home/paloma/cerebrum-artis/data/artemis/dataset/official_data/combined_artemis_with_splits.csv'
    
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo {csv_path} não encontrado.")
        return

    df = pd.read_csv(csv_path)
    
    # Contagem e ordenação
    emotion_counts = df['emotion'].value_counts()
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette="viridis")
    
    plt.title('Distribution of Emotions in ArtEmis Dataset', fontsize=16, pad=20)
    plt.xlabel('Emotion Category', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.xticks(rotation=45)
    
    # Adicionar valores no topo das barras
    for i, v in enumerate(emotion_counts.values):
        ax.text(i, v + 500, f'{v:,}', ha='center', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_dataset_distribution.png'), dpi=300)
    print("Salvo em fig2_dataset_distribution.png")
    plt.close()

def trimf(x, abc):
    """Triangular membership function generator"""
    a, b, c = abc
    y = np.zeros_like(x)
    
    # Left side
    if a != b:
        idx = np.logical_and(a <= x, x < b)
        y[idx] = (x[idx] - a) / (b - a)
    
    # Right side
    if b != c:
        idx = np.logical_and(b <= x, x <= c)
        y[idx] = (c - x[idx]) / (c - b)
        
    # Peak
    y[x == b] = 1.0
    return y

def plot_fuzzy_variables():
    print("Gerando Figure 3: Variáveis Fuzzy...")
    
    universe = np.arange(0, 1.01, 0.001)
    
    # Definição das variáveis (baseado em variables.py)
    # Usando Brightness como exemplo representativo
    terms = {
        'Very Low': [0.0, 0.0, 0.25],
        'Low':      [0.0, 0.25, 0.5],
        'Medium':   [0.25, 0.5, 0.75],
        'High':     [0.5, 0.75, 1.0],
        'Very High':[0.75, 1.0, 1.0]
    }
    
    plt.figure(figsize=(10, 5))
    
    colors = ['#2c3e50', '#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
    
    for (label, params), color in zip(terms.items(), colors):
        y = trimf(universe, params)
        plt.plot(universe, y, label=label, linewidth=2.5, color=color)
        plt.fill_between(universe, y, alpha=0.1, color=color)
        
    plt.title('Fuzzy Membership Functions (Example: Brightness/Warmth)', fontsize=16, pad=20)
    plt.xlabel('Normalized Feature Value [0, 1]', fontsize=14)
    plt.ylabel('Membership Degree ($\mu$)', fontsize=14)
    plt.legend(loc='center right', frameon=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_fuzzy_membership.png'), dpi=300)
    print("Salvo em fig3_fuzzy_membership.png")
    plt.close()

if __name__ == "__main__":
    plot_emotion_distribution()
    plot_fuzzy_variables()
