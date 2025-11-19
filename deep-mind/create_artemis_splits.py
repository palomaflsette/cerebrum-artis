#!/data/paloma/venvs/cerebrum-artis/bin/python
"""
Cria splits train/val/test para ArtEmis seguindo a metodologia oficial.

IMPORTANTE:
- Splits são feitos por ARTWORK (painting), não por annotation
- Evita data leakage (mesmo quadro não aparece em train e test)
- Estratifica por emoção dentro de cada artwork
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Emotion mapping
ARTEMIS_EMOTIONS = [
    'amusement', 'awe', 'contentment', 'excitement',
    'anger', 'disgust', 'fear', 'sadness', 'something else'
]
EMOTION_TO_IDX = {emo: idx for idx, emo in enumerate(ARTEMIS_EMOTIONS)}


def create_artemis_splits(csv_path, 
                          output_path=None,
                          train_size=0.8,
                          val_size=0.1,
                          test_size=0.1,
                          random_seed=42):
    """
    Cria splits seguindo metodologia do ArtEmis:
    1. Agrupa por artwork (art_style + painting)
    2. Divide artworks em train/val/test
    3. Todas as annotations de um artwork vão pro mesmo split
    
    Args:
        csv_path: Path do combined_artemis.csv
        output_path: Onde salvar CSV com splits (None = sobrescreve original)
        train_size, val_size, test_size: Proporções (devem somar 1.0)
        random_seed: Seed para reprodutibilidade
    
    Returns:
        DataFrame com coluna 'split' adicionada
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits devem somar 1.0!"
    
    print(f"📖 Carregando {csv_path}...")
    df = pd.read_csv(csv_path, dtype={'art_style': str, 'painting': str}, low_memory=False)
    
    # Mapear emoções para índices
    df['emotion_idx'] = df['emotion'].map(EMOTION_TO_IDX)
    df = df.dropna(subset=['emotion_idx'])
    
    print(f"✅ Total: {len(df):,} annotations")
    print(f"\n📊 Distribuição de emoções:")
    print(df['emotion'].value_counts())
    
    # Criar unique_id por artwork
    df['artwork_id'] = df['art_style'] + '/' + df['painting']
    
    # Agregar por artwork: pega emoção mais comum de cada quadro
    artwork_groups = df.groupby('artwork_id').agg({
        'emotion': lambda x: x.mode()[0],  # Emoção mais frequente
        'emotion_idx': 'first',
        'art_style': 'first',
        'painting': 'first'
    }).reset_index()
    
    print(f"\n🖼️ Total: {len(artwork_groups):,} artworks únicos")
    
    # Split 1: train vs (val + test) - ESTRATIFICADO POR EMOÇÃO DOMINANTE
    train_artworks, temp_artworks = train_test_split(
        artwork_groups,
        test_size=(val_size + test_size),
        random_state=random_seed,
        stratify=artwork_groups['emotion']
    )
    
    # Split 2: val vs test - ESTRATIFICADO
    val_artworks, test_artworks = train_test_split(
        temp_artworks,
        test_size=(test_size / (val_size + test_size)),
        random_state=random_seed,
        stratify=temp_artworks['emotion']
    )
    
    # Criar dicionário artwork_id -> split
    split_map = {}
    for artwork_id in train_artworks['artwork_id']:
        split_map[artwork_id] = 'train'
    for artwork_id in val_artworks['artwork_id']:
        split_map[artwork_id] = 'val'
    for artwork_id in test_artworks['artwork_id']:
        split_map[artwork_id] = 'test'
    
    # Aplicar splits no DataFrame original
    df['split'] = df['artwork_id'].map(split_map)
    
    # Stats
    print(f"\n✅ Splits criados:")
    print(f"   Train: {len(train_artworks):,} artworks, {len(df[df.split=='train']):,} annotations")
    print(f"   Val:   {len(val_artworks):,} artworks, {len(df[df.split=='val']):,} annotations")
    print(f"   Test:  {len(test_artworks):,} artworks, {len(df[df.split=='test']):,} annotations")
    
    # Verificar estratificação
    print(f"\n📊 Distribuição de emoções por split (%):")
    for split_name in ['train', 'val', 'test']:
        print(f"\n{split_name.upper()}:")
        split_df = df[df['split'] == split_name]
        dist = split_df['emotion'].value_counts(normalize=True).round(3) * 100
        for emo, pct in dist.items():
            print(f"  {emo:15s}: {pct:5.1f}%")
    
    # Verificar que não há overlap de artworks
    train_ids = set(train_artworks['artwork_id'])
    val_ids = set(val_artworks['artwork_id'])
    test_ids = set(test_artworks['artwork_id'])
    
    assert len(train_ids & val_ids) == 0, "ERRO: Overlap train-val!"
    assert len(train_ids & test_ids) == 0, "ERRO: Overlap train-test!"
    assert len(val_ids & test_ids) == 0, "ERRO: Overlap val-test!"
    print("\n✅ Verificação: Nenhum artwork aparece em múltiplos splits!")
    
    # Remover coluna temporária
    df = df.drop(columns=['artwork_id'])
    
    # Salvar
    if output_path is None:
        output_path = csv_path.replace('.csv', '_with_splits.csv')
    
    df.to_csv(output_path, index=False)
    print(f"\n💾 Salvo em: {output_path}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, 
                       default='/home/paloma/cerebrum-artis/artemis/dataset/official_data/combined_artemis.csv',
                       help='Path do CSV do ArtEmis')
    parser.add_argument('--output', type=str, default=None,
                       help='Path de saída (default: adiciona _with_splits.csv)')
    parser.add_argument('--train', type=float, default=0.8, help='Train split')
    parser.add_argument('--val', type=float, default=0.1, help='Val split')
    parser.add_argument('--test', type=float, default=0.1, help='Test split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    create_artemis_splits(
        csv_path=args.csv,
        output_path=args.output,
        train_size=args.train,
        val_size=args.val,
        test_size=args.test,
        random_seed=args.seed
    )
