"""
ArtEmis Dataset loader para treino de classificador multimodal.

Carrega:
- Imagens do WikiArt
- Textos explicativos (utterances)
- Labels de emoção (0-8)
"""

import os
import pandas as pd
import numpy as np
import unicodedata
import difflib
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import RobertaTokenizer
import random

# Fix DecompressionBombWarning for large art images
Image.MAX_IMAGE_PIXELS = None


# ArtEmis emotion labels (ordem importa!)
ARTEMIS_EMOTIONS = [
    'amusement',      # 0
    'awe',            # 1
    'contentment',    # 2
    'excitement',     # 3
    'anger',          # 4
    'disgust',        # 5
    'fear',           # 6
    'sadness',        # 7
    'something else'  # 8
]

EMOTION_TO_IDX = {emo: idx for idx, emo in enumerate(ARTEMIS_EMOTIONS)}


class ArtEmisEmotionDataset(Dataset):
    """
    Dataset para classificação de emoções no ArtEmis.
    """
    
    def __init__(self, 
                 csv_path,
                 img_dir,
                 tokenizer,
                 split='train',
                 max_length=128,
                 img_size=224):
        """
        Args:
            csv_path: Caminho para CSV do ArtEmis (combinado ou preprocessado)
            img_dir: Diretório raiz do WikiArt (style/painting.jpg)
            tokenizer: RobertaTokenizer
            split: 'train', 'val', ou 'test'
            max_length: Comprimento máximo do texto
            img_size: Tamanho da imagem (quadrada)
        """
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Load data with proper dtype handling
        df = pd.read_csv(csv_path, dtype={'art_style': str, 'painting': str}, low_memory=False)
        
        # Filter by split se coluna existe
        if 'split' in df.columns:
            df = df[df['split'] == split].reset_index(drop=True)
        
        # Required columns: art_style, painting, utterance, emotion
        required = ['art_style', 'painting', 'utterance', 'emotion']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        
        # Filter rows with valid data
        df = df.dropna(subset=required)
        
        # Map emotion labels to indices
        df['emotion_idx'] = df['emotion'].map(EMOTION_TO_IDX)
        df = df.dropna(subset=['emotion_idx'])  # Remove unknown emotions
        df['emotion_idx'] = df['emotion_idx'].astype(int)
        
        self.data = df.reset_index(drop=True)
        
        # Image transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        print(f"✅ {split.upper()} dataset: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def _fix_encoding(self, name):
        """Tenta corrigir problemas comuns de encoding no dataset."""
        try:
            # Fix comum para mojibake (utf-8 interpretado como latin1)
            return name.encode('latin1').decode('utf-8')
        except:
            return name

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image - try multiple encoding strategies
        painting_name = row['painting']
        art_style = row['art_style']
        
        # Strategy 1: Direct path
        img_path = os.path.join(self.img_dir, art_style, f"{painting_name}.jpg")
        
        # Strategy 2: Fix encoding (mojibake)
        if not os.path.exists(img_path):
            fixed_name = self._fix_encoding(painting_name)
            img_path = os.path.join(self.img_dir, art_style, f"{fixed_name}.jpg")

        # Strategy 3: Normalize unicode (NFC)
        if not os.path.exists(img_path):
            normalized = unicodedata.normalize('NFC', painting_name)
            img_path = os.path.join(self.img_dir, art_style, f"{normalized}.jpg")
        
        # Strategy 4: Fuzzy match in directory (Last resort)
        if not os.path.exists(img_path):
            try:
                style_dir = os.path.join(self.img_dir, art_style)
                if os.path.exists(style_dir):
                    files = os.listdir(style_dir)
                    # Tenta encontrar o arquivo mais parecido
                    matches = difflib.get_close_matches(f"{painting_name}.jpg", files, n=1, cutoff=0.5)
                    if matches:
                        img_path = os.path.join(style_dir, matches[0])
            except Exception:
                pass
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # Fallback: pick a random valid image instead of returning zeros/crashing
            # This preserves batch size and avoids teaching the model "black image = emotion X"
            print(f"⚠️ Error loading {painting_name}: {str(e)}. Picking random replacement.")
            rand_idx = random.randint(0, len(self.data) - 1)
            return self.__getitem__(rand_idx)
        
        # Tokenize text
        text = str(row['utterance'])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Emotion label
        label = row['emotion_idx']
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'painting': row['painting'],
            'emotion': row['emotion']
        }


def create_dataloaders(csv_path,
                       img_dir,
                       batch_size=32,
                       num_workers=4,
                       val_split=0.1,
                       test_split=0.1):
    """
    Cria DataLoaders para train/val/test.
    
    Args:
        csv_path: CSV do ArtEmis
        img_dir: Diretório WikiArt
        batch_size: Batch size
        num_workers: Workers para DataLoader
        val_split: Fração para validação
        test_split: Fração para teste
    
    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Load CSV
    df = pd.read_csv(csv_path, dtype={'art_style': str, 'painting': str}, low_memory=False)
    
    # Map emotions to indices first (for stratification)
    if 'emotion_idx' not in df.columns:
        df['emotion_idx'] = df['emotion'].map(EMOTION_TO_IDX)
        df = df.dropna(subset=['emotion_idx'])
    
    if 'split' in df.columns:
        # CSV já tem splits
        train_dataset = ArtEmisEmotionDataset(csv_path, img_dir, tokenizer, split='train')
        val_dataset = ArtEmisEmotionDataset(csv_path, img_dir, tokenizer, split='val')
        test_dataset = ArtEmisEmotionDataset(csv_path, img_dir, tokenizer, split='test')
    else:
        # Manual split COM ESTRATIFICAÇÃO
        from sklearn.model_selection import train_test_split
        
        print(f"📊 Distribuição original das emoções:")
        print(df['emotion'].value_counts())
        
        # Split 1: train vs (val + test) - ESTRATIFICADO
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_split + test_split), 
            random_state=42,
            stratify=df['emotion']  # ⭐ ESTRATIFICAÇÃO!
        )
        
        # Split 2: val vs test - ESTRATIFICADO
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(test_split / (val_split + test_split)),
            random_state=42,
            stratify=temp_df['emotion']  # ⭐ ESTRATIFICAÇÃO!
        )
        
        # Criar coluna split
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        # Concatena e salva
        df_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
        temp_csv = csv_path.replace('.csv', '_with_splits.csv')
        df_splits.to_csv(temp_csv, index=False)
        
        print(f"\n✅ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        print(f"📊 Distribuição estratificada em cada split:")
        for split_name in ['train', 'val', 'test']:
            split_df = df_splits[df_splits['split'] == split_name]
            print(f"\n{split_name.upper()}:")
            print(split_df['emotion'].value_counts(normalize=True).round(3))
        
        train_dataset = ArtEmisEmotionDataset(temp_csv, img_dir, tokenizer, split='train')
        val_dataset = ArtEmisEmotionDataset(temp_csv, img_dir, tokenizer, split='val')
        test_dataset = ArtEmisEmotionDataset(temp_csv, img_dir, tokenizer, split='test')
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Validation can use bigger batches
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader, tokenizer


if __name__ == "__main__":
    # Test dataset
    print("🧪 Testing ArtEmisEmotionDataset...")
    
    csv_path = "/home/paloma/cerebrum-artis/artemis/dataset/official_data/combined_artemis.csv"
    img_dir = "/data/paloma/data/paintings/wikiart"
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    dataset = ArtEmisEmotionDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        tokenizer=tokenizer,
        split='train'
    )
    
    print(f"✅ Dataset size: {len(dataset)}")
    
    # Test sample
    sample = dataset[0]
    print(f"✅ Sample keys: {sample.keys()}")
    print(f"✅ Image shape: {sample['image'].shape}")
    print(f"✅ Input IDs shape: {sample['input_ids'].shape}")
    print(f"✅ Label: {sample['label']} ({sample['emotion']})")
    print(f"✅ Painting: {sample['painting']}")
