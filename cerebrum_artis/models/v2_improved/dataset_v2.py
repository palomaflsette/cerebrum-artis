"""
Dataset V2 - With Data Augmentation

Improvements:
- ✅ Heavy augmentation for training
- ✅ Optimized for art images
- ✅ Same interface as v1
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from transformers import RobertaTokenizer
import torchvision.transforms as transforms

# ArtEmis emotion mapping
EMOTION_TO_IDX = {
    'amusement': 0,
    'awe': 1,
    'contentment': 2,
    'excitement': 3,
    'anger': 4,
    'disgust': 5,
    'fear': 6,
    'sadness': 7,
    'something else': 8
}

IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}


class ArtEmisEmotionDatasetV2(Dataset):
    """
    Dataset v2 com data augmentation.
    
    Args:
        csv_path: Caminho pro CSV do ArtEmis
        img_dir: Diretório das imagens WikiArt
        tokenizer: RoBERTa tokenizer
        split: 'train', 'val', 'test'
        augment: Se True, aplica augmentation (só pra train!)
    """
    
    def __init__(self, csv_path, img_dir, tokenizer, split='train', augment=True):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.split = split
        self.augment = augment and (split == 'train')  # Só augment em train!
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Filter by split
        if 'split' in df.columns:
            df = df[df['split'] == split].reset_index(drop=True)
        
        # Map emotions
        df['emotion_idx'] = df['emotion'].map(EMOTION_TO_IDX)
        df = df.dropna(subset=['emotion_idx'])
        df['emotion_idx'] = df['emotion_idx'].astype(int)
        
        self.data = df
        print(f"✅ {split.upper()} dataset: {len(self.data)} samples")
        
        # Transforms
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup transformações (COM augmentation pra train!)"""
        
        # Normalização ImageNet (sempre)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.augment:
            # TRAIN: Augmentation pesada
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop
                transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontal
                transforms.ColorJitter(
                    brightness=0.3,  # ±30% brilho
                    contrast=0.3,    # ±30% contraste
                    saturation=0.2,  # ±20% saturação
                    hue=0.1          # ±10% matiz
                ),
                transforms.RandomRotation(15),  # ±15° rotação
                transforms.ToTensor(),
                normalize
            ])
            print(f"🔥 {self.split.upper()}: Data augmentation ENABLED")
        else:
            # VAL/TEST: Sem augmentation
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
            print(f"📊 {self.split.upper()}: No augmentation (eval mode)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        painting_path = row['painting']
        if painting_path.startswith('wikiart/'):
            painting_path = painting_path.replace('wikiart/', '')
        
        img_path = os.path.join(self.img_dir, painting_path)
        
        # Multi-strategy loading (unicode fix)
        image = None
        for strategy in ['utf-8', 'latin-1', 'ignore']:
            try:
                if strategy == 'ignore':
                    img_path_safe = img_path.encode('utf-8', 'ignore').decode('utf-8')
                    image = Image.open(img_path_safe).convert('RGB')
                else:
                    image = Image.open(img_path).convert('RGB')
                break
            except:
                continue
        
        if image is None:
            # Fallback: imagem preta
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        image = self.transform(image)
        
        # Tokenize text
        utterance = str(row['utterance'])
        encoded = self.tokenizer(
            utterance,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(row['emotion_idx'], dtype=torch.long)
        }


def create_dataloaders_v2(csv_path, img_dir, batch_size=64, num_workers=4,
                          val_split=0.1, test_split=0.1, augment=True):
    """
    Cria dataloaders v2 com augmentation.
    
    Args:
        csv_path: CSV do ArtEmis
        img_dir: Diretório de imagens
        batch_size: Tamanho do batch
        num_workers: Workers do DataLoader
        val_split: Fração de validação
        test_split: Fração de teste
        augment: Ativar augmentation no train
    
    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Create datasets
    train_dataset = ArtEmisEmotionDatasetV2(
        csv_path, img_dir, tokenizer, split='train', augment=augment
    )
    val_dataset = ArtEmisEmotionDatasetV2(
        csv_path, img_dir, tokenizer, split='val', augment=False
    )
    test_dataset = ArtEmisEmotionDatasetV2(
        csv_path, img_dir, tokenizer, split='test', augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, tokenizer
