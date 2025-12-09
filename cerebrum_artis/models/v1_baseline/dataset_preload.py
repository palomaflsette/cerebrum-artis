"""
ArtEmis Dataset com PRÉ-CARREGAMENTO EM RAM.
Carrega todas as imagens na memória antes do treino.
"""

import os
import pandas as pd
import numpy as np
import unicodedata
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import RobertaTokenizer
from tqdm import tqdm


ARTEMIS_EMOTIONS = [
    'amusement', 'awe', 'contentment', 'excitement',
    'anger', 'disgust', 'fear', 'sadness', 'something else'
]

EMOTION_TO_IDX = {emo: idx for idx, emo in enumerate(ARTEMIS_EMOTIONS)}


class ArtEmisPreloadedDataset(Dataset):
    """
    Dataset que CARREGA TODAS AS IMAGENS EM RAM antes do treino.
    Usa ~30-40GB RAM mas elimina 100% do I/O bottleneck.
    """
    
    def __init__(self, 
                 csv_path,
                 img_dir,
                 tokenizer,
                 split='train',
                 max_length=128,
                 img_size=224):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Load CSV
        df = pd.read_csv(csv_path, dtype={'art_style': str, 'painting': str}, low_memory=False)
        
        if 'split' in df.columns:
            df = df[df['split'] == split].reset_index(drop=True)
        
        required = ['art_style', 'painting', 'utterance', 'emotion']
        df = df.dropna(subset=required)
        df['emotion_idx'] = df['emotion'].map(EMOTION_TO_IDX)
        df = df.dropna(subset=['emotion_idx'])
        df['emotion_idx'] = df['emotion_idx'].astype(int)
        
        self.data = df.reset_index(drop=True)
        
        # Transforms
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
        
        # 🚀 PRÉ-CARREGA TODAS AS IMAGENS EM RAM
        print(f"🔥 PRÉ-CARREGANDO {len(self.data)} imagens em RAM ({split})...")
        self.images_cache = {}
        failed = 0
        
        for idx in tqdm(range(len(self.data)), desc=f"Loading {split}"):
            row = self.data.iloc[idx]
            painting_name = row['painting']
            art_style = row['art_style']
            
            # Tenta carregar imagem
            img_path = os.path.join(img_dir, art_style, f"{painting_name}.jpg")
            
            # Unicode normalization
            if not os.path.exists(img_path):
                normalized = unicodedata.normalize('NFC', painting_name)
                img_path = os.path.join(img_dir, art_style, f"{normalized}.jpg")
            
            # Fuzzy match
            if not os.path.exists(img_path):
                try:
                    style_dir = os.path.join(img_dir, art_style)
                    if os.path.exists(style_dir):
                        files = [f for f in os.listdir(style_dir) if f.endswith('.jpg')]
                        clean_name = ''.join(c for c in painting_name if c.isalnum() or c in '-_')
                        for f in files:
                            clean_f = ''.join(c for c in f[:-4] if c.isalnum() or c in '-_')
                            if clean_name.lower() in clean_f.lower():
                                img_path = os.path.join(style_dir, f)
                                break
                except:
                    pass
            
            # Carrega imagem em RAM (PIL Image object)
            try:
                img = Image.open(img_path).convert('RGB')
                self.images_cache[idx] = img
            except:
                self.images_cache[idx] = None  # Fallback depois
                failed += 1
        
        print(f"✅ {len(self.images_cache) - failed}/{len(self.data)} imagens carregadas em RAM")
        if failed > 0:
            print(f"⚠️ {failed} imagens com erro (usarão fallback preto)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Pega imagem da RAM (JÁ CARREGADA!)
        image = self.images_cache.get(idx)
        
        if image is not None:
            image = self.transform(image)
        else:
            # Fallback: imagem preta
            image = torch.zeros(3, 224, 224)
        
        # Tokenize text
        text = str(row['utterance'])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        label = row['emotion_idx']
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'painting': row['painting'],
            'emotion': row['emotion']
        }


def create_dataloaders_preloaded(csv_path,
                                  img_dir,
                                  batch_size=32,
                                  num_workers=4,
                                  val_split=0.1,
                                  test_split=0.1):
    """
    DataLoaders com dataset PRÉ-CARREGADO em RAM.
    
    ATENÇÃO: Usa ~30-40GB RAM total!
    """
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    df = pd.read_csv(csv_path, dtype={'art_style': str, 'painting': str}, low_memory=False)
    
    if 'split' in df.columns:
        # CSV já tem splits
        train_dataset = ArtEmisPreloadedDataset(csv_path, img_dir, tokenizer, split='train')
        val_dataset = ArtEmisPreloadedDataset(csv_path, img_dir, tokenizer, split='val')
        test_dataset = ArtEmisPreloadedDataset(csv_path, img_dir, tokenizer, split='test')
    else:
        # Cria splits
        from sklearn.model_selection import train_test_split
        
        train_val, test = train_test_split(df, test_size=test_split, random_state=42, stratify=df['emotion'])
        train, val = train_test_split(train_val, test_size=val_split/(1-test_split), 
                                      random_state=42, stratify=train_val['emotion'])
        
        # Salva temporariamente
        import tempfile
        tmpdir = tempfile.mkdtemp()
        
        train['split'] = 'train'
        val['split'] = 'val'
        test['split'] = 'test'
        
        tmp_csv = os.path.join(tmpdir, 'temp_splits.csv')
        pd.concat([train, val, test]).to_csv(tmp_csv, index=False)
        
        train_dataset = ArtEmisPreloadedDataset(tmp_csv, img_dir, tokenizer, split='train')
        val_dataset = ArtEmisPreloadedDataset(tmp_csv, img_dir, tokenizer, split='val')
        test_dataset = ArtEmisPreloadedDataset(tmp_csv, img_dir, tokenizer, split='test')
    
    # DataLoaders - PODE USAR 0 WORKERS (dados já em RAM!)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Sem workers! Dados em RAM
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, tokenizer


if __name__ == '__main__':
    # Test
    csv = '/home/paloma/cerebrum-artis/artemis/dataset/official_data/combined_artemis.csv'
    img_dir = '/data/paloma/data/paintings/wikiart/'
    
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders_preloaded(
        csv, img_dir, batch_size=64, val_split=0.05, test_split=0.05
    )
    
    print(f"\n✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    print(f"✅ Test batches: {len(test_loader)}")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\n✅ Batch shapes:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Labels: {batch['label'].shape}")
    print(f"  Emotion: {batch['emotion'][0]}")
