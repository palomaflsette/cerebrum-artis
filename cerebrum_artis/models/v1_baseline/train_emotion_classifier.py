"""
Script de treino para classificador multimodal de emoções.

Usage:
    python train_emotion_classifier.py --epochs 15 --batch-size 64 --gpu 0
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from multimodal_classifier import MultimodalEmotionClassifier
from dataset import create_dataloaders, ARTEMIS_EMOTIONS
from dataset_preload import create_dataloaders_preloaded  # RAM version


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None, accumulation_steps=1):
    """Treina por uma época com suporte a mixed precision e gradient accumulation."""
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [TRAIN]")
    optimizer.zero_grad()  # Zero inicial
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        image = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Mixed precision forward
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(image, input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps  # Scale loss
        else:
            logits = model(image, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps
        
        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (só a cada N batches)
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Metrics
        running_loss += loss.item() * accumulation_steps
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    """Valida o modelo."""
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [VAL]")
    for batch in pbar:
        image = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        logits = model(image, input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Metrics
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


def save_checkpoint(model, optimizer, epoch, val_acc, save_path, is_best=False):
    """Salva checkpoint."""
    # Handle DataParallel
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'num_emotions': model_to_save.num_emotions,
        'dropout': 0.3
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)
        print(f"💾 Best model saved: {best_path} (acc={val_acc:.4f})")


def main(args):
    # Setup devices
    if torch.cuda.is_available():
        if args.multi_gpu and torch.cuda.device_count() > 1:
            device = torch.device('cuda:0')
            gpu_count = torch.cuda.device_count()
            print(f"🖥️  Multi-GPU mode: {gpu_count} GPUs detected")
            for i in range(gpu_count):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            device = torch.device(f'cuda:{args.gpu}')
            print(f"🖥️  Single GPU mode: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print(f"🖥️  CPU mode")
    
    # Create save directory
    if args.resume:
        # Use existing checkpoint directory
        save_dir = os.path.dirname(args.resume)
        print(f"📂 Resuming training in existing directory: {save_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.save_dir, f"multimodal_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"📂 Created new training directory: {save_dir}")
    
    # Save args
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(save_dir, 'tb_logs'))
    
    # Load data
    print("📦 Loading dataset...")
    
    # MODO PRELOAD: Carrega TUDO em RAM (elimina 100% I/O bottleneck)
    if args.preload_ram:
        print("🔥 MODO PRELOAD ATIVADO: Carregando dataset completo em RAM...")
        train_loader, val_loader, test_loader, tokenizer = create_dataloaders_preloaded(
            csv_path=args.csv_path,
            img_dir=args.img_dir,
            batch_size=args.batch_size,
            num_workers=0,  # Sem workers, dados já em RAM
            val_split=args.val_split,
            test_split=args.test_split
        )
    else:
        # Modo normal (lê do disco a cada época)
        train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
            csv_path=args.csv_path,
            img_dir=args.img_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            test_split=args.test_split
        )
    
    # Create model
    print("🧠 Creating model...")
    model = MultimodalEmotionClassifier(
        num_emotions=9,
        freeze_image_encoder=args.freeze_image,
        freeze_text_encoder=args.freeze_text,
        dropout=args.dropout
    )
    model.to(device)
    
    # Enable DataParallel for multi-GPU
    if args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"🔥 Wrapping model with DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        # Adjust batch size
        effective_batch_size = args.batch_size * torch.cuda.device_count()
        print(f"📊 Effective batch size: {effective_batch_size} ({args.batch_size} per GPU)")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Total params: {total_params/1e6:.2f}M")
    print(f"✅ Trainable params: {trainable_params/1e6:.2f}M")
    
    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # Load checkpoint if resuming
    start_epoch = 1
    if args.resume:
        print(f"📂 Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load model state
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get start epoch
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        
        print(f"✅ Resumed from epoch {checkpoint['epoch']}")
        print(f"✅ Best val acc so far: {best_val_acc:.4f}")
        print(f"🚀 Continuing from epoch {start_epoch}")
    else:
        best_val_acc = 0.0
    
    # Mixed precision scaler (FP16 para velocidade)
    use_fp16 = args.mixed_precision or args.fp16
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    if use_fp16:
        print("⚡ Mixed precision (FP16) ENABLED")
    
    # Training loop
    print("\n🚀 Starting training...\n")
    epochs_no_improve = 0
    early_stop_patience = args.early_stop_patience
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, accumulation_steps=args.accumulation_steps
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Logging
        print(f"\n📊 Epoch {epoch}/{args.epochs}")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint & early stopping check
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            epochs_no_improve = 0
            print(f"   🎯 New best val acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement for {epochs_no_improve}/{early_stop_patience} epochs")
        
        if epoch % args.save_every == 0 or is_best:
            save_path = os.path.join(save_dir, f'checkpoint_epoch{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, val_acc, save_path, is_best)
        
        # Early stopping
        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            print(f"\n🛑 Early stopping triggered! No improvement for {early_stop_patience} epochs.")
            print(f"   Best val acc: {best_val_acc:.4f}")
            break
    
    # Final evaluation
    print("\n🎯 Final Test Evaluation...")
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device, 'TEST'
    )
    
    print(f"\n📊 Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")
    
    # Classification report
    report = classification_report(
        test_labels, test_preds,
        target_names=ARTEMIS_EMOTIONS,
        digits=4
    )
    print("\n" + report)
    
    # Save report
    with open(os.path.join(save_dir, 'test_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), cm)
    
    print(f"\n✅ Training complete! Models saved in: {save_dir}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multimodal Emotion Classifier")
    
    # Data
    parser.add_argument('--csv-path', type=str,
                       default='/home/paloma/cerebrum-artis/data/artemis/dataset/official_data/combined_artemis_with_splits.csv',
                       help='Path to ArtEmis CSV (with splits)')
    parser.add_argument('--img-dir', type=str,
                       default='/data/paloma/data/paintings/wikiart',
                       help='Path to WikiArt images')
    
    # Model
    parser.add_argument('--freeze-image', action='store_true', default=True,
                       help='Freeze ResNet50 encoder')
    parser.add_argument('--freeze-text', action='store_true', default=False,
                       help='Freeze RoBERTa encoder')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of epochs')
    parser.add_argument('--early-stop-patience', type=int, default=5,
                       help='Early stopping patience (0 = disabled)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers')
    
    # Splits
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split fraction')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Test split fraction')
    
    # Saving
    parser.add_argument('--save-dir', type=str,
                       default='/data/paloma/cerebrum-artis/checkpoints/v1_baseline',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (for single GPU mode)')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                       help='Use all available GPUs with DataParallel')
    parser.add_argument('--preload-ram', action='store_true', default=False,
                       help='Preload ALL images to RAM (30-40GB RAM, eliminates I/O bottleneck)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                       help='Use FP16 mixed precision training (faster, ~2x speedup)')
    parser.add_argument('--fp16', action='store_true', default=False,
                       help='Alias for --mixed-precision')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps (simulates larger batch)')
    
    args = parser.parse_args()
    main(args)
