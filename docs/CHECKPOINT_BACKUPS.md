# Checkpoint Backup Management

## Backup Location
**Primary Backups:** `/data/paloma/checkpoint-backups/`

## Current Backups

### V3.1 Integrated (Former V4.1) - December 9, 2025
- **Location:** `/data/paloma/checkpoint-backups/v3_1_integrated_backup_20251209/`
- **Size:** 6.1GB (4 checkpoints)
- **Reason:** Backup before V3.1 retraining
- **Files:**
  - `checkpoint_best.pt` - Best validation performance (used in demo notebook)
  - `checkpoint_epoch8_last.pt`
  - `checkpoint_epoch9_last.pt`
  - `checkpoint_epoch10_last.pt`

## Active Training Checkpoints
**Location:** `/data/paloma/deep-mind-checkpoints/`

### Models Being Retrained (Will Overwrite)
- ✅ **v2_fuzzy_features** - V2 improved retraining (safe to overwrite)
- ✅ **v3_1_integrated** - V3.1 retraining (BACKUP CREATED)

### Safe Checkpoints (Not Being Retrained)
- **v3_adaptive_gating** - V3 adaptive gating (no current retraining)

## Backup Strategy
1. **Before retraining:** Create timestamped backup in `/data/paloma/checkpoint-backups/`
2. **Active checkpoints:** Keep in `/data/paloma/deep-mind-checkpoints/` for training scripts
3. **Old/experimental:** Archive in `/data/paloma/checkpoint-backups/archive/` if needed

## Recovery
To restore a backup:
```bash
cp /data/paloma/checkpoint-backups/v3_1_integrated_backup_20251209/*.pt /data/paloma/deep-mind-checkpoints/v3_1_integrated/
```

## Notes
- Total backup space used: ~6.1GB
- Original checkpoints remain in `/data/paloma/deep-mind-checkpoints/` until overwritten
- Training logs in `/data/paloma/training-logs/`
