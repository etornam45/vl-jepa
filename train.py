from model import VL_JEPA
import torch
from dataset import MSRVTTDataset, collate_fn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils import logging
logging.set_verbosity_error()
import os
from pathlib import Path


train_ds = MSRVTTDataset(
    hf_dataset = load_dataset('friedrichor/MSR-VTT', 'train_9k')['train'],
    video_root_dir = "msrvtt_videos/MSRVTT/videos/all/",
    num_frames=8
)

train_ds.filter_valid_videos()
print(f"Valid samples: {len(train_ds.valid_indices)} / {train_ds.original_len}")

device = torch.device("mps" if torch.mps.is_available() else "cpu")


model = VL_JEPA()
model.to(device)

# Print trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

loader = DataLoader(train_ds, batch_size=32, collate_fn=collate_fn)


CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.pth"
NUM_EPOCHS = 50


def load_checkpoint_if_exists(model, optimizer, device):
    start_epoch = 0
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt.get('model_state', {}))
        opt_state = ckpt.get('optimizer_state', None)
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except Exception:
                # optimizer state may be incompatible across devices/backends; ignore if fails
                print("Warning: failed to fully load optimizer state; continuing without it.")
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from checkpoint; starting at epoch {start_epoch}")
    return start_epoch


def save_checkpoint(model, optimizer, epoch):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, CHECKPOINT_PATH)


if __name__ == "__main__":

  start_epoch = load_checkpoint_if_exists(model, optimizer, device)

  for epoch in range(start_epoch, NUM_EPOCHS):
      model.train()
      total_loss = 0.0
      num_batches = 0
      
      pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
      for batch in pbar:
          if batch is None:
              continue
          
          frames = batch['frames'].to(device)  # [B, N_frames, C, H, W]
          queries = batch['query']              # List of strings [B]
          targets = batch['target']             # List of strings [B]

          optimizer.zero_grad()
          loss, accuracy = model(frames, queries, targets)
          loss.backward()
          optimizer.step()
          
          total_loss += loss.item()
          num_batches += 1
          
          pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})

      avg_loss = total_loss / num_batches if num_batches > 0 else 0
      print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

      # save checkpoint after each epoch (minimal implementation)
      save_checkpoint(model, optimizer, epoch+1)
      print(f"Saved checkpoint to {CHECKPOINT_PATH}")