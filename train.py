from model import VL_JEPA
import torch
from dataset import MSRVTTDataset, collate_fn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils import logging
logging.set_verbosity_error()


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



if __name__ == "__main__":

  for epoch in range(50):
      model.train()
      total_loss = 0.0
      num_batches = 0
      
      pbar = tqdm(loader, desc=f"Epoch {epoch+1}/50")
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