import os
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video


def get_frames(video_path, num_frames=8):
    video, _, info = read_video(video_path, pts_unit="sec")
    if len(video) == 0:
        return None
    step = max(1, len(video) // num_frames)
    frames = video[::step][:num_frames]  # (T, H, W, C)
    return frames.permute(0, 3, 1, 2).float() / 255.0


class MSRVTTDataset(Dataset):
    def __init__(self, hf_dataset, video_root_dir, num_frames=8, captions_per_video=20):
        self.hf_dataset = hf_dataset
        self.video_root = video_root_dir
        self.num_frames = num_frames
        self.captions_per_video = captions_per_video  # to get ~180k–200k examples

    def filter_valid_videos(self):
        """Filter dataset to only include samples with existing video files"""
        valid_indices = []
        for i in range(len(self.hf_dataset)):
            example = self.hf_dataset[i]
            video_path = os.path.join(self.video_root, example["video"])
            if os.path.exists(video_path):
                valid_indices.append(i)

        self.valid_indices = valid_indices
        self.original_len = len(self.hf_dataset)

    def __len__(self):
        if hasattr(self, "valid_indices"):
            return len(self.valid_indices) * self.captions_per_video
        return len(self.hf_dataset) * self.captions_per_video

    def __getitem__(self, idx):
        # Handle slice objects for indexing like dataset[:50]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self.__getitem__(i) for i in range(start, stop, step or 1)]
        
        if hasattr(self, "valid_indices"):
            sample_idx = self.valid_indices[idx // self.captions_per_video]
        else:
            sample_idx = idx // self.captions_per_video

        # cap_idx = idx % self.captions_per_video

        example = self.hf_dataset[sample_idx]
        video_path = os.path.join(self.video_root, example["video"])

        if not os.path.exists(video_path):
            return None

        frames = get_frames(video_path, num_frames=self.num_frames)

        target_caption = max(example["caption"], key=len)
        return {
            "frames": frames,
            "query": "Describe this short video clip.",
            "target": target_caption,
        }


def collate_fn(batch):
    """Filter out None values from batch"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    # Stack frames and keep queries/targets as lists
    frames = torch.stack([item["frames"] for item in batch])
    queries = [item["query"] for item in batch]
    targets = [item["target"] for item in batch]

    return {"frames": frames, "query": queries, "target": targets}


if __name__ == "__main__":
    # Usage example
    train_ds = MSRVTTDataset(
        hf_dataset=load_dataset("friedrichor/MSR-VTT", "train_9k")["train"],
        video_root_dir="msrvtt_videos/MSRVTT/videos/all/",
        num_frames=8,
    )

    train_ds.filter_valid_videos()
    print(f"Valid samples: {len(train_ds.valid_indices)} / {train_ds.original_len}")
    print(f"Total batches with batch_size=2: {len(train_ds) // 2}")
