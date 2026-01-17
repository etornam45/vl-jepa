import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoVideoProcessor
# from torchcodec.decoders import VideoDecoder
import torch.nn.functional as F
import numpy as np


class TextEmbedder(nn.Module):
    def __init__(self, model_name="google/gemma-3-270m-it"):
        super().__init__()
        temp_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.embed_tokens = temp_model.embed_tokens
        self.vocab_size = temp_model.config.vocab_size
        self.hidden_size = temp_model.config.hidden_size
        del temp_model  # NOTE: delete the model itself and keep the embedding layer
      
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class Predictor(nn.Module):
    def __init__(self, model_name="google/gemma-3-270m-it", num_layers=4):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model.layers = nn.ModuleList(self.model.layers[-num_layers:])
        # self.partial_freeze()

        self.query_token = nn.Parameter(
            torch.randn(1, 1, self.model.config.hidden_size)
        )
        self.sep_token = nn.Parameter(torch.randn(1, 1, self.model.config.hidden_size))

    def partial_freeze(self):
        """Freeze middle layers, train first and last"""
        total_layers = len(self.model.layers)

        for param in self.model.layers.parameters():
            param.requires_grad = False

        for layer_idx in [0, total_layers - 1]:
            for param in self.model.layers[layer_idx].parameters():
                param.requires_grad = True

        # Always train the final norm
        if hasattr(self.model, "norm"):
            for param in self.model.norm.parameters():
                param.requires_grad = True

    def forward(self, query_embeddings, visual_embeddings, pos_embed=None):
        batch_size = query_embeddings.shape[0]

        query_token = self.query_token.expand(batch_size, -1, -1)  # [B, 1, D]
        sep_token = self.sep_token.expand(batch_size, -1, -1)  # [B, 1, D]

        # CAT [QUERY_TOKEN] + query_embeddings + [SEP_TOKEN] + visual_embeddings
        input_embeddings = torch.cat(
            [query_token, query_embeddings, sep_token, visual_embeddings], dim=1
        )  # [B, 3 + N_visual, D]

        if pos_embed is not None:
            input_embeddings = (
                input_embeddings + pos_embed[:, : input_embeddings.shape[1], :]
            )

        outputs = self.model(
            inputs_embeds=input_embeddings,
            attention_mask=torch.ones(
                batch_size,
                input_embeddings.shape[1],
                dtype=torch.bool,
                device=input_embeddings.device,
            ),
            use_cache=False,
        )

        return outputs.last_hidden_state


class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) loss for contrastive learning
    """
    def __init__(self, temperature=0.07, learnable_temperature=True):
        super().__init__()
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        Compute InfoNCE loss between two sets of embeddings
        z_i: [B, D] - anchor embeddings (predicted)
        z_j: [B, D] - positive embeddings (target)
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i_norm = F.normalize(z_i, dim=-1)
        z_j_norm = F.normalize(z_j, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z_i_norm, z_j_norm.T)  # [B, B]

        # Scale by temperature
        temperature = (
            self.temperature
            if isinstance(self.temperature, float)
            else self.temperature.clamp(min=1e-8)
        )
        sim_matrix = sim_matrix / temperature

        # Labels are the diagonal (positive pairs)
        labels = torch.arange(batch_size, device=z_i.device)

        # Compute cross-entropy loss
        loss = self.cross_entropy(sim_matrix, labels)

        # Compute accuracy (optional)
        with torch.no_grad():
            preds = torch.argmax(sim_matrix, dim=1)
            accuracy = (preds == labels).float().mean()

        return loss, accuracy


class VL_JEPA(nn.Module):
    def __init__(
        self,
        v_enc_name="facebook/vjepa2-vitl-fpc64-256",
        predictor_dim=640,
        text_dim=768,
        max_seq_len=512,
        temperature=0.07,
        learnable_temperature=True,
        y_enc_name="google/embeddinggemma-300m",
        num_frames=64,  # V-JEPA expects 64 frames
        frame_size=224,  # Input frame size for V-JEPA
    ):
        super().__init__()
        # Configurations
        self.predictor_dim = predictor_dim
        self.text_dim = text_dim
        self.max_seq_len = max_seq_len
        self.num_frames = num_frames
        self.frame_size = frame_size

        # V-JEPA video encoder
        self.video_processor = AutoVideoProcessor.from_pretrained(v_enc_name)
        self.xv_encoder = AutoModel.from_pretrained(
            v_enc_name,
            trust_remote_code=True,
        )
        self.freeze_model(self.xv_encoder)
        
        # Get visual dimension from V-JEPA config
        self.visual_dim = self.xv_encoder.config.hidden_size
        
        self.visual_proj = nn.Sequential(
            nn.Linear(self.visual_dim, predictor_dim), 
            nn.GELU(), 
            nn.LayerNorm(predictor_dim)
        )

        # Text encoder for targets
        self.y_encoder = AutoModel.from_pretrained(y_enc_name)
        self.freeze_model(self.y_encoder)

        # Query tokenizer and embedder
        self.query_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
        if self.query_tokenizer.pad_token is None:
            self.query_tokenizer.pad_token = self.query_tokenizer.eos_token

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, predictor_dim), 
            nn.GELU(), 
            nn.LayerNorm(predictor_dim)
        )

        self.query_embedder = TextEmbedder()
        self.predictor = Predictor()

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, predictor_dim))

        # Loss function
        self.loss_fn = InfoNCELoss(
            temperature=temperature, 
            learnable_temperature=learnable_temperature
        )

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def embed_query(self, xq):
        """Embed query text using TextEmbedder."""
        tokenized = self.query_tokenizer(
            xq,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.pos_embed.device)
        sq = self.query_embedder(input_ids)
        sq = sq.to(self.pos_embed.device)
        return sq

    def encode_target(self, y):
        """Encode target text using y_encoder."""
        tokenized = self.query_tokenizer(
            y,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.pos_embed.device)
        attention_mask = tokenized.attention_mask.to(self.pos_embed.device)
        with torch.no_grad():
            sy = self.y_encoder(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state
        return sy

    def embed_visual(self, video_frames):
        """
        Embed video frames using V-JEPA encoder.
        video_frames: Tensor of shape [B, N_frames, C, H, W]
        Returns: Tensor of shape [B, N_frames, D_visual]
        """
        B, N, C, H, W = video_frames.shape
        
        # Ensure we have the right number of frames for V-JEPA
        if N != self.num_frames:
            # Resample frames if needed
            indices = torch.linspace(0, N-1, self.num_frames).long()
            video_frames = video_frames[:, indices]
            N = self.num_frames
        
        # Process each video in the batch
        all_visual_features = []
        for i in range(B):
            # Get single video [N, C, H, W]
            single_video = video_frames[i]
            
            # Process with V-JEPA video processor
            processed_video = self.video_processor(
                single_video.numpy() if isinstance(single_video, torch.Tensor) else single_video,
                return_tensors="pt"
            ).to(self.pos_embed.device)
            
            # Get V-JEPA encoder outputs
            with torch.no_grad():
                outputs = self.xv_encoder(**processed_video)
                encoder_outputs = outputs.last_hidden_state
                
                # Get vision features (use predictor outputs for better representations)
                if hasattr(outputs, 'predictor_output'):
                    visual_features = outputs.predictor_output.last_hidden_state
                else:
                    visual_features = encoder_outputs
                
                # Average over spatial dimensions to get per-frame features
                # V-JEPA outputs might be [1, T, N_patches, D] or [1, T*N_patches, D]
                if len(visual_features.shape) == 4:
                    # [1, T, N_patches, D] -> average over patches
                    visual_features = visual_features.mean(dim=2)  # [1, T, D]
                elif len(visual_features.shape) == 3:
                    # [1, T*N_patches, D] -> reshape and average
                    T = self.num_frames
                    N_patches = visual_features.shape[1] // T
                    visual_features = visual_features.view(1, T, N_patches, -1).mean(dim=2)
                
                all_visual_features.append(visual_features.squeeze(0))
        
        # Stack all batch elements
        sv = torch.stack(all_visual_features, dim=0)  # [B, T, D_visual]
        return sv

    def forward(self, xv, xq, y):
        """
        xv: Tensor of shape [B, N_frames, C, H, W]
        xq: List of strings [B]
        y: List of strings [B]
        """
        # Encode visual inputs with V-JEPA
        sv = self.embed_visual(xv)  # [B, N_frames, D_visual]
        sv = self.visual_proj(sv)  # [B, N_frames, D_p]
        
        # Encode query text
        sq = self.embed_query(xq)  # [B, S_q, D_p]

        # Predictor forward pass
        sp = self.predictor(sq, sv, pos_embed=self.pos_embed)  # [B, S_q + N_frames + 2, D_p]

        # Encode target text
        sy = self.encode_target(y)  # [B, S_y, D_t]
        sy = self.text_proj(sy)  # [B, S_y, D_p]

        # Mean pooling for contrastive learning
        sp_pooled = sp.mean(dim=1)  # [B, D_p]
        sy_pooled = sy.mean(dim=1)  # [B, D_p]

        # Compute contrastive loss
        loss, accuracy = self.loss_fn(z_i=sp_pooled, z_j=sy_pooled)
        return loss, accuracy

    def predict(self, xv, xq):
        """Generate predictions for given video and query."""
        sv = self.embed_visual(xv)
        sv = self.visual_proj(sv)
        sq = self.embed_query(xq)
        
        sp = self.predictor(sq, sv, pos_embed=self.pos_embed)
        return sp


if __name__ == "__main__":
    # Test with V-JEPA
    model = VL_JEPA(
        v_enc_name="facebook/vjepa2-vitl-fpc64-256",
        predictor_dim=1024,  # Match V-JEPA hidden size
    )
    
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)
    
    print("Model Architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Percentage trainable: {(trainable / total_params) * 100:.2f}%")
    
    # Test forward pass
    batch_size = 2
    num_frames = 64  # V-JEPA expects 64 frames
    xv = torch.randn(batch_size, num_frames, 3, 224, 224).to(device)
    xq = ["What is happening in this video?"] * batch_size
    y = ["A person is performing an action."] * batch_size
    
    print("Input shapes:")
    print(f"Video: {xv.shape}")
    print(f"Query: {len(xq)} samples")
    print(f"Target: {len(y)} samples")
    
    loss, accuracy = model(xv, xq, y)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Accuracy: {accuracy.item():.4f}")