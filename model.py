import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


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
        self.partial_freeze()

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
    NOTE: I did not implement this function by myself it may be wrong.
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

        Returns: scalar loss
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i_norm = F.normalize(z_i, dim=-1)
        z_j_norm = F.normalize(z_j, dim=-1)

        # Compute similarity matrix
        # [B, D] @ [D, B] = [B, B]
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
        v_enc_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        visual_dim=384,
        predictor_dim=640,
        text_dim=768,
        max_seq_len=512,
        temperature=0.07,
        learnable_temperature=True,
        y_enc_name="google/embeddinggemma-300m",
    ):
        super().__init__()
        # Configurations
        self.predictor_dim = predictor_dim
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.max_seq_len = max_seq_len

        self.xv_encoder = AutoModel.from_pretrained(v_enc_name)
        self.freeze_model(self.xv_encoder)

        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, predictor_dim), nn.GELU(), nn.LayerNorm(predictor_dim)
        )

        self.y_encoder = AutoModel.from_pretrained(y_enc_name)
        self.freeze_model(self.y_encoder)

        self.query_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
        if self.query_tokenizer.pad_token is None:
            self.query_tokenizer.pad_token = (
                self.query_tokenizer.eos_token
            )  # ADD PAD TOKEN

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, predictor_dim), nn.GELU(), nn.LayerNorm(predictor_dim)
        )

        self.query_embedder = TextEmbedder()

        self.predictor = Predictor()

        # Positional embeddings and temperature for contrastive learning
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, predictor_dim))

        self.loss_fn = InfoNCELoss(
            temperature=temperature, learnable_temperature=learnable_temperature
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
        # video_frames: Tensor of shape [B, N_frames, C, H, W]
        B, N, C, H, W = video_frames.shape
        video_frames = video_frames.view(B * N, C, H, W)
        with torch.no_grad():
            sv = self.xv_encoder(video_frames).last_hidden_state  # [B*N, S_v, D_v]
            sv = sv.mean(dim=1)  # Global average pooling over sequence dimension
            sv = sv.view(B, N, -1)  # [B, N, D_v]
        return sv

    def forward(self, xv, xq, y):
        """
        xv: Tensor of shape [B, N_frames, C, H, W]
        xq: List of strings [B]
        y: List of strings [B]
        """

        sv = self.embed_visual(xv)  # [B, N, D_v]
        sv = self.visual_proj(sv)  # [B, N, D_p]
        sq = self.embed_query(xq)  # [B, S_q, D_p]

        sp = self.predictor(sq, sv, pos_embed=self.pos_embed)  # [B, S_q + N + 2, D_p]

        sy = self.encode_target(y)  # [B, S_y, D_t]
        sy = self.text_proj(sy)  # [B, S_y, D_p]

        # Use mean pooling for both
        sp_pooled = sp.mean(dim=1)  # [B, D_p]
        sy_pooled = sy.mean(dim=1)  # [B, D_p]

        loss, accuracy = self.loss_fn(z_i=sp_pooled, z_j=sy_pooled)
        return loss, accuracy

    def predict(self, xv, xq):
        sv = self.embed_visual(xv)  # [B, N, D_v]
        sv = self.visual_proj(sv)  # [B, N, D_p]
        sq = self.embed_query(xq)  # [B, S_q, D_p]

        sp = self.predictor(sq, sv, pos_embed=self.pos_embed)  # [B, S_q + N + 2, D_p]
        return sp


if __name__ == "__main__":
    model = VL_JEPA()
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(trainable, total_params)
    print("Percentage trainable", (trainable / total_params) * 100, "%")

    xv = torch.randn(2, 4, 3, 224, 224)  # Example video input
    xq = ["This is a sample query."] * 2  # Example text
    y = ["This is a sample target."] * 2  # Example target text
    xv, xq, y = xv.to(device), xq, y
    loss, accuracy = model(xv, xq, y)
    print("Loss:", loss.item())
    print("Accuracy:", accuracy.item())
