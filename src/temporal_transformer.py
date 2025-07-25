import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_modalities, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_modalities = num_modalities

    def forward(self, x):
        # x: (batch, time, modalities, embed_dim)
        b, t, m, e = x.shape
        x = x.view(b * t, m, e)  # (batch*time, modalities, embed_dim)
        attn_out, _ = self.attn(x, x, x)  # self-attention across modalities
        attn_out = self.norm(attn_out + x)
        attn_out = attn_out.mean(dim=1)  # fuse modalities (batch*time, embed_dim)
        attn_out = attn_out.view(b, t, e)  # (batch, time, embed_dim)
        return attn_out

class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_modalities=3, num_timepoints=5, num_heads=8, num_layers=4, num_classes=3):
        super().__init__()
        self.cross_modal = CrossModalAttention(embed_dim, num_modalities, num_heads=4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, num_classes)  # 3 classes: 6, 12, 24 month risk

    def forward(self, x):
        # x: (batch, time, modalities, embed_dim)
        x = self.cross_modal(x)  # (batch, time, embed_dim)
        x = self.transformer(x)  # (batch, time, embed_dim)
        out = self.head(x)       # (batch, time, num_classes)
        return out

# Example usage:
# model = TemporalTransformer()
# dummy = torch.randn(2, 5, 3, 768)  # batch=2, time=5, modalities=3, embed_dim=768
# out = model(dummy)
# print(out.shape)  # (2, 5, 3) 