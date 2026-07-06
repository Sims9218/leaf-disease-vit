import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import timm
from einops import rearrange
from einops.layers.torch import Rearrange

class RobustEdgeDetectionModule(nn.Module):
    def __init__(self, mode='sobel'):
        super().__init__()
        self.mode = mode

        if mode == 'sobel':
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

            self.register_buffer('sobel_x_kernel', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
            self.register_buffer('sobel_y_kernel', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, x):
        if x is None:
            raise ValueError("Input tensor is None")
            
        B, C, H, W = x.shape

        if self.mode == 'sobel':
            grad_x = F.conv2d(x, self.sobel_x_kernel, padding=1, groups=C)
            grad_y = F.conv2d(x, self.sobel_y_kernel, padding=1, groups=C)
            edge_map = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            edge_map = edge_map / (edge_map.max() + 1e-6)

        elif self.mode == 'canny':
            edge_maps = []
            for i in range(B):
                img = x[i].permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if C == 3 else img.squeeze()
                    edges = cv2.Canny(gray, 50, 150)
                    edges = torch.from_numpy(edges.astype(np.float32) / 255.0).unsqueeze(0).to(x.device)
                    edges = edges.repeat(C, 1, 1)
                    edge_maps.append(edges)
                except Exception:
                    edge_maps.append(torch.zeros(C, H, W, device=x.device))
            edge_map = torch.stack(edge_maps)
            
        return edge_map

class SafeCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        if x is None or context is None:
            return torch.zeros_like(x) if x is not None else torch.zeros_like(context)

        qkv = self.to_qkv(torch.cat([x, context], dim=1)).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class RobustAttentionGuidedEdgeViT(nn.Module):
    def __init__(self, num_classes=7, edge_mode='sobel', pretrained=True, vit_model='vit_base_patch16_224'):
        super().__init__()
        self.edge_detector = RobustEdgeDetectionModule(mode=edge_mode)
        self.vit = timm.create_model(vit_model, pretrained=pretrained)
        self.embed_dim = self.vit.embed_dim
        self.vit.head = nn.Identity()

        self.edge_patch_embed = nn.Sequential(
            nn.Conv2d(3, self.embed_dim, kernel_size=16, stride=16),
            nn.Flatten(2),
            Rearrange('b c n -> b n c'),
            nn.LayerNorm(self.embed_dim)
        )

        self.cross_attn = SafeCrossAttention(dim=self.embed_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

        num_patches = (224 // 16) ** 2
        self.edge_pos_embed = nn.Parameter(torch.randn(1, num_patches, self.embed_dim) * 0.02)

    def forward(self, x):
        if x is None:
            raise ValueError("Model input cannot be None")

        edge_maps = self.edge_detector(x)
        rgb_features = self.vit.forward_features(x)

        edge_tokens = self.edge_patch_embed(edge_maps)
        edge_tokens = edge_tokens + self.edge_pos_embed

        cls_token = rgb_features[:, 0:1]
        patch_tokens = rgb_features[:, 1:]

        attended = self.cross_attn(patch_tokens, edge_tokens)
        combined = torch.cat([cls_token, attended], dim=1)

        logits = self.classifier(combined[:, 0])
        return logits, edge_maps
