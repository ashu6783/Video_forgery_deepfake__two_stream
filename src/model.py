import torch
import torch.nn as nn
from torchvision import models


class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b0(weights=weights)

        if in_channels != 3:
            first_conv = base.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

            with torch.no_grad():
                if pretrained:
                    if in_channels == 1:
                        new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                    elif in_channels == 2:
                        new_conv.weight[:, 0:1].copy_(first_conv.weight[:, 0:1])
                        new_conv.weight[:, 1:2].copy_(first_conv.weight[:, 1:2])
                    else:
                        repeat_factor = (in_channels + 2) // 3
                        expanded = first_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :in_channels]
                        expanded *= 3.0 / float(in_channels)
                        new_conv.weight.copy_(expanded)
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)

            base.features[0][0] = new_conv

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 1280

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x


class TwoStreamDeepfakeNet(nn.Module):
    def __init__(
        self,
        pretrained=True,
        dropout=0.3,
        temporal_hidden_dim=256,
        fusion_dim=512,
        max_frames=32,
    ):
        super().__init__()
        self.rgb_encoder = EfficientNetEncoder(pretrained=pretrained, in_channels=3)
        self.flow_encoder = EfficientNetEncoder(pretrained=pretrained, in_channels=2)
        fused_dim = self.rgb_encoder.out_dim + self.flow_encoder.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.pos_embedding = nn.Embedding(max_frames, fusion_dim)
        self.temporal_gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        temporal_out_dim = temporal_hidden_dim
        self.temporal_attention = nn.Linear(temporal_out_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(temporal_out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )

    def forward(self, rgb, flow):
        # rgb/flow: [B, T, C, H, W]
        b, t, c_rgb, h, w = rgb.shape
        _, _, c_flow, _, _ = flow.shape
        rgb_flat = rgb.reshape(b * t, c_rgb, h, w)
        flow_flat = flow.reshape(b * t, c_flow, h, w)

        rgb_feat = self.rgb_encoder(rgb_flat).reshape(b, t, -1)
        flow_feat = self.flow_encoder(flow_flat).reshape(b, t, -1)
        fused = torch.cat([rgb_feat, flow_feat], dim=-1)
        fused = self.fusion(fused)

        if t > self.pos_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {t} exceeds max_frames={self.pos_embedding.num_embeddings}. "
                "Increase max_frames in model config."
            )

        pos_ids = torch.arange(t, device=fused.device).unsqueeze(0).expand(b, -1)
        fused = fused + self.pos_embedding(pos_ids)

        temporal_seq, _ = self.temporal_gru(fused)
        attn_logits = self.temporal_attention(temporal_seq).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        temporal = torch.sum(temporal_seq * attn_weights, dim=1)
        return self.classifier(temporal)

    def freeze_backbones(self):
        for p in self.rgb_encoder.parameters():
            p.requires_grad = False
        for p in self.flow_encoder.parameters():
            p.requires_grad = False

    def unfreeze_top_blocks(self, num_blocks=2):
        for encoder in [self.rgb_encoder, self.flow_encoder]:
            for p in encoder.parameters():
                p.requires_grad = False
            blocks = list(encoder.features.children())
            for block in blocks[-num_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True