import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


def pad_to_stride(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """Right-pad a sequence so its length is divisible by the pooling stride."""
    length = x.shape[-1]
    pad_len = (-length) % stride
    if pad_len > 0:
        x = F.pad(x, (0, pad_len))
    return x


def build_local_attention_bias(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    """Build the Gaussian distance bias used between patch tokens."""
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    pos_i = positions.unsqueeze(1)  # (seq_len, 1)
    pos_j = positions.unsqueeze(0)  # (1, seq_len)
    distance = torch.abs(pos_i - pos_j)  # (seq_len, seq_len)

    sigma = window_size / 2.0
    bias = -distance ** 2 / (2 * sigma ** 2)  # (seq_len, seq_len)

    return bias


class ConvBlock(nn.Module):
    """Residual two-convolution block followed by pooling and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.downsample.weight.data.normal_(0, 0.01)

        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform ``(B, C_in, L)`` into ``(B, C_out, L')``."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        # Even kernels can create a one-step boundary mismatch.
        if identity.shape[2] != out.shape[2]:
            target_len = out.shape[2]
            current_len = identity.shape[2]
            if current_len > target_len:
                identity = identity[:, :, :target_len]
            else:
                pad_len = target_len - current_len
                identity = F.pad(identity, (0, pad_len), mode="constant", value=0)

        out = out + identity
        out = self.last_activation(out)
        out = self.maxpool(out)
        out = self.dropout(out)

        return out

class HybridMultiHeadAttention(nn.Module):
    """Combine local patch attention with global router-to-patch attention.

    Patch-to-patch scores receive a Gaussian distance bias. The final router
    token can attend globally, while patch-to-router attention is masked.
    """

    def __init__(self, d_model: int, nhead: int, window_size: int = 10, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.window_size = window_size

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor, router_idx: int = -1) -> torch.Tensor:
        """Attend over ``(B, L, D)`` tokens with the router in the final slot."""
        B, L, D = x.shape

        Q = self.W_q(x)  # (B, L, D)
        K = self.W_k(x)  # (B, L, D)
        V = self.W_v(x)  # (B, L, D)

        Q = Q.view(B, L, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)
        K = K.view(B, L, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)
        V = V.view(B, L, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, nhead, L, L)

        # The router occupies the final row and column.
        local_bias = build_local_attention_bias(L - 1, self.window_size, scores.device)

        full_bias = torch.zeros(L, L, device=scores.device)
        full_bias[:-1, :-1] = local_bias
        full_bias[:-1, -1] = float("-inf")

        full_bias = full_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        scores = scores + full_bias  # (B, nhead, L, L)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, nhead, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        out = self.W_o(out)
        return out


class HybridAttentionOnly(nn.Module):
    """Hybrid attention without residual, normalization, or FFN layers."""

    def __init__(self, d_model: int, nhead: int, window_size: int = 10, dropout: float = 0.1):
        super().__init__()
        self.attn = HybridMultiHeadAttention(d_model, nhead, window_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)




class LocalHierarchicalCrossAttentionOnly(nn.Module):
    """Local cross-attention without residual, normalization, or FFN layers."""

    def __init__(self, d_model: int, nhead: int, window_size: int = 3, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.window_size = window_size

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> torch.Tensor:
        """Map ``(B, L_q, D)`` queries to local ``(B, L_kv, D)`` regions."""
        B, L_q, D = query.shape
        L_kv = key_value.shape[1]

        Q = self.W_q(query)  # (B, L_q, D)
        K = self.W_k(key_value)  # (B, L_kv, D)
        V = self.W_v(key_value)  # (B, L_kv, D)

        Q = Q.view(B, L_q, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L_q, d_k)
        K = K.view(B, L_kv, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L_kv, d_k)
        V = V.view(B, L_kv, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L_kv, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, nhead, L_q, L_kv)

        if L_kv > 1 and L_q > 1:
            local_bias = build_local_attention_bias(L_kv, self.window_size, scores.device)  # (L_kv, L_kv)
            local_bias = local_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, L_kv, L_kv)
            local_bias = local_bias.expand(B, self.nhead, L_q, -1)  # (B, nhead, L_q, L_kv)
            scores = scores + local_bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, nhead, L_q, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)  # (B, L_q, D)
        out = self.W_o(out)
        return out




class RouterAttentionOnly(nn.Module):
    """Global router attention without residual, normalization, or FFN layers."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global attention to ``G`` router tokens of shape ``(B, G, D)``."""
        B, G, D = x.shape

        Q = self.W_q(x)  # (B, G, D)
        K = self.W_k(x)  # (B, G, D)
        V = self.W_v(x)  # (B, G, D)

        Q = Q.view(B, G, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, G, d_k)
        K = K.view(B, G, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, G, d_k)
        V = V.view(B, G, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, G, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, nhead, G, G)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, nhead, G, d_k)
        out = out.transpose(1, 2).contiguous().view(B, G, D)  # (B, G, D)
        out = self.W_o(out)
        return out




class ConvGranularityBranch(nn.Module):
    """Extract patch and router tokens at one temporal granularity."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        dropout: float,
        max_len: int = 4096,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        mid_dim = embed_dim // 2

        self.stem = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=mid_dim,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
            ),
            ConvBlock(
                in_channels=mid_dim,
                out_channels=mid_dim,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
            ),
            ConvBlock(
                in_channels=mid_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
            ),
        )

        self.max_len = max_len
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return patch tokens ``(B, P, D)`` and a router token ``(B, D)``."""
        x = pad_to_stride(x, self.kernel_size, self.stride)
        tokens = self.stem(x).transpose(1, 2)  # (B, P, D)

        B, P, D = tokens.shape
        if P + 1 > self.max_len:
            raise ValueError(f"Patch count {P} exceeds max_len={self.max_len}")

        pos = self.pos_embed[:, : P + 1, :]  # (1, P+1, D)
        pos = pos.expand(B, -1, -1)  # (B, P+1, D)
        patch_pos = pos[:, :P, :]
        router_pos = pos[:, -1:, :]

        patch_tokens = tokens + patch_pos
        router_token = router_pos.squeeze(1)

        return patch_tokens, router_token




class MultiGranularityAttention(nn.Module):
    """Apply fine-to-coarse, intra-granularity, and cross-router attention."""

    def __init__(
        self,
        num_granularities: int,
        embed_dim: int,
        num_heads: int,
        local_window: int,
        cross_granularity_window: int,
        dropout: float = 0.1,
        enable_cross_granularity: bool = True,
        enable_router_interaction: bool = True,
    ):
        super().__init__()
        self.num_granularities = num_granularities
        self.cross_granularity_window = cross_granularity_window
        self.enable_cross_granularity = enable_cross_granularity
        self.enable_router_interaction = enable_router_interaction

        self.intra_granularity_attns = nn.ModuleList([
            HybridAttentionOnly(
                d_model=embed_dim,
                nhead=num_heads,
                window_size=local_window,
                dropout=dropout,
            )
            for _ in range(num_granularities)
        ])

        self.upward_attns = nn.ModuleList(
            [
                LocalHierarchicalCrossAttentionOnly(
                    d_model=embed_dim,
                    nhead=num_heads,
                    window_size=cross_granularity_window,
                    dropout=dropout,
                )
                for _ in range(num_granularities - 1)
            ]
            if enable_cross_granularity
            else []
        )

        self.router_inter_attention = (
            RouterAttentionOnly(d_model=embed_dim, nhead=num_heads, dropout=dropout)
            if enable_router_interaction
            else None
        )

    def forward(
        self,
        patch_tokens_list: list[torch.Tensor],
        router_tokens_list: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Update per-granularity patch and router token lists."""
        # Coarse queries aggregate aligned windows of finer patch tokens.
        enhanced_patch_tokens = list(patch_tokens_list)

        for i in (
            range(self.num_granularities - 2, -1, -1)
            if self.enable_cross_granularity
            else []
        ):
            coarse_patches = enhanced_patch_tokens[i]  # (B, P_i, D)
            fine_patches = enhanced_patch_tokens[i + 1]  # (B, P_{i+1}, D)
            coarse_len = coarse_patches.shape[1]
            fine_len = fine_patches.shape[1]

            upward_attn = self.upward_attns[i]
            window_size = self.cross_granularity_window

            enhanced_coarse_patches_list = []

            for j in range(coarse_len):
                ratio = fine_len / coarse_len
                center_idx = int((j + 0.5) * ratio)

                half_window = window_size // 2
                start_idx = max(0, center_idx - half_window)
                end_idx = min(fine_len, center_idx + half_window + 1)

                fine_region = fine_patches[:, start_idx:end_idx, :]  # (B, region_len, D)

                if fine_region.shape[1] == 0:
                    enhanced_coarse_patches_list.append(coarse_patches[:, j:j+1, :])
                    continue

                coarse_patch = coarse_patches[:, j:j+1, :]  # (B, 1, D)
                enhanced_coarse_patch = upward_attn(coarse_patch, fine_region)  # (B, 1, D)
                enhanced_coarse_patches_list.append(enhanced_coarse_patch)

            enhanced_patch_tokens[i] = torch.cat(enhanced_coarse_patches_list, dim=1)  # (B, P_i, D)

        # Joint intra-granularity attention updates patches and their router.
        enhanced_router_tokens = []
        for i in range(self.num_granularities):
            current_patches = enhanced_patch_tokens[i]  # (B, P_i, D)
            current_router = router_tokens_list[i].unsqueeze(1)  # (B, 1, D)

            patches_with_router = torch.cat([
                current_patches,
                current_router
            ], dim=1)  # (B, P_i + 1, D)

            intra_attn = self.intra_granularity_attns[i]
            enhanced_all = intra_attn(patches_with_router)  # (B, P_i + 1, D)

            enhanced_patches = enhanced_all[:, :-1, :]  # (B, P_i, D)
            enhanced_router = enhanced_all[:, -1, :]  # (B, D)

            enhanced_patch_tokens[i] = enhanced_patches
            enhanced_router_tokens.append(enhanced_router)

        if not self.enable_router_interaction:
            return enhanced_patch_tokens, enhanced_router_tokens

        router_features = torch.stack(enhanced_router_tokens, dim=1)
        enhanced_router_features = self.router_inter_attention(router_features)
        return enhanced_patch_tokens, [
            enhanced_router_features[:, i, :] for i in range(self.num_granularities)
        ]


class MultiGranularityAttentionBlock(nn.Module):
    """Wrap multi-granularity attention with residual, norm, and FFN layers."""

    def __init__(
        self,
        attention: MultiGranularityAttention,
        embed_dim: int,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        dim_feedforward = dim_feedforward or (embed_dim * 4)
        self.attention = attention
        self.num_granularities = attention.num_granularities

        self.norm1_list = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(self.num_granularities)
        ])
        self.norm2_list = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(self.num_granularities)
        ])

        self.ffn_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=dim_feedforward, kernel_size=1),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(in_channels=dim_feedforward, out_channels=embed_dim, kernel_size=1),
                nn.Dropout(dropout),
            )
            for _ in range(self.num_granularities)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        patch_tokens_list: list[torch.Tensor],
        router_tokens_list: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Apply attention and FFN sublayers with post-norm residuals."""
        new_patch_tokens, new_router_tokens = self.attention(
            patch_tokens_list, router_tokens_list
        )

        enhanced_patch_tokens = [
            patch_tokens_list[i] + self.dropout(new_patch_tokens[i])
            for i in range(self.num_granularities)
        ]
        enhanced_router_tokens = [
            router_tokens_list[i] + self.dropout(new_router_tokens[i])
            for i in range(self.num_granularities)
        ]

        normalized_patch_tokens = [
            self.norm1_list[i](enhanced_patch_tokens[i])
            for i in range(self.num_granularities)
        ]
        normalized_router_tokens = [
            self.norm1_list[i](enhanced_router_tokens[i].unsqueeze(1)).squeeze(1)
            for i in range(self.num_granularities)
        ]

        ffn_patch_tokens = [
            self.ffn_list[i](normalized_patch_tokens[i].transpose(-1, 1)).transpose(-1, 1)
            for i in range(self.num_granularities)
        ]

        ffn_router_tokens = [
            self.ffn_list[i](normalized_router_tokens[i].unsqueeze(1).transpose(-1, 1)).transpose(-1, 1).squeeze(1)
            for i in range(self.num_granularities)
        ]

        final_patch_tokens = [
            self.norm2_list[i](enhanced_patch_tokens[i] + ffn_patch_tokens[i])
            for i in range(self.num_granularities)
        ]
        final_router_tokens = [
            self.norm2_list[i]((enhanced_router_tokens[i] + ffn_router_tokens[i]).unsqueeze(1)).squeeze(1)
            for i in range(self.num_granularities)
        ]

        return final_patch_tokens, final_router_tokens


class PrismWF(nn.Module):
    """Multi-granularity patch-based Transformer for multi-label WF."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 6,
        num_layers: int = 3,
        branch_kernels: tuple[int, ...] = (15, 11, 7, 5),
        enable_cross_granularity: bool = True,
        enable_router_interaction: bool = True,
    ):
        super().__init__()
        embed_dim = 256
        if not branch_kernels:
            raise ValueError("At least one convolution branch is required")
        conv_settings = tuple(
            {"kernel": kernel, "stride": kernel} for kernel in branch_kernels
        )
        num_heads = 8
        dropout = 0.1
        max_len = 4096
        local_window = 5
        cross_granularity_window = 3
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_granularities = len(conv_settings)
        self.cross_granularity_window = cross_granularity_window
        self.branches = nn.ModuleList(
            [
                ConvGranularityBranch(
                    in_channels=in_channels,
                    embed_dim=embed_dim,
                    kernel_size=cfg["kernel"],
                    stride=cfg["stride"],
                    dropout=dropout,
                    max_len=max_len,
                )
                for cfg in conv_settings
            ]
        )

        self.layers = nn.ModuleList([
            MultiGranularityAttentionBlock(
                attention=MultiGranularityAttention(
                    num_granularities=self.num_granularities,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    local_window=local_window,
                    cross_granularity_window=cross_granularity_window,
                    dropout=dropout,
                    enable_cross_granularity=enable_cross_granularity,
                    enable_router_interaction=enable_router_interaction,
                ),
                embed_dim=embed_dim,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
            )
            for _ in range(num_layers)
        ])
        fused_dim = embed_dim * len(conv_settings)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return multi-label logits for an input of shape ``(B, C, L)``."""
        if x.ndim != 3:
            raise ValueError(f"Expected a 3D input (B, C, L), received {x.shape}")

        patch_tokens_list = []
        router_tokens_list = []

        for branch in self.branches:
            patch_tokens, router_token = branch(x)
            patch_tokens_list.append(patch_tokens)
            router_tokens_list.append(router_token)

        for layer in self.layers:
            patch_tokens_list, router_tokens_list = layer(patch_tokens_list, router_tokens_list)

        router_features = torch.stack(router_tokens_list, dim=1)  # (B, G, D)
        fused = router_features.reshape(router_features.size(0), -1)  # (B, G*D)

        logits = self.classifier(fused)  # (B, num_classes)
        return logits
