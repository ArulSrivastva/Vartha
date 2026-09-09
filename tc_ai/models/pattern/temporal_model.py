"""Module 3: Tropical Cyclone Temporal Pattern Recognition.

Models learn from temporal satellite sequences:
T-12h, T-9h, T-6h, T-3h, T0

Models:
  - ConvLSTM
  - Temporal Vision Transformer (TimeSformer-like)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size, padding=self.padding, bias=bias,
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM for temporal satellite sequence analysis."""

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 kernel_size: int = 3, num_layers: int = 3,
                 batch_first: bool = True, bias: bool = True,
                 return_all_layers: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        prev_dim = input_dim
        for i in range(num_layers):
            cell_list.append(ConvLSTMCell(prev_dim, hidden_dims[i], kernel_size, bias))
            prev_dim = hidden_dims[i]
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden_state=None):
        """x: (B, T, C, H, W) if batch_first."""
        if not self.batch_first:
            x = x.transpose(0, 1)  # to (B, T, C, H, W)

        b, seq_len, _, h, w = x.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(b, h, w, x.device)

        layer_output_list = []
        last_state_list = []
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], h, c)
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if self.return_all_layers:
            return layer_output_list, last_state_list
        return layer_output_list[-1], last_state_list[-1]

    def _init_hidden(self, batch_size, h, w, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append([
                torch.zeros(batch_size, self.hidden_dims[i], h, w, device=device),
                torch.zeros(batch_size, self.hidden_dims[i], h, w, device=device),
            ])
        return init_states


class TemporalPatternModel(nn.Module):
    """Full temporal pattern recognition model (satellite sequence -> pattern)."""

    def __init__(self, input_channels: int = 3, hidden_dim: int = 64,
                 num_layers: int = 3, n_pattern_classes: int = 9,
                 kernel_size: int = 3, sequence_length: int = 6,
                 image_size: int = 128, image_encoder: str = "convlstm"):
        super().__init__()
        self.sequence_length = sequence_length
        self.image_encoder = image_encoder

        # Per-frame encoder: encodes each satellite image to feature map
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.encoded_size = image_size // 8

        if image_encoder == "convlstm":
            self.temporal = ConvLSTM(
                input_dim=128, hidden_dims=[hidden_dim] * num_layers,
                num_layers=num_layers, kernel_size=kernel_size,
            )
            self.temporal_out_dim = hidden_dim
        else:
            raise ValueError(f"Unknown image encoder: {image_encoder}")

        # Classification heads
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * (self.encoded_size ** 2), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, n_pattern_classes),
        )

    def forward(self, sequences, return_features: bool = False):
        """sequences: (B, T, C, H, W)"""
        b, t, c, h, w = sequences.shape
        # Encode each frame
        frames = sequences.reshape(b * t, c, h, w)
        features = self.frame_encoder(frames)  # (B*T, 128, Hs, Ws)
        features = features.reshape(b, t, *features.shape[1:])

        temporal_out, _ = self.temporal(features)
        if isinstance(temporal_out, list):
            temporal_out = temporal_out[-1]
        temporal_out = temporal_out[:, -1]  # last timestep

        pooled = temporal_out.reshape(b, -1)  # flatten spatial
        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled
        return logits


class TemporalVisionTransformer(nn.Module):
    """Temporal Transformer for sequence classification (TimeSformer-like)."""

    def __init__(self, input_channels: int, seq_len: int,
                 image_size: int, patch_size: int = 16,
                 embed_dim: int = 384, depth: int = 6, num_heads: int = 8,
                 num_classes: int = 9, temporal_emb: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.temporal_emb = temporal_emb
        n_patches = self.grid_size ** 2

        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Spatial position embeddings + temporal embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

    def forward(self, sequences):
        b, t, c, h, w = sequences.shape
        # Patch embed each frame
        x = sequences.reshape(b * t, c, h, w)
        x = self.patch_embed(x)  # (B*T, D, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (B*T, N, D)
        x = x + self.pos_embed
        _, n, d = x.shape
        x = x.reshape(b, t, n, d)
        if self.temporal_emb:
            x = x + self.temporal_embed.unsqueeze(2)
        x = x.reshape(b, t * n, d)
        x = self.encoder(x)
        x = self.norm(x)
        # Take CLS-like aggregated representation (mean over all tokens)
        x = x.mean(dim=1)
        return self.head(x)


class TrackIntensityModel(nn.Module):
    """Phase 3 track-only Intensity/RI model for REAL satellite-free data.

    Uses only best-track wind/pressure history (IBTrACS), so metrics remain
    valid when no satellite imagery / ERA5 is available.

    Inputs:
      - track_states: (B, H, 2) past (wind_kt, pressure_mb) at each 6h point
    Outputs:
      - delta_wind_24h: (B, 1) predicted wind change over the next 24h (kt)
      - ri_logits: (B, 1) binary logit for Rapid Intensification (>=30 kt/24h)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128,
                 num_layers: int = 2, history_len: int = 6):
        super().__init__()
        self.history_len = history_len
        self.track_gru = nn.GRU(input_dim, hidden_dim, num_layers,
                                batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.delta_wind_head = nn.Linear(64, 1)
        self.ri_head = nn.Linear(64, 1)

    def forward(self, track_states: torch.Tensor, **kwargs):
        """track_states: (B, H, 2)."""
        out, _ = self.track_gru(track_states)
        h = self.head(out[:, -1])
        return {
            "delta_wind_24h": self.delta_wind_head(h),
            "ri_logits": self.ri_head(h),
            "embedding": h,
        }


class TrackIntensityWeatherModel(nn.Module):
    """Phase 3 v2: wind/pressure history GRU + ERA5 environmental MLP.

    Unlike the track-only variant, environmental fields (SST, shear, steering,
    MSL, humidity, etc.) + Coriolis/steering physics are fused into the head.
    Both the 24h wind-change regression and the RI classifier benefit from the
    absolute scale of physical fields (global-normed weather), which the
    track-only model cannot see.

    Inputs:
      - track_states: (B, H, 2) past (wind_kt, pressure_mb), normalized
      - weather: (B, W) global-normed environmental + physical features
    Outputs:
      - delta_wind_24h: (B, 1) predicted wind change over the next 24h (kt)
      - ri_logits: (B, 1) binary logit for Rapid Intensification (>=30 kt/24h)
      - embedding: (B, hidden_dim)
    """

    def __init__(self, input_dim: int = 2, weather_dim: int = 72,
                 hidden_dim: int = 128, num_layers: int = 2,
                 history_len: int = 6):
        super().__init__()
        self.history_len = history_len
        self.track_gru = nn.GRU(input_dim, hidden_dim, num_layers,
                                batch_first=True)
        self.weather_mlp = nn.Sequential(
            nn.Linear(weather_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.delta_wind_head = nn.Linear(64, 1)
        self.ri_head = nn.Linear(64, 1)

    def forward(self, track_states: torch.Tensor, weather: torch.Tensor = None, **kwargs):
        """track_states: (B, H, 2); weather: (B, W) or None."""
        out, _ = self.track_gru(track_states)
        h = out[:, -1]
        if weather is not None:
            w = self.weather_mlp(weather)
            h = self.head(torch.cat([h, w], dim=-1))
        else:
            h = self.head(torch.cat([h, torch.zeros_like(h)], dim=-1))
        return {
            "delta_wind_24h": self.delta_wind_head(h),
            "ri_logits": self.ri_head(h),
            "embedding": h,
        }


class TemporalIntensityRIModel(nn.Module):
    """Phase 3 Temporal Model: Intensity-Change Trend + Rapid Intensification (RI).

    Inputs:
      - sequences: (B, T, C, H, W) satellite sequence (e.g. 4-6 frames)
      - track_history: (B, H, 2) optional track coordinates
    Outputs:
      - delta_wind_24h: (B, 1) predicted wind change over next 24h (kt)
      - ri_logits: (B, 1) binary classification logit for RI (>=30kt/24h)
      - embedding: (B, hidden_dim) learned spatiotemporal feature embedding for Phase 4 fusion
    """

    def __init__(self, input_channels: int = 3, hidden_dim: int = 128,
                 track_history_len: int = 6, use_track: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_track = use_track

        # Per-frame lightweight CNN feature extractor
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 64 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Temporal sequence GRU over satellite frames
        self.seq_gru = nn.GRU(128, hidden_dim, num_layers=2, batch_first=True)

        # Optional track history GRU
        if use_track:
            self.track_gru = nn.GRU(2, 64, num_layers=1, batch_first=True)
            self.fusion_fc = nn.Sequential(
                nn.Linear(hidden_dim + 64, hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.fusion_fc = nn.Identity()

        # Task heads
        self.delta_wind_head = nn.Linear(hidden_dim, 1)
        self.ri_head = nn.Linear(hidden_dim, 1)

    def forward(self, sequences: torch.Tensor, track_history: Optional[torch.Tensor] = None):
        """sequences: (B, T, C, H, W) or (B, C, F, H, W)."""
        # Ensure (B, T, C, H, W)
        if sequences.ndim == 5 and sequences.shape[1] in (1, 2, 3) and sequences.shape[2] > sequences.shape[1]:
            sequences = sequences.permute(0, 2, 1, 3, 4)

        b, t, c, h, w = sequences.shape
        frames = sequences.reshape(b * t, c, h, w)
        frame_feats = self.frame_encoder(frames)  # (B*T, 128)
        frame_feats = frame_feats.reshape(b, t, 128)

        seq_out, _ = self.seq_gru(frame_feats)
        sat_last = seq_out[:, -1]  # (B, hidden_dim)

        if self.use_track and track_history is not None:
            # track_history: (B, H, 2)
            trk_out, _ = self.track_gru(track_history[:, :, :2])
            trk_last = trk_out[:, -1]
            fused = self.fusion_fc(torch.cat([sat_last, trk_last], dim=1))
        else:
            fused = sat_last

        delta_wind = self.delta_wind_head(fused)
        ri_logits = self.ri_head(fused)

        return {
            "delta_wind_24h": delta_wind,
            "ri_logits": ri_logits,
            "embedding": fused,
        }