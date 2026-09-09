"""Module 4: Hybrid Track Prediction Models.

Architecture: NWP + AI residual model.
Inputs: cyclone history, weather/environment, NWP forecast.
Predicts: +6h, +12h, +24h lat/lon + uncertainty; optionally wind/pressure.

Models:
  - GRU track model (baseline, from track history)
  - Transformer track model
  - Track Prediction Network (full: history + weather + NWP)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict


class GRUTrackModel(nn.Module):
    """GRU-based track prediction from cyclone position history.

    Input: relative positions (H, 2) or track states (H, 6).
    Optional: atmospheric/ERA5 weather features (weather_dim).
    Output: positions relative to current at each horizon.
    """

    def __init__(self, input_dim: int = 2, weather_dim: int = 0,
                 hidden_dim: int = 256, num_layers: int = 3,
                 horizons: List[int] = [6, 12, 24],
                 predict_intensity: bool = False,
                 uncertainty_estimation: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.weather_dim = weather_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizons = horizons
        self.predict_intensity = predict_intensity
        self.uncertainty_estimation = uncertainty_estimation

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        if weather_dim > 0:
            self.weather_proj = nn.Sequential(
                nn.Linear(weather_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)

        # Position heads (aggregated to handle heterogeneity of horizons)
        self.position_head = nn.ModuleDict({
            str(h): nn.Linear(hidden_dim, 2) for h in horizons
        })
        if uncertainty_estimation:
            self.uncertainty_head = nn.ModuleDict({
                str(h): nn.Linear(hidden_dim, 2) for h in horizons  # log-variance x2
            })
        if predict_intensity:
            self.intensity_head = nn.ModuleDict({
                str(h): nn.Linear(hidden_dim, 2) for h in horizons  # wind, pressure
            })

    def forward(self, history, current=None, features=None):
        """history: (B, H, 2) relative positions; current: (B, 2) absolute."""
        out, _ = self.gru(history)
        last = out[:, -1]  # (B, D)

        if features is not None and hasattr(self, "weather_proj"):
            w_emb = self.weather_proj(features)
            last = self.fusion_layer(torch.cat([last, w_emb], dim=1))

        result = {}
        for h in self.horizons:
            pred = self.position_head[str(h)](last)
            if current is not None:
                pred = pred + current
            if self.uncertainty_estimation:
                log_var = self.uncertainty_head[str(h)](last)
                result[str(h)] = {"position": pred, "log_variance": log_var}
            else:
                result[str(h)] = {"position": pred}

            if self.predict_intensity:
                intensity = self.intensity_head[str(h)](last)
                result[str(h)]["intensity"] = intensity

        return result


class TransformerTrackModel(nn.Module):
    """Transformer encoder for track prediction."""

    def __init__(self, input_dim: int = 2, weather_dim: int = 0,
                 hidden_dim: int = 256, num_layers: int = 3, num_heads: int = 8,
                 horizons: List[int] = [6, 12, 24],
                 predict_intensity: bool = False,
                 uncertainty_estimation: bool = True):
        super().__init__()
        self.horizons = horizons
        self.weather_dim = weather_dim
        self.predict_intensity = predict_intensity
        self.uncertainty_estimation = uncertainty_estimation

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        if weather_dim > 0:
            self.weather_proj = nn.Sequential(
                nn.Linear(weather_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)

        self.position_head = nn.ModuleDict({
            str(h): nn.Linear(hidden_dim, 2) for h in horizons
        })
        if uncertainty_estimation:
            self.uncertainty_head = nn.ModuleDict({
                str(h): nn.Linear(hidden_dim, 2) for h in horizons
            })
        if predict_intensity:
            self.intensity_head = nn.ModuleDict({
                str(h): nn.Linear(hidden_dim, 2) for h in horizons
            })

    def forward(self, history, current=None, features=None):
        """history: (B, H, 2)."""
        b, h_len, _ = history.shape
        x = self.input_proj(history) + self.pos_embed[:, :h_len]
        x = self.encoder(x)
        x = self.norm(x)
        # Aggregate over time
        x = x.mean(dim=1)

        if features is not None and hasattr(self, "weather_proj"):
            w_emb = self.weather_proj(features)
            x = self.fusion_layer(torch.cat([x, w_emb], dim=1))

        result = {}
        for h in self.horizons:
            pred = self.position_head[str(h)](x)
            if current is not None:
                pred = pred + current
            if self.uncertainty_estimation:
                log_var = self.uncertainty_head[str(h)](x)
                result[str(h)] = {"position": pred, "log_variance": log_var}
            else:
                result[str(h)] = {"position": pred}
            if self.predict_intensity:
                intensity = self.intensity_head[str(h)](x)
                result[str(h)]["intensity"] = intensity
        return result


class NWPResidualTrackModel(nn.Module):
    """NWP + AI residual track model.

    Uses NWP forecast positions as baseline and learns a residual correction
    using AI features (weather/environment).
    """

    def __init__(self, nwp_in_dim: int, env_in_dim: int, hidden_dim: int = 256,
                 horizons: List[int] = [6, 12, 24], num_layers: int = 3,
                 uncertainty_estimation: bool = True):
        super().__init__()
        self.horizons = horizons
        self.nwp_in_dim = nwp_in_dim
        self.env_in_dim = env_in_dim

        # Encode environment
        self.env_encoder = nn.Sequential(
            nn.Linear(env_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Per-horizon NWP context
        self.nwp_encoder = nn.ModuleDict({
            str(h): nn.Linear(nwp_in_dim, hidden_dim) for h in horizons
        })

        # Residual heads
        self.residual_head = nn.ModuleDict({
            str(h): nn.Linear(hidden_dim * 2, 2) for h in horizons
        })
        if uncertainty_estimation:
            self.uncertainty_head = nn.ModuleDict({
                str(h): nn.Linear(hidden_dim * 2, 2) for h in horizons
            })

    def forward(self, nwp_positions, environment, history_features=None):
        """nwp_positions: (B, len(horizons), 2) relative positions from NWP.
        environment: (B, env_dim) environmental features.
        Returns absolute (relative to current) predicted positions + residual.
        """
        env = self.env_encoder(environment)
        result = {}
        for i, h in enumerate(self.horizons):
            nwp_feat = self.nwp_encoder[str(h)](nwp_positions[:, i])
            combined = torch.cat([nwp_feat, env], dim=1)
            residual = self.residual_head[str(h)](combined)
            predicted = nwp_positions[:, i] + residual
            result[str(h)] = {"position": predicted, "residual": residual}
            if hasattr(self, "uncertainty_head"):
                log_var = self.uncertainty_head[str(h)](combined)
                result[str(h)]["log_variance"] = log_var
        return result


class TrackPredictionModel(nn.Module):
    """Full track+intensity model combining history and weather guidance."""

    def __init__(self, history_dim: int = 2, env_dim: int = 64,
                 nwp_dim: int = 2, hidden_dim: int = 256,
                 num_layers: int = 3, horizons: List[int] = [6, 12, 24],
                 model_type: str = "gru",
                 predict_intensity: bool = True,
                 uncertainty_estimation: bool = True,
                 use_nwp: bool = True):
        super().__init__()
        self.horizons = horizons
        self.predict_intensity = predict_intensity
        self.uncertainty_estimation = uncertainty_estimation
        self.use_nwp = use_nwp

        if model_type == "gru":
            self.history_encoder = GRUTrackModel(
                input_dim=history_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                horizons=horizons, predict_intensity=predict_intensity,
                uncertainty_estimation=False,  # we handle uncertainty at fusion layer
            )
        elif model_type == "transformer":
            self.history_encoder = TransformerTrackModel(
                input_dim=history_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                horizons=horizons, predict_intensity=predict_intensity,
                uncertainty_estimation=False,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Environment fusion
        self.env_proj = nn.Linear(env_dim, hidden_dim)
        self.nwp_used = use_nwp

        # Scene embedding / fusion heads
        self.position_head = nn.ModuleDict({
            str(h): nn.Linear(hidden_dim * 3, 2) for h in horizons
        })
        if uncertainty_estimation:
            self.uncertainty_head = nn.ModuleDict({
                str(h): nn.Linear(hidden_dim * 3, 2) for h in horizons
            })
        if predict_intensity:
            self.intensity_head = nn.ModuleDict({
                str(h): nn.Linear(hidden_dim * 3, 2) for h in horizons
            })

    def forward(self, history, current=None, environment=None, nwp_positions=None):
        """history: (B, H, 2); environment: (B, D); nwp_positions: (B, nHor, 2)."""
        # Encode history -> per-horizon hidden vectors
        hist_result = self.history_encoder(history, current=None)
        # hist_result[str(h)] = {"position": (B,2)} hides latent; get GRU output instead
        # We need the actual hidden state. Use history_encoder's GRU directly for sharing.
        if isinstance(self.history_encoder, GRUTrackModel):
            out, _ = self.history_encoder.gru(history)
        else:
            x = self.history_encoder.input_proj(history) + self.history_encoder.pos_embed[:, :history.shape[1]]
            x = self.history_encoder.encoder(x)
            out = self.history_encoder.norm(x).mean(dim=1)

        if environment is None:
            env_dim = out.shape[1]
            env = torch.zeros_like(out)
        else:
            env = self.env_proj(environment)

        result = {}
        for h in self.horizons:
            hist_vec = out[:, -1] if isinstance(self.history_encoder, GRUTrackModel) else out
            # NWP influence
            if self.use_nwp and nwp_positions is not None:
                nwp_idx = self.horizons.index(h)
                nwp_vec = nwp_positions[:, nwp_idx]
                # Simple combination: pass NWP offset as additional hint
                combined = torch.cat([hist_vec, env, nwp_vec], dim=1)
                input_dim = combined.shape[1]
                if input_dim != self.position_head[str(h)].in_features:
                    raise ValueError(
                        f"Dimension mismatch: combined dim {input_dim} != "
                        f"head in_features {self.position_head[str(h)].in_features}"
                    )
            else:
                combined = torch.cat([hist_vec, env, torch.zeros(hist_vec.shape[0], 2, device=hist_vec.device)], dim=1)

            pred = self.position_head[str(h)](combined)
            if current is not None:
                pred = pred + current

            out_h = {"position": pred}
            if self.uncertainty_estimation:
                log_var = self.uncertainty_head[str(h)](combined)
                out_h["log_variance"] = log_var
            if self.predict_intensity:
                intensity = self.intensity_head[str(h)](combined)
                out_h["intensity"] = intensity
            result[str(h)] = out_h

        return result


class DynamicalTrackModel(nn.Module):
    """Dynamical Physics-Kinematic Track Forecast Model.

    Combines:
      - 8-dimensional kinematic track history (relative pos, displacement velocities,
        translation speed, heading sin/cos, pressure/wind trends)
      - Physical environmental & steering context (DLM steering flow, Coriolis f, Beta drift, VWS)
      - Bi-directional GRU with Multi-Head Temporal Self-Attention
      - Dynamical Gated Fusion Unit (GLU)
      - Multi-horizon (+6h, +12h, +24h) residual heads with heteroscedastic 2D uncertainty
    """

    def __init__(self, input_dim: int = 8, weather_dim: int = 72,
                 hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4,
                 horizons: List[int] = [6, 12, 24],
                 dropout: float = 0.1,
                 uncertainty_estimation: bool = True):
        super().__init__()
        self.horizons = horizons
        self.uncertainty_estimation = uncertainty_estimation
        self.hidden_dim = hidden_dim

        # 1. Kinematic sequence projection
        self.kinematic_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # 2. Bi-directional GRU (2 directions * hidden_dim = 256)
        self.bi_gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        gru_dim = hidden_dim * 2

        # 3. Multi-Head Temporal Self-Attention over sequence steps
        self.self_attn = nn.MultiheadAttention(
            embed_dim=gru_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_attn = nn.LayerNorm(gru_dim)

        # 4. Environmental & Physical Steering Projection
        self.env_proj = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, gru_dim),
            nn.LayerNorm(gru_dim),
        )

        # 5. Dynamical Gated Fusion Unit (GLU style)
        self.gate = nn.Sequential(
            nn.Linear(gru_dim * 2, gru_dim),
            nn.Sigmoid()
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(gru_dim * 2, gru_dim),
            nn.LayerNorm(gru_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # 6. Multi-Horizon Position & Uncertainty Heads
        self.position_head = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(gru_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2)
            ) for h in horizons
        })
        if uncertainty_estimation:
            self.uncertainty_head = nn.ModuleDict({
                str(h): nn.Sequential(
                    nn.Linear(gru_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 2)
                ) for h in horizons
            })

    def forward(self, history, current=None, features=None):
        # history: (B, H, input_dim)
        # features: (B, weather_dim)
        x = self.kinematic_proj(history)
        gru_out, _ = self.bi_gru(x)  # (B, H, gru_dim)

        # Self-attention over temporal sequence
        attn_out, _ = self.self_attn(gru_out, gru_out, gru_out)
        h_seq = self.norm_attn(gru_out + attn_out)

        # Attention pooled context (last state enriched with attended history)
        h_pooled = h_seq[:, -1]

        # Environmental & Physical Fusion
        if features is not None and hasattr(self, "env_proj"):
            env_feat = self.env_proj(features)
            combined = torch.cat([h_pooled, env_feat], dim=1)
            g = self.gate(combined)
            fused = self.fuse_proj(combined)
            context = g * fused + (1.0 - g) * h_pooled
        else:
            context = h_pooled

        result = {}
        for h in self.horizons:
            pos = self.position_head[str(h)](context)
            if current is not None:
                pos = pos + current
            out_dict = {"position": pos}
            if self.uncertainty_estimation:
                out_dict["log_variance"] = self.uncertainty_head[str(h)](context)
            result[str(h)] = out_dict

        return result


