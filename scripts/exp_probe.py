"""Fast iteration driver: imports helpers from eval_real_metrics and runs a single phase.

Usage:
  python scripts/exp_probe.py --phase 1 [--epochs N] [--model-dims ...]
"""
import argparse, json, sys, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import TCConfig
from tc_ai.data.dataset import TCDataset
from tc_ai.data.real_data import DATA_SOURCE_TAG
from tc_ai.utils.common import set_seed, get_device

from scripts.eval_real_metrics import (
    HORIZONS,
    CachedSatelliteDataset,
    collate_phase1, collate_phase2_real, collate_phase3_real, collate_phase4,
    eval_baselines, eval_track_model, train_track_model,
    eval_stage_model, train_stage_model,
    eval_intensity_model, train_intensity_model, tune_ri_threshold,
    eval_fusion_model, train_fusion_model,
)
from tc_ai.models.prediction.track_predictor import DynamicalTrackModel
from tc_ai.models.classification.classifier import StageClassifier
from tc_ai.models.pattern.temporal_model import TrackIntensityModel
from tc_ai.models.fusion.multimodal import SatelliteTrackFusionModel


from tc_ai.evaluation.metrics import TrackMetrics


def eval_track_raw(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            hist = batch["history"].to(device)
            cur = batch["current"].to(device)
            weat = batch["weather"].to(device)
            tgt = batch["targets_future"].to(device)
            out = model(hist, current=cur, features=weat)
            for i in range(hist.shape[0]):
                p_dict = {h: {"position": out[h]["position"][i].cpu().numpy()}
                          for h in [str(x) for x in model.horizons]}
                t_dict = {str(h): (cur[i] + tgt[i, j]).cpu().numpy() for j, h in enumerate(HORIZONS)}
                t_dict["current_position"] = cur[i].cpu().numpy()
                preds.append({"prediction": p_dict})
                targets.append(t_dict)
    return preds, targets


def train_track_nll(model, train_loader, val_loader, epochs, device, lr=6e-4, wd=1e-3, seed=None):
    from tc_ai.training.losses import kinematic_track_loss
    if seed is not None:
        torch.manual_seed(seed + 1000)
        np.random.seed(seed + 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    best_val_dpe, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            hist = batch["history"].to(device)
            weat = batch["weather"].to(device)
            tgt = batch["targets_future"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                preds = model(hist, current=None, features=weat)
                loss = kinematic_track_loss(preds, {h: tgt[:, j] for j, h in enumerate(model.horizons)},
                                            lambda_turn=0.4, theta_max_deg=55.0, use_uncertainty=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        model.eval()
        val_dpes = []
        with torch.no_grad():
            for batch in val_loader:
                hist = batch["history"].to(device)
                weat = batch["weather"].to(device)
                tgt = batch["targets_future"].to(device)
                preds = model(hist, current=None, features=weat)
                p24 = preds["24"]["position"]
                dpe24 = torch.sqrt((p24[:, 0] - tgt[:, 2, 0])**2 + ((p24[:, 1] - tgt[:, 2, 1]) * 0.95)**2) * 111.0
                val_dpes.extend(dpe24.cpu().numpy().tolist())
        m_dpe24 = float(np.mean(val_dpes)) if val_dpes else float("inf")
        if m_dpe24 < best_val_dpe:
            best_val_dpe = m_dpe24
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [P1-NLL] epoch {epoch:2d}/{epochs} | train {np.mean(losses):.4f} | val 24h DPE {m_dpe24:.1f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def collate_phase1_v2(batch, wmean, wstd):
    histories, currents, weathers, target_future = [], [], [], []
    for item in batch:
        cur = item.get("current_position", torch.zeros(2))
        currents.append(cur)
        h = item.get("track_history", torch.zeros(6, 2))
        if h.shape[-1] > 2:
            h = h[:, :2]
        st = item.get("track_states", torch.zeros(6, 6))
        step_dx = torch.cat([h[:1, 0], h[1:, 0] - h[:-1, 0]])
        step_dy = torch.cat([h[:1, 1], h[1:, 1] - h[:-1, 1]])
        speeds = (st[:, 5] / 20.0) if st.shape[-1] >= 6 else (torch.sqrt(step_dx**2 + step_dy**2) * 5.0)
        headings = torch.deg2rad(st[:, 4]) if st.shape[-1] >= 6 else torch.atan2(step_dy, step_dx + 1e-6)
        winds = ((st[:, 2] - 50.0) / 30.0) if st.shape[-1] >= 6 else torch.zeros(6)
        kin_seq = torch.stack([h[:, 0], h[:, 1], step_dx, step_dy, speeds,
                               torch.sin(headings), torch.cos(headings), winds], dim=-1)
        histories.append(kin_seq)
        raw_w = item.get("weather", torch.zeros(64))
        w = torch.nan_to_num(raw_w, nan=0.0, posinf=0.0, neginf=0.0)
        w = torch.clamp(w, -1500.0, 1500.0)
        w_norm = (w - wmean) / wstd
        lat = float(cur[0].item()) if hasattr(cur[0], "item") else float(cur[0])
        lon = float(cur[1].item()) if hasattr(cur[1], "item") else float(cur[1])
        phi = math.radians(lat)
        f_cor = 2.0 * 7.2921e-5 * math.sin(phi) * 1e4
        beta = (2.0 * 7.2921e-5 * math.cos(phi) / 6.371e6) * 1e11
        u_beta = -0.5 * beta
        v_beta = 0.5 * beta
        recent_u = (h[-1, 1] - h[max(0, len(h) - 3), 1]).item()
        recent_v = (h[-1, 0] - h[max(0, len(h) - 3), 0]).item()
        phys_vec = torch.tensor([lat / 30.0, lon / 100.0, f_cor, beta, u_beta, v_beta, recent_u, recent_v], dtype=torch.float32)
        weathers.append(torch.cat([w_norm, phys_vec], dim=-1))
        target_future.append(item["labels"].get("future_positions", torch.zeros(3, 2)))
    return {
        "history": torch.stack(histories),
        "current": torch.stack(currents),
        "weather": torch.stack(weathers),
        "targets_future": torch.stack(target_future),
    }


def train_track_mse(model, train_loader, val_loader, epochs, device, lr=6e-4, wd=1e-3, seed=None):
    import torch.nn.functional as F_
    if seed is not None:
        torch.manual_seed(seed + 1000)
        np.random.seed(seed + 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    best_val_dpe, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            hist = batch["history"].to(device)
            weat = batch["weather"].to(device)
            tgt = batch["targets_future"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                preds = model(hist, current=None, features=weat)
                loss = sum(F_.smooth_l1_loss(preds[str(h)]["position"], tgt[:, j])
                           for j, h in enumerate(model.horizons))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        model.eval()
        val_dpes = []
        with torch.no_grad():
            for batch in val_loader:
                hist = batch["history"].to(device)
                weat = batch["weather"].to(device)
                tgt = batch["targets_future"].to(device)
                preds = model(hist, current=None, features=weat)
                p24 = preds["24"]["position"]
                dpe24 = torch.sqrt((p24[:, 0] - tgt[:, 2, 0])**2 + ((p24[:, 1] - tgt[:, 2, 1]) * 0.95)**2) * 111.0
                val_dpes.extend(dpe24.cpu().numpy().tolist())
        m_dpe24 = float(np.mean(val_dpes)) if val_dpes else float("inf")
        if m_dpe24 < best_val_dpe:
            best_val_dpe = m_dpe24
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [P1-MSE] epoch {epoch:2d}/{epochs} | train {np.mean(losses):.4f} | val 24h DPE {m_dpe24:.1f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--data-dir", default="data/real")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--pos-weight", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--loss", choices=["mse", "nll"], default="mse")
    parser.add_argument("--ens", type=int, default=1, help="ensemble size for phase 1")
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--aux", action="store_true", help="Phase 2: use auxiliary intensity/track head fusion")
    parser.add_argument("--tag", default="probe")
    args = parser.parse_args()

    cfg = TCConfig.from_yaml("configs/config.yaml")
    set_seed(cfg.seed)
    device = get_device(require_gpu=True)
    print(f"[probe] device={device} phase={args.phase} data={args.data_dir}", flush=True)

    dss = {s: CachedSatelliteDataset(TCDataset(args.data_dir, s, mode="full", augmented=(s == "train")))
           for s in ["train", "val", "test"]}
    dss_tr = {s: TCDataset(args.data_dir, s, mode="track", augmented=False)
              for s in ["train", "val", "test"]}
    dss3 = {s: TCDataset(args.data_dir, s, mode="temporal", augmented=False)
            for s in ["train", "val", "test"]}

    if args.phase == 1:
        loaders = {s: DataLoader(ds, batch_size=args.batch_size, shuffle=(s == "train"),
                                 drop_last=(s == "train"), collate_fn=collate_phase1)
                   for s, ds in dss_tr.items()}
        model = DynamicalTrackModel(input_dim=8, weather_dim=72, hidden_dim=args.hidden,
                                    num_layers=args.layers, num_heads=4, horizons=HORIZONS,
                                    dropout=args.dropout,
                                    uncertainty_estimation=True).to(device)
        t0 = time.time()
        all_preds, targets = [], None
        for seed in range(args.ens):
            model = DynamicalTrackModel(input_dim=8, weather_dim=72, hidden_dim=args.hidden,
                                        num_layers=args.layers, num_heads=4, horizons=HORIZONS,
                                        dropout=args.dropout,
                                        uncertainty_estimation=True).to(device)
            if args.v2:
                wmean_all, wstd_all = [], []
                for item in dss_tr["train"]:
                    r = item.get("weather", torch.zeros(64)).float()
                    wmean_all.append(r.numpy())
                wmean_all = np.stack(wmean_all)
                wmean_a = torch.from_numpy(wmean_all.mean(0)).float()
                wstd_a = torch.from_numpy(wmean_all.std(0) + 1e-4).float()
                def mk_col(wm, ws):
                    return lambda b: collate_phase1_v2(b, wm, ws)
                ltr = DataLoader(dss_tr["train"], batch_size=args.batch_size, shuffle=True,
                                 drop_last=True, collate_fn=mk_col(wmean_a, wstd_a))
                lval = DataLoader(dss_tr["val"], batch_size=args.batch_size, shuffle=False,
                                  collate_fn=mk_col(wmean_a, wstd_a))
                lte = DataLoader(dss_tr["test"], batch_size=args.batch_size, shuffle=False,
                                 collate_fn=mk_col(wmean_a, wstd_a))
                if args.loss == "nll":
                    model = train_track_nll(model, ltr, lval, args.epochs, device,
                                            lr=args.lr, wd=args.wd, seed=seed)
                else:
                    model = train_track_mse(model, ltr, lval, args.epochs, device,
                                            lr=args.lr, wd=args.wd, seed=seed)
                preds, tgts = eval_track_raw(model, lte, device)
            else:
                model = train_track_model(model, loaders["train"], loaders["val"], args.epochs, device,
                                          lr=args.lr, weight_decay=args.wd)
                preds, tgts = eval_track_raw(model, loaders["test"], device)
            if targets is None:
                targets = tgts
            all_preds.append(preds)
            print(f"[ens] seed={seed} done ({len(preds)} samples)", flush=True)

        if args.ens > 1:
            n = len(targets)
            ens_preds = []
            for i in range(n):
                p_dict = {}
                for h in HORIZONS:
                    pos = np.stack([all_preds[k][i]["prediction"][str(h)]["position"] for k in range(args.ens)])
                    p_dict[str(h)] = {"position": pos.mean(axis=0)}
                ens_preds.append({"prediction": p_dict})
            res = TrackMetrics(HORIZONS).compute(ens_preds, targets)
        else:
            res = TrackMetrics(HORIZONS).compute(all_preds[0], targets)
        dt = time.time() - t0
        print(json.dumps({"elapsed_s": round(dt, 1), "dpe_6h": res["dpe_6h_mean"],
                          "dpe_12h": res["dpe_12h_mean"], "dpe_24h": res["dpe_24h_mean"],
                          "hit6": res["dpe_6h_hit_rate"], "hit12": res["dpe_12h_hit_rate"],
                          "hit24": res["dpe_24h_hit_rate"]}, indent=2))

    elif args.phase == 2:
        loaders = {s: DataLoader(ds, batch_size=args.batch_size, shuffle=(s == "train"),
                                 drop_last=(s == "train"), collate_fn=collate_phase2_real)
                   for s, ds in dss.items()}
        model = StageClassifier(backbone="resnet18", n_classes=6, pretrained=False,
                                input_channels=3, embed_dim=256).to(device)
        t0 = time.time()
        if args.aux:
            c2 = make_collate_phase2_aux(256)
            ltr = DataLoader(dss["train"], batch_size=args.batch_size, shuffle=True,
                             drop_last=True, collate_fn=c2)
            lval = DataLoader(dss["val"], batch_size=args.batch_size, shuffle=False, collate_fn=c2)
            lte = DataLoader(dss["test"], batch_size=args.batch_size, shuffle=False, collate_fn=c2)
            model = AuxStageClassifier(model, aux_dim=12, freeze_base=True).to(device)
            model = train_stage_aux(model, ltr, lval, args.epochs, device)
            res = eval_stage_aux(model, lte, device)
        else:
            model = train_stage_model(model, loaders["train"], loaders["val"], args.epochs, device)
            res = eval_stage_model(model, loaders["test"], device)
        dt = time.time() - t0
        print(json.dumps({"elapsed_s": round(dt, 1), **res}, indent=2))

    elif args.phase == 3:
        loaders = {s: DataLoader(ds, batch_size=args.batch_size, shuffle=(s == "train"),
                                 drop_last=(s == "train"), collate_fn=collate_phase3_real)
                   for s, ds in dss3.items()}
        model = TrackIntensityModel(input_dim=2, hidden_dim=args.hidden, num_layers=args.layers,
                                    history_len=6).to(device)
        t0 = time.time()
        model = train_intensity_model(model, loaders["train"], loaders["val"], args.epochs, device, args.pos_weight)
        thr = tune_ri_threshold(model, loaders["val"], device)
        res = eval_intensity_model(model, loaders["test"], device, ri_threshold=thr)
        print(json.dumps({"elapsed_s": round(time.time() - t0, 1), "ri_threshold": thr, **res}, indent=2))

    elif args.phase == 4:
        loaders = {s: DataLoader(ds, batch_size=args.batch_size, shuffle=(s == "train"),
                                 drop_last=(s == "train"), collate_fn=collate_phase4)
                   for s, ds in dss.items()}
        stage_enc = StageClassifier(backbone="resnet18", n_classes=6, pretrained=False,
                                    input_channels=3, embed_dim=256).to(device)
        fusion = SatelliteTrackFusionModel(track_dim=2, weather_dim=64, sat_embed_dim=256,
                                           hidden_dim=args.hidden, horizons=HORIZONS).to(device)
        t0 = time.time()
        fusion = train_fusion_model(fusion, stage_enc, loaders["train"], loaders["val"],
                                    args.epochs, device, sat_lr=1e-4)
        res = eval_fusion_model(fusion, stage_enc, loaders["test"], device)
        print(json.dumps({"elapsed_s": round(time.time() - t0, 1), "dpe_6h": res["dpe_6h_mean"],
                          "dpe_12h": res["dpe_12h_mean"], "dpe_24h": res["dpe_24h_mean"],
                          "hit6": res["dpe_6h_hit_rate"], "hit12": res["dpe_12h_hit_rate"],
                          "hit24": res["dpe_24h_hit_rate"]}, indent=2))


def make_collate_phase2_aux(img_size=256):
    """Collate for Phase 2 with auxiliary intensity/track/environment features.
    Stage is defined by operational wind thresholds; giving the classifier access
    to the observed wind + motion trends recovers stage far better than
    month-matched SST imagery alone."""
    def _collate(batch):
        images, stages, auxs = [], [], []
        for item in batch:
            if "satellite" not in item or "stage" not in item["labels"]:
                continue
            img = item["satellite"].float()
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.shape[-1] != img_size:
                import torch.nn.functional as F_
                img = F_.interpolate(img, size=(img_size, img_size), mode="bilinear",
                                     align_corners=False)
            images.append(img.squeeze(0))
            raw_stage = int(item["labels"]["stage"].item()) if hasattr(item["labels"]["stage"], "item") else int(item["labels"]["stage"])
            stages.append(torch.tensor(max(0, min(5, raw_stage - 2)), dtype=torch.long))
            st = item.get("track_states", torch.zeros(6, 6))
            cur = item.get("current_position", torch.zeros(2))
            wins = st[:, 2] if st.shape[-1] >= 3 else torch.zeros(st.shape[0])
            lat = float(cur[0].item()) if hasattr(cur[0], "item") else float(cur[0])
            lon = float(cur[1].item()) if hasattr(cur[1], "item") else float(cur[1])
            h = item.get("track_history", torch.zeros(6, 2))
            if h.shape[-1] > 2:
                h = h[:, :2]
            motion = (h[-1] - h[0]) if len(h) > 0 else torch.zeros(2)
            aux = torch.tensor([
                wins[-1] / 60.0, wins.max() / 60.0, wins.min() / 60.0,
                (wins[-1] - wins[0]) / 30.0 if len(wins) else 0.0,
                motion[1] * 0.5, motion[0] * 0.5,
                lat / 30.0, lon / 100.0,
                0.0, 0.0, 0.0, 0.0,
            ], dtype=torch.float32)
            auxs.append(aux)
        if not images:
            return {"satellite": torch.empty(0), "aux": torch.empty(0),
                    "stage": torch.empty(0, dtype=torch.long)}
        return {"satellite": torch.stack(images), "aux": torch.stack(auxs),
                "stage": torch.stack(stages)}
    return _collate


class AuxStageClassifier(nn.Module):
    """StageClassifier + MLP over auxiliary intensity/track features, fused at the head."""

    def __init__(self, base, aux_dim=12, n_classes=6, freeze_base=False):
        super().__init__()
        self.base = base
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
        self.aux_proj = nn.Sequential(
            nn.Linear(aux_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(256 + 128, n_classes)

    def forward(self, x, aux=None):
        if aux is None:
            return self.base(x)
        _, emb = self.base(x, return_embedding=True)
        a = self.aux_proj(aux)
        z = torch.cat([emb, a], dim=-1)
        return self.classifier(z)


def train_stage_aux(model, train_loader, val_loader, epochs, device):
    from tc_ai.training.losses import OrdinalStageLoss
    loss_fn = OrdinalStageLoss(n_classes=6, alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_f1, best_state = -1.0, None
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            if batch["satellite"].size(0) <= 1:
                continue
            images = batch["satellite"].to(device)
            aux = batch["aux"].to(device)
            stages = batch["stage"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                logits = model(images, aux)
                loss = loss_fn(logits, stages)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch["satellite"].size(0) <= 1:
                    continue
                logits = model(batch["satellite"].to(device), batch["aux"].to(device))
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                val_true.extend(batch["stage"].numpy().tolist())
        from sklearn.metrics import f1_score
        vf1 = float(f1_score(val_true, val_preds, average="macro", zero_division=0))
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [P2-Aux] epoch {epoch:2d}/{epochs} | train {np.mean(losses):.4f} | val macro-F1 {vf1:.3f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_stage_aux(model, loader, device):
    from scripts.eval_real_metrics import compute_stage_metrics
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in loader:
            if batch["satellite"].size(0) <= 1:
                continue
            logits = model(batch["satellite"].to(device), batch["aux"].to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            true.extend(batch["stage"].numpy().tolist())
    return compute_stage_metrics(true, preds)


if __name__ == "__main__":
    main()