import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import torch.serialization
import time
from tqdm import tqdm

torch.serialization.add_safe_globals([StandardScaler])
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["SimHei", "Times New Roman"]
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 1. Config Class (Aligned Layer Freezing + No-IDM Testing) --------------------------
class Config:
    # Core input/output config (variable-length compatible)
    input_dim = 4  # 4-dim input: spacing, follower speed, relative speed, leader speed
    output_dim = 1  # Single-step acceleration output
    seq_len = 250
    dt = 0.04

    # Variable-length config (for HighD)
    # HighD
    # min_total_len = 375  # Minimum total sample length (history 100 + min prediction 50 + 1)
    # min_pred_len = 50  # Minimum prediction steps (acceleration sequence length)
    # max_pred_len = 125  # Maximum prediction steps (acceleration sequence length)

    # 30Hz HighD
    min_total_len = 750  # Minimum total sample length (history 100 + min prediction 200 + 1)
    min_pred_len = 200  # Minimum prediction steps (acceleration sequence length)
    max_pred_len = 200  # Maximum prediction steps (acceleration sequence length)
    step_pred = 25  # Recurrent prediction window size

    # Domain config (strictly aligned with training code)
    domains = ["ngsim", "lyft", "waymo", "spmd1", "spmd2"]
    source_domains = ["ngsim", "spmd2"]
    target_domains = ["spmd1", "lyft", "waymo"]

    # Test domain config
    test_domains = ["highd", "30highd"]  # Support both highd and 30highd
    # test_domains = ["highd"]  # Test highd only
    test_domains = ["30highd"]  # Test 30highd only

    # Dataset acceleration limits (training + test + fallback defaults)
    dataset_acc_limits = {
        # Training domain limits (aligned with training code)
        # "ngsim": {"min": -2.7, "max": 2.6},
        # "spmd1": {"min": -3.5, "max": 2.0},
        # "spmd2": {"min": -3.5, "max": 2.0},
        # "waymo": {"min": -4.0, "max": 3.0},
        # "lyft": {"min": -3.0, "max": 2.9},

        # Test domain limits
        # "highd": {"min": -2.9, "max": 1.57},
        "30highd": {"min": -2.7, "max": 1.6},

        # Fallback default values
        "default": {"min": -4.0, "max": 3.0}
    }

    # No-IDM config (fully aligned with training code)
    idm_params = {}
    idm_weight = 0.0  # IDM weight set to 0 → IDM fusion disabled

    # Model parameters (aligned with training layer-freezing config)
    hidden_size = 64
    enc_layers = 2
    dec_layers = 2
    dropout = 0.1
    meta_alpha = 0.01
    freeze_layers = 1  # Number of encoder layers frozen

    # Test config
    batch_size = 1  # Variable-length samples require batch_size=1
    collision_threshold = 0.0  # Collision threshold

    # Paths for layer-frozen + no-IDM models
    save_dir = "output/our_method_ablation/only_layer_freeze"
    result_dir = os.path.join(save_dir, "test_result_layer_freeze_highd")
    data_paths = {
        # "highd": {"test": "dataset/data/HighD_test_data.npy"},
        "30highd": {"test": "dataset/data/30_HighD_test_data.npy"},
    }


config = Config()
# Validate minimum length constraints
assert config.min_total_len - config.seq_len >= config.min_pred_len + 1, \
    f"min_total_len {config.min_total_len} - seq_len {config.seq_len} must be ≥ min_pred_len {config.min_pred_len} + 1"
os.makedirs(config.result_dir, exist_ok=True)


# -------------------------- 2. Utility Functions (Aligned Physics Integration Logic) --------------------------
def integrate_speed(acc_seq, speed_init, dt, pred_len):
    """Integrate speed from acceleration (variable-length, aligned with training logic)"""
    speed_seq = torch.zeros((1, pred_len), device=acc_seq.device, dtype=acc_seq.dtype)
    speed_seq[:, 0] = speed_init
    for t in range(1, pred_len):
        speed_seq[:, t] = speed_seq[:, t - 1] + acc_seq[:, t - 1] * dt
        speed_seq[:, t] = torch.clamp(speed_seq[:, t], min=0.1)
    return speed_seq


def calculate_jerk(acc_seq, dt):
    """Compute absolute jerk (variable-length compatible)"""
    if acc_seq.shape[1] <= 1:
        return torch.zeros((1, 0), device=acc_seq.device)
    jerk = (acc_seq[:, 1:] - acc_seq[:, :-1]) / dt
    abs_jerk = torch.abs(jerk)
    return abs_jerk


def calculate_ttc(spacing_seq, fv_speed_seq, lv_speed_seq, pred_len):
    """Compute Time-To-Collision (TTC, variable-length compatible)"""
    delta_speed = fv_speed_seq - lv_speed_seq
    valid_mask = delta_speed > 1e-3
    ttc = torch.zeros((1, pred_len), device=spacing_seq.device)
    ttc[valid_mask] = spacing_seq[valid_mask] / delta_speed[valid_mask]
    ttc[~valid_mask] = float('inf')

    valid_ttc = ttc[0][valid_mask[0]]
    if len(valid_ttc) > 0:
        min_ttc = torch.min(valid_ttc).item()
    else:
        min_ttc = float('inf')
    return min_ttc


# -------------------------- 3. No-IDM Spacing Integration (Fully Aligned) --------------------------
def calculate_spacing_from_acc(pred_acc, ego_speed_init, front_speed_seq, init_spacing, dt, domain, pred_len):
    """Pure physics-based spacing integration (no IDM, fully aligned with training logic)"""
    device = pred_acc.device
    ego_speed = torch.zeros((1, pred_len), device=device, dtype=pred_acc.dtype)
    spacing_pred = torch.zeros((1, pred_len), device=device, dtype=pred_acc.dtype)

    ego_speed[:, 0] = ego_speed_init
    spacing_pred[:, 0] = init_spacing

    dt_tensor = torch.tensor(dt, dtype=pred_acc.dtype, device=device)
    s0 = torch.tensor(0, dtype=pred_acc.dtype, device=device)

    for t in range(1, pred_len):
        ego_speed_t = ego_speed[:, t - 1] + pred_acc[:, t - 1] * dt_tensor
        ego_speed_t = torch.clamp(ego_speed_t, min=0.1)
        ego_speed[:, t] = ego_speed_t

        front_speed_t = front_speed_seq[:, t - 1] if t - 1 < front_speed_seq.shape[1] else front_speed_seq[:, -1]
        spacing_t = spacing_pred[:, t - 1] + (front_speed_t - ego_speed_t) * dt_tensor
        spacing_t = torch.clamp(spacing_t, min=s0)
        spacing_pred[:, t] = spacing_t

    return spacing_pred


# -------------------------- 4. Test Dataset (Aligned with Layer-Frozen Model Input Format) --------------------------
class MultiDomainTestDataset(Dataset):
    def __init__(self, domain, scaler):
        self.domain = domain
        self.scaler = scaler
        self.data_path = config.data_paths[domain]["test"]
        assert os.path.exists(self.data_path), f"[{domain}] Test dataset file not found: {self.data_path}"
        self.raw_data = np.load(self.data_path, allow_pickle=True)
        self.features, self.labels, self.meta, self.raw_spacing, self.raw_fv_speed, self.raw_lv_speed, self.pred_lens = self._process_data()

    def _process_data(self):
        """Core fix: align feature processing with training code"""
        features = []
        labels = []
        meta = []
        raw_spacing = []
        raw_fv_speed = []
        raw_lv_speed = []
        pred_lens = []

        for sample in self.raw_data:
            try:
                spacing = np.array(sample[0], dtype=np.float32)
                fv_speed = np.array(sample[1], dtype=np.float32)
                rel_speed = np.array(sample[2], dtype=np.float32)
                lv_speed = np.array(sample[3], dtype=np.float32)

                total_len = min(len(spacing), len(fv_speed), len(rel_speed), len(lv_speed))
                if total_len < config.min_total_len:
                    continue

                acc = (fv_speed[1:total_len] - fv_speed[:total_len - 1]) / config.dt
                if len(acc) < config.min_pred_len:
                    continue

                if np.any(fv_speed < 0) or np.any(lv_speed < 0) or np.any(spacing <= 0.3):
                    continue

                input_end = config.seq_len
                label_start = config.seq_len
                max_label_end = min(len(acc), config.seq_len + config.max_pred_len)
                label_end = max_label_end
                pred_len = label_end - label_start

                if pred_len < config.min_pred_len or pred_len > config.max_pred_len:
                    continue

                input_spacing = spacing[:input_end]
                input_fv = fv_speed[:input_end]
                input_rel = rel_speed[:input_end]
                input_lv = lv_speed[:input_end]
                feat = np.stack([input_spacing, input_fv, input_rel, input_lv], axis=0)

                label_acc = acc[label_start:label_end]
                if len(label_acc) != pred_len or len(feat[0]) != config.seq_len:
                    continue

                raw_spacing_window = spacing[label_start:label_end]
                raw_fv_speed_window = fv_speed[label_start:label_end]
                raw_lv_speed_window = lv_speed[label_start:label_end]
                if len(raw_spacing_window) != pred_len or len(raw_fv_speed_window) != pred_len:
                    continue

                speed_mean = np.mean(fv_speed[:input_end])
                speed_std = np.std(fv_speed[:input_end])
                max_acc_window = np.max(np.abs(acc[:label_end]))
                spacing_mean = np.mean(spacing[:input_end])
                rel_speed_mean = np.mean(rel_speed[:input_end])
                scene_feat = np.array([speed_mean, speed_std, max_acc_window, spacing_mean, rel_speed_mean],
                                      dtype=np.float32)

                input_reshaped = feat.transpose(1, 0).reshape(-1, config.input_dim)
                input_scaled = self.scaler.transform(input_reshaped)
                feat_scaled = input_scaled.reshape(config.seq_len, config.input_dim).transpose(1, 0)

                features.append(feat_scaled)
                labels.append(label_acc)
                meta.append(scene_feat)
                raw_spacing.append(raw_spacing_window)
                raw_fv_speed.append(raw_fv_speed_window)
                raw_lv_speed.append(raw_lv_speed_window)
                pred_lens.append(pred_len)

            except Exception as e:
                print(f"[{self.domain}] Error processing sample: {str(e)[:50]}...")
                continue

        features = np.array(features) if features else np.array([])
        meta = np.array(meta) if meta else np.array([])

        if pred_lens:
            print(f"[{self.domain}] Preprocessing complete | Valid samples: {len(features)}")
            print(
                f"[{self.domain}] Prediction length range: {min(pred_lens)}~{max(pred_lens)} "
                f"(required {config.min_pred_len}~{config.max_pred_len})"
            )

        return features, labels, meta, raw_spacing, raw_fv_speed, raw_lv_speed, pred_lens

    def __getitem__(self, idx):
        pred_len = self.pred_lens[idx]
        return {
            "x": torch.FloatTensor(self.features[idx]),
            "y": torch.FloatTensor(self.labels[idx]),
            "meta": torch.FloatTensor(self.meta[idx]),
            "spacing": torch.FloatTensor(self.raw_spacing[idx]),
            "fv_speed": torch.FloatTensor(self.raw_fv_speed[idx]),
            "lv_speed": torch.FloatTensor(self.raw_lv_speed[idx]),
            "domain": self.domain,
            "ego_speed_init": self.raw_fv_speed[idx][0] if pred_len > 0 else 0.0,
            "spacing_init": self.raw_spacing[idx][0] if pred_len > 0 else 0.0,
            "pred_len": pred_len,
            "total_len": config.seq_len + pred_len + 1
        }

    def __len__(self):
        return len(self.features) if hasattr(self, 'features') else 0


# -------------------------- 5. Model Definition (Strictly Aligned with Layer-Freezed + No-IDM Training Code) --------------------------
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False
        )

    def forward(self, x):
        x = x.permute(2, 0, 1)
        out, (hn, cn) = self.lstm(x)
        return hn, cn


class Seq2SeqDecoder(nn.Module):
    def __init__(self, hidden_size, output_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False
        )
        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, dec_input, hidden, cell):
        dec_input = dec_input.unsqueeze(0)
        out, (hidden, cell) = self.lstm(dec_input, (hidden, cell))
        pred = self.fc_out(out.squeeze(0))
        return pred, hidden, cell


class EnhancedDynamicWeight(nn.Module):
    def __init__(self, scene_feat_dim=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(scene_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.gate = nn.Sequential(
            nn.Linear(scene_feat_dim, 2),
            nn.Sigmoid()
        )

    def forward(self, scene_features):
        raw_weights = self.mlp(scene_features)
        gate_values = self.gate(scene_features)
        weighted_weights = raw_weights * gate_values
        return torch.softmax(weighted_weights, dim=1)


class Seq2SeqFollowingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = FeatureEncoder(
            config.input_dim, config.hidden_size, config.enc_layers, config.dropout
        )
        self.decoder = Seq2SeqDecoder(
            config.hidden_size, config.output_dim, config.dec_layers, config.dropout
        )
        self.meta_adapter = nn.Linear(config.hidden_size, config.output_dim)
        self.dynamic_weight = EnhancedDynamicWeight(scene_feat_dim=5)

        self.dataset_acc_min = {
            d: torch.tensor(v["min"], dtype=torch.float32).to(self.device)
            for d, v in config.dataset_acc_limits.items()
        }
        self.dataset_acc_max = {
            d: torch.tensor(v["max"], dtype=torch.float32).to(self.device)
            for d, v in config.dataset_acc_limits.items()
        }
        self.collision_threshold = config.collision_threshold

        for i in range(config.freeze_layers):
            for param in self.encoder.lstm.all_weights[i]:
                param.requires_grad = False

    def init_decoder_input(self, batch_size):
        return torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)

    def step_predict(self, enc_hidden, enc_cell, meta_feat, current_domain, step_num,
                     ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        batch_size = enc_hidden.shape[1]
        dec_input = self.init_decoder_input(batch_size)
        hidden, cell = enc_hidden, enc_cell
        step_preds = []

        acc_min = self.dataset_acc_min.get(current_domain, self.dataset_acc_min["default"])
        acc_max = self.dataset_acc_max.get(current_domain, self.dataset_acc_max["default"])

        for t in range(step_num):
            base_pred, hidden, cell = self.decoder(dec_input, hidden, cell)
            meta_pred = base_pred + self.meta_adapter(hidden[-1])
            weights = self.dynamic_weight(meta_feat)
            fusion_pred = weights[:, 0:1] * base_pred + weights[:, 1:2] * meta_pred

            fusion_pred = torch.clamp(fusion_pred, min=acc_min, max=acc_max)
            step_preds.append(fusion_pred)
            dec_input = fusion_pred

        return torch.cat(step_preds, dim=1), hidden, cell

    def forward(self, x, meta, current_domain, pred_len,
                ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        batch_size = x.shape[0]
        total_preds = []
        enc_hidden, enc_cell = self.encoder(x)
        remaining_pred = pred_len
        current_hidden, current_cell = enc_hidden, enc_cell
        current_x = x

        while remaining_pred > 0:
            step_num = min(self.config.step_pred, remaining_pred)
            step_pred, current_hidden, current_cell = self.step_predict(
                current_hidden, current_cell, meta, current_domain, step_num,
                ego_speed_seq, front_speed_seq, spacing_seq
            )
            total_preds.append(step_pred)
            new_x = self._build_new_input(current_x, step_pred, current_domain, front_speed_seq, step_num)
            current_x = new_x
            remaining_pred -= step_num

        total_preds = torch.cat(total_preds, dim=1)[:, :pred_len]
        if total_preds.shape[1] != pred_len:
            if total_preds.shape[1] > pred_len:
                total_preds = total_preds[:, :pred_len]
            else:
                pad_len = pred_len - total_preds.shape[1]
                padding = torch.zeros((batch_size, pad_len), device=total_preds.device, dtype=total_preds.dtype)
                total_preds = torch.cat([total_preds, padding], dim=1)

        return total_preds, None

    def _build_new_input(self, current_x, step_pred, current_domain, front_speed_seq=None, step_num=None):
        batch_size = current_x.shape[0]
        step_num = step_num if step_num else self.config.step_pred

        keep_len = self.config.seq_len - step_num
        x_slice = current_x[:, :, step_num:] if keep_len > 0 else current_x[:, :, -1:].repeat(1, 1, 1)
        x_slice = x_slice[:, :, :keep_len]
        if x_slice.shape[2] < keep_len:
            pad_len = keep_len - x_slice.shape[2]
            x_slice = torch.cat([x_slice, x_slice[:, :, -1:].repeat(1, 1, pad_len)], dim=2)

        pred_feat = torch.zeros((batch_size, self.config.input_dim, step_num),
                                device=current_x.device, dtype=current_x.dtype)

        ego_speed_init = current_x[:, 1, -1]
        spacing_init = current_x[:, 0, -1]
        dt_tensor = torch.tensor(self.config.dt, dtype=current_x.dtype, device=current_x.device)
        s0 = 0

        ego_speed = ego_speed_init
        spacing = spacing_init
        for t in range(step_num):
            ego_speed = ego_speed + step_pred[:, t] * dt_tensor
            ego_speed = torch.clamp(ego_speed, min=0.1)

            if front_speed_seq is not None:
                front_speed = front_speed_seq[:, t] if t < front_speed_seq.shape[1] else front_speed_seq[:, -1]
            else:
                front_speed = current_x[:, 3, -1]

            spacing = spacing + (front_speed - ego_speed) * dt_tensor
            spacing = torch.clamp(spacing, min=s0)

            pred_feat[:, 0, t] = spacing
            pred_feat[:, 1, t] = ego_speed
            pred_feat[:, 2, t] = front_speed - ego_speed
            pred_feat[:, 3, t] = front_speed

        new_x = torch.cat([x_slice, pred_feat], dim=2)[:, :, :self.config.seq_len]
        return new_x


# -------------------------- 6. Evaluation Functions (Layer-Frozen Model Compatible) --------------------------
def evaluate_model(domain, model, dataloader, device):
    model.eval()
    metrics = {
        "mse_acc": 0.0,
        "mae_speed": 0.0,
        "mae_abs_jerk": 0.0,
        "avg_min_ttc": 0.0,
        "mse_spacing": 0.0,
        "mae_acc": 0.0,
        "mae_spacing": 0.0,
        "collision_count": 0,
        "collision_rate": 0.0,
        "valid_samples": 0,
        "total_pred_steps": 0,
        "total_ttc_samples": 0,
        "pred_len_stats": []
    }

    all_min_ttc = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating {domain.upper()} (Layer Freeze + No IDM)")
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            y_true = batch["y"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            spacing_gt = batch["spacing"].to(device, non_blocking=True)
            fv_speed_gt = batch["fv_speed"].to(device, non_blocking=True)
            lv_speed_gt = batch["lv_speed"].to(device, non_blocking=True)
            current_domain = batch["domain"][0]
            ego_speed_init = batch["ego_speed_init"].to(device, non_blocking=True)
            spacing_init = batch["spacing_init"].to(device, non_blocking=True)
            pred_len = batch["pred_len"].item()

            pred_acc, _ = model(
                x, meta, current_domain, pred_len=pred_len,
                ego_speed_seq=fv_speed_gt, front_speed_seq=lv_speed_gt, spacing_seq=spacing_gt
            )

            if pred_acc.shape[1] != y_true.shape[1]:
                min_len = min(pred_acc.shape[1], y_true.shape[1])
                pred_acc = pred_acc[:, :min_len]
                y_true = y_true[:, :min_len]
                spacing_gt = spacing_gt[:, :min_len] if spacing_gt.shape[1] > min_len else spacing_gt
                fv_speed_gt = fv_speed_gt[:, :min_len] if fv_speed_gt.shape[1] > min_len else fv_speed_gt
                lv_speed_gt = lv_speed_gt[:, :min_len] if lv_speed_gt.shape[1] > min_len else lv_speed_gt
                pred_len = min_len

            spacing_pred = calculate_spacing_from_acc(
                pred_acc, ego_speed_init, lv_speed_gt, spacing_init, config.dt, current_domain, pred_len
            )

            speed_pred = integrate_speed(pred_acc, ego_speed_init, config.dt, pred_len)

            mse_acc = nn.MSELoss()(pred_acc, y_true)
            mae_acc = nn.L1Loss()(pred_acc, y_true)

            if spacing_pred.shape[1] != spacing_gt.shape[1]:
                min_spacing_len = min(spacing_pred.shape[1], spacing_gt.shape[1])
                spacing_pred = spacing_pred[:, :min_spacing_len]
                spacing_gt = spacing_gt[:, :min_spacing_len]
            mse_spacing = nn.MSELoss()(spacing_pred, spacing_gt)
            mae_spacing = nn.L1Loss()(spacing_pred, spacing_gt)

            if speed_pred.shape[1] != fv_speed_gt.shape[1]:
                min_speed_len = min(speed_pred.shape[1], fv_speed_gt.shape[1])
                speed_pred = speed_pred[:, :min_speed_len]
                fv_speed_gt = fv_speed_gt[:, :min_speed_len]
            mae_speed = nn.L1Loss()(speed_pred, fv_speed_gt)

            jerk_pred = calculate_jerk(pred_acc, config.dt)
            jerk_true = calculate_jerk(y_true, config.dt)
            min_jerk_len = min(jerk_pred.shape[1], jerk_true.shape[1]) if jerk_pred.shape[1] > 0 else 0
            mae_abs_jerk = torch.tensor(0.0, device=device)
            if min_jerk_len > 0:
                mae_abs_jerk = nn.L1Loss()(jerk_pred[:, :min_jerk_len], jerk_true[:, :min_jerk_len])

            min_ttc = calculate_ttc(spacing_pred, speed_pred, lv_speed_gt, pred_len)
            if min_ttc != float('inf'):
                all_min_ttc.append(min_ttc)
                metrics["total_ttc_samples"] += 1

            collision = torch.any(spacing_pred <= model.collision_threshold)
            collision_count = 1 if collision else 0

            metrics["mse_acc"] += mse_acc.item() * pred_len
            metrics["mae_acc"] += mae_acc.item() * pred_len
            metrics["mse_spacing"] += mse_spacing.item() * pred_len
            metrics["mae_spacing"] += mae_spacing.item() * pred_len
            metrics["mae_speed"] += mae_speed.item() * pred_len
            if min_jerk_len > 0:
                metrics["mae_abs_jerk"] += mae_abs_jerk.item() * (pred_len - 1)
            metrics["collision_count"] += collision_count
            metrics["valid_samples"] += 1
            metrics["total_pred_steps"] += pred_len
            metrics["pred_len_stats"].append(pred_len)

            pbar.set_postfix({
                "PredLen": pred_len,
                "MSE_acc": f"{mse_acc.item():.6f}",
                "MAE_speed": f"{mae_speed.item():.6f}",
                "Collision": collision_count,
                "Mode": "Layer Freeze + No IDM"
            })

    if metrics["total_pred_steps"] > 0:
        metrics["mse_acc"] /= metrics["total_pred_steps"]
        metrics["mae_acc"] /= metrics["total_pred_steps"]
        metrics["mse_spacing"] /= metrics["total_pred_steps"]
        metrics["mae_spacing"] /= metrics["total_pred_steps"]
        metrics["mae_speed"] /= metrics["total_pred_steps"]
        if metrics["total_pred_steps"] - len(metrics["pred_len_stats"]) > 0:
            metrics["mae_abs_jerk"] /= (metrics["total_pred_steps"] - len(metrics["pred_len_stats"]))

    if metrics["valid_samples"] > 0:
        metrics["collision_rate"] = (metrics["collision_count"] / metrics["valid_samples"]) * 100
        metrics["avg_min_ttc"] = np.mean(all_min_ttc) if len(all_min_ttc) > 0 else float('inf')

    if metrics["pred_len_stats"]:
        print(f"\n[{domain}] Prediction length statistics (Layer Freeze + No IDM):")
        print(
            f"  Min: {min(metrics['pred_len_stats'])} | "
            f"Max: {max(metrics['pred_len_stats'])} | "
            f"Mean: {np.mean(metrics['pred_len_stats']):.1f}"
        )

    return metrics


# -------------------------- 7. Main Test Pipeline (Layer-Frozen + No-IDM Model) --------------------------
def main():
    print("=" * 80)
    print("Testing Layer-Frozen + No-IDM Transfer Learning Model on HighD Dataset")
    print(f"Training domains: {config.source_domains} | Test domains: {config.test_domains}")
    print(f"Frozen layers: {config.freeze_layers} | IDM fusion: Disabled")
    print("=" * 80)

    model_path = os.path.join(config.save_dir, "pretrain_best.pth")
    assert os.path.exists(model_path), f"Layer-frozen model file not found: {model_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Loading layer-frozen model to device: {device}")

    model = Seq2SeqFollowingModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    scaler = checkpoint["scaler"]
    print(f"✅ Successfully loaded layer-frozen model (best loss: {checkpoint.get('best_loss', 'N/A')})")

    all_results = {}
    result_df = pd.DataFrame(columns=[
        "Dataset", "Valid Samples", "Total Prediction Steps", "Avg Prediction Length",
        "Acc MSE", "Acc MAE", "Speed MAE (m/s)", "Abs Jerk MAE (m/s³)",
        "Spacing MSE (m²)", "Spacing MAE (m)", "Min Avg TTC (s)",
        "Collision Count", "Collision Rate (%)", "Model Version"
    ])

    for domain in config.test_domains:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {domain.upper()} Dataset (Layer Freeze + No IDM)")
        print(f"{'=' * 60}")

        try:
            test_dataset = MultiDomainTestDataset(domain, scaler)
            if len(test_dataset) == 0:
                print(f"⚠️ {domain.upper()} has no valid samples, skipping")
                continue

            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0
            )

            metrics = evaluate_model(domain, model, test_loader, device)

            avg_pred_len = np.mean(metrics["pred_len_stats"]) if metrics["pred_len_stats"] else 0

            all_results[domain] = metrics
            result_df.loc[len(result_df)] = [
                domain.upper(),
                metrics["valid_samples"],
                metrics["total_pred_steps"],
                round(avg_pred_len, 1),
                round(metrics["mse_acc"], 6),
                round(metrics["mae_acc"], 6),
                round(metrics["mae_speed"], 6),
                round(metrics["mae_abs_jerk"], 6),
                round(metrics["mse_spacing"], 3),
                round(metrics["mae_spacing"], 3),
                round(metrics["avg_min_ttc"], 3) if metrics["avg_min_ttc"] != float('inf') else "N/A",
                metrics["collision_count"],
                round(metrics["collision_rate"], 2),
                "Layer Freeze + No IDM"
            ]

            print(f"\n📊 {domain.upper()} Test Results (Layer Freeze + No IDM):")
            print(
                f"   Valid samples: {metrics['valid_samples']} | "
                f"Total prediction steps: {metrics['total_pred_steps']} | "
                f"Avg prediction length: {avg_pred_len:.1f}"
            )
            print(f"   Acc MSE: {metrics['mse_acc']:.6f} | Acc MAE: {metrics['mae_acc']:.6f}")
            print(f"   Speed MAE: {metrics['mae_speed']:.6f} m/s | Abs Jerk MAE: {metrics['mae_abs_jerk']:.6f} m/s³")
            print(f"   Spacing MSE: {metrics['mse_spacing']:.3f} m² | Spacing MAE: {metrics['mae_spacing']:.3f} m")
            if metrics["avg_min_ttc"] != float('inf'):
                print(f"   Min Avg TTC: {metrics['avg_min_ttc']:.3f} s")
            else:
                print("   Min Avg TTC: N/A")
            print(f"   Collisions: {metrics['collision_count']} | Collision Rate: {metrics['collision_rate']:.2f}%")

        except Exception as e:
            print(f"❌ {domain.upper()} evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if not result_df.empty:
        result_excel_path = os.path.join(config.result_dir, "test_results_highd_layer_freeze_no_idm.xlsx")
        result_df.to_excel(result_excel_path, index=False, engine="openpyxl")
        print(f"\n✅ Test results saved to: {result_excel_path}")

        print(f"\n{'=' * 80}")
        print("Layer-Frozen + No-IDM Model Test Summary (HighD Dataset)")
        print(f"{'=' * 80}")
        print(result_df.to_string(index=False))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Layer-Frozen + No-IDM Transfer Learning Model Performance (HighD Dataset)",
                     fontsize=16, fontweight="bold")
        domains = result_df["Dataset"].tolist()

        axes[0, 0].bar(domains, result_df["Acc MSE"], color="#2E86AB", alpha=0.8)
        axes[0, 0].set_title("Acceleration MSE", fontweight="bold")
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        axes[0, 1].bar(domains, result_df["Speed MAE (m/s)"], color="#F18F01", alpha=0.8)
        axes[0, 1].set_title("Speed MAE (m/s)", fontweight="bold")
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        axes[0, 2].bar(domains, result_df["Abs Jerk MAE (m/s³)"], color="#A23B72", alpha=0.8)
        axes[0, 2].set_title("Abs Jerk MAE (m/s³)", fontweight="bold")
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3, axis='y')

        axes[1, 0].bar(domains, result_df["Spacing MSE (m²)"], color="#C73E1D", alpha=0.8)
        axes[1, 0].set_title("Spacing MSE (m²)", fontweight="bold")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        axes[1, 1].bar(domains, result_df["Collision Rate (%)"], color="#6A994E", alpha=0.8)
        axes[1, 1].set_title("Collision Rate (%)", fontweight="bold")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        axes[1, 2].bar(domains, result_df["Avg Prediction Length"], color="#F77F00", alpha=0.8)
        axes[1, 2].set_title("Avg Prediction Length (steps)", fontweight="bold")
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig_path = os.path.join(config.result_dir, "test_performance_highd_layer_freeze_no_idm.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Performance visualization saved to: {fig_path}")

    print(f"\n{'=' * 80}")
    print("HighD Dataset Layer-Frozen + No-IDM Variable-Length Sample Testing Complete!")
    print(f"Results directory: {config.result_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()