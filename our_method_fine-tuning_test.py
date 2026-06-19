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


# -------------------------- 1. Configuration Class (For fine-tuning only, no IDM training) --------------------------
class Config:
    # Core input & output settings (Consistent with fine-tuning only, no IDM training code)
    input_dim = 4  # 4-dimensional input: gap distance, following vehicle speed, relative speed, leading vehicle speed
    output_dim = 1  # Single-step acceleration output
    seq_len = 100  # Fixed historical sequence length (100 previous steps)
    dt = 0.1  # Fixed 10Hz time interval (0.1 seconds per step)

    # Variable-length settings (Core modification: support 50~80-step prediction)
    min_total_len = 151  # Minimum total sample length (100 history + min 50 prediction + 1)
    min_pred_len = 50  # Minimum prediction steps
    max_pred_len = 80  # Maximum prediction steps
    step_pred = 10  # Recursive prediction step size
    slide_step = 80  # Dataset sliding window step (Consistent with training code)

    # Domain settings (Consistent with fine-tuning only, no IDM training code)
    domains = ["ngsim", "lyft", "waymo", "spmd1", "spmd2"]
    source_domains = ["ngsim", "spmd2"]  # Source domains
    target_domains = ["spmd1", "lyft", "waymo"]  # Target domains

    # Acceleration constraints for each dataset
    dataset_acc_limits = {
        "ngsim": {"min": -2.7, "max": 2.6},
        "spmd1": {"min": -3.5, "max": 2.0},
        "spmd2": {"min": -3.5, "max": 2.0},
        "waymo": {"min": -4.0, "max": 3.0},
        "lyft": {"min": -3.0, "max": 2.9}
    }

    # Model hyperparameters (Fully consistent with training code)
    hidden_size = 64
    enc_layers = 2
    dec_layers = 2
    dropout = 0.1
    freeze_layers = 0  # Full layers fine-tuning without freezing

    # Test settings
    batch_size = 1  # Compatible with variable-length sequences
    collision_threshold = 0.0  # Threshold to judge collision

    # Path settings (For fine-tuning only, no IDM models)
    save_dir = "output/our_method_ablation/only_finetune_no_idm"  # Training directory for fine-tuning only without IDM
    result_dir = os.path.join(save_dir, "test_result_finetune")  # Directory for variable-length test results
    data_paths = {
        "ngsim": {"test": "dataset/data/NGSIM_I_80_test_data.npy"},
        "waymo": {"test": "dataset/data/Waymo_test_data.npy"},
        "lyft": {"test": "dataset/data/Lyft_test_data.npy"},
        "spmd1": {"test": "dataset/data/SPMD1_test_data.npy"},
        "spmd2": {"test": "dataset/data/SPMD2_test_data.npy"}
    }


config = Config()
# Verify minimum length constraint
assert config.min_total_len - config.seq_len >= config.min_pred_len + 1, \
    f"Minimum total length {config.min_total_len} minus history length {config.seq_len} should be no less than minimum prediction length {config.min_pred_len} + 1"
os.makedirs(config.result_dir, exist_ok=True)


# -------------------------- 2. Utility Functions (Consistent with training code) --------------------------
def calculate_spacing_from_acc(pred_acc, ego_speed_init, front_speed_seq, init_spacing, dt, domain):
    """Calculate gap distance purely via physical integration, no IDM constraint"""
    batch_size, seq_len = pred_acc.shape
    device = pred_acc.device
    ego_speed = torch.zeros((batch_size, seq_len), device=device, dtype=pred_acc.dtype)
    spacing_pred = torch.zeros((batch_size, seq_len), device=device, dtype=pred_acc.dtype)

    ego_speed[:, 0] = ego_speed_init
    spacing_pred[:, 0] = init_spacing
    dt_tensor = torch.tensor(dt, dtype=pred_acc.dtype, device=device)
    s0 = torch.tensor(0, dtype=pred_acc.dtype, device=device)  # Minimum safe gap distance

    # Adapt to mismatched length of leading vehicle speed sequence
    front_speed_seq = front_speed_seq[:, :seq_len] if front_speed_seq.shape[1] > seq_len else front_speed_seq
    if front_speed_seq.shape[1] < seq_len:
        pad_len = seq_len - front_speed_seq.shape[1]
        front_speed_seq = torch.cat([front_speed_seq, front_speed_seq[:, -1:].repeat(1, pad_len)], dim=1)

    for t in range(1, seq_len):
        # Velocity integration rule
        ego_speed_t = ego_speed[:, t - 1] + pred_acc[:, t - 1] * dt_tensor
        ego_speed_t = torch.clamp(ego_speed_t, min=0.1)
        ego_speed[:, t] = ego_speed_t

        # Leading vehicle speed at current timestep
        front_speed_t = front_speed_seq[:, t - 1]

        # Gap distance integration without IDM
        spacing_t = spacing_pred[:, t - 1] + (front_speed_t - ego_speed_t) * dt_tensor
        spacing_t = torch.clamp(spacing_t, min=s0)
        spacing_pred[:, t] = spacing_t

    return spacing_pred


def integrate_speed(acc_seq, speed_init, dt, pred_len):
    """Integrate acceleration to get speed sequence, support variable prediction length"""
    speed_seq = torch.zeros((1, pred_len), device=acc_seq.device, dtype=acc_seq.dtype)
    speed_seq[:, 0] = speed_init
    for t in range(1, pred_len):
        speed_seq[:, t] = speed_seq[:, t - 1] + acc_seq[:, t - 1] * dt
        speed_seq[:, t] = torch.clamp(speed_seq[:, t], min=0.1)  # Non-negative speed constraint
    return speed_seq


def calculate_jerk(acc_seq, dt):
    """Compute absolute jerk metric for variable-length sequences"""
    if acc_seq.shape[1] <= 1:
        return torch.zeros((1, 0), device=acc_seq.device)
    jerk = (acc_seq[:, 1:] - acc_seq[:, :-1]) / dt
    abs_jerk = torch.abs(jerk)
    return abs_jerk


def calculate_ttc(spacing_seq, fv_speed_seq, lv_speed_seq, pred_len):
    """Compute minimum Time-To-Collision for variable-length prediction"""
    delta_speed = fv_speed_seq - lv_speed_seq
    valid_mask = delta_speed > 1e-3  # Avoid division by zero
    ttc = torch.zeros((1, pred_len), device=spacing_seq.device)
    ttc[valid_mask] = spacing_seq[valid_mask] / delta_speed[valid_mask]
    ttc[~valid_mask] = float('inf')

    # Extract minimum valid TTC
    valid_ttc = ttc[0][valid_mask]
    if len(valid_ttc) > 0:
        min_ttc = torch.min(valid_ttc).item()
    else:
        min_ttc = float('inf')
    return min_ttc


# -------------------------- 3. Dataset Class (Core: support variable prediction length) --------------------------
class MultiDomainTestDataset(Dataset):
    def __init__(self, domain, scaler):
        self.domain = domain
        self.scaler = scaler
        self.data_path = config.data_paths[domain]["test"]
        assert os.path.exists(self.data_path), f"Test dataset file for domain {domain} not found: {self.data_path}"
        self.raw_data = np.load(self.data_path, allow_pickle=True)
        # Process raw data into features, labels, meta info and prediction lengths
        self.features, self.labels, self.meta, self.raw_spacing, self.raw_fv_speed, self.raw_lv_speed, self.pred_lens = self._process_data()

    def _process_data(self):
        """Complete raw data preprocessing pipeline"""
        features = []  # Fixed shape [4, 100]
        labels = []  # Variable-length acceleration labels [pred_len]
        meta = []  # Scene statistics vector [5]
        raw_spacing = []  # Ground-truth gap distance with variable length
        raw_fv_speed = []  # Ground-truth following vehicle speed with variable length
        raw_lv_speed = []  # Ground-truth leading vehicle speed with variable length
        pred_lens = []  # Record prediction length for each sample

        for sample in self.raw_data:
            try:
                # Read raw trajectory records
                spacing = np.array(sample[0], dtype=np.float32)
                fv_speed = np.array(sample[1], dtype=np.float32)
                rel_speed = np.array(sample[2], dtype=np.float32)
                lv_speed = np.array(sample[3], dtype=np.float32)

                # Filter samples insufficient in total timesteps
                total_len = min(len(spacing), len(fv_speed), len(rel_speed), len(lv_speed))
                if total_len < config.min_total_len:
                    continue

                # Compute acceleration sequence from speed difference
                acc = (fv_speed[1:] - fv_speed[:-1]) / config.dt

                # Filter abnormal physical values
                if np.any(fv_speed < 0) or np.any(lv_speed < 0) or np.any(spacing <= 0.3):
                    continue

                # Split fixed history segment and variable prediction segment
                input_end = config.seq_len  # End index of historical input window
                max_label_end = min(len(acc), config.seq_len + config.max_pred_len)
                label_end = max_label_end
                pred_len = label_end - config.seq_len

                # Filter prediction length within valid range
                if pred_len < config.min_pred_len or pred_len > config.max_pred_len:
                    continue

                # Extract fixed-length input feature and variable-length acceleration label
                feat = np.stack(
                    [spacing[:input_end], fv_speed[:input_end], rel_speed[:input_end], lv_speed[:input_end]],
                    axis=0)
                label_acc = acc[config.seq_len:label_end]  # Label length equals pred_len

                # Length validation
                if len(label_acc) != pred_len or len(feat[0]) != config.seq_len:
                    continue

                # Extract ground-truth trajectory segments matching prediction range
                raw_spacing_window = spacing[config.seq_len + 1: config.seq_len + 1 + pred_len]
                raw_fv_speed_window = fv_speed[config.seq_len + 1: config.seq_len + 1 + pred_len]
                raw_lv_speed_window = lv_speed[config.seq_len + 1: config.seq_len + 1 + pred_len]

                if len(raw_spacing_window) != pred_len or len(raw_fv_speed_window) != pred_len:
                    continue

                # Compute scene statistical features
                speed_mean = np.mean(fv_speed[:input_end])
                speed_std = np.std(fv_speed[:input_end])
                max_acc_window = np.max(np.abs(acc[:label_end]))
                spacing_mean = np.mean(spacing[:input_end])
                rel_speed_mean = np.mean(rel_speed[:input_end])
                scene_feat = np.array([speed_mean, speed_std, max_acc_window, spacing_mean, rel_speed_mean],
                                      dtype=np.float32)

                # Standardization transformation
                input_reshaped = feat.transpose(1, 0).reshape(-1, config.input_dim)
                input_scaled = self.scaler.transform(input_reshaped)
                feat_scaled = input_scaled.reshape(config.seq_len, config.input_dim).transpose(1, 0)

                # Append processed data lists
                features.append(feat_scaled)
                labels.append(label_acc)
                meta.append(scene_feat)
                raw_spacing.append(raw_spacing_window)
                raw_fv_speed.append(raw_fv_speed_window)
                raw_lv_speed.append(raw_lv_speed_window)
                pred_lens.append(pred_len)

            except Exception as e:
                print(f"Error processing sample from domain {domain}: {str(e)[:50]}...")
                continue

        # Print dataset statistics
        if pred_lens:
            print(f"Preprocessing finished for domain {domain} | Valid sample count: {len(features)}")
            print(f"Prediction length range: {min(pred_lens)}~{max(pred_lens)} (required range {config.min_pred_len}~{config.max_pred_len})")

        return features, labels, meta, raw_spacing, raw_fv_speed, raw_lv_speed, pred_lens

    def __getitem__(self, idx):
        pred_len = self.pred_lens[idx]  # Variable prediction length per sample
        return {
            "x": torch.FloatTensor(self.features[idx]),  # Fixed shape [4,100]
            "y": torch.FloatTensor(self.labels[idx]),  # Variable length [pred_len]
            "meta": torch.FloatTensor(self.meta[idx]),  # Scene statistic vector [5]
            "spacing": torch.FloatTensor(self.raw_spacing[idx]),  # Ground-truth gap distance [pred_len]
            "fv_speed": torch.FloatTensor(self.raw_fv_speed[idx]),  # Ground-truth following speed [pred_len]
            "lv_speed": torch.FloatTensor(self.raw_lv_speed[idx]),  # Ground-truth leading speed [pred_len]
            "domain": self.domain,
            "ego_speed_init": self.raw_fv_speed[idx][0] if pred_len > 0 else 0.0,
            "spacing_init": self.raw_spacing[idx][0] if pred_len > 0 else 0.0,
            "pred_len": pred_len
        }

    def __len__(self):
        return len(self.features) if hasattr(self, 'features') else 0


# -------------------------- 4. Model Definition (Support variable-length prediction) --------------------------
class FeatureEncoder(nn.Module):
    """LSTM Feature Encoder"""

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
        x = x.permute(2, 0, 1)  # Convert to (seq_len, batch_size, input_dim)
        out, (hn, cn) = self.lstm(x)
        return hn, cn


class Seq2SeqDecoder(nn.Module):
    """LSTM Sequence Decoder"""

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
        dec_input = dec_input.unsqueeze(0)  # (1, batch_size, output_dim)
        out, (hidden, cell) = self.lstm(dec_input, (hidden, cell))
        pred = self.fc_out(out.squeeze(0))  # (batch_size, output_dim)
        return pred, hidden, cell


class Seq2SeqFollowingModel(nn.Module):
    """Core Seq2Seq Car-Following Model without IDM & Domain Adversarial Module"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Collision judgment threshold attribute
        self.collision_threshold = config.collision_threshold

        # Encoder & Decoder initialization
        self.encoder = FeatureEncoder(
            config.input_dim, config.hidden_size, config.enc_layers, config.dropout
        )
        self.decoder = Seq2SeqDecoder(
            config.hidden_size, config.output_dim, config.dec_layers, config.dropout
        )

        # Scene feature adaptation layer
        self.meta_adapter = nn.Linear(config.hidden_size, config.output_dim)

        # Per-dataset acceleration clipping bounds
        self.acc_min = {d: torch.tensor(v["min"], dtype=torch.float32).to(self.device) for d, v in
                        config.dataset_acc_limits.items()}
        self.acc_max = {d: torch.tensor(v["max"], dtype=torch.float32).to(self.device) for d, v in
                        config.dataset_acc_limits.items()}

        # All layers trainable without freezing
        for param in self.parameters():
            param.requires_grad = True

    def step_predict(self, hidden, cell, meta_feat, current_domain, step_num=10):
        """Single-window recursive prediction without IDM"""
        batch_size = hidden.shape[1]
        dec_input = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
        step_preds = []

        acc_min = self.acc_min[current_domain]
        acc_max = self.acc_max[current_domain]

        for _ in range(step_num):
            base_pred, hidden, cell = self.decoder(dec_input, hidden, cell)
            meta_pred = base_pred + self.meta_adapter(hidden[-1])
            fusion_pred = 0.5 * base_pred + 0.5 * meta_pred
            fusion_pred = torch.clamp(fusion_pred, min=acc_min, max=acc_max)

            step_preds.append(fusion_pred)
            dec_input = fusion_pred

        return torch.cat(step_preds, dim=1), hidden, cell

    def _build_new_input(self, current_x, step_pred, current_domain, step_num=10):
        """Update sliding input window with predicted acceleration sequence"""
        batch_size = current_x.shape[0]
        device = current_x.device

        keep_len = self.config.seq_len - step_num
        if keep_len > 0:
            if step_num < current_x.shape[2]:
                x_slice = current_x[:, :, step_num:]
            else:
                x_slice = current_x[:, :, -1:].repeat(1, keep_len)
        else:
            x_slice = current_x[:, :, -1:].repeat(1, 1)

        x_slice = x_slice[:, :, :keep_len]
        if x_slice.shape[2] < keep_len:
            pad_len = keep_len - x_slice.shape[2]
            x_slice = torch.cat([x_slice, x_slice[:, :, -1:].repeat(1, pad_len)], dim=2)

        pred_feat = torch.zeros((batch_size, self.config.input_dim, step_num),
                                device=device, dtype=current_x.dtype)

        ego_speed_init = current_x[:, 1, -1]
        spacing_init = current_x[:, 0, -1]
        dt_tensor = torch.tensor(self.config.dt, dtype=current_x.dtype, device=device)
        s0 = 0

        ego_speed = ego_speed_init
        spacing = spacing_init
        for t in range(step_num):
            ego_speed = ego_speed + step_pred[:, t] * dt_tensor
            ego_speed = torch.clamp(ego_speed, min=0.1)

            front_speed = current_x[:, 3, -1]

            spacing = spacing + (front_speed - ego_speed) * dt_tensor
            spacing = torch.clamp(spacing, min=s0)

            pred_feat[:, 0, t] = spacing
            pred_feat[:, 1, t] = ego_speed
            pred_feat[:, 2, t] = front_speed - ego_speed
            pred_feat[:, 3, t] = front_speed

        new_x = torch.cat([x_slice, pred_feat], dim=2)[:, :, :self.config.seq_len]
        return new_x

    def forward(self, x, meta, current_domain, pred_len, is_train=False, y_gt=None,
                ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        """Forward propagation supporting arbitrary prediction length"""
        batch_size = x.shape[0]
        total_preds = []
        enc_hidden, enc_cell = self.encoder(x)  # Encode historical input sequence

        remaining_pred = pred_len
        current_hidden, current_cell = enc_hidden, enc_cell
        current_x = x

        while remaining_pred > 0:
            step_num = min(self.config.step_pred, remaining_pred)
            step_pred, current_hidden, current_cell = self.step_predict(
                current_hidden, current_cell, meta, current_domain, step_num
            )
            total_preds.append(step_pred)

            new_x = self._build_new_input(current_x, step_pred, current_domain, step_num)
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

        return total_preds


# -------------------------- 5. Evaluation Function (Variable-length compatible) --------------------------
def evaluate_model(domain, model, dataloader, device):
    model.eval()
    metrics = {
        "mse_acc": 0.0,  # MSE of acceleration
        "mae_acc": 0.0,  # MAE of acceleration
        "mae_speed": 0.0,  # MAE of following vehicle speed
        "mae_abs_jerk": 0.0,  # MAE of absolute jerk
        "avg_min_ttc": 0.0,  # Average minimum TTC
        "mse_spacing": 0.0,  # MSE of gap distance
        "mae_spacing": 0.0,  # MAE of gap distance
        "collision_count": 0,  # Count of collision samples
        "collision_rate": 0.0,  # Collision percentage
        "valid_samples": 0,  # Total valid test samples
        "total_pred_steps": 0,  # Sum of all prediction timesteps
        "total_ttc_samples": 0,  # Valid samples with computable TTC
        "pred_len_metrics": {pl: {"mse_acc": 0.0, "mse_spacing": 0.0, "samples": 0}
                             for pl in range(config.min_pred_len, config.max_pred_len + 1)}
    }

    all_min_ttc = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating {domain.upper()} (Fine-tune Only, No IDM, Variable Length)")
        for batch in pbar:
            try:
                x = batch["x"].to(device, non_blocking=True)
                y_true = batch["y"].to(device, non_blocking=True)
                meta = batch["meta"].to(device, non_blocking=True)
                spacing_gt = batch["spacing"].to(device, non_blocking=True)
                fv_speed_gt = batch["fv_speed"].to(device, non_blocking=True)
                lv_speed_gt = batch["lv_speed"].to(device, non_blocking=True)
                current_domain = batch["domain"][0]
                ego_speed_init = torch.tensor([batch["ego_speed_init"]], device=device, dtype=torch.float32)
                spacing_init = torch.tensor([batch["spacing_init"]], device=device, dtype=torch.float32)
                pred_len = batch["pred_len"] if isinstance(batch["pred_len"], int) else batch["pred_len"].item()

                pred_acc = model(
                    x, meta, current_domain, pred_len,
                    is_train=False,
                    ego_speed_seq=fv_speed_gt, front_speed_seq=lv_speed_gt
                )

                if pred_acc.shape[1] != pred_len or y_true.shape[1] != pred_len:
                    min_len = min(pred_acc.shape[1], y_true.shape[1], pred_len)
                    pred_acc = pred_acc[:, :min_len]
                    y_true = y_true[:, :min_len]
                    spacing_gt = spacing_gt[:, :min_len]
                    fv_speed_gt = fv_speed_gt[:, :min_len]
                    lv_speed_gt = lv_speed_gt[:, :min_len]
                    pred_len = min_len

                mse_acc = nn.MSELoss()(pred_acc, y_true)
                mae_acc = nn.L1Loss()(pred_acc, y_true)

                spacing_pred = calculate_spacing_from_acc(
                    pred_acc, ego_speed_init, lv_speed_gt, spacing_init, config.dt, current_domain
                )
                mse_spacing = nn.MSELoss()(spacing_pred, spacing_gt)
                mae_spacing = nn.L1Loss()(spacing_pred, spacing_gt)

                speed_pred = integrate_speed(pred_acc, ego_speed_init, config.dt, pred_len)
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

                if pred_len in metrics["pred_len_metrics"]:
                    metrics["pred_len_metrics"][pred_len]["mse_acc"] += mse_acc.item() * pred_len
                    metrics["pred_len_metrics"][pred_len]["mse_spacing"] += mse_spacing.item() * pred_len
                    metrics["pred_len_metrics"][pred_len]["samples"] += 1

                pbar.set_postfix({
                    "Pred_Len": pred_len,
                    "MSE_acc": f"{mse_acc.item():.6f}",
                    "MSE_spacing": f"{mse_spacing.item():.3f}",
                    "MAE_speed": f"{mae_speed.item():.6f}",
                    "Collision": collision_count,
                    "Var_Len": "True"
                })
            except Exception as e:
                print(f"Error processing sample from domain {domain}: {str(e)[:50]}...")
                continue

    if metrics["total_pred_steps"] > 0:
        metrics["mse_acc"] /= metrics["total_pred_steps"]
        metrics["mae_acc"] /= metrics["total_pred_steps"]
        metrics["mse_spacing"] /= metrics["total_pred_steps"]
        metrics["mae_spacing"] /= metrics["total_pred_steps"]
        metrics["mae_speed"] /= metrics["total_pred_steps"]
        if (metrics["total_pred_steps"] - metrics["valid_samples"]) > 0:
            metrics["mae_abs_jerk"] /= (metrics["total_pred_steps"] - metrics["valid_samples"])

    if metrics["valid_samples"] > 0:
        metrics["collision_rate"] = (metrics["collision_count"] / metrics["valid_samples"]) * 100
        if len(all_min_ttc) > 0:
            metrics["avg_min_ttc"] = np.mean(all_min_ttc)
        else:
            metrics["avg_min_ttc"] = float('inf')

    for pl in metrics["pred_len_metrics"]:
        if metrics["pred_len_metrics"][pl]["samples"] > 0:
            total_steps = pl * metrics["pred_len_metrics"][pl]["samples"]
            metrics["pred_len_metrics"][pl]["mse_acc"] /= total_steps
            metrics["pred_len_metrics"][pl]["mse_spacing"] /= total_steps

    return metrics


# -------------------------- 6. Main Test Pipeline (Variable-length compatible) --------------------------
def main():
    # Configuration validation
    print("=" * 80)
    print("Start testing pipeline: Fine-tune Only Model without IDM (Variable-length prediction 50~80 steps)")
    print(f"Valid prediction length range: {config.min_pred_len} - {config.max_pred_len} steps")
    print("=" * 80)

    # Load trained model and scaler
    finetune_model_path = os.path.join(config.save_dir, "finetune_best.pth")
    pretrain_model_path = os.path.join(config.save_dir, "pretrain_best.pth")

    if os.path.exists(finetune_model_path):
        model_path = finetune_model_path
        model_type = "Best Fine-tuned Checkpoint"
    elif os.path.exists(pretrain_model_path):
        model_path = pretrain_model_path
        model_type = "Pre-trained Checkpoint"
    else:
        raise FileNotFoundError(f"No model checkpoint found under directory: {config.save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Loading {model_type} to computing device: {device}")
    print(f"   Checkpoint file path: {model_path}")

    model = Seq2SeqFollowingModel(config).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    scaler = checkpoint["scaler"]
    best_loss = checkpoint.get("best_loss", "N/A")
    print(f"✅ Successfully loaded {model_type} weights (Best validation loss: {best_loss})")
    print(f"✅ Layer freeze setting: {config.freeze_layers} | IDM fusion: Disabled | Domain adversarial: Disabled")
    print(f"✅ Prediction length range: {config.min_pred_len}-{config.max_pred_len} | Recursive step size: {config.step_pred} steps")

    all_results = {}
    result_df = pd.DataFrame(columns=[
        "Dataset", "Valid Samples", "Total Prediction Steps",
        "Acceleration MSE", "Acceleration MAE", "Speed MAE (m/s)", "Absolute Jerk MAE (m/s³)",
        "Gap MSE (m²)", "Gap MAE (m)", "Avg Minimum TTC (s)",
        "Collision Count", "Collision Rate (%)"
    ])

    pred_len_result_df = pd.DataFrame(columns=[
        "Dataset", "Prediction Length (step)", "Sample Count", "Acceleration MSE", "Gap MSE (m²)"
    ])

    for domain in config.domains:
        print(f"\n{'=' * 60}")
        print(f"Evaluate dataset {domain.upper()} (Fine-tune Only, No IDM, Variable Length)")
        print(f"{'=' * 60}")

        try:
            test_dataset = MultiDomainTestDataset(domain, scaler)
            if len(test_dataset) == 0:
                print(f"⚠️ No valid samples for {domain.upper()}, skip evaluation")
                continue

            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0
            )

            metrics = evaluate_model(domain, model, test_loader, device)

            all_results[domain] = metrics
            result_df.loc[len(result_df)] = [
                domain.upper(),
                metrics["valid_samples"],
                metrics["total_pred_steps"],
                round(metrics["mse_acc"], 6),
                round(metrics["mae_acc"], 6),
                round(metrics["mae_speed"], 6),
                round(metrics["mae_abs_jerk"], 6),
                round(metrics["mse_spacing"], 3),
                round(metrics["mae_spacing"], 3),
                round(metrics["avg_min_ttc"], 3) if metrics["avg_min_ttc"] != float('inf') else "N/A",
                metrics["collision_count"],
                round(metrics["collision_rate"], 2)
            ]

            for pl in metrics["pred_len_metrics"]:
                pl_metrics = metrics["pred_len_metrics"][pl]
                if pl_metrics["samples"] > 0:
                    pred_len_result_df.loc[len(pred_len_result_df)] = [
                        domain.upper(),
                        pl,
                        pl_metrics["samples"],
                        round(pl_metrics["mse_acc"], 6),
                        round(pl_metrics["mse_spacing"], 3)
                    ]

            print(f"\n📊 Test metrics of {domain.upper()} (Fine-tune Only, Variable Length):")
            print(f"   Valid sample count: {metrics['valid_samples']} | Total prediction timesteps: {metrics['total_pred_steps']}")
            print(f"   Acceleration MSE: {metrics['mse_acc']:.6f} | Acceleration MAE: {metrics['mae_acc']:.6f}")
            print(f"   Speed MAE: {metrics['mae_speed']:.6f} m/s | Absolute Jerk MAE: {metrics['mae_abs_jerk']:.6f} m/s³")
            print(f"   Gap MSE: {metrics['mse_spacing']:.3f} m² | Gap MAE: {metrics['mae_spacing']:.3f} m")
            print(f"   Average minimum TTC: {metrics['avg_min_ttc']:.3f} s" if metrics["avg_min_ttc"] != float('inf') else "   Average minimum TTC: N/A")
            print(f"   Collision count: {metrics['collision_count']} | Collision rate: {metrics['collision_rate']:.2f}%")

        except Exception as e:
            print(f"❌ Evaluation failed on {domain.upper()} (Fine-tune Only, Variable Length): {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if not result_df.empty:
        result_excel_path = os.path.join(config.result_dir, "test_results_finetune_variable_len.xlsx")
        with pd.ExcelWriter(result_excel_path, engine="openpyxl") as writer:
            result_df.to_excel(writer, sheet_name="Global Metrics", index=False)
            pred_len_result_df.to_excel(writer, sheet_name="Grouped by Prediction Length", index=False)
        print(f"\n✅ Test result excel saved to: {result_excel_path}")

        print(f"\n{'=' * 80}")
        print(f"Summary of Fine-tune Only Model Metrics ({model_type} + Variable Length Prediction)")
        print(f"{'=' * 80}")
        print("📈 Global Evaluation Metrics:")
        print(result_df.to_string(index=False))

        print(f"\n📈 Metrics grouped by prediction length (Top 10 rows):")
        print(pred_len_result_df.head(10).to_string(index=False))

        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        fig.suptitle(f"Performance of Seq2Seq Model (Fine-tune Only, No IDM, Variable Prediction Length)", fontsize=16, fontweight="bold")
        domains = result_df["Dataset"].tolist()

        axes[0, 0].bar(domains, result_df["Acceleration MSE"], color="#2E86AB", alpha=0.8)
        axes[0, 0].set_title("Acceleration MSE", fontweight="bold")
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        axes[0, 1].bar(domains, result_df["Speed MAE (m/s)"], color="#F18F01", alpha=0.8)
        axes[0, 1].set_title("Speed MAE (m/s)", fontweight="bold")
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        axes[0, 2].bar(domains, result_df["Absolute Jerk MAE (m/s³)"], color="#A23B72", alpha=0.8)
        axes[0, 2].set_title("Absolute Jerk MAE (m/s³)", fontweight="bold")
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3, axis='y')

        axes[0, 3].bar(domains, result_df["Gap MSE (m²)"], color="#C73E1D", alpha=0.8)
        axes[0, 3].set_title("Gap MSE (m²)", fontweight="bold")
        axes[0, 3].tick_params(axis='x', rotation=45)
        axes[0, 3].grid(True, alpha=0.3, axis='y')

        axes[1, 0].bar(domains, result_df["Collision Rate (%)"], color="#6A994E", alpha=0.8)
        axes[1, 0].set_title("Collision Rate (%)", fontweight="bold")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        axes[1, 1].bar(domains, result_df["Acceleration MAE"], color="#F77F00", alpha=0.8)
        axes[1, 1].set_title("Acceleration MAE", fontweight="bold")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        axes[1, 2].bar(domains, result_df["Gap MAE (m)"], color="#7209B7", alpha=0.8)
        axes[1, 2].set_title("Gap MAE (m)", fontweight="bold")
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3, axis='y')

        axes[1, 3].bar(domains, result_df["Valid Samples"], color="#FCBF49", alpha=0.8)
        axes[1, 3].set_title("Valid Sample Count", fontweight="bold")
        axes[1, 3].tick_params(axis='x', rotation=45)
        axes[1, 3].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig_path = os.path.join(config.result_dir, "test_performance_finetune_variable_len.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Performance comparison figure saved to: {fig_path}")

        if not pred_len_result_df.empty:
            fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
            fig2.suptitle(f"Prediction Length vs Model Performance ({model_type})", fontsize=14, fontweight="bold")

            pl_acc_mse = pred_len_result_df.groupby("Prediction Length (step)")["Acceleration MSE"].mean()
            axes2[0].plot(pl_acc_mse.index, pl_acc_mse.values, marker='o', linewidth=2, color="#2E86AB")
            axes2[0].set_title("Prediction Length vs Avg Acceleration MSE", fontweight="bold")
            axes2[0].set_xlabel("Prediction Length (step)")
            axes2[0].set_ylabel("Acceleration MSE")
            axes2[0].grid(True, alpha=0.3)

            pl_spacing_mse = pred_len_result_df.groupby("Prediction Length (step)")["Gap MSE (m²)"].mean()
            axes2[1].plot(pl_spacing_mse.index, pl_spacing_mse.values, marker='s', linewidth=2, color="#C73E1D")
            axes2[1].set_title("Prediction Length vs Avg Gap MSE", fontweight="bold")
            axes2[1].set_xlabel("Prediction Length (step)")
            axes2[1].set_ylabel("Gap MSE (m²)")
            axes2[1].grid(True, alpha=0.3)

            plt.tight_layout()
            fig2_path = os.path.join(config.result_dir, "pred_len_vs_performance.png")
            plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✅ Prediction length performance curve saved to: {fig2_path}")

    print(f"\n{'=' * 80}")
    print("Test pipeline finished: Fine-tune Only Model without IDM (Variable-length prediction)")
    print(f"Result storage directory: {config.result_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()