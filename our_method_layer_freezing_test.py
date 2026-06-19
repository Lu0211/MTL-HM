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


# -------------------------- 1. Configuration Class (Match training config with layer freezing and no IDM) --------------------------
class Config:
    # Core input & output settings (Adapt to variable-length prediction)
    input_dim = 4  # 4-dimensional input: gap distance, following vehicle speed, relative speed, leading vehicle speed
    output_dim = 1  # Single-step acceleration output
    seq_len = 100  # Fixed historical sequence length (first 100 timesteps)
    dt = 0.1  # Fixed 10Hz time interval (0.1s per step)

    # Variable-length prediction settings
    min_total_len = 151  # Minimum total sample length (100 history steps + min 50 prediction steps + 1)
    min_pred_len = 50  # Minimum number of acceleration prediction timesteps
    max_pred_len = 80  # Maximum number of acceleration prediction timesteps
    step_pred = 10  # Predict 10 timesteps per sliding window iteration

    # Domain settings (Consistent with training script)
    domains = ["ngsim", "lyft", "waymo", "spmd1", "spmd2"]
    source_domains = ["ngsim", "spmd2"]
    target_domains = ["spmd1", "lyft", "waymo"]

    # Acceleration constraints for each dataset (Identical to training script)
    dataset_acc_limits = {
        "ngsim": {"min": -2.7, "max": 2.6},
        "spmd1": {"min": -3.5, "max": 2.0},
        "spmd2": {"min": -3.5, "max": 2.0},
        "waymo": {"min": -4.0, "max": 3.0},
        "lyft": {"min": -3.0, "max": 2.9}
    }

    # Model hyperparameters (Fully consistent with layer-freeze & no-IDM training code)
    hidden_size = 64
    enc_layers = 2
    dec_layers = 2
    dropout = 0.1
    meta_alpha = 0.01
    freeze_layers = 1  # Number of frozen encoder layers (Same as training)

    # Test hyperparameters
    batch_size = 1  # Must set to 1 to avoid batch dimension mismatch for variable-length samples
    collision_threshold = 0.0  # Threshold for collision detection

    # File path settings (For model trained with layer freezing and no IDM)
    save_dir = "output/our_method_ablation/only_layer_freeze"  # Training directory of layer-freeze no-IDM model
    result_dir = os.path.join(save_dir, "test_result_layer_freeze_no_idm")  # Test output directory
    data_paths = {
        "ngsim": {"test": "dataset/data/NGSIM_I_80_test_data.npy"},
        "waymo": {"test": "dataset/data/Waymo_test_data.npy"},
        "lyft": {"test": "dataset/data/Lyft_test_data.npy"},
        "spmd1": {"test": "dataset/data/SPMD1_test_data.npy"},
        "spmd2": {"test": "dataset/data/SPMD2_test_data.npy"}
    }


config = Config()
# Verify total length constraint
assert config.min_total_len - config.seq_len >= config.min_pred_len + 1, \
    f"Minimum total length {config.min_total_len} minus history length {config.seq_len} should be greater than or equal to minimum prediction length {config.min_pred_len} + 1"
os.makedirs(config.result_dir, exist_ok=True)


# -------------------------- 2. Utility Functions (Removed IDM logic, keep core calculation) --------------------------
def integrate_speed(acc_seq, speed_init, dt, pred_len):
    """Integrate acceleration to derive speed sequence, support variable-length input"""
    speed_seq = torch.zeros((1, pred_len), device=acc_seq.device, dtype=acc_seq.dtype)
    speed_seq[:, 0] = speed_init
    for t in range(1, pred_len):
        speed_seq[:, t] = speed_seq[:, t - 1] + acc_seq[:, t - 1] * dt
        speed_seq[:, t] = torch.clamp(speed_seq, min=0.1)  # Non-negative speed constraint
    return speed_seq


def calculate_jerk(acc_seq, dt):
    """Compute absolute jerk values, support variable-length input"""
    if acc_seq.shape[1] <= 1:
        return torch.zeros((1, 0), device=acc_seq.device)  # Prevent empty sequence error
    jerk = (acc_seq[:, 1:] - acc_seq[:, :-1]) / dt
    abs_jerk = torch.abs(jerk)
    return abs_jerk


def calculate_ttc(spacing_seq, fv_speed_seq, lv_speed_seq, pred_len):
    """Calculate Time-To-Collision metric, support variable-length input"""
    delta_speed = fv_speed_seq - lv_speed_seq
    valid_mask = delta_speed > 1e-3  # Avoid division by zero error
    ttc = torch.zeros((1, pred_len), device=spacing_seq.device)
    ttc[valid_mask] = spacing_seq[valid_mask] / delta_speed[valid_mask]
    ttc[~valid_mask] = float('inf')

    # Extract minimum valid TTC value
    valid_ttc = ttc[0][valid_mask[0]]
    if len(valid_ttc) > 0:
        min_ttc = torch.min(valid_ttc).item()
    else:
        min_ttc = float('inf')
    return min_ttc


# -------------------------- 2. Gap Integration Function (No IDM, consistent with training code) --------------------------
def calculate_spacing_from_acc(pred_acc, ego_speed_init, front_speed_seq, init_spacing, dt, domain, pred_len):
    """Pure physical gap integration without IDM constraints, consistent with training script"""
    device = pred_acc.device
    speed_len = pred_len + 1
    spacing_len = pred_len + 1

    ego_speed = torch.zeros((1, speed_len), device=device, dtype=pred_acc.dtype)
    spacing_pred = torch.zeros((1, spacing_len), device=device, dtype=pred_acc.dtype)
    ego_speed[:, 0] = ego_speed_init
    spacing_pred[:, 0] = init_spacing

    dt_tensor = torch.tensor(dt, dtype=pred_acc.dtype, device=device)
    s0 = torch.tensor(0, dtype=pred_acc.dtype, device=device)  # Basic minimum gap without IDM

    for t in range(1, speed_len):
        # Speed integration
        ego_speed_t = ego_speed[:, t - 1] + pred_acc[:, t - 1] * dt_tensor if t - 1 < pred_len else ego_speed[:, t - 1]
        ego_speed_t = torch.clamp(ego_speed_t, min=0)
        ego_speed[:, t] = ego_speed_t

        # Fetch leading vehicle speed, avoid index out of range for variable-length data
        front_speed_t = front_speed_seq[:, t] if t < front_speed_seq.shape[1] else front_speed_seq[:, -1]

        # Gap integration without IDM lower bound
        spacing_t = spacing_pred[:, t - 1] + (front_speed_t - ego_speed_t) * dt_tensor
        spacing_t = torch.clamp(spacing_t, min=s0)
        spacing_pred[:, t] = spacing_t

    # Return gap sequence aligned with acceleration length (last pred_len timesteps)
    return spacing_pred[:, 1:1 + pred_len]


# -------------------------- 3. Dataset Class (Input format matches layer-freeze training) --------------------------
class MultiDomainTestDataset(Dataset):
    def __init__(self, domain, scaler):
        self.domain = domain
        self.scaler = scaler
        self.data_path = config.data_paths[domain]["test"]
        assert os.path.exists(self.data_path), f"Test dataset file for domain {domain} not found: {self.data_path}"
        self.raw_data = np.load(self.data_path, allow_pickle=True)
        # Process variable-length trajectory samples
        self.features, self.labels, self.meta, self.raw_spacing, self.raw_fv_speed, self.raw_lv_speed, self.pred_lens = self._process_data()

    def _process_data(self):
        """Data processing logic consistent with training script, fix acceleration length alignment"""
        features = []  # Fixed dimension [4, 100]
        labels = []  # Variable-length acceleration sequence [pred_len]
        meta = []  # Fixed 5-dimensional statistical scene features
        raw_spacing = []  # Variable-length ground truth gap sequence matched to acceleration
        raw_fv_speed = []  # Variable-length following speed sequence matched to acceleration
        raw_lv_speed = []  # Variable-length leading speed sequence matched to acceleration
        pred_lens = []  # Record prediction length of each sample

        for sample in self.raw_data:
            try:
                # Load raw trajectory data
                spacing = np.array(sample[0], dtype=np.float32)
                fv_speed = np.array(sample[1], dtype=np.float32)
                rel_speed = np.array(sample[2], dtype=np.float32)
                lv_speed = np.array(sample[3], dtype=np.float32)

                # Filter short samples
                total_len = min(len(spacing), len(fv_speed), len(rel_speed), len(lv_speed))
                if total_len < config.min_total_len:
                    continue

                # Calculate acceleration from speed difference
                acc = (fv_speed[1:] - fv_speed[:-1]) / config.dt
                lv_acc = (lv_speed[1:] - lv_speed[:-1]) / config.dt

                # Filter abnormal physical values
                if np.any(fv_speed < 0) or np.any(lv_speed < 0) or np.any(spacing < 0):
                    continue

                # Fix first seq_len timesteps as historical input
                input_end = config.seq_len
                label_start = config.seq_len
                max_label_end = min(len(acc), config.seq_len + config.max_pred_len)
                label_end = max_label_end

                pred_len = label_end - label_start

                # Discard samples with insufficient prediction timesteps
                if pred_len < config.min_pred_len:
                    continue

                # Extract fixed-length historical input
                input_spacing = spacing[:input_end]
                input_fv = fv_speed[:input_end]
                input_rel = rel_speed[:input_end]
                input_lv = lv_speed[:input_end]
                feat = np.stack([input_spacing, input_fv, input_rel, input_lv], axis=0)

                # Extract acceleration ground truth strictly aligned with pred_len
                label_acc = acc[label_start:label_end]
                if len(label_acc) != pred_len or len(feat[0]) != config.seq_len:
                    continue

                # Extract ground truth window matching prediction range
                raw_spacing_window = spacing[label_start + 1: label_end + 1]
                raw_fv_speed_window = fv_speed[label_start + 1: label_end + 1]
                raw_lv_speed_window = lv_speed[label_start + 1: label_end + 1]

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

                # Standardization same as training
                input_reshaped = feat.transpose(1, 0).reshape(-1, config.input_dim)
                input_scaled = self.scaler.transform(input_reshaped)
                feat_scaled = input_scaled.reshape(config.seq_len, config.input_dim).transpose(1, 0)

                # Append processed data
                features.append(feat_scaled)
                labels.append(label_acc)
                meta.append(scene_feat)
                raw_spacing.append(raw_spacing_window)
                raw_fv_speed.append(raw_fv_speed_window)
                raw_lv_speed.append(raw_lv_speed_window)
                pred_lens.append(pred_len)

            except Exception as e:
                print(f"Error processing sample from domain {self.domain}: {str(e)[:50]}...")
                continue

        # Convert fixed-length lists to numpy arrays
        features = np.array(features) if features else np.array([])
        meta = np.array(meta) if meta else np.array([])

        # Print dataset statistics
        if pred_lens:
            print(f"Preprocessing finished for domain {self.domain} | Valid sample count: {len(features)}")
            print(f"Prediction length range: {min(pred_lens)}~{max(pred_lens)} (Min requirement: {config.min_pred_len}, Max limit: {config.max_pred_len})")

        return features, labels, meta, raw_spacing, raw_fv_speed, raw_lv_speed, pred_lens

    def __getitem__(self, idx):
        pred_len = self.pred_lens[idx]  # Dynamic prediction length per sample
        return {
            "x": torch.FloatTensor(self.features[idx]),  # Fixed [4,100] historical input
            "y": torch.FloatTensor(self.labels[idx]),  # Variable-length acceleration ground truth
            "meta": torch.FloatTensor(self.meta[idx]),  # Fixed 5D scene feature
            "spacing": torch.FloatTensor(self.raw_spacing[idx]),  # Variable-length ground truth gap
            "fv_speed": torch.FloatTensor(self.raw_fv_speed[idx]),  # Variable-length following speed
            "lv_speed": torch.FloatTensor(self.raw_lv_speed[idx]),  # Variable-length leading speed
            "domain": self.domain,
            "ego_speed_init": self.raw_fv_speed[idx][0] if pred_len > 0 else 0.0,
            "spacing_init": self.raw_spacing[idx][0] if pred_len > 0 else 0.0,
            "pred_len": pred_len,
            "total_len": config.seq_len + pred_len + 1
        }

    def __len__(self):
        return len(self.features) if hasattr(self, 'features') else 0


# -------------------------- 4. Model Definition (Identical to layer-freeze no-IDM training code) --------------------------
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
        x = x.permute(2, 0, 1)  # [1,4,100] -> [100,1,4]
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
        dec_input = dec_input.unsqueeze(0)  # [1,1] -> [1,1,1]
        out, (hidden, cell) = self.lstm(dec_input, (hidden, cell))
        pred = self.fc_out(out.squeeze(0))  # [1,1]
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
        # Domain discriminator removed (consistent with training script without domain adversarial)
        self.dataset_acc_min = {d: torch.tensor(v["min"], dtype=torch.float32).to(self.device) for d, v in config.dataset_acc_limits.items()}
        self.dataset_acc_max = {d: torch.tensor(v["max"], dtype=torch.float32).to(self.device) for d, v in config.dataset_acc_limits.items()}
        self.collision_threshold = config.collision_threshold

        # Layer freezing same as training
        for i in range(config.freeze_layers):
            for param in self.encoder.lstm.all_weights[i]:
                param.requires_grad = False

    def init_decoder_input(self, batch_size):
        return torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)

    def step_predict(self, enc_hidden, enc_cell, meta_feat, current_domain, step_num,
                     ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        """Single window prediction without IDM and domain adversarial module"""
        batch_size = enc_hidden.shape[1]
        dec_input = self.init_decoder_input(batch_size)
        hidden, cell = enc_hidden, enc_cell
        step_preds = []

        # Domain-specific acceleration clipping bounds
        acc_min = self.dataset_acc_min.get(current_domain, self.dataset_acc_min["ngsim"])
        acc_max = self.dataset_acc_max.get(current_domain, self.dataset_acc_max["ngsim"])

        for t in range(step_num):
            # Base network prediction
            base_pred, hidden, cell = self.decoder(dec_input, hidden, cell)
            meta_pred = base_pred + self.meta_adapter(hidden[-1])
            weights = self.dynamic_weight(meta_feat)
            fusion_pred = weights[:, 0:1] * base_pred + weights[:, 1:2] * meta_pred

            # Only apply acceleration range limit, no IDM fusion
            fusion_pred = torch.clamp(fusion_pred, min=acc_min, max=acc_max)
            step_preds.append(fusion_pred)
            dec_input = fusion_pred

        return torch.cat(step_preds, dim=1), hidden, cell

    def forward(self, x, meta, current_domain, pred_len,
                ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        """Forward pass matching layer-freeze no-IDM training script"""
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

            # Rebuild sliding input window without IDM
            new_x = self._build_new_input(current_x, step_pred, current_domain, front_speed_seq, step_num)
            current_x = new_x
            remaining_pred -= step_num

        # Concatenate and trim output to target prediction length
        total_preds = torch.cat(total_preds, dim=1)[:, :pred_len]
        if total_preds.shape[1] != pred_len:
            if total_preds.shape[1] > pred_len:
                total_preds = total_preds[:, :pred_len]
            else:
                pad_len = pred_len - total_preds.shape[1]
                padding = torch.zeros((batch_size, pad_len), device=total_preds.device, dtype=total_preds.dtype)
                total_preds = torch.cat([total_preds, padding], dim=1)

        return total_preds, None  # No domain prediction output

    def _build_new_input(self, current_x, step_pred, current_domain, front_speed_seq=None, step_num=None):
        """Sliding window update logic identical to training script without IDM"""
        batch_size = current_x.shape[0]
        step_num = step_num if step_num else self.config.step_num

        keep_len = self.config.seq_len - step_num
        if keep_len > 0:
            x_slice = current_x[:, :, step_num:]
        else:
            x_slice = current_x[:, :, -1:].repeat(1, 1, 1)
        x_slice = x_slice[:, :, :keep_len]
        if x_slice.shape[2] < keep_len:
            pad_len = keep_len - x_slice.shape[2]
            x_slice = torch.cat([x_slice, x_slice[:, :, -1:].repeat(1, 1, pad_len)], dim=2)

        pred_feat = torch.zeros((batch_size, self.config.input_dim, step_num),
                                device=current_x.device, dtype=current_x.dtype)

        ego_speed_init = current_x[:, 1, -1]
        spacing_init = current_x[:, 0, -1]
        dt_tensor = torch.tensor(self.config.dt, dtype=current_x.dtype, device=current_x.device)
        s0 = 0  # No IDM minimum gap

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


# -------------------------- 5. Evaluation Function (For layer-freeze no-IDM model) --------------------------
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
        "collision_count": 0,  # Number of collision samples
        "collision_rate": 0.0,  # Collision percentage
        "valid_samples": 0,  # Total valid test samples
        "total_pred_steps": 0,  # Sum of all prediction timesteps
        "total_ttc_samples": 0,  # Valid samples with computable TTC
        "pred_len_stats": []  # Record prediction length of each sample
    }

    all_min_ttc = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating {domain.upper()} (Layer Freeze + No IDM)")
        for batch in pbar:
            try:
                # Load batch tensors
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

                assert y_true.shape[1] == pred_len, f"Label length {y_true.shape[1]} mismatches prediction length {pred_len}"
                metrics["pred_len_stats"].append(pred_len)

                # Model inference
                pred_acc, _ = model(
                    x, meta, current_domain, pred_len=pred_len,
                    ego_speed_seq=fv_speed_gt, front_speed_seq=lv_speed_gt, spacing_seq=spacing_gt
                )

                # Align sequence lengths
                if pred_acc.shape[1] != y_true.shape[1]:
                    min_len = min(pred_acc.shape[1], y_true.shape[1])
                    pred_acc = pred_acc[:, :min_len]
                    y_true = y_true[:, :min_len]
                    spacing_gt = spacing_gt[:, :min_len] if spacing_gt.shape[1] > min_len else spacing_gt
                    fv_speed_gt = fv_speed_gt[:, :min_len] if fv_speed_gt.shape[1] > min_len else fv_speed_gt
                    lv_speed_gt = lv_speed_gt[:, :min_len] if lv_speed_gt.shape[1] > min_len else lv_speed_gt
                    pred_len = min_len

                # Compute predicted gap
                spacing_pred = calculate_spacing_from_acc(
                    pred_acc, ego_speed_init, lv_speed_gt, spacing_init, config.dt, current_domain, pred_len
                )

                # Integrate predicted speed
                speed_pred = integrate_speed(pred_acc, ego_speed_init, config.dt, pred_len)

                # Acceleration error metrics
                mse_acc = nn.MSELoss()(pred_acc, y_true)
                mae_acc = nn.L1Loss()(pred_acc, y_true)

                # Gap error metrics
                if spacing_pred.shape[1] != spacing_gt.shape[1]:
                    min_spacing_len = min(spacing_pred.shape[1], spacing_gt.shape[1])
                    spacing_pred = spacing_pred[:, :min_spacing_len]
                    spacing_gt = spacing_gt[:, :min_spacing_len]
                mse_spacing = nn.MSELoss()(spacing_pred, spacing_gt)
                mae_spacing = nn.L1Loss()(spacing_pred, spacing_gt)

                # Speed error metrics
                if speed_pred.shape[1] != fv_speed_gt.shape[1]:
                    min_speed_len = min(speed_pred.shape[1], fv_speed_gt.shape[1])
                    speed_pred = speed_pred[:, :min_speed_len]
                    fv_speed_gt = fv_speed_gt[:, :min_speed_len]
                mae_speed = nn.L1Loss()(speed_pred, fv_speed_gt)

                # Jerk calculation
                jerk_pred = calculate_jerk(pred_acc, config.dt)
                jerk_true = calculate_jerk(y_true, config.dt)
                min_jerk_len = min(jerk_pred.shape[1], jerk_true.shape[1]) if jerk_pred.shape[1] > 0 else 0
                mae_abs_jerk = torch.tensor(0.0, device=device)
                if min_jerk_len > 0:
                    mae_abs_jerk = nn.L1Loss()(jerk_pred[:, :min_jerk_len], jerk_true[:, :min_jerk_len])

                # Minimum TTC calculation
                min_ttc = calculate_ttc(spacing_pred, speed_pred, lv_speed_gt, pred_len)
                if min_ttc != float('inf'):
                    all_min_ttc.append(min_ttc)
                    metrics["total_ttc_samples"] += 1

                # Collision detection
                collision = torch.any(spacing_pred <= model.collision_threshold)
                collision_count = 1 if collision else 0

                # Accumulate weighted metrics
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

                pbar.set_postfix({
                    "PredLen": pred_len,
                    "MSE_acc": f"{mse_acc.item():.6f}",
                    "MAE_speed": f"{mae_speed.item():.6f}",
                    "Collision": collision_count
                })
            except Exception as e:
                print(f"Error processing sample from domain {domain}: {str(e)[:50]}...")
                continue

    # Normalize metrics by total prediction steps
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
        if len(all_min_ttc) > 0:
            metrics["avg_min_ttc"] = np.mean(all_min_ttc)
        else:
            metrics["avg_min_ttc"] = float('inf')

    # Print prediction length statistics
    if metrics["pred_len_stats"]:
        print(f"\nPrediction length statistics for domain {domain} (Layer Freeze + No IDM):")
        print(f"  Min: {min(metrics['pred_len_stats'])} | Max: {max(metrics['pred_len_stats'])} | Mean: {np.mean(metrics['pred_len_stats']):.1f}")

    return metrics


# -------------------------- 6. Main Test Pipeline (Layer Freeze + No IDM Model) --------------------------
def main():
    # Header print
    print("=" * 80)
    print("Start testing pipeline for layer-frozen transfer learning model without IDM fusion (Variable-length samples)")
    print("=" * 80)

    # Load pretrained checkpoint
    model_path = os.path.join(config.save_dir, "pretrain_best.pth")
    assert os.path.exists(model_path), f"Layer-freeze no-IDM model file not found: {model_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Loading layer-frozen no-IDM model to device: {device}")

    # Initialize model matching training architecture
    model = Seq2SeqFollowingModel(config).to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    scaler = checkpoint["scaler"]
    best_pretrain_loss = checkpoint.get("best_loss", "N/A")
    print(f"✅ Successfully loaded layer-frozen no-IDM model weights (Best pretrain loss: {best_pretrain_loss})")
    print(f"Frozen encoder layers: {config.freeze_layers} | IDM fusion: Disabled")

    all_results = {}
    result_df = pd.DataFrame(columns=[
        "Dataset", "Valid Samples", "Total Prediction Steps", "Avg Prediction Length",
        "Acceleration MSE", "Acceleration MAE", "Speed MAE (m/s)", "Absolute Jerk MAE (m/s³)",
        "Gap MSE (m²)", "Gap MAE (m)", "Avg Minimum TTC (s)",
        "Collision Count", "Collision Rate (%)"
    ])

    # Iterate all test domains
    for domain in config.domains:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {domain.upper()} dataset (Layer Freeze + No IDM)")
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
                round(metrics["collision_rate"], 2)
            ]

            # Print per-domain results
            print(f"\n📊 {domain.upper()} layer-freeze no-IDM test metrics:")
            print(f"Valid samples: {metrics['valid_samples']}, total prediction timesteps: {metrics['total_pred_steps']}, average prediction length: {avg_pred_len:.1f}")
            print(f"Acceleration MSE: {metrics['mse_acc']:.6f}, MAE: {metrics['mae_acc']:.6f}")
            print(f"Speed MAE: {metrics['mae_speed']:.6f} m/s, Absolute Jerk MAE: {metrics['mae_abs_jerk']:.6f} m/s³")
            print(f"Gap MSE: {metrics['mse_spacing']:.3f} m², Gap MAE: {metrics['mae_spacing']:.3f} m")
            print(f"Average minimum TTC: {metrics['avg_min_ttc']:.3f} s" if metrics["avg_min_ttc"] != float('inf') else "Average minimum TTC: N/A")
            print(f"Collision count: {metrics['collision_count']}, Collision rate: {metrics['collision_rate']:.2f}%")

        except Exception as e:
            print(f"❌ Evaluation failed on {domain.upper()} (Layer Freeze + No IDM): {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save and visualize outputs
    if not result_df.empty:
        result_excel_path = os.path.join(config.result_dir, "test_results_layer_freeze_no_idm.xlsx")
        result_df.to_excel(result_excel_path, index=False, engine="openpyxl")
        print(f"\n✅ Layer-freeze no-IDM test results saved to: {result_excel_path}")

        print(f"\n{'=' * 80}")
        print("Layer-frozen no-IDM model overall performance summary (Variable-length samples)")
        print(f"{'=' * 80}")
        print(result_df.to_string(index=False))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Performance of Layer-Frozen Seq2Seq Model Without IDM Fusion (Variable-Length Samples)", fontsize=16, fontweight="bold")
        domains = result_df["Dataset"].tolist()

        axes[0, 0].bar(domains, result_df["Acceleration MSE"], color="#2E86AB", alpha=0.8)
        axes[0, 0].set_title("Acceleration MSE", fontweight="bold")
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].bar(domains, result_df["Speed MAE (m/s)"], color="#F18F01", alpha=0.8)
        axes[0, 1].set_title("Speed MAE (m/s)", fontweight="bold")
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].bar(domains, result_df["Absolute Jerk MAE (m/s³)"], color="#A23B72", alpha=0.8)
        axes[0, 2].set_title("Absolute Jerk MAE (m/s³)", fontweight="bold")
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].bar(domains, result_df["Gap MSE (m²)"], color="#C73E1D", alpha=0.8)
        axes[1, 0].set_title("Gap MSE (m²)", fontweight="bold")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].bar(domains, result_df["Collision Rate (%)"], color="#6A994E", alpha=0.8)
        axes[1, 1].set_title("Collision Rate (%)", fontweight="bold")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].bar(domains, result_df["Avg Prediction Length"], color="#F77F00", alpha=0.8)
        axes[1, 2].set_title("Average Prediction Length (step)", fontweight="bold")
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(config.result_dir, "test_performance_layer_freeze_no_idm.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Layer-freeze no-IDM performance plot saved to: {fig_path}")

    print(f"\n{'=' * 80}")
    print("Variable-length sample test pipeline completed for layer-frozen no-IDM model!")
    print(f"Result storage directory: {config.result_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()