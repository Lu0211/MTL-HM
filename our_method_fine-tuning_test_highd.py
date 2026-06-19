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


# -------------------------- 1. Configuration Class (Aligned with Finetune Transfer Training without IDM) --------------------------
class Config:
    # Core input & output settings (Adapt to variable-length prediction + finetune transfer)
    input_dim = 4  # 4-dimensional input: gap distance, following vehicle speed, relative speed, leading vehicle speed
    output_dim = 1  # Single-step acceleration output
    seq_len = 250  # Historical sequence length (Consistent with finetune training)
    dt = 0.04  # Time step of HighD dataset

    # Variable-length prediction settings (Aligned with HighD and finetune training)
    # highd
    min_total_len = 375  # Minimum total sample length (100 history steps + min 50 prediction steps + 1)
    min_pred_len = 50  # Minimum prediction steps of acceleration sequence
    max_pred_len = 125  # Maximum prediction steps of acceleration sequence
    # 30highd
    # min_total_len = 750  # Minimum total sample length (100 history steps + min 50 prediction steps + 1)
    # min_pred_len = 200  # Minimum prediction steps of acceleration sequence
    # max_pred_len = 200  # Maximum prediction steps of acceleration sequence
    step_pred = 25  # Sliding window step size

    # Domain settings for finetune transfer training
    source_domains = ["ngsim", "spmd2"]  # Source domains for pre-training
    target_domains = ["spmd1", "lyft", "waymo"]  # Target domains for finetuning
    # test_domains = ["30highd"]  # HighD test domain
    test_domains = ["highd"]             # Option to test HighD alone

    # Acceleration constraints for each dataset (Adapt to HighD and finetune training)
    dataset_acc_limits = {
        # "30highd": {"min": -2.7, "max": 1.6},
        "highd": {"min": -2.9, "max": 1.57},
        "default": {"min": -4.0, "max": 3.0}  # Fallback value
    }

    # Settings without IDM module
    idm_params = {}
    idm_weight = 0.0  # IDM weight set to zero to disable fusion

    # Model hyperparameters (Fully consistent with finetune transfer training)
    hidden_size = 64
    enc_layers = 2
    dec_layers = 2
    dropout = 0.1
    meta_alpha = 0.01
    freeze_layers = 0  # Layer freezing disabled during finetune transfer training

    # Test hyperparameters
    batch_size = 1  # Must set to 1 for variable-length samples
    collision_threshold = 0.0  # Threshold to judge collision occurrence

    # Model path settings for finetune transfer without IDM (Core modification)
    save_dir = "output/our_method_ablation/only_finetune_no_idm"  # Save directory of finetune transfer model
    result_dir = os.path.join(save_dir, "test_result_finetune_highd")  # Directory for test outputs
    data_paths = {
        # "30highd": {"test": "dataset/data/30_HighD_test_data.npy"},
        "highd": {"test": "dataset/data/HighD_test_data.npy"},
    }


config = Config()
# Verify length constraints
assert config.min_total_len - config.seq_len >= config.min_pred_len + 1, \
    f"Minimum total length {config.min_total_len} minus history length {config.seq_len} should be greater than or equal to minimum prediction length {config.min_pred_len} + 1"
os.makedirs(config.result_dir, exist_ok=True)


# -------------------------- 2. Utility Functions (Aligned with physical integration logic of finetune training) --------------------------
def integrate_speed(acc_seq, speed_init, dt, pred_len):
    """Integrate acceleration to obtain speed sequence (Adapt to variable-length, consistent with finetune training logic)"""
    speed_seq = torch.zeros((1, pred_len), device=acc_seq.device, dtype=acc_seq.dtype)
    speed_seq[:, 0] = speed_init
    for t in range(1, pred_len):
        speed_seq[:, t] = speed_seq[:, t - 1] + acc_seq[:, t - 1] * dt
        speed_seq[:, t] = torch.clamp(speed_seq, min=0.1)  # Non-negative speed constraint
    return speed_seq


def calculate_jerk(acc_seq, dt):
    """Compute absolute jerk metric (Support variable-length input)"""
    if acc_seq.shape[1] <= 1:
        return torch.zeros((1, 0), device=acc_seq.device)
    jerk = (acc_seq[:, 1:] - acc_seq[:, :-1]) / dt
    abs_jerk = torch.abs(jerk)
    return abs_jerk


def calculate_ttc(spacing_seq, fv_speed_seq, lv_speed_seq, pred_len):
    """Calculate Time-To-Collision (Support variable-length input)"""
    delta_speed = fv_speed_seq - lv_speed_seq
    valid_mask = delta_speed > 1e-3  # Avoid division by zero error
    ttc = torch.zeros((1, pred_len), device=spacing_seq.device)
    ttc[valid_mask] = spacing_seq[valid_mask] / delta_speed[valid_mask]
    ttc[~valid_mask] = float('inf')

    # Compute minimum valid TTC value
    valid_ttc = ttc[0][valid_mask[0]]
    if len(valid_ttc) > 0:
        min_ttc = torch.min(valid_ttc).item()
    else:
        min_ttc = float('inf')
    return min_ttc


# -------------------------- 3. Spacing Integration Function without IDM (Aligned with finetune training code) --------------------------
def calculate_spacing_from_acc(pred_acc, ego_speed_init, front_speed_seq, init_spacing, dt, domain, pred_len):
    """Pure physical integration to compute gap distance, IDM disabled, fully consistent with finetune training logic"""
    device = pred_acc.device
    ego_speed = torch.zeros((1, pred_len), device=device, dtype=pred_acc.dtype)
    spacing_pred = torch.zeros((1, pred_len), device=device, dtype=pred_acc.dtype)
    ego_speed[:, 0] = ego_speed_init
    spacing_pred[:, 0] = init_spacing

    dt_tensor = torch.tensor(dt, dtype=pred_acc.dtype, device=device)
    s0 = torch.tensor(0, dtype=pred_acc.dtype, device=device)  # Basic minimum safe gap

    for t in range(1, pred_len):
        # Velocity integration (Consistent with finetune training)
        ego_speed_t = ego_speed[:, t - 1] + pred_acc[:, t - 1] * dt_tensor
        ego_speed_t = torch.clamp(ego_speed_t, min=0.1)
        ego_speed[:, t] = ego_speed_t

        # Leading vehicle speed (Adapt to variable-length sequence)
        front_speed_t = front_speed_seq[:, t - 1] if t - 1 < front_speed_seq.shape[1] else front_speed_seq[:, -1]

        # Pure physical gap integration without IDM module
        spacing_t = spacing_pred[:, t - 1] + (front_speed_t - ego_speed_t) * dt_tensor
        spacing_t = torch.clamp(spacing_t, min=s0)
        spacing_pred[:, t] = spacing_t

    return spacing_pred


# -------------------------- 4. Dataset Class (Adapt to input format of finetune transfer model) --------------------------
class MultiDomainTestDataset(Dataset):
    def __init__(self, domain, scaler):
        self.domain = domain
        self.scaler = scaler
        self.data_path = config.data_paths[domain]["test"]
        assert os.path.exists(self.data_path), f"Test dataset file for domain {domain} not found: {self.data_path}"
        self.raw_data = np.load(self.data_path, allow_pickle=True)
        self.features, self.labels, self.meta, self.raw_spacing, self.raw_fv_speed, self.raw_lv_speed, self.pred_lens = self._process_data()

    def _process_data(self):
        """Feature processing pipeline aligned with finetune training code"""
        features = []  # Fixed shape [4, 250]
        labels = []  # Variable-length acceleration sequence [pred_len]
        meta = []  # Fixed 4-dimensional scene feature vector (Consistent with training code)
        raw_spacing = []
        raw_fv_speed = []
        raw_lv_speed = []
        pred_lens = []  # Record prediction length of each sample

        for sample in self.raw_data:
            try:
                # Load raw trajectory records
                spacing = np.array(sample[0], dtype=np.float32)
                fv_speed = np.array(sample[1], dtype=np.float32)
                rel_speed = np.array(sample[2], dtype=np.float32)
                lv_speed = np.array(sample[3], dtype=np.float32)

                # Filter short samples (Aligned with finetune training)
                total_len = min(len(spacing), len(fv_speed), len(rel_speed), len(lv_speed))
                if total_len < config.min_total_len:
                    continue

                # Compute acceleration sequence (Aligned with finetune training)
                acc = (fv_speed[1:total_len] - fv_speed[:total_len - 1]) / config.dt
                if len(acc) < config.min_pred_len:
                    continue

                # Filter abnormal physical values
                if np.any(fv_speed < 0) or np.any(lv_speed < 0) or np.any(spacing <= 0.3):
                    continue

                # Fix first seq_len steps as historical input (250 steps, consistent with training)
                input_end = config.seq_len
                label_start = config.seq_len
                max_label_end = min(len(acc), config.seq_len + config.max_pred_len)
                label_end = max_label_end
                pred_len = label_end - label_start

                # Filter samples with invalid prediction length
                if pred_len < config.min_pred_len or pred_len > config.max_pred_len:
                    continue

                # Extract historical input features
                input_spacing = spacing[:input_end]
                input_fv = fv_speed[:input_end]
                input_rel = rel_speed[:input_end]
                input_lv = lv_speed[:input_end]
                feat = np.stack([input_spacing, input_fv, input_rel, input_lv], axis=0)

                # Extract acceleration ground truth labels
                label_acc = acc[label_start:label_end]
                if len(label_acc) != pred_len or len(feat[0]) != config.seq_len:
                    continue

                # Extract ground truth trajectory segments matching prediction window
                raw_spacing_window = spacing[label_start:label_end]
                raw_fv_speed_window = fv_speed[label_start:label_end]
                raw_lv_speed_window = lv_speed[label_start:label_end]
                if len(raw_spacing_window) != pred_len or len(raw_fv_speed_window) != pred_len:
                    continue

                # Scene statistical features (4D, fully consistent with training code)
                speed_mean = np.mean(fv_speed[:input_end])
                speed_std = np.std(fv_speed[:input_end])
                max_acc_window = np.max(np.abs(acc[:label_end]))
                spacing_mean = np.mean(spacing[:input_end])
                scene_feat = np.array([speed_mean, speed_std, max_acc_window, spacing_mean],
                                      dtype=np.float32)

                # Standardization transformation (Aligned with finetune training)
                input_reshaped = feat.transpose(1, 0).reshape(-1, config.input_dim)
                input_scaled = self.scaler.transform(input_reshaped)
                feat_scaled = input_scaled.reshape(config.seq_len, config.input_dim).transpose(1, 0)

                # Append processed data to lists
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

        # Convert lists to numpy arrays
        features = np.array(features) if features else np.array([])
        meta = np.array(meta) if meta else np.array([])

        # Print dataset statistics
        if pred_lens:
            print(f"Preprocessing finished for domain {self.domain} | Valid sample count: {len(features)}")
            print(f"Prediction length range: {min(pred_lens)}~{max(pred_lens)} (Required: {config.min_pred_len}~{config.max_pred_len})")

        return features, labels, meta, raw_spacing, raw_fv_speed, raw_lv_speed, pred_lens

    def __getitem__(self, idx):
        pred_len = self.pred_lens[idx]
        return {
            "x": torch.FloatTensor(self.features[idx]),  # Fixed shape [4,250], consistent with training history length
            "y": torch.FloatTensor(self.labels[idx]),  # Variable-length acceleration [pred_len]
            "meta": torch.FloatTensor(self.meta[idx]),  # Fixed 4D scene feature, consistent with training
            "spacing": torch.FloatTensor(self.raw_spacing[idx]),  # Ground-truth gap [pred_len]
            "fv_speed": torch.FloatTensor(self.raw_fv_speed[idx]),  # Ground-truth following speed [pred_len]
            "lv_speed": torch.FloatTensor(self.raw_lv_speed[idx]),  # Ground-truth leading speed [pred_len]
            "domain": self.domain,
            "ego_speed_init": self.raw_fv_speed[idx][0] if pred_len > 0 else 0.0,
            "spacing_init": self.raw_spacing[idx][0] if pred_len > 0 else 0.0,
            "pred_len": pred_len,
            "total_len": config.seq_len + pred_len + 1
        }

    def __len__(self):
        return len(self.features) if hasattr(self, 'features') else 0


# -------------------------- 5. Model Definition (Fully consistent with finetune training code without IDM) --------------------------
class FeatureEncoder(nn.Module):
    """LSTM Feature Encoder (Identical to finetune training code)"""

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
        x = x.permute(2, 0, 1)  # [1,4,250] -> [250,1,4]
        out, (hn, cn) = self.lstm(x)
        return hn, cn


class Seq2SeqDecoder(nn.Module):
    """LSTM Sequence Decoder (Identical to finetune training code)"""

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


class Seq2SeqFollowingModel(nn.Module):
    """Core car-following model without IDM, aligned with finetune transfer training code"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core modules fully consistent with training script
        self.encoder = FeatureEncoder(
            config.input_dim, config.hidden_size, config.enc_layers, config.dropout
        )
        self.decoder = Seq2SeqDecoder(
            config.hidden_size, config.output_dim, config.dec_layers, config.dropout
        )
        self.meta_adapter = nn.Linear(config.hidden_size, config.output_dim)

        # Per-dataset acceleration clipping bounds (Aligned with finetune training)
        self.dataset_acc_min = {d: torch.tensor(v["min"], dtype=torch.float32).to(self.device) for d, v in
                                config.dataset_acc_limits.items()}
        self.dataset_acc_max = {d: torch.tensor(v["max"], dtype=torch.float32).to(self.device) for d, v in
                                config.dataset_acc_limits.items()}
        self.collision_threshold = config.collision_threshold

        # All layers trainable for finetuning (layer freezing disabled)
        for param in self.parameters():
            param.requires_grad = True

    def init_decoder_input(self, batch_size):
        return torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)

    def step_predict(self, enc_hidden, enc_cell, meta_feat, current_domain, step_num,
                     ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        """Single window prediction without IDM fusion, consistent with fixed weighting logic in training code"""
        batch_size = enc_hidden.shape[1]
        dec_input = self.init_decoder_input(batch_size)
        hidden, cell = enc_hidden, enc_cell
        step_preds = []

        # Acceleration range clipping bounds per dataset
        acc_min = self.dataset_acc_min.get(current_domain, self.dataset_acc_min["default"])
        acc_max = self.dataset_acc_max.get(current_domain, self.dataset_acc_max["default"])

        for t in range(step_num):
            # Pure network prediction without IDM
            base_pred, hidden, cell = self.decoder(dec_input, hidden, cell)
            meta_pred = base_pred + self.meta_adapter(hidden[-1])
            # Fixed weighted fusion consistent with training: 0.5*base + 0.5*meta
            fusion_pred = 0.5 * base_pred + 0.5 * meta_pred

            # Only apply acceleration range constraint
            fusion_pred = torch.clamp(fusion_pred, min=acc_min, max=acc_max)
            step_preds.append(fusion_pred)
            dec_input = fusion_pred

        return torch.cat(step_preds, dim=1), hidden, cell

    def forward(self, x, meta, current_domain, pred_len,
                ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        """Forward propagation aligned with finetune training without IDM"""
        batch_size = x.shape[0]
        total_preds = []
        enc_hidden, enc_cell = self.encoder(x)

        remaining_pred = pred_len
        current_hidden, current_cell = enc_hidden, enc_cell
        current_x = x

        while remaining_pred > 0:
            step_num = min(self.config.step_pred, remaining_pred)
            # Predict acceleration for current sliding window
            step_pred, current_hidden, current_cell = self.step_predict(
                current_hidden, current_cell, meta, current_domain, step_num,
                ego_speed_seq, front_speed_seq, spacing_seq
            )
            total_preds.append(step_pred)

            # Update historical input window without IDM logic
            new_x = self._build_new_input(current_x, step_pred, current_domain, front_speed_seq, step_num)
            current_x = new_x
            remaining_pred -= step_num

        # Concatenate all segments and trim to target prediction length
        total_preds = torch.cat(total_preds, dim=1)[:, :pred_len]
        if total_preds.shape[1] != pred_len:
            if total_preds.shape[1] > pred_len:
                total_preds = total_preds[:, :pred_len]
            else:
                pad_len = pred_len - total_preds.shape[1]
                padding = torch.zeros((batch_size, pad_len), device=total_preds.device, dtype=total_preds.dtype)
                total_preds = torch.cat([total_preds, padding], dim=1)

        return total_preds

    def _build_new_input(self, current_x, step_pred, current_domain, front_speed_seq=None, step_num=None):
        """Update sliding historical window aligned with finetune training without IDM"""
        batch_size = current_x.shape[0]
        step_num = step_num if step_num else self.config.step_pred

        # Reserve last (seq_len - step_num) steps of original input
        keep_len = self.config.seq_len - step_num
        x_slice = current_x[:, :, step_num:] if keep_len > 0 else current_x[:, :, -1:].repeat(1, 1, 1)
        x_slice = x_slice[:, :, :keep_len]
        if x_slice.shape[2] < keep_len:
            pad_len = keep_len - x_slice.shape[2]
            x_slice = torch.cat([x_slice, x_slice[:, :, -1:].repeat(1, 1, pad_len)], dim=2)

        # Construct new feature frames from predicted acceleration
        pred_feat = torch.zeros((batch_size, self.config.input_dim, step_num),
                                device=current_x.device, dtype=current_x.dtype)

        # Initial physical state variables consistent with training
        ego_speed_init = current_x[:, 1, -1]
        spacing_init = current_x[:, 0, -1]
        dt_tensor = torch.tensor(self.config.dt, dtype=current_x.dtype, device=current_x.device)
        s0 = 0  # Fixed minimum safe gap

        ego_speed = ego_speed_init
        spacing = spacing_init
        for t in range(step_num):
            # Integrate acceleration to get speed
            ego_speed = ego_speed + step_pred[:, t] * dt_tensor
            ego_speed = torch.clamp(ego_speed, min=0.1)

            # Fetch leading vehicle speed
            if front_speed_seq is not None:
                front_speed = front_speed_seq[:, t] if t < front_speed_seq.shape[1] else front_speed_seq[:, -1]
            else:
                front_speed = current_x[:, 3, -1]

            # Pure gap integration without IDM
            spacing = spacing + (front_speed - ego_speed) * dt_tensor
            spacing = torch.clamp(spacing, min=s0)

            # Fill four-dimensional traffic features
            pred_feat[:, 0, t] = spacing
            pred_feat[:, 1, t] = ego_speed
            pred_feat[:, 2, t] = front_speed - ego_speed
            pred_feat[:, 3, t] = front_speed

        new_x = torch.cat([x_slice, pred_feat], dim=2)[:, :, :self.config.seq_len]
        return new_x


# -------------------------- 6. Evaluation Function (Adapt to finetune model without IDM) --------------------------
def evaluate_model(domain, model, dataloader, device):
    model.eval()
    metrics = {
        "mse_acc": 0.0,  # MSE of acceleration
        "mae_acc": 0.0,  # MAE of acceleration
        "mae_speed": 0.0,  # MA of following vehicle speed
        "mae_abs_jerk": 0.0,  # MAE of absolute jerk
        "avg_min_ttc": 0.0,  # Average minimum TTC
        "mse_spacing": 0.0,  # MSE of gap distance
        "mae_spacing": 0.0,  # MAE of gap distance
        "collision_count": 0,  # Number of collision samples
        "collision_rate": 0.0,  # Collision percentage
        "valid_samples": 0,  # Total valid test samples
        "total_pred_steps": 0,  # Sum of all prediction timesteps
        "total_ttc_samples": 0,  # Valid samples with computable TTC
        "pred_len_stats": []  # Statistics of prediction lengths per sample
    }

    all_min_ttc = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating {domain.upper()} (Finetune + No IDM)")
        for batch in pbar:
            try:
                # Load batch data
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

                # Model forward pass (Finetune without IDM)
                pred_acc = model(
                    x, meta, current_domain, pred_len=pred_len,
                    ego_speed_seq=fv_speed_gt, front_speed_seq=lv_speed_gt, spacing_seq=spacing_gt
                )

                # Align prediction and ground truth length
                if pred_acc.shape[1] != y_true.shape[1]:
                    min_len = min(pred_acc.shape[1], y_true.shape[1])
                    pred_acc = pred_acc[:, :min_len]
                    y_true = y_true[:, :min_len]
                    spacing_gt = spacing_gt[:, :min_len] if spacing_gt.shape[1] > min_len else spacing_gt
                    fv_speed_gt = fv_speed_gt[:, :min_len] if fv_speed_gt.shape[1] > min_len else fv_speed_gt
                    lv_speed_gt = lv_speed_gt[:, :min_len] if lv_speed_gt.shape[1] > min_len else lv_speed_gt
                    pred_len = min_len

                # Predict gap distance via physical integration
                spacing_pred = calculate_spacing_from_acc(
                    pred_acc, ego_speed_init, lv_speed_gt, spacing_init, config.dt, current_domain, pred_len
                )

                # Integrate acceleration to predicted speed sequence
                speed_pred = integrate_speed(pred_acc, ego_speed_init, config.dt, pred_len)

                # 1. Core prediction error metrics
                mse_acc = nn.MSELoss()(pred_acc, y_true)
                mae_acc = nn.L1Loss()(pred_acc, y_true)

                mse_spacing = nn.MSELoss()(spacing_pred, spacing_gt)
                mae_spacing = nn.L1Loss()(spacing_pred, spacing_gt)

                mae_speed = nn.L1Loss()(speed_pred, fv_speed_gt)

                # 2. Jerk metric calculation
                jerk_pred = calculate_jerk(pred_acc, config.dt)
                jerk_true = calculate_jerk(y_true, config.dt)
                min_jerk_len = min(jerk_pred.shape[1], jerk_true.shape[1]) if jerk_pred.shape[1] > 0 else 0
                mae_abs_jerk = torch.tensor(0.0, device=device)
                if min_jerk_len > 0:
                    mae_abs_jerk = nn.L1Loss()(jerk_pred[:, :min_jerk_len], jerk_true[:, :min_jerk_len])

                # 3. Minimum TTC calculation
                min_ttc = calculate_ttc(spacing_pred, speed_pred, lv_speed_gt, pred_len)
                if min_ttc != float('inf'):
                    all_min_ttc.append(min_ttc)
                    metrics["total_ttc_samples"] += 1

                # 4. Collision detection
                collision = torch.any(spacing_pred <= model.collision_threshold)
                collision_count = 1 if collision else 0

                # Accumulate total metric values
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

                # Progress bar display
                pbar.set_postfix({
                    "PredLen": pred_len,
                    "MSE_acc": f"{mse_acc.item():.6f}",
                    "MAE_speed": f"{mae_speed.item():.6f}",
                    "Collision": collision_count,
                    "Mode": "Fine-tune + No IDM"
                })
            except Exception as e:
                print(f"Error processing sample from domain {domain}: {str(e)[:50]}...")
                continue

    # Compute average metrics weighted by prediction steps
    if metrics["total_pred_steps"] > 0:
        metrics["mse_acc"] /= metrics["total_pred_steps"]
        metrics["mae_acc"] /= metrics["total_pred_steps"]
        metrics["mse_spacing"] /= metrics["total_pred_steps"]
        metrics["mae_spacing"] /= metrics["total_pred_steps"]
        metrics["mae_speed"] /= metrics["total_pred_steps"]
        if metrics["total_pred_steps"] - len(metrics["pred_len_stats"]) > 0:
            metrics["mae_abs_jerk"] /= (metrics["total_pred_steps"] - len(metrics["pred_len_stats"]))

    # Calculate collision rate and average TTC
    if metrics["valid_samples"] > 0:
        metrics["collision_rate"] = (metrics["collision_count"] / metrics["valid_samples"]) * 100
        metrics["avg_min_ttc"] = np.mean(all_min_ttc) if len(all_min_ttc) > 0 else float('inf')

    # Print prediction length statistics
    if metrics["pred_len_stats"]:
        print(f"\nPrediction length statistics for domain {domain} (Finetune + No IDM):")
        print(f"  Min: {min(metrics['pred_len_stats'])} | Max: {max(metrics['pred_len_stats'])} | Mean: {np.mean(metrics['pred_len_stats']):.1f}")

    return metrics


# -------------------------- 7. Main Test Pipeline (Finetune model without IDM) --------------------------
def main():
    # Configuration validation
    print("=" * 80)
    print("Start testing pipeline: Finetune transfer model without IDM on HighD dataset")
    print(f"Model pre-training domains: {config.source_domains} | Finetune target domains: {config.target_domains} | Test domains: {config.test_domains}")
    print(f"Finetune transfer enabled | IDM fusion disabled | Layer freezing disabled")
    print("=" * 80)

    # Load saved finetune model without IDM
    finetune_model_path = os.path.join(config.save_dir, "finetune_best.pth")
    pretrain_model_path = os.path.join(config.save_dir, "pretrain_best.pth")

    if os.path.exists(finetune_model_path):
        model_path = finetune_model_path
        model_type = "Best Finetuned Checkpoint"
    elif os.path.exists(pretrain_model_path):
        model_path = pretrain_model_path
        model_type = "Pre-trained Checkpoint"
    else:
        raise FileNotFoundError(f"No model checkpoint found under directory: {config.save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Loading {model_type} to computing device: {device}")
    print(f"   Checkpoint file path: {model_path}")

    # Initialize model matching training hyperparameters
    model = Seq2SeqFollowingModel(config).to(device)

    # Load checkpoint weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    scaler = checkpoint["scaler"]
    best_loss = checkpoint.get("best_loss", "N/A")
    print(f"✅ Successfully loaded {model_type} weights (Best validation loss: {best_loss})")
    print(f"✅ Layer freeze setting: {config.freeze_layers} | IDM fusion: Disabled | Finetune transfer: Enabled")
    print(f"✅ Historical sequence length: {config.seq_len} steps | Recursive prediction step size: {config.step_pred} steps")

    all_results = {}
    result_df = pd.DataFrame(columns=[
        "Dataset", "Valid Samples", "Total Prediction Steps", "Avg Prediction Length",
        "Acceleration MSE", "Acceleration MAE", "Speed MAE (m/s)", "Absolute Jerk MAE (m/s³)",
        "Gap MSE (m²)", "Gap MAE (m)", "Avg Minimum TTC (s)",
        "Collision Count", "Collision Rate (%)", "Model Version"
    ])

    # Iterate all target test domains
    for domain in config.test_domains:
        print(f"\n{'=' * 60}")
        print(f"Evaluate dataset {domain.upper()} (Finetune + No IDM)")
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
                round(metrics["collision_rate"], 2),
                "Fine-tune + No IDM"
            ]

            # Print per-domain evaluation metrics
            print(f"\n📊 Test metrics of {domain.upper()} (Finetune + No IDM):")
            print(f"   Valid sample count: {metrics['valid_samples']} | Total prediction timesteps: {metrics['total_pred_steps']} | Avg prediction length: {avg_pred_len:.1f}")
            print(f"   Acceleration MSE: {metrics['mse_acc']:.6f} | Acceleration MAE: {metrics['mae_acc']:.6f}")
            print(f"   Speed MAE: {metrics['mae_speed']:.6f} m/s | Absolute Jerk MAE: {metrics['mae_abs_jerk']:.6f} m/s³")
            print(f"   Gap MSE: {metrics['mse_spacing']:.3f} m² | Gap MAE: {metrics['mae_spacing']:.3f} m")
            print(f"   Average minimum TTC: {metrics['avg_min_ttc']:.3f} s" if metrics["avg_min_ttc"] != float('inf') else "   Average minimum TTC: N/A")
            print(f"   Collision count: {metrics['collision_count']} | Collision rate: {metrics['collision_rate']:.2f}%")

        except Exception as e:
            print(f"❌ Evaluation failed on {domain.upper()} (Finetune + No IDM): {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save and plot results if data exists
    if not result_df.empty:
        result_excel_path = os.path.join(config.result_dir, "test_results_highd_finetune_no_idm.xlsx")
        result_df.to_excel(result_excel_path, index=False, engine="openpyxl")
        print(f"\n✅ Test result excel saved to: {result_excel_path}")

        # Print summary table
        print(f"\n{'=' * 80}")
        print(f"Summary of Finetune Model Metrics on HighD Dataset (No IDM)")
        print(f"{'=' * 80}")
        print(result_df.to_string(index=False))

        # Draw performance comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Performance of Finetune Transfer Model without IDM - HighD Dataset", fontsize=16, fontweight="bold")
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

        axes[1, 0].bar(domains, result_df["Gap MSE (m²)"], color="#C73E1D", alpha=0.8)
        axes[1, 0].set_title("Gap MSE (m²)", fontweight="bold")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        axes[1, 1].bar(domains, result_df["Collision Rate (%)"], color="#6A994E", alpha=0.8)
        axes[1, 1].set_title("Collision Rate (%)", fontweight="bold")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        axes[1, 2].bar(domains, result_df["Avg Prediction Length"], color="#F77F00", alpha=0.8)
        axes[1, 2].set_title("Average Prediction Length (step)", fontweight="bold")
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig_path = os.path.join(config.result_dir, "test_performance_highd_finetune_no_idm.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Performance comparison figure saved to: {fig_path}")

    print(f"\n{'=' * 80}")
    print("Variable-length test pipeline finished for Finetune Transfer Model without IDM on HighD dataset")
    print(f"Result storage directory: {config.result_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()