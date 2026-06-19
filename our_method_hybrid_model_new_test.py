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

# Ensure StandardScaler can be safely serialized
torch.serialization.add_safe_globals([StandardScaler])
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["SimHei", "Times New Roman"]
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 1. Configuration Class (Adapt to variable-length samples) --------------------------
class Config:
    # Core input & output settings (Adapt to variable-length)
    input_dim = 4  # 4-dimensional input: gap distance, following vehicle speed, relative speed, leading vehicle speed
    output_dim = 1  # Single-step acceleration output
    # seq_len = 100  # Fixed historical length (first 100 steps)
    dt = 0.1  # Fixed 10Hz time interval (0.1s per step)
    # Ablation studies for different history and prediction lengths
    seq_len = 100    #10, 30, 50, 80, 100
    step_pred = 1   #0.4, 0.6, 0.8, 1, 1.8,3,4,8(4,6,8,10,18,30,40,80)

    # Variable-length settings
    min_total_len = 151  # Minimum total sample length (100 history steps + min 50 prediction steps + 1)
    min_pred_len = 50  # Minimum number of acceleration prediction steps
    max_pred_len = 80  # Maximum number of acceleration prediction steps
    # step_pred = 10  # Predict 10 steps per sliding window iteration

    # Domain settings
    domains = ["ngsim", "lyft", "waymo", "spmd1", "spmd2"]
    source_domains = ["ngsim", "spmd2"]
    target_domains = ["spmd1", "lyft", "waymo"]

    # Acceleration constraints for each dataset
    dataset_acc_limits = {
        "ngsim": {"min": -2.7, "max": 2.6},
        "spmd1": {"min": -3.5, "max": 2.0},
        "spmd2": {"min": -3.5, "max": 2.0},
        "waymo": {"min": -4.0, "max": 3.0},
        "lyft": {"min": -3.0, "max": 2.9}
    }

    # IDM hyperparameters
    idm_params = {
        "ngsim": {"v0": 38.11, "T": 1.39, "a": 1.66, "b": 1.18, "s0": 6.63, "delta": 1.57},
        "spmd1": {"v0": 41.58, "T": 0.66, "a": 1.21, "b": 0.50, "s0": 5.26, "delta": 2.85},
        "spmd2": {"v0": 41.36, "T": 0.74, "a": 0.65, "b": 0.49, "s0": 0.94, "delta": 2.26},
        "waymo": {"v0": 41.32, "T": 1.59, "a": 1.80, "b": 0.25, "s0": 7.29, "delta": 7.17},
        "lyft": {"v0": 13.41, "T": 1.62, "a": 1.70, "b": 0.27, "s0": 12, "delta": 1.90},
    }
    # λ(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    # idm_weight = 0.3  # Fusion weight for IDM acceleration
    idm_weight = 0.1

    # Model hyperparameters
    hidden_size = 64
    enc_layers = 2
    dec_layers = 2
    dropout = 0.1
    meta_alpha = 0.01
    freeze_layers = 1

    # Test settings
    batch_size = 1  # Must be set to 1 for variable-length samples to avoid dimension mismatch
    collision_threshold = 0.0  # Threshold to judge collision occurrence

    # File path settings
    save_dir = "output/our_method_with_IDM_final_new"
    # result_dir = os.path.join(save_dir, "test_results_variable_len")
    # result_dir = os.path.join(save_dir, "ablation_test_result")  # Ablation for different λ values
    # result_dir = os.path.join(save_dir, "different_history_step_test_result")  # Different history length experiments
    result_dir = os.path.join(save_dir, "different_slipe_step_test_result")  # Different sliding window length experiments
    data_paths = {
        "ngsim": {"test": "dataset/data/NGSIM_I_80_test_data.npy"},
        "waymo": {"test": "dataset/data/Waymo_test_data.npy"},
        "lyft": {"test": "dataset/data/Lyft_test_data.npy"},
        "spmd1": {"test": "dataset/data/SPMD1_test_data.npy"},
        "spmd2": {"test": "dataset/data/SPMD2_test_data.npy"}
    }


config = Config()
# Verify total length constraint: total length = history length + prediction length + 1
assert config.min_total_len - config.seq_len >= config.min_pred_len + 1, \
    f"Minimum total length {config.min_total_len} minus history length {config.seq_len} should be greater than or equal to minimum prediction length {config.min_pred_len} + 1"
os.makedirs(config.result_dir, exist_ok=True)


# -------------------------- 2. Utility Functions (Adapt to variable-length input) --------------------------
def integrate_speed(acc_seq, speed_init, dt, pred_len):
    """Integrate acceleration sequence to obtain speed sequence (support variable length)"""
    speed_seq = torch.zeros((1, pred_len), device=acc_seq.device, dtype=acc_seq.dtype)
    speed_seq[:, 0] = speed_init
    for t in range(1, pred_len):
        speed_seq[:, t] = speed_seq[:, t - 1] + acc_seq[:, t - 1] * dt
        speed_seq[:, t] = torch.clamp(speed_seq, min=0.1)  # Non-negative speed constraint
    return speed_seq


def calculate_jerk(acc_seq, dt):
    """Compute absolute jerk values (support variable length)"""
    if acc_seq.shape[1] <= 1:
        return torch.zeros((1, 0), device=acc_seq.device)  # Avoid error for short sequences
    jerk = (acc_seq[:, 1:] - acc_seq[:, :-1]) / dt
    abs_jerk = torch.abs(jerk)
    return abs_jerk


def calculate_ttc(spacing_seq, fv_speed_seq, lv_speed_seq, pred_len):
    """Calculate Time-To-Collision (TTC) metric (support variable length)"""
    delta_speed = fv_speed_seq - lv_speed_seq
    valid_mask = delta_speed > 1e-3  # Prevent division by zero
    ttc = torch.zeros((1, pred_len), device=spacing_seq.device)
    ttc[valid_mask] = spacing_seq[valid_mask] / delta_speed[valid_mask]
    ttc[~valid_mask] = float('inf')

    # Calculate minimum valid TTC
    valid_ttc = ttc[0][valid_mask[0]]
    if len(valid_ttc) > 0:
        min_ttc = torch.min(valid_ttc).item()
    else:
        min_ttc = float('inf')
    return min_ttc


# -------------------------- 2. Core IDM Functions (Adapt to variable-length input) --------------------------
def idm_desired_spacing(ego_speed, front_speed, params):
    """Calculate desired safe gap distance of IDM model"""
    delta_v = ego_speed - front_speed
    s_star = params["s0"] + ego_speed * params["T"] + (ego_speed * delta_v) / (2 * np.sqrt(params["a"] * params["b"]))
    return np.maximum(s_star, params["s0"])


def idm_acceleration(ego_speed, front_speed, current_spacing, params):
    """Compute ID-model predicted acceleration (vectorized calculation)"""
    # Convert tensor to numpy array for computation
    if torch.is_tensor(ego_speed):
        ego_speed = ego_speed.cpu().numpy()
    if torch.is_tensor(front_speed):
        front_speed = front_speed.cpu().numpy()
    if torch.is_tensor(current_spacing):
        current_spacing = current_spacing.cpu().numpy()

    s_star = idm_desired_spacing(ego_speed, front_speed, params)
    current_spacing = np.maximum(current_spacing, 0.1)
    acc_idm = params["a"] * (1 - (ego_speed / params["v0"]) ** params["delta"] - (s_star / current_spacing) ** 2)
    return acc_idm


def calculate_spacing_from_acc(pred_acc, ego_speed_init, front_speed_seq, init_spacing, dt, domain, pred_len):
    """Integrate gap distance with IDM physical constraints (support variable length)"""
    device = pred_acc.device
    speed_len = pred_len + 1
    spacing_len = pred_len + 1

    ego_speed = torch.zeros((1, speed_len), device=device, dtype=pred_acc.dtype)
    spacing_pred = torch.zeros((1, spacing_len), device=device, dtype=pred_acc.dtype)
    ego_speed[:, 0] = ego_speed_init
    spacing_pred[:, 0] = init_spacing

    dt_tensor = torch.tensor(dt, dtype=pred_acc.dtype, device=device)
    idm_params = config.idm_params.get(domain, config.idm_params["ngsim"])
    s0 = torch.tensor(idm_params["s0"], dtype=pred_acc.dtype, device=device)

    for t in range(1, speed_len):
        # Integrate speed from acceleration
        ego_speed_t = ego_speed[:, t - 1] + pred_acc[:, t - 1] * dt_tensor if t - 1 < pred_len else ego_speed[:, t - 1]
        ego_speed_t = torch.clamp(ego_speed_t, min=0.1)
        ego_speed[:, t] = ego_speed_t

        # Fetch leading vehicle speed, avoid out-of-bounds for variable-length sequence
        front_speed_t = front_speed_seq[:, t] if t < front_speed_seq.shape[1] else front_speed_seq[:, -1]

        # Integrate gap distance with IDM safety lower bound
        spacing_t = spacing_pred[:, t - 1] + (front_speed_t - ego_speed_t) * dt_tensor
        if config.idm_weight == 0:
            spacing_t = torch.clamp(spacing_t, min=0)
        else:
            spacing_t = torch.clamp(spacing_t, min=s0)
        spacing_pred[:, t] = spacing_t

    # Return gap sequence matched with prediction length (take last pred_len steps)
    return spacing_pred[:, 1:1 + pred_len]


# -------------------------- 3. Dataset Class (Core fix: length alignment) --------------------------
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
        """Core fix: strictly align acceleration label length with prediction length"""
        features = []  # Fixed dimension: [4, 100]
        labels = []  # Variable dimension: acceleration sequence [pred_len]
        meta = []  # Fixed dimension: [5] scene statistical features
        raw_spacing = []  # Variable-length ground-truth gap sequence
        raw_fv_speed = []  # Variable-length following vehicle speed sequence
        raw_lv_speed = []  # Variable-length leading vehicle speed sequence
        pred_lens = []  # Store prediction length of each sample

        for sample in self.raw_data:
            try:
                # Load raw trajectory data
                spacing = np.array(sample[0], dtype=np.float32)
                fv_speed = np.array(sample[1], dtype=np.float32)
                rel_speed = np.array(sample[2], dtype=np.float32)
                lv_speed = np.array(sample[3], dtype=np.float32)

                # Filter short samples below minimum total length threshold
                total_len = min(len(spacing), len(fv_speed), len(rel_speed), len(lv_speed))
                if total_len < config.min_total_len:
                    continue

                # Calculate acceleration sequence from speed difference
                acc = (fv_speed[1:] - fv_speed[:-1]) / config.dt
                lv_acc = (lv_speed[1:] - lv_speed[:-1]) / config.dt

                # Filter samples with abnormal physical values
                if np.any(fv_speed < 0) or np.any(lv_speed < 0) or np.any(spacing <= 0.3):
                    continue
                # if np.any(fv_speed < 0.1) or np.any(lv_speed < 0.1) or np.any(spacing <= 0.3):
                #     continue
                # if np.any(acc < -15) or np.any(acc > 15) or np.any(lv_acc < -15) or np.any(lv_acc > 15):
                #     continue

                # Fix first seq_len steps as historical input segment
                input_end = config.seq_len
                # Start index of acceleration label sequence
                label_start = config.seq_len
                # Max valid end index for label sequence
                max_label_end = min(len(acc), config.seq_len + config.max_pred_len)
                label_end = max_label_end

                # Actual prediction length of current sample
                pred_len = label_end - label_start

                # Discard samples with insufficient prediction steps
                if pred_len < config.min_pred_len:
                    continue

                # Extract fixed-length historical input features
                input_spacing = spacing[:input_end]
                input_fv = fv_speed[:input_end]
                input_rel = rel_speed[:input_end]
                input_lv = lv_speed[:input_end]
                feat = np.stack([input_spacing, input_fv, input_rel, input_lv], axis=0)

                # Extract acceleration ground truth matched to pred_len
                label_acc = acc[label_start:label_end]
                if len(label_acc) != pred_len or len(feat[0]) != config.seq_len:
                    continue

                # Extract ground truth trajectory window matched with prediction range
                raw_spacing_window = spacing[label_start + 1: label_end + 1]
                raw_fv_speed_window = fv_speed[label_start + 1: label_end + 1]
                raw_lv_speed_window = lv_speed[label_start + 1: label_end + 1]

                if len(raw_spacing_window) != pred_len or len(raw_fv_speed_window) != pred_len:
                    continue

                # Compute 5-dimensional scene statistical features
                speed_mean = np.mean(fv_speed[:input_end])
                speed_std = np.std(fv_speed[:input_end])
                max_acc_window = np.max(np.abs(acc[:label_end]))
                spacing_mean = np.mean(spacing[:input_end])
                rel_speed_mean = np.mean(rel_speed[:input_end])
                scene_feat = np.array([speed_mean, speed_std, max_acc_window, spacing_mean, rel_speed_mean],
                                      dtype=np.float32)

                # Standardization for historical input
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
                print(f"Error processing sample from domain {self.domain}: {str(e)[:50]}...")
                continue

        # Convert fixed-length arrays to numpy format
        features = np.array(features) if features else np.array([])
        meta = np.array(meta) if meta else np.array([])

        # Print dataset preprocessing statistics
        if pred_lens:
            print(f"Preprocessing finished for domain {self.domain} | Valid sample count: {len(features)}")
            print(f"Prediction length range: {min(pred_lens)}~{max(pred_lens)} (required range: {config.min_pred_len}~{config.max_pred_len})")

        return features, labels, meta, raw_spacing, raw_fv_speed, raw_lv_speed, pred_lens

    def __getitem__(self, idx):
        pred_len = self.pred_lens[idx]  # Variable prediction length per sample
        return {
            "x": torch.FloatTensor(self.features[idx]),  # Fixed [4,100] historical input
            "y": torch.FloatTensor(self.labels[idx]),  # Variable-length acceleration label
            "meta": torch.FloatTensor(self.meta[idx]),  # Fixed 5D scene feature
            "spacing": torch.FloatTensor(self.raw_spacing[idx]),  # Ground-truth gap sequence
            "fv_speed": torch.FloatTensor(self.raw_fv_speed[idx]),  # Ground-truth following speed
            "lv_speed": torch.FloatTensor(self.raw_lv_speed[idx]),  # Ground-truth leading speed
            "domain": self.domain,
            "ego_speed_init": self.raw_fv_speed[idx][0] if pred_len > 0 else 0.0,
            "spacing_init": self.raw_spacing[idx][0] if pred_len > 0 else 0.0,
            "pred_len": pred_len,
            "total_len": config.seq_len + pred_len + 1
        }

    def __len__(self):
        return len(self.features) if hasattr(self, 'features') else 0


# -------------------------- 4. Model Definition (Support variable-length prediction) --------------------------
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


class DomainDiscriminator(nn.Module):
    def __init__(self, hidden_size, num_domains):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_domains)
        )

    def forward(self, x):
        x = x.squeeze(1) if x.dim() == 3 else x
        return self.fc(x)


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
        return torch.softmax(weighted_weights, dim=2 if raw_weights.dim() == 3 else 1)

class LearnableWeights(nn.Module):
    """⭐ Learn adaptive loss weights automatically"""
    def __init__(self, num_tasks=3):
        super().__init__()
        # Learnable log variance parameter to ensure positive standard deviation
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        losses: [loss_acc, loss_spacing, loss_adv]
        return weighted total loss + regularization term
        """
        total_loss = 0
        regularization = 0

        for i, loss in enumerate(losses):
            # Weight = 1 / (2 * σ²) = exp(-log_var) / 2
            precision = torch.exp(-self.log_vars[i])  # 1/σ²
            weight = precision / 2

            total_loss += weight * loss
            regularization += self.log_vars[i]  # log(σ)

        # Total loss = weighted loss sum + regularization
        return total_loss + regularization
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
        self.domain_discriminator = DomainDiscriminator(config.hidden_size, len(config.domains))
        self.dataset_acc_min = {d: torch.tensor(v["min"], dtype=torch.float32).to(self.device) for d, v in config.dataset_acc_limits.items()}
        self.dataset_acc_max = {d: torch.tensor(v["max"], dtype=torch.float32).to(self.device) for d, v in config.dataset_acc_limits.items()}
        self.collision_threshold = config.collision
        # Newly added task weight learning module
        self.learnable_weights = LearnableWeights(num_tasks=3)

    def init_decoder_input(self, batch_size):
        return torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)

    def step_predict(self, enc_hidden, enc_cell, meta_feat, current_domain, step_num,
                     ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        """Single window iterative prediction for variable-length output"""
        batch_size = enc_hidden.shape[1]
        dec_input = self.init_decoder_input(batch_size)
        hidden, cell = enc_hidden, enc_cell
        step_preds = []

        # Domain-specific acceleration bounds
        acc_min = self.dataset_acc_min.get(current_domain, self.dataset_acc_min["ngsim"])
        acc_max = self.dataset_acc_max.get(current_domain, self.dataset_acc_max["ngsim"])
        idm_params = self.config.idm_params.get(current_domain, self.config.idm_params["ngsim"])

        for t in range(step_num):
            # Base prediction from decoder
            base_pred, hidden, cell = self.decoder(dec_input, hidden, cell)
            meta_pred = base_pred + self.meta_adapter(hidden[-1])
            weights = self.dynamic_weight(meta_feat)
            fusion_pred = weights[:, 0:1] * base_pred + weights[:, 1:2] * meta_pred

            # Fuse IDM acceleration if trajectory sequence is provided
            if ego_speed_seq is not None and front_speed_seq is not None and spacing_seq is not None:
                ego_speed = ego_speed[:, t] if t < ego_speed_seq.shape[1] else ego_speed_seq[:, -1]
                front_speed = front_speed[:, t] if t < front_speed_seq.shape[1] else front_speed_seq[:, -1]
                spacing = spacing_seq[:, t] if t < spacing_seq.shape[1] else spacing_seq[:, -1]

                idm_acc = idm_acceleration(ego_speed, front_speed, spacing, idm_params)
                idm_acc = torch.tensor(idm_acc, dtype=fusion_pred.dtype, device=fusion_pred.device).unsqueeze(1)

                fusion_pred = (1 - self.config.idm_weight) * fusion_pred + self.config.idm_weight * idm_acc

            # Constrain acceleration within physical range
            a_low = torch.tensor(-idm_params["b"], dtype=fusion_pred.dtype, device=fusion_pred.device)
            a_high = idm_params["a"] * (1.0 - (ego_speed[:, 0:1] / idm_params["v0"]) ** idm_params["delta"])
            fusion_pred = torch.clamp(fusion_pred, min=a_low, max=a_high).reshape(-1, 1)
            step_preds.append(fusion_pred)
            dec_input = fusion_pred

        return torch.cat(step_preds, dim=1), hidden, cell

    def forward(self, x, meta, current_domain, pred_len,
                ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        """Forward propagation supporting arbitrary variable prediction length"""
        batch_size = x.shape[0]
        total_preds = []
        enc_hidden, enc_cell = self.encoder(x)
        domain_feat = enc_hidden[-1]
        domain_pred = self.domain_discriminator(domain_feat)
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

            # Update sliding historical input window
            new_x = self._build_new_input(current_x, step_pred, current_domain, front_speed_seq, step_num)
            current_x = new_x
            remaining_pred -= step_num

        # Concatenate and trim total prediction to target pred_len
        total_preds = torch.cat(total_preds, dim=1)[:, :pred_len]
        if total_preds.shape[1] != pred_len:
            if total_preds.shape[1] > pred_len:
                total_preds = total_preds[:, :pred_len]
            else:
                pad_len = pred_len - total_preds.shape[1]
                padding = torch.zeros((batch_size, pad_len), device=total_preds.device, dtype=total_preds.dtype)
                total_preds = torch.cat([total_preds, padding], dim=1)

        return total_preds, domain_pred

    def _build_new_input(self, current_x, step_pred, current_domain, front_speed_seq=None, step_num=None):
        """Reconstruct sliding input window after each prediction iteration"""
        batch_size = current_x.shape[0]
        step_num = step_num if step_num else self.config.step_pred
        step_num = min(step_num, self.config.seq_len)
        step_num = max(step_num, 1)

        keep_len = self.config.seq_len - step_num
        if keep_len > 0:
            x_slice = current_x[:, :, step_num:]
        else:
            x_slice = current_x[:, :, -1:].repeat(1, 1, keep_len if keep_len > 0 else 1)
        x_slice = x_slice[:, :, :keep_len]
        if x_slice.shape[2] < keep_len:
            pad_len = keep_len - x_slice.shape[2]
            x_slice = torch.cat([x_slice, x_slice[:, :, -1:].repeat(1, 1, pad_len)], dim=2)

        pred_feat = torch.zeros((batch_size, self.config.input_dim, step_num),
                                device=current_x.device, dtype=current_x.dtype)

        ego_speed_init = current_x[:, 1, -1]
        spacing_init = current_x[:, 0, -1]
        dt_tensor = torch.tensor(self.config.dt, dtype=current_x.dtype, device=current_x.device)
        idm_params = self.config.idm_params.get(current_domain, self.config.idm_params["ngsim"])
        s0 = idm_params["s0"]

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


# -------------------------- 5. Evaluation Function (Variable-length compatible) --------------------------
def evaluate_model(domain, model, dataloader, device):
    model.eval()
    metrics = {
        "mse_acc": 0.0,  # MSE of acceleration
        "mae_acc": 0.0,  # MAE of acceleration
        "mae_speed": 0.0,  # MAE of following vehicle speed
        "mae_abs_jerk": 0.0,  # MAE of absolute jerk
        "avg_pred_abs_jerk": 0.0,  # Average absolute jerk of prediction
        "max_pred_abs_jerk": 0.0,  # Maximum absolute jerk of prediction
        "std_pred_abs_jerk": 0.0,  # Standard deviation of predicted absolute jerk
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
        pbar = tqdm(dataloader, desc=f"Evaluating {domain.upper()}")
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

                # Model forward inference
                pred_acc, _ = model(
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

                # Predict gap from acceleration integration
                spacing_pred = calculate_spacing_from_acc(
                    pred_acc, ego_speed_init, lv_speed_gt, spacing_init, config.dt, current_domain, pred_len
                )

                # Integrate predicted speed sequence
                speed_pred = integrate_speed(pred_acc, ego_speed_init, config.dt, pred_len)

                # Compute acceleration error metrics
                mse_acc = nn.MSELoss()(pred_acc, y_true)
                mae_acc = nn.L1Loss()(pred_acc, y_true)

                # Compute gap error metrics
                if spacing_pred.shape[1] != spacing_gt.shape[1]:
                    min_spacing_len = min(spacing_pred.shape[1], spacing_gt.shape[1])
                    spacing_pred = spacing_pred[:, :min_spacing_len]
                    spacing_gt = spacing_gt[:, :min_spacing_len]
                mse_spacing = nn.MSELoss()(spacing_pred, spacing_gt)
                mae_spacing = nn.L1Loss()(spacing_pred, spacing_gt)

                # Compute speed error metrics
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

                # Record predicted jerk statistics
                if jerk_pred.shape[1] > 0:
                    current_pred_abs_jerk = jerk_pred.abs().mean().item()
                    current_max_abs_jerk = jerk_pred.abs().max().item()
                    current_std_abs_jerk = jerk_pred.abs().std().item()
                    metrics["avg_pred_abs_jerk"] += current_pred_abs_jerk * pred_len
                    metrics["max_pred_abs_jerk"] = max(metrics["max_pred_abs_jerk"], current_max_abs_jerk)
                    metrics["std_pred_abs_jerk"] += current_std_abs_jerk * pred_len

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
                metrics["pred_len_stats"].append(pred_len)

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
        if (metrics["total_pred_steps"] - len(metrics["pred_len_stats"])) > 0:
            metrics["mae_abs_jerk"] /= (metrics["total_pred_steps"] - len(metrics["pred_len_stats"]))
        metrics["avg_pred_abs_jerk"] /= metrics["total_pred_steps"]
        metrics["std_pred_abs_jerk"] /= metrics["total_pred_steps"]

    # Calculate collision rate and average TTC
    if metrics["valid_samples"] > 0:
        metrics["collision_rate"] = (metrics["collision_count"] / metrics["valid_samples"]) * 100
        if len(all_min_ttc) > 0:
            metrics["avg_min_ttc"] = np.mean(all_min_ttc)
        else:
            metrics["avg_min_ttc"] = float('inf')

    # Print prediction length statistics
    if metrics["pred_len_stats"]:
        print(f"\nPrediction length statistics for domain {domain}:")
        print(f"  Min: {min(metrics['pred_len_stats'])} | Max: {max(metrics['pred_len_stats'])} | Mean: {np.mean(metrics['pred_len_stats']):.1f}")

    return metrics


# -------------------------- 6. Main Test Pipeline --------------------------
def main():
    # Configuration check header
    print("=" * 80)
    print("Start testing pipeline for IDM-fused Seq2Seq model with variable-length prediction")
    print("=" * 80)

    # Load trained checkpoint
    # model_path = os.path.join(save_dir, "pretrain_best.pth")
    model_path = os.path.join(config.save_dir, "finetune_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at path: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Loading model to computing device: {device}")

    # Initialize model instance
    model = Seq2SeqFollowingModel(config).to(device)

    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    scaler = checkpoint["scaler"]
    best_loss = checkpoint.get("finetune_loss", "N/A")
    print(f"✅ Successfully loaded model weights (Best validation loss: {best_loss})")
    print(f"Freeze layer setting: {config.freeze_layers} | IDM fusion weight: {config.idm_weight}")

    all_results = {}
    result_df = pd.DataFrame(columns=[
        "Dataset", "Valid Samples", "Total Prediction Steps", "Avg Prediction Length",
        "Acceleration MSE", "Acceleration MAE", "Speed MAE (m/s)", "Absolute Jerk MAE (m/s³)",
        "Avg Pred Jerk", "Max Pred Jerk", "Std Pred Jerk",
        "Gap MSE (m²)", "Gap MAE (m)", "Avg Min TTC (s)",
        "Collision Count", "Collision Rate (%)"
    ])

    # Iterate over all test domains
    for domain in config.domains:
        print(f"\n{'=' * 60}")
        print(f"Evaluating dataset {domain.upper()}")
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

            metrics = evaluate_model(domain, model, test_loader)
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
                round(metrics["avg_pred_abs_jerk"], 6),
                round(metrics["max_pred_abs_jerk"], 6),
                round(metrics["std_pred_abs_jerk"], 6),
                round(metrics["mse_spacing"], 3),
                round(metrics["mae_spacing"], 3),
                round(metrics["avg_min_ttc"], 3) if metrics["avg_min_ttc"] != float('inf') else "N/A",
                metrics["collision_count"],
                round(metrics["collision_rate"], 2)
            ]

            # Print per-domain test metrics
            print(f"\n📊 Test metrics for {domain.upper()}:")
            print(f"Valid samples: {metrics['valid_samples']}, total prediction steps: {metrics['total_pred_steps']}, average pred length: {avg_pred_len:.1f}")
            print(f"Acceleration MSE: {metrics['mse_acc']:.6f}, MAE: {metrics['mae_acc']:.6f}")
            print(f"Speed MAE: {metrics['mae_speed']:.6f} m/s, Absolute Jerk MAE: {metrics['mae_abs_jerk']:.6f} m/s³")
            print(f"Gap MSE: {metrics['mse_spacing']:.3f} m², MAE: {metrics['mae_spacing']:.3f} m")
            print(f"Average minimum TTC: {metrics['avg_min_ttc']:.3f} s" if metrics["avg_min_ttc"] != float('inf') else "Average minimum TTC: N/A")
            print(f"Collision count: {metrics['collision_count']}, collision rate: {metrics['collision_rate']:.2f}%")

        except Exception as e:
            print(f"❌ Evaluation failed on {domain.upper()}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save output files if results exist
    if not result_df.empty:
        excel_path = os.path.join(config.result_dir, "test_results_variable_len.xlsx")
        result_df.to_excel(excel_path, index=False, engine="openpyxl")
        print(f"\n✅ Test result table saved to {excel_path}")

        # Print overall summary table
        print(f"\n{'=' * 80}")
        print("Overall test performance summary")
        print(f"{'=' * 80}")
        print(result_df.to_string(index=False))

        # Draw performance figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Performance of IDM-fused Seq2Seq Model (Variable-length Prediction)", fontsize=16, fontweight="bold")
        domain_list = result_df["Dataset"].tolist()

        axes[0, 0].bar(domain_list, result_df["Acceleration MSE"], color="#2E86AB", alpha=0.8)
        axes[0, 0].set_title("Acceleration MSE", fontweight="bold")
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].bar(domain_list, result_df["Speed MAE (m/s)"], color="#F18F01", alpha=0.8)
        axes[0, 1].set_title("Speed MAE (m/s)", fontweight="bold")
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].bar(domain_list, result_df["Absolute Jerk MAE (m/s³)"], color="#A23B72", alpha=0.8)
        axes[0, 2].set_title("Absolute Jerk MAE (m/s³)", fontweight="bold")
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].bar(domain_list, result_df["Gap MSE (m²)"], color="#C73E1D", alpha=0.8)
        axes[1, 0].set_title("Gap MSE (m²)", fontweight="bold")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].bar(domain_list, result_df["Collision Rate (%)"], color="#6A994E", alpha=0.8)
        axes[1, 1].set_title("Collision Rate (%)", fontweight="bold")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].bar(domain_list, result_df["Avg Prediction Length"], color="#F77F00", alpha=0.8)
        axes[1, 2].set_title("Average Prediction Length (step)", fontweight="bold")
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_save_path = os.path.join(config.result_dir, "test_performance_variable_len.png")
        plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Performance plot saved to {fig_save_path}")

    print(f"\n{'=' * 80}")
    print("Variable-length test pipeline completed")
    print(f"Result directory: {config.result_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()