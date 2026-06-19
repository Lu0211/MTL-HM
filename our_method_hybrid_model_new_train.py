import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
import os
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from tqdm import tqdm
import joblib

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["SimHei", "Times New Roman"]
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 1. Hyperparameter Configuration --------------------------
class Config:
    input_dim = 4  # 4D input: spacing, follower speed, relative speed, leader speed
    output_dim = 1  # Single-step acceleration output
    seq_len = 100  # History sequence length
    pred_len = 80  # Total prediction length
    step_pred = 10  # Recursive prediction steps per window
    slide_step = 80  # Sliding step for dataset generation
    total_len = pred_len + seq_len + 1
    dt = 0.1
    domains = ["ngsim", "lyft", "waymo", "spmd1", "spmd2"]
    source_domains = ["ngsim", "spmd2"]
    target_domains = ["spmd1", "lyft", "waymo"]
    # Model related
    hidden_size = 64
    enc_layers = 2
    dec_layers = 2
    dropout = 0.1
    meta_alpha = 0.01
    # Dataset acceleration constraints
    dataset_acc_limits = {
        "ngsim": {"min": -2.7, "max": 2.6},
        "spmd1": {"min": -3.5, "max": 2.0},
        "spmd2": {"min": -3.5, "max": 2.0},
        "waymo": {"min": -4.0, "max": 3.0},
        "lyft": {"min": -3.0, "max": 2.9}
    }
    # IDM parameters (domain-specific)
    idm_params = {
        "ngsim": {"v0": 38.11, "T": 1.39, "a": 1.66, "b": 1.18, "s0": 6.63, "delta": 1.57},
        "spmd1": {"v0": 41.58, "T": 0.66, "a": 1.21, "b": 0.50, "s0": 5.26, "delta": 2.85},
        "spmd2": {"v0": 41.36, "T": 0.74, "a": 0.65, "b": 0.49, "s0": 0.94, "delta": 2.26},
        "waymo": {"v0": 41.32, "T": 1.59, "a": 1.80, "b": 0.25, "s0": 7.29, "delta": 7.17},
        "lyft": {"v0": 13.41, "T": 1.62, "a": 1.70, "b": 0.27, "s0": 12, "delta": 1.90},
    }
    idm_weight = 0.1  # IDM acceleration fusion weight (model 70% + IDM 30%)
    # Data augmentation configuration
    noise_std = 0.02  # Gaussian noise standard deviation
    noise_prob = 0.8  # Noise application probability
    # Training configuration
    pretrain_epochs = 80  # Source-domain pretraining epochs
    finetune_epochs = 40  # Target-domain fine-tuning epochs
    batch_size = 256
    pretrain_lr = 0.001
    finetune_lr = 0.0001  # Fine-tuning learning rate (1/10 of pretraining)
    meta_lr = 0.0001
    weight_decay = 1e-5  # Weight decay coefficient (L2 regularization)
    patience = 10
    freeze_layers = 1
    # Path configuration
    data_paths = {
        "ngsim": {"train": "dataset/data/NGSIM_I_80_train_data.npy", "val": "dataset/data/NGSIM_I_80_val_data.npy",
                  "test": "dataset/data/NGSIM_I_80_test_data.npy"},
        "waymo": {"train": "dataset/data/Waymo_train_data.npy", "val": "dataset/data/Waymo_val_data.npy",
                  "test": "dataset/data/Waymo_test_data.npy"},
        "lyft": {"train": "dataset/data/Lyft_train_data.npy", "val": "dataset/data/Lyft_val_data.npy",
                 "test": "dataset/data/Lyft_test_data.npy"},
        "spmd1": {"train": "dataset/data/SPMD1_train_data.npy", "val": "dataset/data/SPMD1_val_data.npy",
                  "test": "dataset/data/SPMD1_test_data.npy"},
        "spmd2": {"train": "dataset/data/SPMD2_train_data.npy", "val": "dataset/data/SPMD2_val_data.npy",
                  "test": "dataset/data/SPMD2_test_data.npy"}
    }
    save_dir = "output/our_method_with_IDM_final_new"
    time_log_path = os.path.join(save_dir, "training_time_log.csv")


config = Config()
os.makedirs(config.save_dir, exist_ok=True)


# -------------------------- 2. IDM Core Functions + Spacing Prediction --------------------------
def idm_desired_spacing(ego_speed, front_speed, params):
    """Compute IDM desired safe spacing"""
    delta_v = ego_speed - front_speed
    s_star = params["s0"] + ego_speed * params["T"] + (ego_speed * delta_v) / (2 * np.sqrt(params["a"] * params["b"]))
    return np.maximum(s_star, params["s0"])  # Ensure >= minimum safe spacing


def idm_acceleration(ego_speed, front_speed, current_spacing, params):
    """Compute IDM acceleration (physics constraint)"""
    s_star = idm_desired_spacing(ego_speed, front_speed, params)
    current_spacing = np.maximum(current_spacing, 0.1)
    acc_idm = params["a"] * (1 - (ego_speed / params["v0"]) ** params["delta"] - (s_star / current_spacing) ** 2)
    return acc_idm


def calculate_spacing_from_acc(pred_acc, ego_speed_init, front_speed_seq, init_spacing, dt, domain):
    """Spacing integration with IDM fusion"""
    batch_size, seq_len = pred_acc.shape
    device = pred_acc.device
    ego_speed = torch.zeros((batch_size, seq_len), device=device, dtype=pred_acc.dtype)
    spacing_pred = torch.zeros((batch_size, seq_len), device=device, dtype=pred_acc.dtype)
    ego_speed[:, 0] = ego_speed_init
    spacing_pred[:, 0] = init_spacing

    dt_tensor = torch.tensor(dt, dtype=pred_acc.dtype, device=device)
    idm_params = config.idm_params.get(domain, config.idm_params["ngsim"])
    s0 = torch.tensor(idm_params["s0"], dtype=pred_acc.dtype, device=device)

    for t in range(1, seq_len):
        # Speed integration constraint (avoid negative speed)
        ego_speed_t = ego_speed[:, t - 1] + pred_acc[:, t - 1] * dt_tensor
        ego_speed_t = torch.clamp(ego_speed_t, min=0.1)  # Minimum speed 0.1 m/s
        ego_speed[:, t] = ego_speed_t

        # Spacing integration + IDM safety constraint
        front_speed_t = front_speed_seq[:, t - 1] if t - 1 < front_speed_seq.shape[1] else front_speed_seq[:, -1]
        spacing_t = spacing_pred[:, t - 1] + (front_speed_t - ego_speed_t) * dt_tensor
        # spacing_t = torch.clamp(spacing_t, min=s0)  # IDM minimum safe spacing constraint
        spacing_pred[:, t] = spacing_t

    return spacing_pred


# -------------------------- 3. Dataset Class (With Data Augmentation) --------------------------
class MultiDomainDataset(Dataset):
    def __init__(self, domain, split, scaler=None, is_train=True):
        self.domain = domain
        self.split = split
        self.is_train = is_train
        self.data_path = config.data_paths[domain][split]
        self.raw_data = np.load(self.data_path, allow_pickle=True)
        self.features, self.labels, self.meta, self.raw_spacing, self.raw_fv_speed, self.raw_lv_speed = self._process_all_domains()
        # Standardization
        if is_train:
            self.scaler = StandardScaler()
            input_reshaped = self.features.transpose(0, 2, 1).reshape(-1, config.input_dim)
            self.scaler.fit(input_reshaped)
        else:
            self.scaler = scaler
        input_reshaped = self.features.transpose(0, 2, 1).reshape(-1, config.input_dim)
        input_scaled = self.scaler.transform(input_reshaped)
        self.features = input_scaled.reshape(-1, self.features.shape[2], config.input_dim).transpose(0, 2, 1)

    def _process_all_domains(self):
        features = []
        labels = []
        meta = []
        raw_spacing = []
        raw_fv_speed = []
        raw_lv_speed = []

        for sample in self.raw_data:
            spacing = np.array(sample[0], dtype=np.float32)[:config.total_len]
            fv_speed = np.array(sample[1], dtype=np.float32)[:config.total_len]
            rel_speed = np.array(sample[2], dtype=np.float32)[:config.total_len]
            lv_speed = np.array(sample[3], dtype=np.float32)[:config.total_len]

            # Data augmentation: add Gaussian noise during training (solve overly similar adjacent steps)
            if self.is_train and random.random() < config.noise_prob:
                # Add noise to core physical quantities
                spacing += np.random.normal(0, config.noise_std, spacing.shape)
                fv_speed += np.random.normal(0, config.noise_std, fv_speed.shape)
                lv_speed += np.random.normal(0, config.noise_std, lv_speed.shape)
                rel_speed = lv_speed - fv_speed  # Recalculate relative speed
                # Ensure physical quantity rationality
                spacing = np.maximum(spacing, 0.1)
                fv_speed = np.maximum(fv_speed, 0.1)
                lv_speed = np.maximum(lv_speed, 0.1)

            # Filter short samples and outliers
            if len(spacing) < config.total_len or len(fv_speed) < config.total_len or len(rel_speed) < config.total_len or len(lv_speed) < config.total_len:
                continue
            acc = (fv_speed[1:config.total_len] - fv_speed[:config.total_len-1]) / config.dt
            lv_acc = (lv_speed[1:config.total_len] - lv_speed[:config.total_len-1]) / config.dt
            if np.any(fv_speed <= 0.1) or np.any(lv_speed <= 0.1) or np.any(spacing <= 0.3):
                continue

            # Generate samples via sliding window
            max_start = config.total_len - config.seq_len - config.pred_len
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, config.slide_step):
                input_end = start + config.seq_len
                label_end = start + config.seq_len + config.pred_len
                input_spacing = spacing[start:input_end]
                input_fv = fv_speed[start:input_end]
                input_rel = rel_speed[start:input_end]
                input_lv = lv_speed[start:input_end]
                feat = np.stack([input_spacing, input_fv, input_rel, input_lv], axis=0)
                label_acc = acc[start + config.seq_len: label_end]

                if len(feat[0]) != config.seq_len or len(label_acc) != config.pred_len:
                    continue

                # Scene features
                speed_mean = np.mean(fv_speed[start:input_end])
                speed_std = np.std(fv_speed[start:input_end])
                max_acc_window = np.max(np.abs(acc[start:label_end]))
                spacing_mean = np.mean(spacing[start:input_end])
                rel_speed_mean = np.mean(rel_speed[start:input_end])
                scene_feat = np.array([speed_mean, speed_std, max_acc_window, spacing_mean, rel_speed_mean],
                                      dtype=np.float32)

                # Raw data storage
                raw_spacing_window = spacing[start + config.seq_len: label_end]
                raw_fv_speed_window = fv_speed[start + config.seq_len: label_end]
                raw_lv_speed_window = lv_speed[start + config.seq_len: label_end]

                features.append(feat)
                labels.append(label_acc)
                meta.append(scene_feat)
                raw_spacing.append(raw_spacing_window)
                raw_fv_speed.append(raw_fv_speed_window)
                raw_lv_speed.append(raw_lv_speed_window)

        return (np.array(features), np.array(labels), np.array(meta),
                np.array(raw_spacing), np.array(raw_fv_speed), np.array(raw_lv_speed))

    def __getitem__(self, idx):
        return {
            "x": torch.FloatTensor(self.features[idx]),
            "y": torch.FloatTensor(self.labels[idx]),
            "meta": torch.FloatTensor(self.meta[idx]),
            "spacing": torch.FloatTensor(self.raw_spacing[idx]),
            "fv_speed": torch.FloatTensor(self.raw_fv_speed[idx]),
            "lv_speed": torch.FloatTensor(self.raw_lv_speed[idx]),
            "domain": self.domain
        }

    def __len__(self):
        return len(self.features) if hasattr(self, 'features') else 0


# -------------------------- 4. Load Datasets --------------------------
def load_datasets(scaler=None, is_train=True):
    """Unified dataset loading function"""
    # Source-domain datasets
    source_datasets = []
    source_scaler = scaler
    for domain in config.source_domains:
        if source_scaler is None and is_train:
            ds = MultiDomainDataset(domain, "train", is_train=True)
            source_scaler = ds.scaler
        else:
            ds = MultiDomainDataset(domain, "train", scaler=source_scaler, is_train=False)
        source_datasets.append(ds)

    # Target-domain datasets
    target_datasets = {}
    for domain in config.target_domains:
        target_datasets[domain] = MultiDomainDataset(domain, "train", scaler=source_scaler, is_train=False)

    # Validation sets
    val_datasets = {domain: MultiDomainDataset(domain, "val", scaler=source_scaler, is_train=False) for domain in
                    config.domains}
    val_loaders = {
        domain: DataLoader(
            val_datasets[domain],
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0
        ) for domain in config.domains
    }

    return source_datasets, target_datasets, source_scaler, val_loaders


# -------------------------- 5. Seq2Seq Model Definition (With IDM) --------------------------
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
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=False)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, dec_input, hidden, cell):
        # 🔥 Force shape to (B, 1)
        dec_input = dec_input.view(-1, 1)
        # 🔥 Convert to LSTM expected shape (1, B, 1)
        dec_input = dec_input.unsqueeze(0)
        out, (hidden, cell) = self.lstm(dec_input, (hidden, cell))
        pred = self.fc_out(out.squeeze(0))
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
        return torch.softmax(weighted_weights, dim=1)


class LearnableWeights(nn.Module):
    """⭐ Dynamic weight learning: let the model learn α, β, γ automatically"""

    def __init__(self, num_tasks=3):
        super().__init__()
        # Learnable log-variance (use log to ensure positivity)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        losses: [loss_acc, loss_spacing, loss_adv]
        Returns: weighted total loss + regularization term
        """
        total_loss = 0
        regularization = 0

        for i, loss in enumerate(losses):
            # weight = 1 / (2 * σ²) = exp(-log_var) / 2
            precision = torch.exp(-self.log_vars[i])  # 1/σ²
            weight = precision / 2

            total_loss += weight * loss
            regularization += self.log_vars[i]  # log(σ)

        # Total loss = weighted loss + regularization
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
        self.dataset_acc_min = {d: torch.tensor(v["min"], dtype=torch.float32).to(self.device) for d, v in
                                config.dataset_acc_limits.items()}
        self.dataset_acc_max = {d: torch.tensor(v["max"], dtype=torch.float32).to(self.device) for d, v in
                                config.dataset_acc_limits.items()}
        # ⭐ New: dynamic weight learning
        self.learnable_weights = LearnableWeights(num_tasks=3)
        # Freeze first N layers of encoder (pretraining stage)
        for i in range(config.freeze_layers):
            for param in self.encoder.lstm.all_weights[i]:
                param.requires_grad = False

    def init_decoder_input(self, batch_size):
        return torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)

    def step_predict(self, enc_hidden, enc_cell, meta_feat, current_domain, step_num=10,
                     ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        batch_size = enc_hidden.shape[1]
        dec_input = self.init_decoder_input(batch_size)
        hidden, cell = enc_hidden, enc_cell
        step_preds = []

        # Domain-specific parameters
        acc_min = self.dataset_acc_min.get(current_domain, self.dataset_acc_min["ngsim"])
        acc_max = self.dataset_acc_max.get(current_domain, self.dataset_acc_max["ngsim"])
        idm_params = self.config.idm_params.get(current_domain, self.config.idm_params["ngsim"])

        for t in range(step_num):
            # 1. Model predicts acceleration
            base_pred, hidden, cell = self.decoder(dec_input.view(-1, 1), hidden, cell)
            meta_pred = base_pred + self.meta_adapter(hidden[-1])
            weights = self.dynamic_weight(meta_feat)
            fusion_pred = weights[:, 0:1] * base_pred + weights[:, 1:2] * meta_pred

            # 2. Fuse IDM acceleration (physics constraint)
            if ego_speed_seq is not None and front_speed_seq is not None and spacing_seq is not None:
                # Get current-step physical quantities
                ego_speed = ego_speed_seq[:, t] if t < ego_speed_seq.shape[1] else ego_speed_seq[:, -1]
                front_speed = front_speed_seq[:, t] if t < front_speed_seq.shape[1] else front_speed_seq[:, -1]
                spacing = spacing_seq[:, t] if t < spacing_seq.shape[1] else spacing_seq[:, -1]

                # Batch compute IDM acceleration
                idm_acc = torch.zeros_like(fusion_pred)
                for i in range(batch_size):
                    idm_acc[i] = idm_acceleration(
                        ego_speed[i].cpu().numpy(),
                        front_speed[i].cpu().numpy(),
                        spacing[i].cpu().numpy(),
                        idm_params
                    )
                idm_acc = idm_acc.to(fusion_pred.device)

                # Fuse model and IDM acceleration
                fusion_pred = (1 - self.config.idm_weight) * fusion_pred + self.config.idm_weight * idm_acc

            # 3. Acceleration range constraint
            # fusion_pred = torch.clamp(fusion_pred, min=acc_min, max=acc_max)
            # Only restrict behavioral rationality without compromising safety
            # 3. Use IDM natural reasonable acceleration interval constraint
            # Safe IDM acceleration constraint (does not change tensor shape!)
            a_low = torch.tensor(-idm_params["b"], dtype=fusion_pred.dtype, device=fusion_pred.device)
            a_high = idm_params["a"] * (1.0 - (ego_speed_seq[:, 0:1] / idm_params["v0"]) ** idm_params["delta"])
            # Keep shape (B,1) after constraint
            fusion_pred = torch.clamp(fusion_pred, min=a_low, max=a_high).reshape(-1, 1)
            step_preds.append(fusion_pred)

            dec_input = fusion_pred.reshape(-1, 1)

        return torch.cat(step_preds, dim=1), hidden, cell

    def forward(self, x, meta, current_domain, is_train=True, y_gt=None,
                ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        batch_size = x.shape[0]
        total_preds = []
        enc_hidden, enc_cell = self.encoder(x)
        domain_feat = enc_hidden[-1]
        domain_pred = self.domain_discriminator(domain_feat)
        remaining_pred = self.config.pred_len
        current_hidden, current_cell = enc_hidden, enc_cell
        current_x = x

        while remaining_pred > 0:
            step_num = min(self.config.step_pred, remaining_pred)
            # Predict acceleration for current window (with IDM fusion)
            step_pred, current_hidden, current_cell = self.step_predict(
                current_hidden, current_cell, meta, current_domain, step_num,
                ego_speed_seq, front_speed_seq, spacing_seq
            )
            total_preds.append(step_pred)

            # Build new history window
            if is_train and y_gt is not None:
                gt_slice = y_gt[:,
                           self.config.pred_len - remaining_pred: self.config.pred_len - remaining_pred + step_num]
                new_x = self._build_new_input(current_x, gt_slice, current_domain, front_speed_seq)
            else:
                new_x = self._build_new_input(current_x, step_pred, current_domain, front_speed_seq)

            current_x = new_x
            remaining_pred -= step_num

        total_preds = torch.cat(total_preds, dim=1)
        return total_preds, domain_pred

    def _build_new_input(self, current_x, step_pred, current_domain, front_speed_seq=None):
        """History window update with IDM fusion"""
        batch_size = current_x.shape[0]
        x_slice = current_x[:, :, self.config.step_pred:]  # Take last 40 steps (50-10)
        pred_feat = torch.zeros((batch_size, self.config.input_dim, self.config.step_pred),
                                device=current_x.device, dtype=current_x.dtype)

        # Initial physical quantities
        ego_speed_init = current_x[:, 1, -1]
        spacing_init = current_x[:, 0, -1]
        dt_tensor = torch.tensor(self.config.dt, dtype=current_x.dtype, device=current_x.device)
        idm_params = self.config.idm_params.get(current_domain, self.config.idm_params["ngsim"])
        s0 = idm_params["s0"]

        ego_speed = ego_speed_init
        spacing = spacing_init
        for t in range(self.config.step_pred):
            # 1. Speed integration
            ego_speed = ego_speed + step_pred[:, t] * dt_tensor
            # ego_speed = torch.clamp(ego_speed, min=0.1)

            # 2. Real leader speed (key fix: no longer fixed)
            if front_speed_seq is not None:
                front_speed = front_speed_seq[:, t] if t < front_speed_seq.shape[1] else front_speed_seq[:, -1]
            else:
                front_speed = current_x[:, 3, -1]  # Fallback

            # 3. Spacing integration + IDM safety constraint
            spacing = spacing + (front_speed - ego_speed) * dt_tensor
            # spacing = torch.clamp(spacing, min=s0)  # IDM minimum safe spacing

            # 4. Fill features
            pred_feat[:, 0, t] = spacing  # IDM-constrained spacing
            pred_feat[:, 1, t] = ego_speed  # Integrated speed
            pred_feat[:, 2, t] = front_speed - ego_speed  # Relative speed
            pred_feat[:, 3, t] = front_speed  # Real leader speed

        new_x = torch.cat([x_slice, pred_feat], dim=2)
        return new_x


# -------------------------- 6. Training and Validation Functions --------------------------
def train_one_epoch(model, loader, optimizer, epoch, stage="pretrain"):
    """Single training epoch (supports pretraining/fine-tuning)"""
    model.train()
    total_loss = 0.0
    total_acc_loss = 0.0
    total_spacing_loss = 0.0
    domain_map = {d: i for i, d in enumerate(config.domains)}
    epoch_start_time = time.time()

    pbar = tqdm(loader, desc=f"{stage.capitalize()} Epoch {epoch + 1} [Train]")
    for batch in pbar:
        x = batch["x"].to(model.device, non_blocking=True)
        y = batch["y"].to(model.device, non_blocking=True)
        meta = batch["meta"].to(model.device, non_blocking=True)
        spacing_gt = batch["spacing"].to(model.device, non_blocking=True)
        fv_speed = batch["fv_speed"].to(model.device, non_blocking=True)
        lv_speed = batch["lv_speed"].to(model.device, non_blocking=True)
        current_domain = batch["domain"][0]
        domain_labels = torch.tensor([domain_map[d] for d in batch["domain"]], device=model.device)

        # Model forward (pass physical sequences, fuse IDM)
        out, domain_pred = model(
            x, meta, current_domain, is_train=True, y_gt=y,
            ego_speed_seq=fv_speed, front_speed_seq=lv_speed, spacing_seq=spacing_gt
        )

        # Loss computation (weights match validation logic)
        loss_acc = nn.MSELoss()(out, y)
        spacing_pred = calculate_spacing_from_acc(out, fv_speed[:, 0], lv_speed, spacing_gt[:, 0], config.dt,
                                                  current_domain)
        loss_spacing = nn.MSELoss()(spacing_pred, spacing_gt)
        loss_adv = nn.CrossEntropyLoss()(domain_pred, domain_labels)
        # loss_gen = 0.2 * loss_acc + 0.8 * loss_spacing + 0.1 * loss_adv  # Match validation weights
        # loss_gen = α * loss_acc + β * loss_spacing + γ * loss_adv
        # ⭐ New code: dynamic weights
        losses = [loss_acc, loss_spacing, loss_adv]
        loss_gen = model.learnable_weights(losses)

        optimizer.zero_grad()
        loss_gen.backward()
        optimizer.step()

        # ⭐ Print dynamic weights (see what the model learns)
        with torch.no_grad():
            weights = [torch.exp(-w / 2).item() for w in model.learnable_weights.log_vars]
            print(f"   Dynamic weights: α={weights[0]:.3f}, β={weights[1]:.3f}, γ={weights[2]:.3f}")

        total_loss += loss_gen.item() * x.shape[0]
        total_acc_loss += loss_acc.item() * x.shape[0]
        total_spacing_loss += loss_spacing.item() * x.shape[0]

        pbar.set_postfix({
            "acc_loss": f"{loss_acc.item():.6f}",
            "spacing_loss": f"{loss_spacing.item():.6f}",
            "total_loss": f"{loss_gen.item():.6f}"
        })

    epoch_duration = (time.time() - epoch_start_time) / 60
    avg_loss = total_loss / len(loader.dataset)
    avg_acc_loss = total_acc_loss / len(loader.dataset)
    avg_spacing_loss = total_spacing_loss / len(loader.dataset)

    print(
        f"  {stage} | Duration: {epoch_duration:.2f} min | Acc MSE: {avg_acc_loss:.6f} | Spacing MSE: {avg_spacing_loss:.6f}")
    return avg_loss, epoch_duration


def validate(model, val_loaders):
    """Validation function"""
    model.eval()
    val_loss = {d: {"acc": 0.0, "spacing": 0.0, "total": 0.0} for d in config.domains}
    val_duration = {}

    with torch.no_grad():
        for domain in config.domains:
            val_start_time = time.time()
            loader = val_loaders[domain]
            total_acc = 0.0
            total_spacing = 0.0
            total_samples = 0

            pbar = tqdm(loader, desc=f"Validate {domain.upper()}")
            for batch in pbar:
                x = batch["x"].to(model.device, non_blocking=True)
                y = batch["y"].to(model.device, non_blocking=True)
                meta = batch["meta"].to(model.device, non_blocking=True)
                spacing_gt = batch["spacing"].to(model.device, non_blocking=True)
                fv_speed = batch["fv_speed"].to(model.device, non_blocking=True)
                lv_speed = batch["lv_speed"].to(model.device, non_blocking=True)

                # Model forward
                out, _ = model(
                    x, meta, domain, is_train=False,
                    ego_speed_seq=fv_speed, front_speed_seq=lv_speed, spacing_seq=spacing_gt
                )

                # Loss computation
                loss_acc = nn.MSELoss()(out, y)
                spacing_pred = calculate_spacing_from_acc(out, fv_speed[:, 0], lv_speed, spacing_gt[:, 0], config.dt,
                                                          domain)
                loss_spacing = nn.MSELoss()(spacing_pred, spacing_gt)

                batch_size = x.shape[0]
                total_acc += loss_acc.item() * batch_size
                total_spacing += loss_spacing.item() * batch_size
                total_samples += batch_size

                pbar.set_postfix({
                    "acc_loss": f"{loss_acc.item():.6f}",
                    "spacing_loss": f"{loss_spacing.item():.6f}"
                })

            # Compute average losses
            if total_samples > 0:
                val_loss[domain]["acc"] = total_acc / total_samples
                val_loss[domain]["spacing"] = total_spacing / total_samples
                val_loss[domain]["total"] = 0.5 * val_loss[domain]["acc"] + 0.5 * val_loss[domain]["spacing"]
            else:
                val_loss[domain]["total"] = float("inf")

            val_duration[domain] = (time.time() - val_start_time) / 60

    # Target-domain average loss
    target_avg_loss = sum(val_loss[d]["total"] for d in config.target_domains) / len(config.target_domains)
    return val_loss, target_avg_loss, val_duration


# -------------------------- 7. Main Training Pipeline (Pretrain + Finetune) --------------------------
# The following code executes Stage 1 training when needed, skips Stage 1 if already trained,
# and always executes Stage 2 fine-tuning
if __name__ == "__main__":
    # Initialize device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqFollowingModel(config).to(device)
    print(f"✅ Model loaded to device: {device}")
    print(f"✅ CUDA available: {torch.cuda.is_available()} | GPU count: {torch.cuda.device_count()}")
    print(f"✅ Source domains: {config.source_domains} | Target domains: {config.target_domains}")
    print(f"✅ IDM fusion weight: {config.idm_weight} | Noise std: {config.noise_std}")

    # Load datasets (required regardless of whether pretraining was done)
    source_datasets, target_datasets, source_scaler, val_loaders = load_datasets()
    pretrain_loader = DataLoader(
        ConcatDataset(source_datasets),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0
    )

    # -------------------------- Core Modification: Pretrained Model Check --------------------------
    pretrain_model_path = os.path.join(config.save_dir, "pretrain_best.pth")
    pretrain_done = os.path.exists(pretrain_model_path)  # Check if pretrained model exists

    # -------------------------- Stage 1: Source-Domain Pretraining (Only if model does not exist) --------------------------
    if not pretrain_done:  # Execute pretraining only if model does not exist
        print("\n" + "=" * 60)
        print("Stage 1: Source-Domain Pretraining")
        print("=" * 60)

        # Pretraining optimizer
        base_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + \
                      list(model.domain_discriminator.parameters()) + list(model.dynamic_weight.parameters())
        meta_params = list(model.meta_adapter.parameters())
        pretrain_optimizer = optim.Adam([
            {"params": base_params, "lr": config.pretrain_lr},
            {"params": meta_params, "lr": config.meta_lr}
        ], weight_decay=config.weight_decay)
        pretrain_scheduler = optim.lr_scheduler.ReduceLROnPlateau(pretrain_optimizer, mode="min", factor=0.5, patience=3)

        # Pretraining variables
        best_pretrain_loss = float("inf")
        pretrain_early_stop = 0
        pretrain_log = []
        total_start_time = time.time()

        for epoch in range(config.pretrain_epochs):
            # Train
            train_loss, train_time = train_one_epoch(model, pretrain_loader, pretrain_optimizer, epoch, "pretrain")
            # Validate
            val_loss, target_avg_loss, val_duration = validate(model, val_loaders)
            # Learning rate scheduling
            pretrain_scheduler.step(target_avg_loss)

            # Log recording
            pretrain_log.append({
                "Epoch": epoch + 1,
                "Stage": "Pretrain",
                "Train_Loss": train_loss,
                "Train_Time_Min": train_time,
                "Target_Avg_Loss": target_avg_loss,
                "Cumulative_Time_Min": (time.time() - total_start_time) / 60
            })

            # Print logs
            print(f"Pretrain Epoch {epoch + 1}/{config.pretrain_epochs} | Target-domain avg loss: {target_avg_loss:.6f}")
            for d in config.target_domains:
                print(f"  {d} | Acc MSE: {val_loss[d]['acc']:.6f} | Spacing MSE: {val_loss[d]['spacing']:.6f}")

            # Save best pretrained model
            if target_avg_loss < best_pretrain_loss:
                best_pretrain_loss = target_avg_loss
                pretrain_early_stop = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "scaler": source_scaler,
                    "epoch": epoch,
                    "best_loss": best_pretrain_loss,
                    "config": config.__dict__
                }, pretrain_model_path)
                print(f"  ✅ Best pretrained model saved! Loss: {best_pretrain_loss:.6f}")
            else:
                pretrain_early_stop += 1
                if pretrain_early_stop >= config.patience:
                    print(f"  ⚠️  Pretraining early stopping triggered!")
                    break
    else:
        # Pretraining completed, load scaler directly from model file
        checkpoint = torch.load(pretrain_model_path, map_location=device, weights_only=False)
        source_scaler = checkpoint["scaler"]
        print(f"\n✅ Detected existing pretrained model: {pretrain_model_path}")
        print(f"✅ Skip pretraining stage, enter fine-tuning directly!")
        total_start_time = time.time()  # Initialize time without affecting later statistics

    # -------------------------- Stage 2: Target-Domain Fine-Tuning (Always Executed) --------------------------
    print("\n" + "=" * 60)
    print("Stage 2: Target-Domain Fine-Tuning")
    print("=" * 60)

    # Load best pretrained model (reload best weights regardless of whether just trained)
    checkpoint = torch.load(pretrain_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Loaded best pretrained model (loss: {checkpoint['best_loss']:.6f})")

    # Unfreeze all layers (fine-tuning)
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True
    print("✅ All layers unfrozen, start target-domain fine-tuning")

    # Fine-tuning optimizer (reduced learning rate)
    finetune_optimizer = optim.Adam([
        {"params": model.encoder.parameters(), "lr": config.finetune_lr},
        {"params": model.decoder.parameters(), "lr": config.finetune_lr},
        {"params": model.meta_adapter.parameters(), "lr": config.meta_lr * 0.1},
        {"params": model.dynamic_weight.parameters(), "lr": config.finetune_lr}
    ], weight_decay=config.weight_decay)
    finetune_scheduler = optim.lr_scheduler.ReduceLROnPlateau(finetune_optimizer, mode="min", factor=0.5, patience=2)

    # Target-domain dataset loading
    finetune_datasets = [target_datasets[d] for d in config.target_domains]
    finetune_loader = DataLoader(
        ConcatDataset(finetune_datasets),
        batch_size=config.batch_size // 2,  # Reduce batch size during fine-tuning
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0
    )

    # Fine-tuning variables
    best_finetune_loss = float("inf")
    finetune_early_stop = 0
    finetune_log = []

    for epoch in range(config.finetune_epochs):
        # Fine-tuning training
        train_loss, train_time = train_one_epoch(model, finetune_loader, finetune_optimizer, epoch, "finetune")
        # Validate
        val_loss, target_avg_loss, val_duration = validate(model, val_loaders)
        # Learning rate scheduling
        finetune_scheduler.step(target_avg_loss)

        # Log recording
        finetune_log.append({
            "Epoch": epoch + 1,
            "Stage": "Finetune",
            "Train_Loss": train_loss,
            "Train_Time_Min": train_time,
            "Target_Avg_Loss": target_avg_loss,
            "Cumulative_Time_Min": (time.time() - total_start_time) / 60
        })

        # Print logs
        print(f"Finetune Epoch {epoch + 1}/{config.finetune_epochs} | Target-domain avg loss: {target_avg_loss:.6f}")
        for d in config.target_domains:
            print(f"  {d} | Acc MSE: {val_loss[d]['acc']:.6f} | Spacing MSE: {val_loss[d]['spacing']:.6f}")

        # Save best fine-tuned model
        if target_avg_loss < best_finetune_loss:
            best_finetune_loss = target_avg_loss
            finetune_early_stop = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler": source_scaler,
                "pretrain_loss": checkpoint["best_loss"],
                "finetune_loss": best_finetune_loss,
                "config": config.__dict__
            }, os.path.join(config.save_dir, "finetune_best.pth"))
            print(f"  ✅ Best fine-tuned model saved! Loss: {best_finetune_loss:.6f}")
        else:
            finetune_early_stop += 1
            if finetune_early_stop >= config.patience:
                print(f"  ⚠️  Fine-tuning early stopping triggered!")
                break

    # -------------------------- Result Saving --------------------------
    # Merge logs (handle case where pretraining was skipped)
    if pretrain_done:
        pretrain_log = []  # Pretraining already done, empty log
    all_log = pretrain_log + finetune_log
    log_df = pd.DataFrame(all_log)
    log_df.to_csv(config.time_log_path, index=False)

    # Total training time
    total_training_time = (time.time() - total_start_time) / 60

    # Plot loss curves (compatible with skipped pretraining)
    plt.figure(figsize=(18, 6))
    # Subplot 1: Pretrain/Finetune loss
    plt.subplot(1, 3, 1)
    if pretrain_log:  # Only plot pretraining curve if executed
        pretrain_epochs = [log["Epoch"] for log in pretrain_log]
        pretrain_losses = [log["Target_Avg_Loss"] for log in pretrain_log]
        plt.plot(pretrain_epochs, pretrain_losses, label="Pretrain target loss", linewidth=2, color="blue")
    finetune_epochs = [log["Epoch"] + len(pretrain_log) for log in finetune_log]
    finetune_losses = [log["Target_Avg_Loss"] for log in finetune_log]
    plt.plot(finetune_epochs, finetune_losses, label="Finetune target loss", linewidth=2, color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Target-Domain Average Loss")
    plt.title("Pretrain + Finetune Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    # Subplot 2: SPMD1/SPMD2 spacing loss
    plt.subplot(1, 3, 2)
    val_loss_final, _, _ = validate(model, val_loaders)
    spmd1_loss = [val_loss_final["spmd1"]["spacing"]]
    spmd2_loss = [val_loss_final["spmd2"]["spacing"]]
    plt.bar(["SPMD1", "SPMD2"], [spmd1_loss[0], spmd2_loss[0]], color=["orange", "green"], alpha=0.8)
    plt.xlabel("Dataset")
    plt.ylabel("Spacing MSE (m²)")
    plt.title("SPMD Dataset Spacing Loss")
    plt.grid(alpha=0.3, axis="y")

    # Subplot 3: Training time
    plt.subplot(1, 3, 3)
    cumulative_time = [log["Cumulative_Time_Min"] for log in all_log]
    plt.plot(range(1, len(cumulative_time) + 1), cumulative_time, label="Cumulative training time",
             linewidth=2, color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Time (minutes)")
    plt.title("Training Time Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, "training_results.png"), dpi=300)
    plt.close()

    # Final statistics
    print("\n" + "=" * 60)
    print("Training Completed! Final Results Summary")
    print("=" * 60)
    if pretrain_done:
        print(f"📌 Best pretraining loss: {checkpoint['best_loss']:.6f} (loaded)")
    else:
        print(f"📌 Best pretraining loss: {best_pretrain_loss:.6f}")
    print(f"📌 Best fine-tuning loss: {best_finetune_loss:.6f}")
    print(f"📌 Total training time: {total_training_time:.2f} minutes")
    print(f"📌 Pretrained model: pretrain_best.pth")
    print(f"📌 Fine-tuned model: finetune_best.pth")
    print(f"📌 Training log: training_time_log.csv")
    print(f"📌 Visualization: training_results.png")
    print(f"📌 Save directory: {config.save_dir}")