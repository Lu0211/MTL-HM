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


# -------------------------- 1. Hyperparameter Configuration (Fine-tuning Only, No IDM) --------------------------
class Config:
    # Data dimension configuration
    input_dim = 4  # Input: spacing, follower speed, relative speed, leader speed
    output_dim = 1  # Output: single-step acceleration
    seq_len = 100  # History sequence length
    pred_len = 80  # Prediction length
    step_pred = 10  # Recursive prediction step size
    slide_step = 80  # Sliding step for dataset
    total_len = pred_len + seq_len + 1
    dt = 0.1  # Time step

    # Domain configuration
    domains = ["ngsim", "lyft", "waymo", "spmd1", "spmd2"]
    source_domains = ["ngsim", "spmd2"]  # Source domains
    target_domains = ["spmd1", "lyft", "waymo"]  # Target domains

    # Model parameters
    hidden_size = 64
    enc_layers = 2
    dec_layers = 2
    dropout = 0.1

    # Acceleration constraints (basic range only, no IDM)
    dataset_acc_limits = {
        "ngsim": {"min": -2.7, "max": 2.6},
        "spmd1": {"min": -3.5, "max": 2.0},
        "spmd2": {"min": -3.5, "max": 2.0},
        "waymo": {"min": -4.0, "max": 3.0},
        "lyft": {"min": -3.0, "max": 2.9}
    }

    # Training configuration (fine-tuning strategy only)
    pretrain_epochs = 20  # Lightweight source-domain pretraining (no special strategy)
    finetune_epochs = 60  # Core: target-domain fine-tuning
    batch_size = 256
    pretrain_lr = 0.001  # Pretraining learning rate
    finetune_lr = 0.0001  # Low learning rate for fine-tuning (critical)
    weight_decay = 1e-5
    patience = 10  # Early stopping patience
    freeze_layers = 0  # Disable layer freezing (full fine-tuning)

    # Data augmentation
    noise_std = 0.02
    noise_prob = 0.8

    # Path configuration (modify according to your actual paths)
    data_paths = {
        "ngsim": {"train": "dataset/data/NGSIM_I_80_train_data.npy", "val": "dataset/data/NGSIM_I_80_val_data.npy"},
        "waymo": {"train": "dataset/data/Waymo_train_data.npy", "val": "dataset/data/Waymo_val_data.npy"},
        "lyft": {"train": "dataset/data/Lyft_train_data.npy", "val": "dataset/data/Lyft_val_data.npy"},
        "spmd1": {"train": "dataset/data/SPMD1_train_data.npy", "val": "dataset/data/SPMD1_val_data.npy"},
        "spmd2": {"train": "dataset/data/SPMD2_train_data.npy", "val": "dataset/data/SPMD2_val_data.npy"}
    }
    save_dir = "output/our_method_ablation/only_finetune_no_idm"  # Independent save path


config = Config()
os.makedirs(config.save_dir, exist_ok=True)


# -------------------------- 2. Core Utility Functions (No IDM) --------------------------
def calculate_spacing_from_acc(pred_acc, ego_speed_init, front_speed_seq, init_spacing, dt, domain):
    """Basic physics-based spacing integration (no IDM constraint)"""
    batch_size, seq_len = pred_acc.shape
    device = pred_acc.device
    ego_speed = torch.zeros((batch_size, seq_len), device=device, dtype=pred_acc.dtype)
    spacing_pred = torch.zeros((batch_size, seq_len), device=device, dtype=pred_acc.dtype)

    ego_speed[:, 0] = ego_speed_init
    spacing_pred[:, 0] = init_spacing
    dt_tensor = torch.tensor(dt, dtype=pred_acc.dtype, device=device)
    s0 = torch.tensor(0.1, dtype=pred_acc.dtype, device=device)  # Minimum safe spacing

    for t in range(1, seq_len):
        # Speed integration (basic physics rule)
        ego_speed_t = ego_speed[:, t - 1] + pred_acc[:, t - 1] * dt_tensor
        ego_speed_t = torch.clamp(ego_speed_t, min=0.1)
        ego_speed[:, t] = ego_speed_t

        # Leader vehicle speed
        front_speed_t = front_speed_seq[:, t - 1] if t - 1 < front_speed_seq.shape[1] else front_speed_seq[:, -1]

        # Spacing integration (no IDM)
        spacing_t = spacing_pred[:, t - 1] + (front_speed_t - ego_speed_t) * dt_tensor
        spacing_t = torch.clamp(spacing_t, min=s0)
        spacing_pred[:, t] = spacing_t

    return spacing_pred


# -------------------------- 3. Dataset Class (Unchanged, No IDM) --------------------------
class MultiDomainDataset(Dataset):
    def __init__(self, domain, split, scaler=None, is_train=True):
        self.domain = domain
        self.split = split
        self.is_train = is_train
        self.data_path = config.data_paths[domain][split]
        self.raw_data = np.load(self.data_path, allow_pickle=True)
        self.features, self.labels, self.meta, self.raw_spacing, self.raw_fv_speed, self.raw_lv_speed = self._process_data()

        # Standardization
        if is_train:
            self.scaler = StandardScaler()
            input_reshaped = self.features.transpose(0, 2, 1).reshape(-1, config.input_dim)
            self.scaler.fit(input_reshaped)
        else:
            self.scaler = scaler

        # Apply standardization
        input_reshaped = self.features.transpose(0, 2, 1).reshape(-1, config.input_dim)
        input_scaled = self.scaler.transform(input_reshaped)
        self.features = input_scaled.reshape(-1, self.features.shape[2], config.input_dim).transpose(0, 2, 1)

    def _process_data(self):
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

            # Data augmentation (training set only)
            if self.is_train and random.random() < config.noise_prob:
                spacing += np.random.normal(0, config.noise_std, spacing.shape)
                fv_speed += np.random.normal(0, config.noise_std, fv_speed.shape)
                lv_speed += np.random.normal(0, config.noise_std, lv_speed.shape)
                rel_speed = lv_speed - fv_speed
                spacing = np.maximum(spacing, 0.1)
                fv_speed = np.maximum(fv_speed, 0.1)
                lv_speed = np.maximum(lv_speed, 0.1)

            # Filter abnormal data
            if len(spacing) < config.total_len or np.any(fv_speed <= 0.1) or np.any(spacing <= 0.3):
                continue

            # Compute acceleration (basic physics)
            acc = (fv_speed[1:config.total_len] - fv_speed[:config.total_len - 1]) / config.dt

            # Generate samples via sliding window
            max_start = config.total_len - config.seq_len - config.pred_len
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, config.slide_step):
                input_end = start + config.seq_len
                label_end = start + config.seq_len + config.pred_len

                # Input features
                input_spacing = spacing[start:input_end]
                input_fv = fv_speed[start:input_end]
                input_rel = rel_speed[start:input_end]
                input_lv = lv_speed[start:input_end]
                feat = np.stack([input_spacing, input_fv, input_rel, input_lv], axis=0)

                # Labels (acceleration)
                label_acc = acc[start + config.seq_len: label_end]

                # Scene features (no IDM)
                speed_mean = np.mean(fv_speed[start:input_end])
                speed_std = np.std(fv_speed[start:input_end])
                max_acc = np.max(np.abs(acc[start:label_end]))
                spacing_mean = np.mean(spacing[start:input_end])
                scene_feat = np.array([speed_mean, speed_std, max_acc, spacing_mean], dtype=np.float32)

                # Raw data (for validation)
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
        return len(self.features)


# -------------------------- 4. Model Definition (No IDM, No Domain Adversarial) --------------------------
class FeatureEncoder(nn.Module):
    """Feature encoder (LSTM)"""

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
        x = x.permute(2, 0, 1)  # (seq_len, batch_size, input_dim)
        out, (hn, cn) = self.lstm(x)
        return hn, cn


class Seq2SeqDecoder(nn.Module):
    """Sequence decoder (LSTM)"""

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
    """Core model (basic Seq2Seq only, no IDM / domain adversarial)"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder + Decoder
        self.encoder = FeatureEncoder(
            config.input_dim, config.hidden_size, config.enc_layers, config.dropout
        )
        self.decoder = Seq2SeqDecoder(
            config.hidden_size, config.output_dim, config.dec_layers, config.dropout
        )

        # Scene feature adapter
        self.meta_adapter = nn.Linear(config.hidden_size, config.output_dim)

        # Acceleration constraints
        self.acc_min = {d: torch.tensor(v["min"]).to(self.device) for d, v in config.dataset_acc_limits.items()}
        self.acc_max = {d: torch.tensor(v["max"]).to(self.device) for d, v in config.dataset_acc_limits.items()}

        # Fully trainable (no freezing)
        for param in self.parameters():
            param.requires_grad = True

    def step_predict(self, hidden, cell, meta_feat, current_domain, step_num=10):
        """Single-step prediction (no IDM)"""
        batch_size = hidden.shape[1]
        dec_input = torch.zeros((batch_size, 1), device=self.device)
        step_preds = []

        # Acceleration constraints
        acc_min = self.acc_min[current_domain]
        acc_max = self.acc_max[current_domain]

        for _ in range(step_num):
            # Base prediction
            base_pred, hidden, cell = self.decoder(dec_input, hidden, cell)
            # Scene feature fusion
            meta_pred = base_pred + self.meta_adapter(hidden[-1])
            # Weighted fusion
            fusion_pred = 0.5 * base_pred + 0.5 * meta_pred
            # Acceleration clamping
            fusion_pred = torch.clamp(fusion_pred, min=acc_min, max=acc_max)

            step_preds.append(fusion_pred)
            dec_input = fusion_pred

        return torch.cat(step_preds, dim=1), hidden, cell

    def forward(self, x, meta, current_domain, is_train=True, y_gt=None,
                ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        """Forward pass (no IDM / domain adversarial)"""
        batch_size = x.shape[0]
        total_preds = []

        # Encode
        enc_hidden, enc_cell = self.encoder(x)
        remaining_pred = self.config.pred_len
        current_hidden, current_cell = enc_hidden, enc_cell
        current_x = x

        # Recursive prediction
        while remaining_pred > 0:
            step_num = min(self.config.step_pred, remaining_pred)
            step_pred, current_hidden, current_cell = self.step_predict(
                current_hidden, current_cell, meta, current_domain, step_num
            )
            total_preds.append(step_pred)

            # Update input window (no IDM)
            if is_train and y_gt is not None:
                gt_slice = y_gt[:,
                           self.config.pred_len - remaining_pred: self.config.pred_len - remaining_pred + step_num]
                new_x = self._update_input_window(current_x, gt_slice, front_speed_seq)
            else:
                new_x = self._update_input_window(current_x, step_pred, front_speed_seq)

            current_x = new_x
            remaining_pred -= step_num

        return torch.cat(total_preds, dim=1)

    def _update_input_window(self, current_x, step_pred, front_speed_seq=None):
        """Update input window (basic physics only, no IDM)"""
        batch_size = current_x.shape[0]
        x_slice = current_x[:, :, self.config.step_pred:]
        pred_feat = torch.zeros((batch_size, self.config.input_dim, self.config.step_pred),
                                device=current_x.device)

        # Initial states
        ego_speed_init = current_x[:, 1, -1]
        spacing_init = current_x[:, 0, -1]
        dt = self.config.dt
        s0 = 0.1

        ego_speed = ego_speed_init
        spacing = spacing_init

        for t in range(self.config.step_pred):
            # Speed integration
            ego_speed = ego_speed + step_pred[:, t] * dt
            ego_speed = torch.clamp(ego_speed, min=0.1)

            # Leader vehicle speed
            if front_speed_seq is not None:
                front_speed = front_speed_seq[:, t] if t < front_speed_seq.shape[1] else front_speed_seq[:, -1]
            else:
                front_speed = current_x[:, 3, -1]

            # Spacing integration
            spacing = spacing + (front_speed - ego_speed) * dt
            spacing = torch.clamp(spacing, min=s0)

            # Fill features
            pred_feat[:, 0, t] = spacing
            pred_feat[:, 1, t] = ego_speed
            pred_feat[:, 2, t] = front_speed - ego_speed
            pred_feat[:, 3, t] = front_speed

        return torch.cat([x_slice, pred_feat], dim=2)


# -------------------------- 5. Training / Validation Functions --------------------------
def train_one_epoch(model, loader, optimizer, epoch, stage):
    """Single training epoch"""
    model.train()
    total_loss = 0.0
    total_acc_loss = 0.0
    total_spacing_loss = 0.0
    pbar = tqdm(loader, desc=f"{stage} Epoch {epoch + 1}")

    for batch in pbar:
        # Load data
        x = batch["x"].to(model.device)
        y = batch["y"].to(model.device)
        meta = batch["meta"].to(model.device)
        spacing_gt = batch["spacing"].to(model.device)
        fv_speed = batch["fv_speed"].to(model.device)
        lv_speed = batch["lv_speed"].to(model.device)
        current_domain = batch["domain"][0]

        # Forward pass
        pred_acc = model(
            x, meta, current_domain, is_train=True, y_gt=y,
            ego_speed_seq=fv_speed, front_speed_seq=lv_speed
        )

        # Loss computation (no IDM loss)
        loss_acc = nn.MSELoss()(pred_acc, y)
        # Spacing loss (basic physics integration)
        spacing_pred = calculate_spacing_from_acc(
            pred_acc, fv_speed[:, 0], lv_speed, spacing_gt[:, 0], config.dt, current_domain
        )
        loss_spacing = nn.MSELoss()(spacing_pred, spacing_gt)
        # Total loss
        loss = 0.2 * loss_acc + 0.8 * loss_spacing

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item() * x.shape[0]
        total_acc_loss += loss_acc.item() * x.shape[0]
        total_spacing_loss += loss_spacing.item() * x.shape[0]

        pbar.set_postfix({
            "acc_loss": f"{loss_acc.item():.6f}",
            "spacing_loss": f"{loss_spacing.item():.6f}",
            "total_loss": f"{loss.item():.6f}"
        })

    # Average losses
    avg_loss = total_loss / len(loader.dataset)
    avg_acc_loss = total_acc_loss / len(loader.dataset)
    avg_spacing_loss = total_spacing_loss / len(loader.dataset)

    print(
        f"\n{stage} Epoch {epoch + 1} | Avg Loss: {avg_loss:.6f} | Acc MSE: {avg_acc_loss:.6f} | Spacing MSE: {avg_spacing_loss:.6f}")
    return avg_loss


@torch.no_grad()
def validate(model, val_loaders):
    """Validation function"""
    model.eval()
    val_loss = {d: {"acc": 0.0, "spacing": 0.0, "total": 0.0} for d in config.domains}

    for domain in config.domains:
        loader = val_loaders[domain]
        if len(loader) == 0:
            continue

        total_acc = 0.0
        total_spacing = 0.0
        total_samples = 0

        for batch in loader:
            x = batch["x"].to(model.device)
            y = batch["y"].to(model.device)
            meta = batch["meta"].to(model.device)
            spacing_gt = batch["spacing"].to(model.device)
            fv_speed = batch["fv_speed"].to(model.device)
            lv_speed = batch["lv_speed"].to(model.device)

            # Prediction
            pred_acc = model(x, meta, domain, is_train=False)
            # Loss computation
            loss_acc = nn.MSELoss()(pred_acc, y)
            spacing_pred = calculate_spacing_from_acc(
                pred_acc, fv_speed[:, 0], lv_speed, spacing_gt[:, 0], config.dt, domain
            )
            loss_spacing = nn.MSELoss()(spacing_pred, spacing_gt)

            # Statistics
            batch_size = x.shape[0]
            total_acc += loss_acc.item() * batch_size
            total_spacing += loss_spacing.item() * batch_size
            total_samples += batch_size

        # Average losses
        if total_samples > 0:
            val_loss[domain]["acc"] = total_acc / total_samples
            val_loss[domain]["spacing"] = total_spacing / total_samples
            val_loss[domain]["total"] = 0.5 * val_loss[domain]["acc"] + 0.5 * val_loss[domain]["spacing"]

    # Target-domain average loss
    target_avg_loss = sum(val_loss[d]["total"] for d in config.target_domains) / len(config.target_domains)
    print(f"\nValidation | Target-Domain Avg Loss: {target_avg_loss:.6f}")
    for d in config.target_domains:
        print(f"  {d} | Acc MSE: {val_loss[d]['acc']:.6f} | Spacing MSE: {val_loss[d]['spacing']:.6f}")

    return val_loss, target_avg_loss


# -------------------------- 6. Main Training Pipeline --------------------------
def main():
    # 1. Device initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 60)
    print("Start Training: Fine-Tuning Only Transfer Learning (No IDM)")
    print("=" * 60)

    # 2. Load datasets
    print("\nLoading datasets...")
    # Source-domain datasets (pretraining)
    source_datasets = []
    source_scaler = None
    for domain in config.source_domains:
        if source_scaler is None:
            ds = MultiDomainDataset(domain, "train", is_train=True)
            source_scaler = ds.scaler
        else:
            ds = MultiDomainDataset(domain, "train", scaler=source_scaler, is_train=False)
        source_datasets.append(ds)
    pretrain_loader = DataLoader(
        ConcatDataset(source_datasets),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    # Target-domain datasets (fine-tuning)
    target_datasets = [MultiDomainDataset(d, "train", scaler=source_scaler, is_train=False) for d in
                       config.target_domains]
    finetune_loader = DataLoader(
        ConcatDataset(target_datasets),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    # Validation sets
    val_loaders = {}
    for domain in config.domains:
        val_ds = MultiDomainDataset(domain, "val", scaler=source_scaler, is_train=False)
        val_loaders[domain] = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # 3. Initialize model
    model = Seq2SeqFollowingModel(config).to(device)
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -------------------------- Key Modification: Pretrained Model Detection --------------------------
    pretrain_ckpt_path = os.path.join(config.save_dir, "pretrain_best.pth")
    if os.path.exists(pretrain_ckpt_path):
        # Load existing pretrained model, skip pretraining
        print("\n" + "=" * 60)
        print("Detected saved best pretrained model, loading directly and entering fine-tuning stage")
        print("=" * 60)
        pretrain_ckpt = torch.load(pretrain_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(pretrain_ckpt["model_state_dict"])
        source_scaler = pretrain_ckpt["scaler"]
        best_pretrain_loss = pretrain_ckpt["best_loss"]
        print(f"✅ Successfully loaded pretrained model | Best Pretrain Loss: {best_pretrain_loss:.6f}")
    else:
        # No pretrained model found, run pretraining
        print("\n" + "=" * 60)
        print("Stage 1: Lightweight Source-Domain Pretraining")
        print("=" * 60)
        pretrain_optimizer = optim.Adam(model.parameters(), lr=config.pretrain_lr, weight_decay=config.weight_decay)
        pretrain_scheduler = optim.lr_scheduler.ReduceLROnPlateau(pretrain_optimizer, mode="min", factor=0.5,
                                                                  patience=3)

        best_pretrain_loss = float("inf")
        pretrain_early_stop = 0

        for epoch in range(config.pretrain_epochs):
            # Train
            train_loss = train_one_epoch(model, pretrain_loader, pretrain_optimizer, epoch, "Pretrain")
            # Validate
            _, target_avg_loss = validate(model, val_loaders)
            # LR scheduling
            pretrain_scheduler.step(target_avg_loss)

            # Save best pretrained model
            if target_avg_loss < best_pretrain_loss:
                best_pretrain_loss = target_avg_loss
                pretrain_early_stop = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": pretrain_optimizer.state_dict(),
                    "best_loss": best_pretrain_loss,
                    "scaler": source_scaler
                }, pretrain_ckpt_path)
                print(f"Saved best pretrained model | Loss: {best_pretrain_loss:.6f}")
            else:
                pretrain_early_stop += 1
                if pretrain_early_stop >= config.patience:
                    print("Early stopping triggered during pretraining!")
                    break

    # -------------------------- Stage 2: Target-Domain Fine-Tuning (Core) --------------------------
    print("\n" + "=" * 60)
    print("Stage 2: Target-Domain Fine-Tuning")
    print("=" * 60)
    finetune_optimizer = optim.Adam(model.parameters(), lr=config.finetune_lr, weight_decay=config.weight_decay)
    finetune_scheduler = optim.lr_scheduler.ReduceLROnPlateau(finetune_optimizer, mode="min", factor=0.5, patience=3)

    best_finetune_loss = float("inf")
    finetune_early_stop = 0
    finetune_log = []

    for epoch in range(config.finetune_epochs):
        # Train
        train_loss = train_one_epoch(model, finetune_loader, finetune_optimizer, epoch, "Finetune")
        # Validate
        val_loss, target_avg_loss = validate(model, val_loaders)
        # LR scheduling
        finetune_scheduler.step(target_avg_loss)

        # Logging
        finetune_log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "target_avg_loss": target_avg_loss,
            "spmd1_acc": val_loss["spmd1"]["acc"],
            "spmd1_spacing": val_loss["spmd1"]["spacing"]
        })

        # Save best fine-tuned model
        if target_avg_loss < best_finetune_loss:
            best_finetune_loss = target_avg_loss
            finetune_early_stop = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": finetune_optimizer.state_dict(),
                "best_loss": best_finetune_loss,
                "scaler": source_scaler,
                "config": config.__dict__
            }, os.path.join(config.save_dir, "finetune_best.pth"))
            print(f"Saved best fine-tuned model | Loss: {best_finetune_loss:.6f}")
        else:
            finetune_early_stop += 1
            if finetune_early_stop >= config.patience:
                print("Early stopping triggered during fine-tuning!")
                break

    # 6. Save training log
    log_df = pd.DataFrame(finetune_log)
    log_df.to_csv(os.path.join(config.save_dir, "training_log.csv"), index=False)

    # 7. Plot training curves
    plt.figure(figsize=(12, 4))
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss", color="blue")
    plt.plot(log_df["epoch"], log_df["target_avg_loss"], label="Target Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fine-Tuning Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    # SPMD1 performance
    plt.subplot(1, 2, 2)
    plt.plot(log_df["epoch"], log_df["spmd1_acc"], label="Acc MSE", color="green")
    plt.plot(log_df["epoch"], log_df["spmd1_spacing"], label="Spacing MSE", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("SPMD1 Validation Performance")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, "training_curves.png"), dpi=300)
    plt.close()

    # 8. Final results
    print("\n" + "=" * 60)
    print("Training Completed!")
    print(f"Best Fine-Tune Loss: {best_finetune_loss:.6f}")
    print(f"Model Save Path: {config.save_dir}")
    print(f"Training Log: {os.path.join(config.save_dir, 'training_log.csv')}")
    print(f"Visualization Curves: {os.path.join(config.save_dir, 'training_curves.png')}")
    print("=" * 60)

if __name__ == "__main__":
    main()