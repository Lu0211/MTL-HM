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

# Fix random seeds
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

# -------------------------- 1. Hyperparameter Configuration (Only Layer Freezing) --------------------------
class Config:
    input_dim = 4  # 4-dimensional input: gap distance, following vehicle speed, relative speed, leading vehicle speed
    output_dim = 1  # Single-step acceleration output
    seq_len = 100  # Historical sequence length
    pred_len = 80  # Total prediction length
    step_pred = 10  # Predict 10 steps per recursive iteration
    slide_step = 80  # Sliding window step for dataset generation
    total_len = pred_len + seq_len + 1
    dt = 0.1
    domains = ["ngsim", "lyft", "waymo", "spmd1", "spmd2"]
    source_domains = ["ngsim", "spmd2"]
    target_domains = ["spmd1", "lyft", "waymo"]
    # Model hyperparameters
    hidden_size = 64
    enc_layers = 2
    dec_layers = 2
    dropout = 0.1
    meta_alpha = 0.01
    # Acceleration constraints for each dataset
    dataset_acc_limits = {
        "ngsim": {"min": -2.7, "max": 2.6},
        "spmd1": {"min": -3.5, "max": 2.0},
        "spmd2": {"min": -3.5, "max": 2.0},
        "waymo": {"min": -4.0, "max": 3.0},
        "lyft": {"min": -3.0, "max": 2.9}
    }
    # Remove IDM related settings
    idm_params = {}
    idm_weight = 0.0
    # Data augmentation
    noise_std = 0.02
    noise_prob = 0.8
    # Training configuration (Only layer freezing)
    pretrain_epochs = 80  # Source domain pretraining only
    finetune_epochs = 0   # Disable finetuning
    batch_size = 256
    pretrain_lr = 0.001
    finetune_lr = 0.0001
    meta_lr = 0.0001
    weight_decay = 1e-5
    patience = 10
    freeze_layers = 1     # Only freeze the first encoder layer
    # File path configuration
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
    save_dir = "output/our_method_ablation/only_layer_freeze"  # Independent storage path for ablation experiment
    time_log_path = os.path.join(save_dir, "training_time_log.csv")

config = Config()
os.makedirs(config.save_dir, exist_ok())

# -------------------------- 2. Core Calculation Functions (No IDM) --------------------------
def calculate_spacing_from_acc(pred_acc, ego_speed_init, front_speed_seq, init_spacing, dt, domain):
    """Basic gap integration without IDM constraints"""
    batch_size, seq_len = pred_acc.shape
    device = pred_acc.device
    ego_speed = torch.zeros((batch_size, seq_len), device=device, dtype=pred_acc.dtype)
    spacing_pred = torch.zeros((batch_size, seq_len), device=device, dtype=pred_acc.dtype)
    ego_speed[:, 0] = ego_speed_init
    spacing_pred[:, 0] = init_spacing

    dt_tensor = torch.tensor(dt, dtype=pred_acc.dtype, device=device)
    s0 = torch.tensor(0.1, dtype=pred_acc.dtype, device=device)  # Basic minimum gap threshold

    for t in range(1, seq_len):
        # Speed integration
        ego_speed_t = ego_speed[:, t - 1] + pred_acc[:, t - 1] * dt_tensor
        ego_speed_t = torch.clamp(ego_speed_t, min=0.1)
        ego_speed[:, t] = ego_speed_t

        # Gap integration without IDM constraints
        front_speed_t = front_speed_seq[:, t - 1] if t - 1 < front_speed_seq.shape[1] else front_speed_seq[:, -1]
        spacing_t = spacing_pred[:, t - 1] + (front_speed_t - ego_speed_t) * dt_tensor
        spacing_t = torch.clamp(spacing_t, min=s0)
        spacing_pred[:, t] = spacing_t

    return spacing_pred

# -------------------------- 3. Dataset Class (Unmodified Logic) --------------------------
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

            # Data augmentation
            if self.is_train and random.random() < config.noise_prob:
                spacing += np.random.normal(0, config.noise_std, spacing.shape)
                fv_speed += np.random.normal(0, config.noise_std, fv_speed.shape)
                lv_speed += np.random.normal(0, config.noise_std, lv_speed.shape)
                rel_speed = lv_speed - fv_speed
                spacing = np.maximum(spacing, 0.1)
                fv_speed = np.maximum(fv_speed, 0.1)
                lv_speed = np.maximum(lv_speed, 0.1)

            # Filter abnormal trajectories
            if len(spacing) < config.total_len or len(fv_speed) < config.total_len or len(rel_speed) < config.total_len or len(lv_speed) < config.total_len:
                continue
            acc = (fv_speed[1:config.total_len] - fv_speed[:config.total_len-1]) / config.dt
            lv_acc = (lv_speed[1:config.total_len] - lv_speed[:config.total_len-1]) / config.dt
            if np.any(fv_speed <= 0.1) or np.any(lv_speed <= 0.1) or np.any(spacing <= 0.3):
                continue

            # Sliding window sampling
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

                # Scene statistical features
                speed_mean = np.mean(fv_speed[start:input_end])
                speed_std = np.std(fv_speed[start:input_end])
                max_acc_window = np.max(np.abs(acc[start:label_end]))
                spacing_mean = np.mean(spacing[start:input_end])
                rel_speed_mean = np.mean(rel_speed[start:input_end])
                scene_feat = np.array([speed_mean, speed_std, max_acc_window, spacing_mean, rel_speed_mean],
                                      dtype=np.float32)

                # Save raw ground truth segments
                raw_spacing_window = spacing[start + config.seq_len: label_end]
                raw_fv_speed_window = fv_speed[start + config.seq_len: label_end]
                raw_lv_speed_window = lv_speed[start + config.seq_len: label_end]

                features.append(feat)
                labels.append(label_acc)
                meta.append(scene_feat)
                raw_spacing.append(raw_spacing_window)
                raw_fv_speed.append(raw_fv_speed_window)
                raw_lv_speed.append(raw_lv_window)

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

# -------------------------- 4. Dataset Loading Function --------------------------
def load_datasets(scaler=None, is_train=True):
    """Unified dataset loading pipeline"""
    # Source domain datasets
    source_datasets = []
    source_scaler = scaler
    for domain in config.source_domains:
        if source_scaler is None and is_train:
            ds = MultiDomainDataset(domain, "train", is_train=True)
            source_scaler = ds.scaler
        else:
            ds = MultiDomainDataset(domain, "train", scaler=source_scaler, is_train=False)
        source_datasets.append(ds)

    # Target domain datasets
    target_datasets = {}
    for domain in config.target_domains:
        target_datasets[domain] = MultiDomainDataset(domain, "train", scaler=source_scaler, is_train=False)

    # Validation datasets
    val_datasets = {domain: MultiDomainDataset(domain, "val", scaler=source_scaler, is_train=False) for domain in config.domains}
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

# -------------------------- 5. Model Definition (Layer Freeze Only, No Domain Adversarial) --------------------------
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
        # Remove domain discriminator module
        self.dataset_acc_min = {d: torch.tensor(v["min"], dtype=torch.float32).to(self.device) for d, v in config.dataset_acc_limits.items()}
        self.dataset_acc_max = {d: torch.tensor(v["max"], dtype=torch.float32).to(self.device) for d, v in config.dataset_acc_limits.items()}

        # Only perform layer freezing operation
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

        # Only physical acceleration clipping without IDM fusion
        acc_min = self.dataset_acc_min.get(current_domain, self.dataset_acc_min["ngsim"])
        acc_max = self.dataset_acc_max.get(current_domain, self.dataset_acc_max["ngsim"])

        for t in range(step_num):
            # Pure model prediction branch, no IDM
            base_pred, hidden, cell = self.decoder(dec_input, hidden, cell)
            meta_pred = base_pred + self.meta_adapter(hidden[-1])
            weights = self.dynamic_weight(meta_feat)
            fusion_pred = weights[:, 0:1] * base_pred + weights[:, 1:2] * meta_pred

            # Only clamp acceleration to physical range
            fusion_pred = torch.clamp(fusion_pred, min=acc_min, max=acc_max)
            step_preds.append(fusion_pred)
            dec_input = fusion_pred

        return torch.cat(step_preds, dim=1), hidden, cell

    def forward(self, x, meta, current_domain, is_train=True, y_gt=None,
                ego_speed_seq=None, front_speed_seq=None, spacing_seq=None):
        batch_size = x.shape[0]
        total_preds = []
        enc_hidden, enc_cell = self.encoder(x)
        # Remove domain adversarial branch calculation
        remaining_pred = self.config.pred_len
        current_hidden, current_cell = enc_hidden, enc_cell
        current_x = x

        while remaining_pred > 0:
            step_num = min(self.config.step_pred, remaining_pred)
            step_pred, current_hidden, current_cell = self.step_predict(
                current_hidden, current_cell, meta, current_domain, step_num,
                ego_speed_seq, front_speed_seq, spacing_seq
            )
            total_preds.append(step_pred)

            # Rebuild sliding input window
            if is_train and y_gt is not None:
                gt_slice = y_gt[:, self.config.pred_len - remaining_pred: self.config.pred_len - remaining_pred + step_num]
                new_x = self._build_new_input(current_x, gt_slice, current_domain, front_speed_seq)
            else:
                new_x = self._build_new_input(current_x, step_pred, current_domain, front_speed_seq)

            current_x = new_x
            remaining_pred -= step_num

        total_preds = torch.cat(total_preds, dim=1)
        return total_preds, None  # No domain prediction output

    def _build_new_input(self, current_x, step_pred, current_domain, front_speed_seq=None):
        """Update sliding history window without IDM logic"""
        batch_size = current_x.shape[0]
        x_slice = current_x[:, :, self.config.step_pred:]
        pred_feat = torch.zeros((batch_size, self.config.input_dim, self.config.step_pred),
                                device=current_x.device, dtype=current_x.dtype)

        # Initial physical state
        ego_speed_init = current_x[:, 1, -1]
        spacing_init = current_x[:, 0, -1]
        dt_tensor = torch.tensor(self.config.dt, dtype=current_x.dtype, device=current_x.device)
        s0 = 0.1  # Basic minimum gap threshold

        ego_speed = ego_speed_init
        spacing = spacing_init
        for t in range(self.config.step_pred):
            # Speed integration
            ego_speed = ego_speed + step_pred[:, t] * dt_tensor
            ego_speed = torch.clamp(ego_speed, min=0.1)

            # Fetch leading vehicle speed
            if front_speed_seq is not None:
                front_speed = front_speed_seq[:, t] if t < front_speed_seq.shape[1] else front_speed_seq[:, -1]
            else:
                front_speed = current_x[:, 3, -1]

            # Gap integration without IDM
            spacing = spacing + (front_speed - ego_speed) * dt_tensor
            spacing = torch.clamp(spacing, min=s0)

            # Fill feature channels
            pred_feat[:, 0, t] = spacing
            pred_feat[:, 1, t] = ego_speed
            pred_feat[:, 2, t] = front_speed - ego_speed
            pred_feat[:, 3, t] = front_speed

        new_x = torch.cat([x_slice, pred_feat], dim=2)
        return new_x

# -------------------------- 6. Training & Validation Functions (No Domain Adversarial Loss) --------------------------
def train_one_epoch(model, loader, optimizer, epoch, stage="pretrain"):
    model.train()
    total_loss = 0.0
    total_acc_loss = 0.0
    total_spacing_loss = 0.0
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

        # Model forward pass without domain adversarial
        out, _ = model(
            x, meta, current_domain, is_train=True, y_gt=y,
            ego_speed_seq=fv_speed, front_speed_seq=lv_speed, spacing_seq=spacing_gt
        )

        # Loss calculation (remove domain adversarial loss term)
        loss_acc = nn.MSELoss()(out, y)
        spacing_pred = calculate_spacing_from_acc(out, fv_speed[:, 0], lv_speed, spacing_gt[:, 0], config.dt, current_domain)
        loss_spacing = nn.MSELoss()(spacing_pred, spacing_gt)
        loss_gen = 0.2 * loss_acc + 0.8 * loss_spacing  # Only generation loss

        optimizer.zero_grad()
        loss_gen.backward()
        optimizer.step()

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

    print(f"  {stage} | Elapsed Time: {epoch_duration:.2f} min | Acceleration MSE: {avg_acc_loss:.6f} | Gap MSE: {avg_spacing_loss:.6f}")
    return avg_loss, epoch_duration

def validate(model, val_loaders):
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

                # Model inference
                out, _ = model(
                    x, meta, domain, is_train=False,
                    ego_speed_seq=fv_speed, front_speed_seq=lv_speed, spacing_seq=spacing_gt
                )

                # Compute losses
                loss_acc = nn.MSELoss()(out, y)
                spacing_pred = calculate_spacing_from_acc(out, fv_speed[:, 0], lv_speed, spacing_gt[:, 0], config.dt, domain)
                loss_spacing = nn.MSELoss()(spacing_pred, spacing_gt)

                batch_size = x.shape[0]
                total_acc += loss_acc.item() * batch_size
                total_spacing += loss_spacing.item() * batch_size
                total_samples += batch_size

                pbar.set_postfix({
                    "acc_loss": f"{loss_acc.item():.6f}",
                    "spacing_loss": f"{loss_spacing.item():.6f}"
                })

            # Average loss per sample
            if total_samples > 0:
                val_loss[domain]["acc"] = total_acc / total_samples
                val_loss[domain]["spacing"] = total_spacing / total_samples
                val_loss[domain]["total"] = 0.5 * val_loss[domain]["acc"] + 0.5 * val_loss[domain]["spacing"]
            else:
                val_loss[domain]["total"] = float("inf")

            val_duration[domain] = (time.time() - val_start_time) / 60

    # Average validation loss over all target domains
    target_avg_loss = sum(val_loss[d]["total"] for d in config.target_domains) / len(config.target_domains)
    return val_loss, target_avg_loss, val_duration

# -------------------------- 7. Main Training Pipeline (Pretrain Only with Layer Freeze) --------------------------
if __name__ == "__main__":
    # Initialize environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqFollowingModel(config).to(device)
    print(f"✅ Model loaded to device: {device}")
    print(f"✅ Ablation Experiment: Only layer freezing (No IDM + No domain adversarial + No finetuning)")

    # Load datasets
    source_datasets, target_datasets, source_scaler, val_loaders = load_datasets()
    pretrain_loader = DataLoader(
        ConcatDataset(source_datasets),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0
    )

    # Check if pretrained checkpoint exists
    pretrain_model_path = os.path.join(config.save_dir, "pretrain_best.pth")
    pretrain_done = os.path.exists(pretrain_model_path)

    # Only run source domain pretraining
    if not pretrain_done:
        print("\n" + "=" * 60)
        print("Stage 1: Source Domain Pretraining (Layer Freeze Only)")
        print("=" * 60)

        # Optimizer setup for pretraining
        base_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + \
                      list(model.dynamic_weight.parameters())
        meta_params = list(model.meta_adapter.parameters())
        pretrain_optimizer = optim.Adam([
            {"params": base_params, "lr": config.pretrain_lr},
            {"params": meta_params, "lr": config.meta_lr}
        ], weight_decay=config.weight_decay)
        pretrain_scheduler = optim.lr_scheduler.ReduceLROnPlateau(pretrain_optimizer, mode="min", factor=0.5, patience=3)

        # Training tracking variables
        best_pretrain_loss = float("inf")
        pretrain_early_stop = 0
        pretrain_log = []
        total_start_time = time.time()

        for epoch in range(config.pretrain_epochs):
            # Train one epoch
            train_loss, train_time = train_one_epoch(model, pretrain_loader, pretrain_optimizer, epoch, "pretrain")
            # Validate all domains
            val_loss, target_avg_loss, val_duration = validate(model, val_loaders)
            # Adjust learning rate
            pretrain_scheduler.step(target_avg_loss)

            # Record log
            pretrain_log.append({
                "Epoch": epoch + 1,
                "Stage": "Pretrain",
                "Train_Loss": train_loss,
                "Train_Time_Min": train_time,
                "Target_Avg_Loss": target_avg_loss,
                "Cumulative_Time_Min": (time.time() - total_start_time) / 60
            })

            # Print epoch information
            print(f"Pretrain Epoch {epoch + 1}/{config.pretrain_epochs} | Target Domain Avg Loss: {target_avg_loss:.6f}")
            for d in config.target_domains:
                print(f"  {d} | Acceleration MSE: {val_loss[d]['acc']:.6f} | Gap MSE: {val_loss[d]['spacing']:.6f}")

            # Save best checkpoint
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
                print(f"  ✅ Best pretrain model saved! Minimum loss: {best_pretrain_loss:.6f}")
            else:
                pretrain_early_stop += 1
                if pretrain_early_stop >= config.patience:
                    print(f"  ⚠️ Early stopping triggered for pretraining!")
                    break
    else:
        # Load existing pretrained weights
        checkpoint = torch.load(pretrain_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        source_scaler = checkpoint["scaler"]
        best_pretrain_loss = checkpoint["best_loss"]
        pretrain_log = []
        total_start_time = time.time()
        print(f"\n✅ Detected existing pretrained weights, loaded directly!")

    # Save training log
    log_df = pd.DataFrame(pretrain_log)
    log_df.to_csv(config.time_log_path, index=False)

    # Total training time calculation
    total_training_time = (time.time() - total_start_time) / 60

    # Plot training curves
    plt.figure(figsize=(18, 6))
    # Subplot 1: Pretraining loss curve
    plt.subplot(1, 3, 1)
    if pretrain_log:
        pretrain_epochs = [log["Epoch"] for log in pretrain_log]
        pretrain_losses = [log["Target_Avg_Loss"] for log in pretrain_log]
        plt.plot(pretrain_epochs, pretrain_losses, label="Pretrain Target Domain Loss", linewidth=2, color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss of Target Domains")
    plt.title("Pretraining Loss Curve (Layer Freeze Only, No IDM)")
    plt.legend()
    plt.grid(alpha=0.3)

    # Subplot 2: SPMD domain gap loss comparison
    plt.subplot(1, 3, 2)
    val_loss_final, _, _ = validate(model, val_loaders)
    spmd1_loss = [val_loss_final["spmd1"]["spacing"]]
    spmd2_loss = [val_loss_final["spmd2"]["spacing"]]
    plt.bar(["SPMD1", "SPMD2"], [spmd1_loss[0], spmd2_loss[0]], color=["orange", "green"], alpha=0.8)
    plt.xlabel("Dataset")
    plt.ylabel("Gap MSE (m²)")
    plt.title("SPMD Gap Loss (Layer Freeze Only, No IDM)")
    plt.grid(alpha=0.3, axis="y")

    # Subplot 3: Cumulative training time
    plt.subplot(1, 3)
    if pretrain_log:
        cumulative_time = [log["Cumulative_Time_Min"] for log in pretrain_log]
        plt.plot(range(1, len(cumulative_time) + 1), cumulative_time, label="Cumulative Training Time", linewidth=2, color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Time (Minutes)")
    plt.title("Training Time Curve (Layer Freeze Only, No IDM)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, "training_results.png"), dpi=300)
    plt.close()

    # Final summary print
    print("\n" + "=" * 60)
    print("Training Complete - Final Statistics (Layer Freeze Only)")
    print("=" * 60)
    print(f"📌 Best pretrain loss: {best_pretrain_loss:.6f}")
    print(f"📌 Total training duration: {total_training_time:.2f} minutes")
    print(f"📌 Pretrain checkpoint: pretrain_best.pth")
    print(f"📌 Training log file: training_time_log.csv")
    print(f"📌 Training visualization: training_results.png")
    print(f"📌 Output directory: {config.save_dir}")