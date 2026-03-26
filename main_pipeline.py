import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import numpy as np

from src.lightcurve_simulation import generate_single_agn_curve, generate_binary_agn_curve
from src.data_processing import build_whisper_dataset
from src.whisper_classifier import WhisperAGNClassifier


# ================= SETTINGS =================

n_curves = 100                     # number of light curves per class (noise vs binary)
whisper_n_points = 480_000        # fixed input length required by Whisper

# ----- Physical / observational parameters -----
fedd = 0.1                        # Eddington fraction (AGN luminosity = 10% of L_Edd)
dl_pc = 1.5e9                     # luminosity distance in parsecs (~5 billion light years)
obs_nu = 1e15                     # observing frequency in Hz (UV band ~10^15 Hz)
ZTF_area = 1.8e3                  # telescope collecting area in cm^2 (approx ZTF scale)

# ----- Time sampling -----
N = 12_000_000                    # number of native time steps (~ long baseline observation)
dt = 1.0                          # time resolution in seconds


# ================= GENERATE DATA =================
print("Generating data...")

noise_curves = []
signal_curves = []

for i in range(n_curves):

    # Black hole mass sampled in LISA-relevant range
    # LISA is sensitive to ~10^5–10^7 solar mass binaries
    BH_mass = 10 ** np.random.uniform(5, 7)

    # ----- Noise-only (single AGN red noise) -----
    noise = generate_single_agn_curve(
        BH_mass, fedd, dl_pc, obs_nu,
        ZTF_area, N, dt, whisper_n_points,
        seed=i
    )
    noise_curves.append(noise)

    # ----- Signal + noise (binary AGN) -----
    signal = generate_binary_agn_curve(
        ecc=np.random.uniform(0, 0.8),   # orbital eccentricity (circular → moderately eccentric)
        n_orbits=100,                    # number of orbital cycles simulated
        BH_mass=BH_mass,
        period_yr=1.0,                   # orbital period (~1 year, simplified)
        fedd=fedd,
        dl_pc=dl_pc,
        obs_nu=obs_nu,
        ZTF_area=ZTF_area,
        N=N,
        dt=dt,
        target_len=whisper_n_points,
        seed=i
    )
    signal_curves.append(signal)

noise_curves = np.array(noise_curves)
signal_curves = np.array(signal_curves)


# ================= DATASET =================
print("Building dataset...")

dataset = build_whisper_dataset(signal_curves, noise_curves)

train_size = int(0.8 * len(dataset))   # 80/20 split
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)


# ================= MODEL =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WhisperAGNClassifier(use_lora=True, use_dora=True).to(device)

criterion = nn.CrossEntropyLoss()

# Only train LoRA parameters (efficient fine-tuning)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)


# ================= TRAIN =================
n_epochs = 10

print("Training...")

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ----- Validation -----
    model.eval()
    val_loss, correct = 0.0, 0

    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)

            outputs = model(X_val)
            loss = criterion(outputs, y_val)

            val_loss += loss.item()
            correct += (outputs.argmax(1) == y_val).sum().item()

    val_acc = correct / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{n_epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.3f}")


print("Done.")
