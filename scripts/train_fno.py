import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

from neuralop.models import FNO


# ============================================================
# Settings (CLI-overridable; defaults reproduce the long-horizon model)
# ============================================================

parser = argparse.ArgumentParser(description="Train the inverse-delay FNO.")
parser.add_argument("--data", default="../dataset/psi_dataset.pt",
                    help="path to the dataset .pt")
parser.add_argument("--save_dir", default="../models/fno_psi",
                    help="directory to save fno_model.pt")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--no_plots", action="store_true",
                    help="skip the (blocking) training/prediction plots")
args = parser.parse_args()

DATA_PATH = args.data
SAVE_DIR = args.save_dir

SEED = 0

EPOCHS = args.epochs
BATCH_SIZE = 32
LR = 1e-3

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

N_MODES = 32
HIDDEN_CHANNELS = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utilities
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


# ============================================================
# Dataset
# ============================================================

class PsiDataset(Dataset):

    def __init__(self, path):

        payload = torch.load(path, map_location="cpu")

        self.X = payload["X"].float()        # (N,1,N_grid)
        self.Y = payload["Y"].float()
        self.t_grid = payload["t_grid"].float()

        self.x_mean = payload["x_mean"]
        self.x_std = payload["x_std"]
        self.y_mean = payload["y_mean"]
        self.y_std = payload["y_std"]

        # normalize
        self.Xn = (self.X - self.x_mean) / self.x_std
        self.Yn = (self.Y - self.y_mean) / self.y_std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        x = self.Xn[idx]   # (1,N)
        y = self.Yn[idx]   # (1,N)

        return x, y


# ============================================================
# Load dataset
# ============================================================

set_seed(SEED)

dataset = PsiDataset(DATA_PATH)

N = len(dataset)
n_train = int(TRAIN_FRAC * N)
n_val = int(VAL_FRAC * N)
n_test = N - n_train - n_val

train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

print("Dataset size:", N)
print("Train / Val / Test:", n_train, n_val, n_test)


# ============================================================
# Model
# ============================================================

model = FNO(
    n_modes=(N_MODES,),
    hidden_channels=HIDDEN_CHANNELS,
    in_channels=1,
    out_channels=1,
).to(DEVICE)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ============================================================
# Training
# ============================================================

train_history = []
val_history = []

for epoch in range(EPOCHS):

    # ---- train ----

    model.train()

    train_rmse = 0
    n_batches = 0

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)

        loss = rmse(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_rmse += loss.item()
        n_batches += 1

    train_rmse /= n_batches


    # ---- validation ----

    model.eval()

    val_rmse = 0
    n_batches = 0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            val_rmse += rmse(pred, y).item()
            n_batches += 1

    val_rmse /= n_batches

    train_history.append(train_rmse)
    val_history.append(val_rmse)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Train RMSE {train_rmse:.6e} | Val RMSE {val_rmse:.6e}")


# ============================================================
# Test evaluation
# ============================================================

model.eval()

test_rmse = 0
n_batches = 0

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)

        test_rmse += rmse(pred, y).item()
        n_batches += 1

test_rmse /= n_batches

print("\nTest RMSE:", test_rmse)


# ============================================================
# Save model
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "fno_model.pt")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "t_grid": dataset.t_grid,
        "x_mean": dataset.x_mean,
        "x_std": dataset.x_std,
        "y_mean": dataset.y_mean,
        "y_std": dataset.y_std,
    },
    MODEL_PATH,
)

print("Model saved to:", MODEL_PATH)


# ============================================================
# Plot training curves
# ============================================================

if args.no_plots:
    print("Skipping plots (--no_plots).")
    raise SystemExit(0)

plt.figure()

plt.plot(train_history, label="train")
plt.plot(val_history, label="val")

plt.xlabel("Epoch")
plt.ylabel("RMSE")

plt.title("Training Curve")
plt.grid(True)
plt.legend()

plt.show()


# ============================================================
# Visualize predictions
# ============================================================

x_test, y_test = next(iter(test_loader))

x_test = x_test.to(DEVICE)

with torch.no_grad():
    pred = model(x_test).cpu()

y_true = y_test

t = dataset.t_grid.numpy()

plt.figure(figsize=(10,4))

for i in range(3):

    plt.subplot(1,3,i+1)

    plt.plot(t, y_true[i,0], label="true")
    plt.plot(t, pred[i,0], "--", label="pred")

    plt.title(f"sample {i}")
    plt.grid(True)

plt.legend()
plt.tight_layout()
plt.show()