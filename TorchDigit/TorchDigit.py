#-------------------------------------------------------------------------------
# Filename: TorchDigit.py
# Program: Neural Network with PyTorch
# Author: Heike Fasold
# Last Change: 26.08.2025
#-------------------------------------------------------------------------------
#---------------------------- Import libraries ---------------------------------
import os
import time
import numpy as np
import pandas as pd
from typing import Any, Callable
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                            roc_curve, auc, precision_recall_curve,
                            average_precision_score)
from matplotlib import pyplot as plt
#-------------------------------------------------------------------------------
# Measure runtime start time
start = time.time()
#--------------------------- Logging configuration -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
#--------------------------- Hyperparameters -----------------------------------
INPUT_SIZE = 784             # Number of pixels per image (28x28)
OUTPUT_SIZE = 10             # Number of classes (digits 0–9)
HIDDEN_1 = 512               # Neurons in the first hidden layer
H1 = HIDDEN_1
HIDDEN_2 = (HIDDEN_1 / 2)    # Neurons in the second hidden layer
H2 = int(HIDDEN_2)
LEARNING_RATE = 1e-3         # Learning rate
LR = LEARNING_RATE
BATCH_SIZE = 64              # Batch size
# BatchNorm as a class so BN(H1)/BN(H2) creates layers
BATCH_NORM = nn.BatchNorm1d
BN = BATCH_NORM
EPOCHS = 5                   # Number of epochs
USE_NORMALIZE = True         # Normalize MNIST (better convergence)
# Activation function/Dropout as instances (in Sequential WITHOUT parentheses)
ACTIVATION_FUNKTION = nn.ReLU()
AF = ACTIVATION_FUNKTION
DROPOUT = nn.Dropout(0.5)
DO = DROPOUT
#---------------------------- Storage paths ------------------------------------
ROOT_WAY = (r"+++++")
SAVE_WAY = (r"+++++")
os.makedirs(SAVE_WAY, exist_ok=True)
SAVE_FILE = os.path.join(SAVE_WAY, "mnist_nn_state.pth")
SAVE_METRICS_CSV = os.path.join(SAVE_WAY, "mnist_metrics.csv")
SAVE_CHECKPOINT = os.path.join(SAVE_WAY, "mnist_checkpoint.pt")
#---------------------- Reproducibility & Device -------------------------------
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Strict deterministic algorithms, if supported.
try:
    torch.use_deterministic_algorithms(True)
except Exception as _e:
    logging.warning("Deterministic algorithms not fully available: %r", _e)

device: torch.device = torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
#----------------------------- Worker seeding ----------------------------------
# Consistent seeds for DataLoader workers.
def _seed_worker(worker_id: int) -> None:
    '''
    Set per-worker random seeds for NumPy and PyTorch.
    '''
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
#----------------------------- Dataset class -----------------------------------
class MyDataset(Dataset):
    '''
    Dataset class for MNIST with integrated transformation.

    Loads training or test data depending on 'train'. Optionally normalizes
    images with mean/std (0.5/0.5).
    '''
    def __init__(self, ds: Any = MNIST, train: bool = True) -> None:
        # Define transformation (ToTensor + optional Normalize).
        if USE_NORMALIZE:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
        # Load dataset (downloaded if needed).
        self.dataset = ds(
            root=ROOT_WAY,
            train=train,
            download=True,
            transform=self.transform,
        )
    def __len__(self) -> int:
        '''
        Return the number of examples in the dataset.
        '''
        return len(self.dataset)
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        '''
        Return an (image, label) tuple for index 'idx'.
        '''
        image, label = self.dataset[idx]
        return image, label
#------------------------ Neural Network class ---------------------------------
class NN(nn.Module):
    '''
    Simple feedforward neural network.

    Architecture:
    - Flatten
    - Linear(INPUT_SIZE -> H1) + BatchNorm + ReLU + Dropout
    - Linear(H1 -> H2)          + BatchNorm + ReLU + Dropout
    - Linear(H2 -> OUTPUT_SIZE) (Logits; Softmax applied only at evaluation)
    '''
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_SIZE, H1),
            BN(H1),
            AF,
            DO,
            nn.Linear(H1, H2),
            BN(H2),
            AF,
            DO,
            nn.Linear(H2, OUTPUT_SIZE),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch, shape (N, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Logits with shape (N, OUTPUT_SIZE).
        '''
        return self.network(x)
#--------------------------- Visualization -------------------------------------
class Visualisation:
    '''
    Visualization of learning curves and test results.

    Notes:
    - `train_losses`, `val_accuracies`, `test_accuracy` are lists with
        one value per epoch.
    - For validation accuracy, the attribute `val_accuracies` is expected
        (can be set later).
    '''
    def __init__(
        self,
        model: nn.Module | None = None,
        train_losses: list[float] | None = None,
        test_accuracy: list[float] | None = None,
        train_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        test_dataset: Dataset | None = None,
    ) -> None:
        self.model = model
        self.train_losses = train_losses
        self.test_accuracy = test_accuracy
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_dataset = test_dataset
    # -------------------------- Loss vs Accuracy ------------------------------
    def plot_loss_train_val_accuracy(self, epochs: int) -> None:
        '''
        Plot train loss, validation accuracy, and test accuracy per epoch.
        '''
        train_losses = self.train_losses or []
        val_accuracy = getattr(self, "val_accuracies", [])
        test_accuracy = self.test_accuracy or []
        plt.figure(figsize=(8, 6))
        if train_losses:
            plt.plot(
                range(1, len(train_losses) + 1),
                train_losses,
                label="Train Loss",
                marker="o",
            )
        if val_accuracy:
            plt.plot(
                range(1, len(val_accuracy) + 1),
                val_accuracy,
                label="Val Accuracy",
                marker="o",
            )
        if test_accuracy:
            plt.plot(
                range(1, len(test_accuracy) + 1),
                test_accuracy,
                label="Test Accuracy",
                marker="o",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Learning Curves (Loss/Accuracy)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    # -------------------------- Confusion Matrix ------------------------------
    def plot_confusion_matrix_roc_pr(self) -> None:
        '''
        Plot confusion matrix as well as ROC and PR curves (one-vs-rest)
        for the test dataloader.
        '''
        if self.test_loader is None or self.model is None:
            print("Test DataLoader or model not available.")
            return
        # Collect predictions
        all_labels: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []
        self.model.eval()
        # No gradient computation in test mode
        with torch.inference_mode():
            self.model.eval()
            for images, labels in self.test_loader:
                images = images.to(device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        y_pred = np.argmax(y_prob, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        # ROC/PR (one-vs-rest) and AP/mAP
        fpr: dict[int, np.ndarray] = {}
        tpr: dict[int, np.ndarray] = {}
        roc_auc: dict[int, float] = {}
        precision: dict[int, np.ndarray] = {}
        recall: dict[int, np.ndarray] = {}
        pr_auc: dict[int, float] = {}
        ap_per_class: dict[int, float] = {}  # [NEW]
        for i in range(OUTPUT_SIZE):
            y_true_bin = (y_true == i).astype(int)
            y_score = y_prob[:, i]
            fpr[i], tpr[i], _ = roc_curve(y_true_bin, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])
            precision[i], recall[i], _ = precision_recall_curve(
                y_true_bin, y_score
            )
            pr_auc[i] = auc(recall[i], precision[i])
            # Average Precision (AP) per class:
            ap_per_class[i] = average_precision_score(y_true_bin, y_score)
        mAP = float(np.mean(list(ap_per_class.values())))
        logging.info("mAP (one-vs-rest, mean AP across classes): %.4f", mAP)
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ConfusionMatrixDisplay(cm).plot(
            ax=axes[0], cmap="Blues", colorbar=False
        )
        axes[0].set_title("Confusion Matrix")
        for i in range(OUTPUT_SIZE):
            axes[1].plot(
                fpr[i], tpr[i], label=f"C{i} (AUC={roc_auc[i]:.2f})"
            )
        axes[1].plot([0, 1], [0, 1], "k--")
        axes[1].set_title("ROC Curves")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend()
        for i in range(OUTPUT_SIZE):
            axes[2].plot(
                recall[i], precision[i],
                label=f"C{i} (PR-AUC={pr_auc[i]:.2f}, AP={ap_per_class[i]:.2f})"
            )
        axes[2].set_title("Precision–Recall Curves (+ AP)")
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].legend()
        plt.tight_layout()
        plt.show()
    #-------------------------- Misclassified examples -------------------------
    def show_wrong_classified_images(self, k: int = 12, cols: int = 6) -> None:
        '''
        Show top-k misclassifications from the test dataloader.
        '''
        if (self.test_dataset is None or self.test_loader is None or
                self.model is None):
            print("Test data or model not available.")
            return
        self.model.eval()
        mistakes: list[dict[str, Any]] = []
        # No gradient computation in test mode
        with torch.inference_mode():
            self.model.eval()
            for images, labels in self.test_loader:
                images = images.to(device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)
                mism = preds.cpu() != labels
                if mism.any():
                    for img, y_true, y_pred, conf in zip(
                        images[mism],
                        labels[mism],
                        preds[mism].cpu(),
                        confs[mism].cpu(),
                    ):
                        mistakes.append(
                            {
                                "image": img.cpu(),
                                "true": y_true.item(),
                                "pred": y_pred.item(),
                                "prob": conf.item(),
                            }
                        )
        if not mistakes:
            logging.info("No misclassifications found.")
            return
        mistakes_sorted = sorted(
            mistakes, key=lambda d: d["prob"], reverse=True
        )
        topk = mistakes_sorted[:k]
        rows = int(np.ceil(len(topk) / cols))
        plt.figure(figsize=(2.2 * cols, 2.6 * rows))
        for i, item in enumerate(topk):
            img = item["image"].squeeze(0).cpu().numpy()
            y_true = item["true"]
            y_pred = item["pred"]
            prob = item["prob"] * 100.0
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(img, cmap="gray")
            ax.set_title(
                f"Pred {y_pred} (p={prob:.1f}%)\nTrue {y_true}", fontsize=9
            )
            ax.axis("off")
        plt.suptitle(
            "Top-k Misclassifications (highest confidence)", fontsize=12
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    # ----------------------- Weight visualization -----------------------------
    def plot_weights(self, n: int = 10) -> None:
        '''
        Visualize weights of the first linear layer as 28x28 maps.
        '''
        if self.model is None:
            print("Model not available.")
            return
        # First linear layer: index 1 (index 0 is Flatten)
        weights = self.model.network[1].weight.data
        n = max(1, min(n, weights.shape[0]))
        cols = max(1, n // 2)
        rows = 2 if n > 1 else 1
        plt.figure(figsize=(2 * cols, 2 * rows))
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(weights[i].view(28, 28), cmap="seismic")
            plt.axis("off")
        plt.suptitle("Visualization of the first layer's weights")
        plt.tight_layout()
        plt.show()
#------------------------------- Helper functions ------------------------------
def _print_metrics_table(epochs: int,
                        train_losses: list[float],
                        val_accuracies: list[float],
                        test_accuracies: list[float]) -> None:
    '''
    Print a nicely formatted console table with epoch/loss/accuracy.
    '''
    h_ep, h_tl, h_va, h_ta = "Epoch", "Train-Loss", "Val-Acc", "Test-Acc"
    w_ep, w_tl, w_va, w_ta = 6, 11, 8, 9
    def line(sep: str = "-") -> str:
        return (
            "+" + sep * (w_ep + 2) +
            "+" + sep * (w_tl + 2) +
            "+" + sep * (w_va + 2) +
            "+" + sep * (w_ta + 2) + "+"
        )
    table_lines = [line("-"),
                    f"| {h_ep:>{w_ep}} | {h_tl:>{w_tl}} | {h_va:>{w_va}} | "
                    f"{h_ta:>{w_ta}} |",
                    line("-")]
    for i in range(epochs):
        tl = f"{train_losses[i]:.4f}"
        va = f"{val_accuracies[i]:.2f}%"
        ta = f"{test_accuracies[i]:.2f}%"
        table_lines.append(
            f"| {i+1:>{w_ep}} | {tl:>{w_tl}} | {va:>{w_va}} | "
            f"{ta:>{w_ta}} |"
        )
    table_lines.append(line("-"))
    logging.info("\n" + "\n".join(table_lines))
#------------------------------- Main logic ------------------------------------
def main() -> None:
    '''
    Run training, validation, and testing; save weights; trigger
    visualizations; and create tables/CSV/checkpoint of metrics and states.
    '''

    #--------------------------- Train & eval helpers --------------------------
    def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ) -> float:
        '''
        Train the model for one epoch.
        Return: average loss (float) across batches.
        '''
        model.train()
        running_loss = 0.0
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(loader)
    def evaluate(model: nn.Module,
                loader: DataLoader,
                device: torch.device) -> float:
        '''
        Evaluate accuracy (%) on the given DataLoader.
        '''
        correct = 0
        total = 0
        with torch.inference_mode():  # [NEW]
            model.eval()
            for data, labels in loader:
                data = data.to(device)
                labels = labels.to(device)
                logits = model(data)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / float(total)
    # Full training dataset
    full_dataset = MyDataset(train=True)
    # 80/20 split into train/validation
    train_data_len = int(0.8 * len(full_dataset))
    val_data_len = len(full_dataset) - train_data_len
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_data_len, val_data_len],
        generator=torch.Generator().manual_seed(42),
    )
    #------------------------ DataLoader parameters (performance) ---------------
    # Workers / pinned memory / persistent workers
    has_cuda = device.type == "cuda"
    auto_workers = max(1, (os.cpu_count() or 2) // 2)
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": auto_workers,
        "pin_memory": has_cuda,
        "persistent_workers": auto_workers > 0,
        "worker_init_fn": _seed_worker,
        "generator": torch.Generator().manual_seed(42),
    }
    train_data = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_data = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_dataset = MyDataset(train=False)
    test_data = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    # Model, loss, optimizer, scheduler
    model = NN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    lr_gamma = 0.85
    lr_step_size = 1
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step_size, gamma=lr_gamma
    )
    # Tracking lists
    train_losses: list[float] = []
    val_accuracies: list[float] = []
    test_accuracies: list[float] = []
    lr_per_epoch: list[float] = []
    #------------------------------- Training ----------------------------------
    for epoch in range(1, EPOCHS + 1):
        lr_per_epoch.append(optimizer.param_groups[0]["lr"])
        loss = train_one_epoch(model, train_data, criterion, optimizer, device)
        val_acc = evaluate(model, val_data, device)
        test_acc = evaluate(model, test_data, device)
        train_losses.append(loss)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
        logging.info(
            "Epoch %02d/%d | Train-Loss: %.4f | Val-Acc: %.2f%% | "
            "Test-Acc: %.2f%%",
            epoch, EPOCHS, loss, val_acc, test_acc
        )
        scheduler.step()
    #-------------------------- Table/CSV output --------------------------------
    _print_metrics_table(EPOCHS, train_losses, val_accuracies, test_accuracies)
    if pd is not None:
        df = pd.DataFrame(
            {
                "epoch": list(range(1, EPOCHS + 1)),
                "train_loss": train_losses,
                "val_accuracy": val_accuracies,
                "test_accuracy": test_accuracies,
                "lr": lr_per_epoch,
            }
        )
        try:
            df.to_csv(SAVE_METRICS_CSV, index=False, encoding="utf-8")
            logging.info("Metrics CSV saved: %s",
                        os.path.abspath(SAVE_METRICS_CSV))
        except Exception as e:
            logging.error("Saving CSV failed: %r", e)
            logging.info("Tip: Check write speed/folder permissions.")
    else:
        logging.info(
            "Pandas not available. Skipped CSV export. "
            "Install 'pandas' for CSV output."
        )
    #--------------------- Checkpoint (resume possible) ------------------------
    # Save full state incl. optimizer/scheduler.
    try:
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": EPOCHS,
            "hparams": {
                "lr": LR, "batch_size": BATCH_SIZE, "hidden1": H1, "hidden2": H2
            },
            "metrics": {
                "train_losses": train_losses,
                "val_accuracies": val_accuracies,
                "test_accuracies": test_accuracies,
                "lr_per_epoch": lr_per_epoch,
            },
        }
        torch.save(ckpt, SAVE_CHECKPOINT)
        logging.info("Checkpoint saved: %s",
                    os.path.abspath(SAVE_CHECKPOINT))
    except Exception as e:
        logging.error("Saving checkpoint failed: %r", e)
    # Additionally save pure state_dict (compact)
    try:
        torch.save(model.state_dict(), SAVE_FILE)
        logging.info("Weights saved: %s", os.path.abspath(SAVE_FILE))
    except Exception as e:
        logging.error("Saving weights failed: %r", e)
        logging.info("Tip: Choose a different target directory (without sync).")
    #------------------- Enable visualizations ---------------------------------
    viz = Visualisation(
        model=model,
        train_losses=train_losses,
        test_accuracy=test_accuracies,
        train_loader=train_data,
        test_loader=test_data,
        test_dataset=test_dataset,
    )
    viz.val_accuracies = val_accuracies
    viz.plot_loss_train_val_accuracy(EPOCHS)
    viz.plot_confusion_matrix_roc_pr()
    viz.show_wrong_classified_images(k=10)
    viz.plot_weights(n=10)
#------------------------------ CLI (argparse) ---------------------------------
# Minimal CLI to control paths/hyperparameters without code changes.
def _parse_args() -> argparse.Namespace:
    '''
    Parse optional arguments for epochs, batch size, learning rate, and paths.
    '''
    p = argparse.ArgumentParser(
        prog="NN_PyTorch_INT",
        description="Train a simple MNIST feed-forward neural network."
    )
    p.add_argument("--epochs", type=int, default=EPOCHS,
                    help="Number of epochs (Default: %(default)s)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="Batch size (Default: %(default)s)")
    p.add_argument("--lr", type=float, default=LR,
                    help="Learning rate (Default: %(default)s)")
    p.add_argument("--root", type=str, default=ROOT_WAY,
                    help="MNIST root (Default: current code value)")
    p.add_argument("--save-dir", type=str, default=SAVE_WAY,
                    help="Output directory (Default: current code value)")
    return p.parse_args()
def _apply_cli_overrides(ns: argparse.Namespace) -> None:
    '''
    Apply CLI overrides to the global parameters.
    '''
    global EPOCHS, BATCH_SIZE, LR, LEARNING_RATE
    global ROOT_WAY, SAVE_WAY, SAVE_FILE, SAVE_METRICS_CSV, SAVE_CHECKPOINT
    EPOCHS = int(ns.epochs)
    BATCH_SIZE = int(ns.batch_size)
    LR = float(ns.lr)
    LEARNING_RATE = LR
    ROOT_WAY = ns.root
    SAVE_WAY = ns.save_dir
    os.makedirs(SAVE_WAY, exist_ok=True)
    SAVE_FILE = os.path.join(SAVE_WAY, "mnist_nn_state.pth")
    SAVE_METRICS_CSV = os.path.join(SAVE_WAY, "mnist_metrics.csv")
    SAVE_CHECKPOINT = os.path.join(SAVE_WAY, "mnist_checkpoint.pt")
    logging.info("Config applied: epochs=%d, batch=%d, lr=%.4g",
                EPOCHS, BATCH_SIZE, LR)
    logging.info("Paths: root=%s | save_dir=%s", ROOT_WAY, SAVE_WAY)
#-------------------------- Main program ---------------------------------------
if __name__ == "__main__":
    # Apply CLI without changing structure.
    try:
        args = _parse_args()
        _apply_cli_overrides(args)
    except SystemExit:
        # argparse showed --help; exit normally.
        raise
    main()
    # ---------------------------- Runtime measurement -------------------------
    end = time.time()
    logging.info("Runtime: %.2f seconds", (end - start))
