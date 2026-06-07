import json
import random
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import articulate as art
from config import amass, datasets, model_config, paths
from models.tic_calibrator import TICTransformerCalibrator, simulate_imu_drift_offset


COMBO_MAP = {
    "lw_rp_h": [0, 3, 4],
}


class AMASSTICWindowDataset(Dataset):
    def __init__(
        self,
        combo_name: str,
        seq_len: int = 256,
        stride: int = 128,
        split: str = "train",
        split_ratio: float = 0.95,
        seed: int = 1234,
        max_windows: int = 0,
    ):
        self.combo = COMBO_MAP[combo_name]
        self.seq_len = seq_len
        self.stride = stride
        self.cache = {}

        files = [paths.processed_datasets / f"{name}.pt" for name in datasets.amass_datasets]
        files = [f for f in files if f.exists()]
        seq_refs = []
        for file_idx, fpath in enumerate(files):
            data = torch.load(fpath, map_location="cpu")
            for seq_idx, (acc, ori) in enumerate(zip(data["acc"], data["ori"])):
                length = min(acc.shape[0], ori.shape[0])
                if length < seq_len:
                    continue
                starts = list(range(0, length - seq_len + 1, stride))
                if starts and starts[-1] != length - seq_len:
                    starts.append(length - seq_len)
                elif not starts:
                    starts = [0]
                for start in starts:
                    seq_refs.append((file_idx, seq_idx, start))

        rng = random.Random(seed)
        rng.shuffle(seq_refs)
        cut = int(len(seq_refs) * split_ratio)
        self.refs = seq_refs[:cut] if split == "train" else seq_refs[cut:]
        if max_windows > 0:
            self.refs = self.refs[:max_windows]
        self.files = files

    def __len__(self):
        return len(self.refs)

    def _get_file(self, file_idx: int):
        if file_idx not in self.cache:
            self.cache[file_idx] = torch.load(self.files[file_idx], map_location="cpu")
        return self.cache[file_idx]

    def __getitem__(self, idx):
        file_idx, seq_idx, start = self.refs[idx]
        data = self._get_file(file_idx)
        acc = data["acc"][seq_idx][start : start + self.seq_len, self.combo].float()
        ori = data["ori"][seq_idx][start : start + self.seq_len, self.combo].float()
        return ori, acc


def build_input(acc: torch.Tensor, ori: torch.Tensor):
    acc = (acc / amass.acc_scale).reshape(acc.shape[0], acc.shape[1], -1)
    ori = ori.reshape(ori.shape[0], ori.shape[1], -1)
    return torch.cat([acc, ori], dim=-1)


def batch_angle_error(pred_r6d: torch.Tensor, target_r6d: torch.Tensor):
    pred = art.math.r6d_to_rotation_matrix(pred_r6d.reshape(-1, 6))
    target = art.math.r6d_to_rotation_matrix(target_r6d.reshape(-1, 6))
    rel = pred.transpose(-2, -1).matmul(target)
    trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    angle = torch.acos(((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)) * 180.0 / torch.pi
    return angle.mean()


def evaluate(model, loader, device, imu_num):
    model.eval()
    mse = torch.nn.MSELoss()
    total_loss = 0.0
    total_drift = 0.0
    total_offset = 0.0
    total_count = 0
    with torch.no_grad():
        for rot, acc in loader:
            rot = rot.to(device)
            acc = acc.to(device)
            noisy_rot, noisy_acc, drift, offset = simulate_imu_drift_offset(
                imu_rot=rot,
                imu_acc=acc,
                imu_num=imu_num,
                ego_imu_id=imu_num - 1,
            )
            x = build_input(noisy_acc, noisy_rot)
            drift_hat, offset_hat = model(x)
            loss = mse(drift_hat, drift) + mse(offset_hat, offset)
            drift_err = batch_angle_error(drift_hat, drift)
            offset_err = batch_angle_error(offset_hat, offset)
            batch_size = rot.shape[0]
            total_loss += loss.item() * batch_size
            total_drift += drift_err.item() * batch_size
            total_offset += offset_err.item() * batch_size
            total_count += batch_size
    return {
        "total_loss": total_loss / max(total_count, 1),
        "drift_err_deg": total_drift / max(total_count, 1),
        "offset_err_deg": total_offset / max(total_count, 1),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--combo", type=str, default="lw_rp_h", choices=sorted(COMBO_MAP))
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split-ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--stack", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save-dir", type=str, default="data/checkpoints/tic_calibrator_amass")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_dataset = AMASSTICWindowDataset(
        combo_name=args.combo,
        seq_len=args.seq_len,
        stride=args.stride,
        split="train",
        split_ratio=args.split_ratio,
        seed=args.seed,
        max_windows=args.max_train_windows,
    )
    val_dataset = AMASSTICWindowDataset(
        combo_name=args.combo,
        seq_len=args.seq_len,
        stride=args.stride,
        split="val",
        split_ratio=args.split_ratio,
        seed=args.seed,
        max_windows=args.max_val_windows,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    imu_num = len(COMBO_MAP[args.combo])
    model = TICTransformerCalibrator(
        imu_num=imu_num,
        n_input=imu_num * 12,
        stack=args.stack,
        multi_head=args.nhead,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_drift = 0.0
        train_offset = 0.0
        count = 0
        for rot, acc in train_loader:
            rot = rot.to(device)
            acc = acc.to(device)
            noisy_rot, noisy_acc, drift, offset = simulate_imu_drift_offset(
                imu_rot=rot,
                imu_acc=acc,
                imu_num=imu_num,
                ego_imu_id=imu_num - 1,
            )
            x = build_input(noisy_acc, noisy_rot)
            drift_hat, offset_hat = model(x)
            loss = mse(drift_hat, drift) + mse(offset_hat, offset)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = rot.shape[0]
            train_loss += loss.item() * batch_size
            train_drift += batch_angle_error(drift_hat.detach(), drift).item() * batch_size
            train_offset += batch_angle_error(offset_hat.detach(), offset).item() * batch_size
            count += batch_size

        train_metrics = {
            "train_loss": train_loss / max(count, 1),
            "train_drift_err_deg": train_drift / max(count, 1),
            "train_offset_err_deg": train_offset / max(count, 1),
        }
        val_metrics = evaluate(model, val_loader, device, imu_num)
        epoch_log = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(epoch_log)

        writer.add_scalar("loss/train_total", train_metrics["train_loss"], epoch)
        writer.add_scalar("loss/train_drift_deg", train_metrics["train_drift_err_deg"], epoch)
        writer.add_scalar("loss/train_offset_deg", train_metrics["train_offset_err_deg"], epoch)
        writer.add_scalar("loss/val_total", val_metrics["total_loss"], epoch)
        writer.add_scalar("loss/val_drift_deg", val_metrics["drift_err_deg"], epoch)
        writer.add_scalar("loss/val_offset_deg", val_metrics["offset_err_deg"], epoch)
        print(
            f"epoch {epoch}: "
            f"train={train_metrics['train_loss']:.6f} "
            f"val={val_metrics['total_loss']:.6f} "
            f"drift={val_metrics['drift_err_deg']:.3f} "
            f"offset={val_metrics['offset_err_deg']:.3f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "model_type": "tic_transformer_calibrator",
            "epoch": epoch,
            "history": history,
        }
        torch.save(checkpoint, last_path)
        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            torch.save(checkpoint, best_path)

    (save_dir / "history.json").write_text(json.dumps(history, indent=2))
    writer.close()
    print(f"BEST_CHECKPOINT={best_path}")


if __name__ == "__main__":
    main()
