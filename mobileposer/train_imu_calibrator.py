import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from config import datasets, paths
from models.imu_calibrator import TemporalIMUCalibrator, build_imu_input


class HuaweiCalibratorWindowDataset(Dataset):
    def __init__(self, data_path: Path, window_size: int = 120, stride: int = 60):
        data = torch.load(data_path, map_location="cpu")
        windows = []
        for input_acc, input_ori, target_acc, target_ori, valid_mask in zip(
            data["input_acc"],
            data["input_ori"],
            data["target_acc"],
            data["target_ori"],
            data["valid_mask"],
        ):
            input_feat = build_imu_input(input_acc.float(), input_ori.float())
            seq_len = input_feat.shape[0]
            if seq_len <= window_size:
                starts = [0]
            else:
                starts = list(range(0, seq_len - window_size + 1, stride))
                if starts[-1] != seq_len - window_size:
                    starts.append(seq_len - window_size)

            for start in starts:
                end = min(start + window_size, seq_len)
                windows.append(
                    {
                        "input": input_feat[start:end],
                        "target_acc": target_acc[start:end].float(),
                        "target_ori": target_ori[start:end].float(),
                        "valid_mask": valid_mask[start:end].bool(),
                    }
                )
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def collate_batch(batch):
    max_len = max(item["input"].shape[0] for item in batch)
    num_devices = batch[0]["input"].shape[1]

    input_tensor = torch.zeros(len(batch), max_len, num_devices, batch[0]["input"].shape[-1])
    target_acc = torch.zeros(len(batch), max_len, num_devices, 3)
    target_ori = torch.zeros(len(batch), max_len, num_devices, 3, 3)
    valid_mask = torch.zeros(len(batch), max_len, num_devices, dtype=torch.bool)
    seq_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for idx, item in enumerate(batch):
        seq_len = item["input"].shape[0]
        input_tensor[idx, :seq_len] = item["input"]
        target_acc[idx, :seq_len] = item["target_acc"]
        target_ori[idx, :seq_len] = item["target_ori"]
        valid_mask[idx, :seq_len] = item["valid_mask"]
        seq_mask[idx, :seq_len] = True

    return {
        "input": input_tensor,
        "target_acc": target_acc,
        "target_ori": target_ori,
        "valid_mask": valid_mask,
        "seq_mask": seq_mask,
    }


def masked_mse(pred, target, mask):
    expand_shape = [1] * (pred.dim() - mask.dim())
    mask = mask.view(*mask.shape, *expand_shape).expand_as(pred)
    diff = (pred - target)[mask]
    if diff.numel() == 0:
        return pred.new_tensor(0.0)
    return (diff ** 2).mean()


def evaluate(model, loader, device):
    model.eval()
    total_acc = 0.0
    total_ori = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            target_acc = batch["target_acc"].to(device)
            target_ori = batch["target_ori"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            valid_mask = batch["valid_mask"].to(device) & seq_mask.unsqueeze(-1)

            pred_acc, pred_ori = model(inputs, seq_mask=seq_mask)
            acc_loss = masked_mse(pred_acc, target_acc, valid_mask)
            ori_loss = masked_mse(pred_ori, target_ori, valid_mask)

            batch_size = inputs.shape[0]
            total_acc += acc_loss.item() * batch_size
            total_ori += ori_loss.item() * batch_size
            total_count += batch_size

    return {
        "acc_loss": total_acc / max(total_count, 1),
        "ori_loss": total_ori / max(total_count, 1),
        "total_loss": (total_acc + total_ori) / max(total_count, 1),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--train-data", type=str, default=str(paths.eval_dir / datasets.huawei_new_calibrator_train))
    parser.add_argument("--val-data", type=str, default=str(paths.eval_dir / datasets.huawei_new_calibrator_test))
    parser.add_argument("--save-dir", type=str, default="data/checkpoints/imu_calibrator_temporal")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=125)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_dataset = HuaweiCalibratorWindowDataset(
        Path(args.train_data), window_size=args.window_size, stride=args.stride
    )
    val_dataset = HuaweiCalibratorWindowDataset(
        Path(args.val_data), window_size=args.window_size, stride=args.stride
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    device = torch.device(args.device)
    model = TemporalIMUCalibrator(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
        nhead=args.nhead,
        max_seq_len=args.window_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        for batch in train_loader:
            inputs = batch["input"].to(device)
            target_acc = batch["target_acc"].to(device)
            target_ori = batch["target_ori"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            valid_mask = batch["valid_mask"].to(device) & seq_mask.unsqueeze(-1)

            pred_acc, pred_ori = model(inputs, seq_mask=seq_mask)
            acc_loss = masked_mse(pred_acc, target_acc, valid_mask)
            ori_loss = masked_mse(pred_ori, target_ori, valid_mask)
            loss = acc_loss + ori_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.shape[0]
            running_count += inputs.shape[0]

        train_loss = running_loss / max(running_count, 1)
        val_metrics = evaluate(model, val_loader, device)
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(epoch_log)
        print(
            f"epoch {epoch}: train={train_loss:.6f} "
            f"val_total={val_metrics['total_loss']:.6f} "
            f"val_acc={val_metrics['acc_loss']:.6f} "
            f"val_ori={val_metrics['ori_loss']:.6f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "model_type": "temporal_transformer",
            "epoch": epoch,
            "history": history,
        }
        torch.save(checkpoint, last_path)
        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            torch.save(checkpoint, best_path)

    (save_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"BEST_CHECKPOINT={best_path}")


if __name__ == "__main__":
    main()
