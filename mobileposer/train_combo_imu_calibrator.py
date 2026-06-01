import json
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from config import datasets, model_config, paths
from evaluate_direct import load_direct_model
from models.imu_calibrator import ComboTemporalIMUCalibrator, build_imu_input


COMBO_MAP = {
    "lw_rp_h": [0, 3, 4],
}


class ComboCalibratorWindowDataset(Dataset):
    def __init__(self, data_path: Path, combo_name: str, window_size: int = 125, stride: int = 60):
        data = torch.load(data_path, map_location="cpu")
        combo = COMBO_MAP[combo_name]
        windows = []
        for input_acc, input_ori, target_acc, target_ori, valid_mask, pose in zip(
            data["input_acc"],
            data["input_ori"],
            data["target_acc"],
            data["target_ori"],
            data["valid_mask"],
            data["pose"],
        ):
            input_feat = build_imu_input(input_acc.float(), input_ori.float())[:, combo]
            target_acc = target_acc[:, combo].float()
            target_ori = target_ori[:, combo].float()
            valid_mask = valid_mask[:, combo].bool()
            pose = pose.float()
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
                        "target_acc": target_acc[start:end],
                        "target_ori": target_ori[start:end],
                        "valid_mask": valid_mask[start:end],
                        "target_pose": pose[start:end],
                    }
                )
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def collate_batch(batch):
    max_len = max(item["input"].shape[0] for item in batch)
    combo_size = batch[0]["input"].shape[1]
    feat_dim = batch[0]["input"].shape[2]

    input_tensor = torch.zeros(len(batch), max_len, combo_size, feat_dim)
    target_acc = torch.zeros(len(batch), max_len, combo_size, 3)
    target_ori = torch.zeros(len(batch), max_len, combo_size, 3, 3)
    valid_mask = torch.zeros(len(batch), max_len, combo_size, dtype=torch.bool)
    target_pose = torch.zeros(len(batch), max_len, 24, 3, 3)
    seq_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for idx, item in enumerate(batch):
        seq_len = item["input"].shape[0]
        input_tensor[idx, :seq_len] = item["input"]
        target_acc[idx, :seq_len] = item["target_acc"]
        target_ori[idx, :seq_len] = item["target_ori"]
        valid_mask[idx, :seq_len] = item["valid_mask"]
        target_pose[idx, :seq_len] = item["target_pose"]
        seq_mask[idx, :seq_len] = True

    return {
        "input": input_tensor,
        "target_acc": target_acc,
        "target_ori": target_ori,
        "valid_mask": valid_mask,
        "target_pose": target_pose,
        "seq_mask": seq_mask,
    }


def masked_mse(pred, target, mask):
    expand_shape = [1] * (pred.dim() - mask.dim())
    mask = mask.view(*mask.shape, *expand_shape).expand_as(pred)
    diff = (pred - target)[mask]
    if diff.numel() == 0:
        return pred.new_tensor(0.0)
    return (diff ** 2).mean()


def compute_jerk_loss(pred_ori6d, seq_mask):
    if pred_ori6d.shape[1] < 4:
        return pred_ori6d.new_tensor(0.0)
    jerk = pred_ori6d[:, 3:] - 3 * pred_ori6d[:, 2:-1] + 3 * pred_ori6d[:, 1:-2] - pred_ori6d[:, :-3]
    valid = seq_mask[:, 3:] & seq_mask[:, 2:-1] & seq_mask[:, 1:-2] & seq_mask[:, :-3]
    if not valid.any():
        return pred_ori6d.new_tensor(0.0)
    jerk = jerk[valid]
    l1_norm = torch.norm(jerk, p=1, dim=-1)
    return l1_norm.mean()


def build_combo_imu(input_acc, pred_ori, combo):
    batch_size, seq_len = input_acc.shape[:2]
    full_acc = input_acc.new_zeros(batch_size, seq_len, 7, 3)
    full_ori = torch.eye(3, device=pred_ori.device).view(1, 1, 1, 3, 3).repeat(batch_size, seq_len, 7, 1, 1)
    full_acc[:, :, combo] = input_acc
    full_ori[:, :, combo] = pred_ori
    return torch.cat([full_acc.flatten(2), full_ori.flatten(2)], dim=2)


def compute_pose_loss(mocap_model, imu_inputs, seq_mask, target_pose):
    seq_lengths = seq_mask.sum(dim=1).tolist()
    pred_reduced = mocap_model(imu_inputs, seq_lengths)
    pred_pose = mocap_model._reduced_global_to_full(pred_reduced)
    mask = seq_mask.view(seq_mask.shape[0], seq_mask.shape[1], 1, 1, 1).expand_as(pred_pose)
    return F.mse_loss(pred_pose[mask], target_pose[mask])


def evaluate(model, loader, mocap_model, combo, device, imu_loss_weight, pose_loss_weight, jerk_loss_weight):
    model.eval()
    total_ori = 0.0
    total_pose = 0.0
    total_jerk = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            target_ori = batch["target_ori"].to(device)
            target_pose = batch["target_pose"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            valid_mask = batch["valid_mask"].to(device) & seq_mask.unsqueeze(-1)

            _, pred_ori, pred_ori6d = model(inputs, seq_mask=seq_mask, return_ori6d=True)
            ori_loss = masked_mse(pred_ori, target_ori, valid_mask)
            jerk_loss = compute_jerk_loss(pred_ori6d, seq_mask)
            pose_loss = pred_ori6d.new_tensor(0.0)

            batch_size = inputs.shape[0]
            total_ori += ori_loss.item() * batch_size
            total_pose += pose_loss.item() * batch_size
            total_jerk += jerk_loss.item() * batch_size
            total_count += batch_size

    return {
        "acc_loss": 0.0,
        "ori_loss": total_ori / max(total_count, 1),
        "pose_loss": total_pose / max(total_count, 1),
        "jerk_loss": total_jerk / max(total_count, 1),
        "total_loss": (
            imu_loss_weight * total_ori + pose_loss_weight * total_pose + jerk_loss_weight * total_jerk
        ) / max(total_count, 1),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--train-data", type=str, default=str(paths.eval_dir / datasets.huawei_new_calibrator_train))
    parser.add_argument("--val-data", type=str, default=str(paths.eval_dir / datasets.huawei_new_calibrator_test))
    parser.add_argument("--combo", type=str, default="lw_rp_h", choices=sorted(COMBO_MAP))
    parser.add_argument("--mocap-model", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="data/checkpoints/combo_imu_calibrator")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=125)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--imu-loss-weight", type=float, default=1.0)
    parser.add_argument("--pose-loss-weight", type=float, default=0.0)
    parser.add_argument("--jerk-loss-weight", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_dataset = ComboCalibratorWindowDataset(Path(args.train_data), args.combo, args.window_size, args.stride)
    val_dataset = ComboCalibratorWindowDataset(Path(args.val_data), args.combo, args.window_size, args.stride)
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
    combo = COMBO_MAP[args.combo]
    model = ComboTemporalIMUCalibrator(
        combo_size=len(COMBO_MAP[args.combo]),
        predict_acc=False,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
        nhead=args.nhead,
        max_seq_len=args.window_size,
    ).to(device)
    mocap_model = load_direct_model(
        args.mocap_model,
        backbone="transformer",
        transformer_kwargs={
            "transformer_d_model": 192,
            "transformer_nhead": 8,
            "transformer_num_layers": 6,
            "transformer_dim_feedforward": 768,
            "transformer_dropout": 0.4,
        },
    )
    mocap_model.eval()
    for param in mocap_model.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_ori = 0.0
        running_pose = 0.0
        running_jerk = 0.0
        running_count = 0
        for batch in train_loader:
            inputs = batch["input"].to(device)
            target_ori = batch["target_ori"].to(device)
            target_pose = batch["target_pose"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            valid_mask = batch["valid_mask"].to(device) & seq_mask.unsqueeze(-1)

            _, pred_ori, pred_ori6d = model(inputs, seq_mask=seq_mask, return_ori6d=True)
            ori_loss = masked_mse(pred_ori, target_ori, valid_mask)
            jerk_loss = compute_jerk_loss(pred_ori6d, seq_mask)
            pose_loss = pred_ori6d.new_tensor(0.0)
            loss = (
                args.imu_loss_weight * ori_loss
                + args.pose_loss_weight * pose_loss
                + args.jerk_loss_weight * jerk_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.shape[0]
            running_ori += ori_loss.item() * inputs.shape[0]
            running_pose += pose_loss.item() * inputs.shape[0]
            running_jerk += jerk_loss.item() * inputs.shape[0]
            running_count += inputs.shape[0]

        train_loss = running_loss / max(running_count, 1)
        train_ori = running_ori / max(running_count, 1)
        train_pose = running_pose / max(running_count, 1)
        train_jerk = running_jerk / max(running_count, 1)
        val_metrics = evaluate(
            model,
            val_loader,
            mocap_model,
            combo,
            device,
            args.imu_loss_weight,
            args.pose_loss_weight,
            args.jerk_loss_weight,
        )
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ori_loss": train_ori,
            "train_pose_loss": train_pose,
            "train_jerk_loss": train_jerk,
            **val_metrics,
        }
        history.append(epoch_log)
        writer.add_scalar("loss/train_total", train_loss, epoch)
        writer.add_scalar("loss/train_ori", train_ori, epoch)
        writer.add_scalar("loss/train_pose", train_pose, epoch)
        writer.add_scalar("loss/train_jerk", train_jerk, epoch)
        writer.add_scalar("loss/val_total", val_metrics["total_loss"], epoch)
        writer.add_scalar("loss/val_acc", val_metrics["acc_loss"], epoch)
        writer.add_scalar("loss/val_ori", val_metrics["ori_loss"], epoch)
        writer.add_scalar("loss/val_pose", val_metrics["pose_loss"], epoch)
        writer.add_scalar("loss/val_jerk", val_metrics["jerk_loss"], epoch)
        print(
            f"epoch {epoch}: train={train_loss:.6f} "
            f"val_total={val_metrics['total_loss']:.6f} "
            f"val_acc={val_metrics['acc_loss']:.6f} "
            f"val_ori={val_metrics['ori_loss']:.6f} "
            f"val_pose={val_metrics['pose_loss']:.6f} "
            f"val_jerk={val_metrics['jerk_loss']:.6f}"
        )

        checkpoint_args = vars(args).copy()
        checkpoint_args["predict_acc"] = False
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": checkpoint_args,
            "model_type": "combo_temporal_transformer",
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
