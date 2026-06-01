import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

from config import amass, datasets, model_config, paths
from evaluate import PoseEvaluator
from evaluate_direct import load_direct_model
from models.imu_calibrator import ComboTemporalIMUCalibrator, build_imu_input
from run_calibrated_directposer import COMBOS


def load_combo_calibrator(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get("args", {})
    model = ComboTemporalIMUCalibrator(
        combo_size=len(COMBOS[args.get("combo", "lw_rp_h")]),
        predict_acc=args.get("predict_acc", True),
        hidden_dim=args.get("hidden_dim", 128),
        dropout=args.get("dropout", 0.1),
        num_layers=args.get("num_layers", 3),
        nhead=args.get("nhead", 4),
        max_seq_len=args.get("window_size", 125),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def pad_imu(acc: torch.Tensor, ori: torch.Tensor, num_devices: int = 7):
    if acc.shape[1] < num_devices:
        acc = torch.cat([acc, torch.zeros(acc.shape[0], num_devices - acc.shape[1], 3)], dim=1)
        ori = torch.cat([ori, torch.zeros(ori.shape[0], num_devices - ori.shape[1], 3, 3)], dim=1)
    return acc, ori


@torch.no_grad()
def run_combo_calibrator_online(calibrator, acc: torch.Tensor, ori: torch.Tensor, combo):
    combo_feat = build_imu_input(acc, ori)[:, combo].to(model_config.device)
    calibrator.reset()
    pred_acc, pred_ori = [], []
    for frame_idx in range(combo_feat.shape[0]):
        acc_i, ori_i = calibrator.forward_frame_windowed(combo_feat[frame_idx])
        if acc_i is not None:
            pred_acc.append(acc_i.cpu())
        pred_ori.append(ori_i.cpu())
    pred_acc = None if not pred_acc else torch.stack(pred_acc)
    return pred_acc, torch.stack(pred_ori)


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--calibrator", type=str, required=True)
    parser.add_argument("--mocap-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="imuposer")
    parser.add_argument("--combo", type=str, default="lw_rp_h", choices=sorted(COMBOS))
    parser.add_argument("--output-dir", type=str, default="data/eval/imuposer/lw_rp_h/combo_calibrated_directposer_online")
    args = parser.parse_args()

    if args.dataset != "imuposer":
        raise ValueError("This script currently supports only imuposer test evaluation.")

    device = model_config.device
    combo = COMBOS[args.combo]
    calibrator = load_combo_calibrator(args.calibrator, device)
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

    data = torch.load(paths.eval_dir / datasets.imuposer_test, map_location="cpu")
    evaluator = PoseEvaluator()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_errs = []
    for seq_idx, (acc, ori, pose_t) in enumerate(zip(data["acc"], data["ori"], data["pose"]), start=1):
        print(f"Evaluating sample {seq_idx}/{len(data['acc'])}...")
        acc = acc.float()
        ori = ori.float()
        pose_t = pose_t.float().view(-1, 24, 3, 3)
        acc, ori = pad_imu(acc, ori)

        pred_acc_combo, pred_ori_combo = run_combo_calibrator_online(calibrator, acc, ori, combo)
        combo_acc = acc[:, combo].clone() if pred_acc_combo is None else pred_acc_combo
        pred_acc = torch.zeros(acc.shape[0], 7, 3)
        pred_ori = torch.eye(3).view(1, 1, 3, 3).repeat(acc.shape[0], 7, 1, 1)
        pred_acc[:, combo] = combo_acc
        pred_ori[:, combo] = pred_ori_combo

        imu_acc = torch.zeros_like(pred_acc)
        combo_ori = torch.zeros_like(pred_ori)
        imu_acc[:, combo] = pred_acc[:, combo] / amass.acc_scale
        combo_ori[:, combo] = pred_ori[:, combo]
        imu = torch.cat([imu_acc.flatten(1), combo_ori.flatten(1)], dim=1)

        mocap_model.reset()
        pose_p = []
        for frame_idx in tqdm(range(imu.shape[0])):
            pose_p.append(mocap_model.forward_frame(imu[frame_idx].to(device)).cpu())
        pose_p = torch.stack(pose_p)

        err = evaluator.eval(pose_p, pose_t)
        seq_errs.append(err)
        torch.save(
            {
                "pose_p": pose_p,
                "pose_t": pose_t,
                "calibrated_acc_combo": combo_acc,
                "calibrated_ori_combo": pred_ori_combo,
            },
            out_dir / f"{seq_idx}.pt",
        )

    summary = torch.stack(seq_errs).mean(dim=0)
    PoseEvaluator.print(summary)
    names = [
        "SIP Error (deg)",
        "Angular Error (deg)",
        "Masked Angular Error (deg)",
        "Positional Error (cm)",
        "Masked Positional Error (cm)",
        "Mesh Error (cm)",
        "Jitter Error (100m/s^3)",
        "Distance Error (cm)",
    ]
    report = {name: {"mean": float(summary[i, 0]), "std": float(summary[i, 1])} for i, name in enumerate(names)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    with open(out_dir / "report.txt", "w") as f:
        for name in names:
            f.write(f"{name}: {report[name]['mean']:.2f} (+/- {report[name]['std']:.2f})\n")


if __name__ == "__main__":
    main()
