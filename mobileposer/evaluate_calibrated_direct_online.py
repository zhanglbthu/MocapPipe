import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

from config import amass, datasets, model_config, paths
from evaluate import PoseEvaluator
from evaluate_direct import load_direct_model
from models.imu_calibrator import MultiDeviceIMUCalibrator, TemporalIMUCalibrator, build_imu_input
from run_calibrated_directposer import COMBOS, load_calibrator


def pad_imu(acc: torch.Tensor, ori: torch.Tensor, num_devices: int = 7):
    if acc.shape[1] < num_devices:
        acc = torch.cat([acc, torch.zeros(acc.shape[0], num_devices - acc.shape[1], 3)], dim=1)
        ori = torch.cat([ori, torch.zeros(ori.shape[0], num_devices - ori.shape[1], 3, 3)], dim=1)
    return acc, ori


@torch.no_grad()
def run_calibrator_online(calibrator, acc: torch.Tensor, ori: torch.Tensor):
    valid_mask = torch.zeros(acc.shape[0], acc.shape[1], dtype=torch.bool)
    valid_mask[:, :5] = True

    calibrator_input = build_imu_input(acc, ori).to(model_config.device)
    if isinstance(calibrator, TemporalIMUCalibrator):
        calibrator.reset()
        pred_acc, pred_ori = [], []
        for frame_idx in range(calibrator_input.shape[0]):
            acc_i, ori_i = calibrator.forward_frame(calibrator_input[frame_idx])
            pred_acc.append(acc_i.cpu())
            pred_ori.append(ori_i.cpu())
        pred_acc = torch.stack(pred_acc)
        pred_ori = torch.stack(pred_ori)
    else:
        pred_acc, pred_ori = calibrator(calibrator_input)
        pred_acc = pred_acc.cpu()
        pred_ori = pred_ori.cpu()

    pred_acc[~valid_mask] = 0
    identity = torch.eye(3).view(1, 1, 3, 3).expand_as(pred_ori)
    pred_ori = torch.where(valid_mask.view(valid_mask.shape[0], valid_mask.shape[1], 1, 1), pred_ori, identity)
    return pred_acc, pred_ori


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--calibrator", type=str, required=True)
    parser.add_argument("--mocap-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="imuposer")
    parser.add_argument("--combo", type=str, default="lw_rp_h", choices=sorted(COMBOS))
    parser.add_argument("--output-dir", type=str, default="data/eval/imuposer/lw_rp_h/calibrated_directposer_online")
    args = parser.parse_args()

    if args.dataset != "imuposer":
        raise ValueError("This script currently supports only imuposer test evaluation.")

    device = model_config.device
    calibrator = load_calibrator(args.calibrator, device)
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
    calibrator.eval()
    mocap_model.eval()

    data_path = paths.eval_dir / datasets.imuposer_test
    data = torch.load(data_path, map_location="cpu")
    evaluator = PoseEvaluator()
    combo = COMBOS[args.combo]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_errs = []
    for seq_idx, (acc, ori, pose_t, tran_t) in enumerate(zip(data["acc"], data["ori"], data["pose"], data["tran"]), start=1):
        print(f"Evaluating sample {seq_idx}/{len(data['acc'])}...")
        acc = acc.float()
        ori = ori.float()
        pose_t = pose_t.float().view(-1, 24, 3, 3)
        tran_t = tran_t.float().view(-1, 3)

        acc, ori = pad_imu(acc, ori)
        pred_acc, pred_ori = run_calibrator_online(calibrator, acc, ori)

        combo_acc = torch.zeros_like(pred_acc)
        combo_ori = torch.zeros_like(pred_ori)
        combo_acc[:, combo] = pred_acc[:, combo] / amass.acc_scale
        combo_ori[:, combo] = pred_ori[:, combo]
        imu = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)

        mocap_model.reset()
        pose_p = []
        for frame_idx in tqdm(range(imu.shape[0])):
            pose_p.append(mocap_model.forward_frame(imu[frame_idx].to(device)).cpu())
        pose_p = torch.stack(pose_p)

        err = evaluator.eval(pose_p, pose_t, tran_t=tran_t)
        seq_errs.append(err)
        torch.save(
            {
                "pose_p": pose_p,
                "pose_t": pose_t,
                "tran_t": tran_t,
                "calibrated_acc": pred_acc,
                "calibrated_ori": pred_ori,
            },
            out_dir / f"{seq_idx}.pt",
        )

    errors = torch.stack(seq_errs)
    mean = errors.mean(dim=0)
    std = errors.std(dim=0)
    stats = torch.stack([mean, std], dim=1)
    PoseEvaluator.print(stats)

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
    report = {
        name: {"mean": float(stats[idx, 0]), "std": float(stats[idx, 1])}
        for idx, name in enumerate(names)
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    with open(out_dir / "report.txt", "w") as f:
        for name in names:
            f.write(f"{name}: {report[name]['mean']:.2f} (+/- {report[name]['std']:.2f})\n")


if __name__ == "__main__":
    main()
