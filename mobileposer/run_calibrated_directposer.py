from argparse import ArgumentParser
from pathlib import Path

import torch

from config import amass, model_config
from evaluate_direct import load_direct_model
from models.imu_calibrator import MultiDeviceIMUCalibrator, TemporalIMUCalibrator, build_imu_input


COMBOS = {
    "lw_rp_h": [0, 3, 4],
    "lw": [0],
}


def load_calibrator(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get("args", {})
    model_type = checkpoint.get("model_type", "mlp")
    if model_type == "temporal_transformer":
        model = TemporalIMUCalibrator(
            hidden_dim=args.get("hidden_dim", 128),
            dropout=args.get("dropout", 0.1),
            num_layers=args.get("num_layers", 3),
            nhead=args.get("nhead", 4),
            max_seq_len=args.get("window_size", 120),
        ).to(device)
    else:
        model = MultiDeviceIMUCalibrator(
            hidden_dim=args.get("hidden_dim", 128),
            dropout=args.get("dropout", 0.1),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def run_temporal_calibrator_online(
    calibrator: TemporalIMUCalibrator,
    calibrator_input: torch.Tensor,
):
    calibrator.reset()
    pred_acc_parts = []
    pred_ori_parts = []
    for frame_idx in range(calibrator_input.shape[0]):
        acc_part, ori_part = calibrator.forward_frame(calibrator_input[frame_idx])
        pred_acc_parts.append(acc_part)
        pred_ori_parts.append(ori_part)
    return torch.stack(pred_acc_parts), torch.stack(pred_ori_parts)


def build_valid_mask(sample: dict, seq_len: int):
    num_devices = 7
    valid_mask = torch.zeros(seq_len, num_devices, dtype=torch.bool)
    observed = sample["aM"].shape[1]
    valid_mask[:, :observed] = True
    if "synthetic_device_indices" in sample:
        valid_mask[:, [int(i) for i in sample["synthetic_device_indices"]]] = False
    return valid_mask


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--calibrator", type=str, required=True)
    parser.add_argument("--mocap-model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--combo", type=str, default="lw_rp_h", choices=sorted(COMBOS))
    parser.add_argument("--output", type=str, default="data/eval/huawei_new/calibrated_directposer.pt")
    args = parser.parse_args()

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
    mocap_model.eval()

    sample = torch.load(args.input, map_location="cpu")
    seq_len = sample["aM"].shape[0]

    input_acc = sample["aM"].float().view(seq_len, sample["aM"].shape[1], 3)
    input_ori = sample["RMB"].float().view(seq_len, sample["RMB"].shape[1], 3, 3)
    if input_acc.shape[1] < 7:
        input_acc = torch.cat([input_acc, torch.zeros(seq_len, 7 - input_acc.shape[1], 3)], dim=1)
        input_ori = torch.cat([input_ori, torch.zeros(seq_len, 7 - input_ori.shape[1], 3, 3)], dim=1)

    valid_mask = build_valid_mask(sample, seq_len)
    calibrator_input = build_imu_input(input_acc, input_ori).to(device)
    if isinstance(calibrator, TemporalIMUCalibrator):
        pred_acc, pred_ori = run_temporal_calibrator_online(
            calibrator,
            calibrator_input,
        )
    else:
        pred_acc, pred_ori = calibrator(calibrator_input)

    pred_acc = pred_acc.cpu()
    pred_ori = pred_ori.cpu()
    pred_acc[~valid_mask] = 0
    identity = torch.eye(3).view(1, 1, 3, 3).expand_as(pred_ori)
    pred_ori = torch.where(valid_mask.view(seq_len, 7, 1, 1), pred_ori, identity)

    combo = COMBOS[args.combo]
    combo_acc = torch.zeros_like(pred_acc)
    combo_ori = torch.zeros_like(pred_ori)
    combo_acc[:, combo] = pred_acc[:, combo] / amass.acc_scale
    combo_ori[:, combo] = pred_ori[:, combo]
    imu = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)

    pose_p = []
    mocap_model.reset()
    for frame_idx in range(seq_len):
        pose_p.append(mocap_model.forward_frame(imu[frame_idx].to(device)).cpu())
    pose_p = torch.stack(pose_p)

    tran_t = sample["tran_gt"].float().view(-1, 3)[:seq_len].clone()
    tran_t[:, 0].neg_()
    tran_t[:, 2].neg_()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pose_p": pose_p,
            "pose_t": sample["pose_gt"].float().view(-1, 24, 3, 3),
            "tran_t": tran_t,
            "calibrated_acc": pred_acc,
            "calibrated_ori": pred_ori,
        },
        out_path,
    )
    print(f"SAVED={out_path}")


if __name__ == "__main__":
    main()
