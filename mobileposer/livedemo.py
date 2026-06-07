import torch
from pygame.time import Clock

import articulate as art
import os
import copy
from config import *
from articulate.utils.unity import MotionViewer

from utils.model_utils import load_model
import numpy as np
import matplotlib
from argparse import ArgumentParser
import keyboard
from sensor_huawei.sensor import CalibratedHuaweiSensor
import traceback
import datetime
from models.imu_calibrator import ComboTemporalIMUCalibrator, build_imu_input

colors = matplotlib.colormaps['tab10'].colors
body_model = art.ParametricModel(paths.smpl_file, device='cuda')
CALIBRATOR_COMBO = [0, 3, 4]


def load_combo_calibrator(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get("args", {})
    model = ComboTemporalIMUCalibrator(
        combo_size=len(CALIBRATOR_COMBO),
        predict_acc=args.get("predict_acc", False),
        hidden_dim=args.get("hidden_dim", 128),
        dropout=args.get("dropout", 0.1),
        num_layers=args.get("num_layers", 3),
        nhead=args.get("nhead", 4),
        max_seq_len=args.get("window_size", 125),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.reset()
    return model


def make_mocap_input(acc: torch.Tensor, ori: torch.Tensor):
    ori = ori[:model_config.n_joints].view(model_config.n_joints, 3, 3)
    acc = acc[:model_config.n_joints].view(model_config.n_joints, 3)
    acc = acc / amass.acc_scale
    return torch.cat([acc.flatten(), ori.flatten()], dim=0).to(model_config.device)


def apply_combo_calibrator(
    calibrator: ComboTemporalIMUCalibrator,
    acc: torch.Tensor,
    ori: torch.Tensor,
):
    calibrated_ori = ori.clone()
    calibrator_input = build_imu_input(
        acc.unsqueeze(0),
        ori.unsqueeze(0),
    )[0, CALIBRATOR_COMBO]
    _, pred_ori_combo = calibrator.forward_frame_windowed(calibrator_input)
    calibrated_ori[CALIBRATOR_COMBO] = pred_ori_combo
    return calibrated_ori


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--sub', type=str, default='chaoran_0529')
    parser.add_argument('--mocap', action='store_true', help='use mocap')
    parser.add_argument(
        '--calibrator',
        type=str,
        default='data/checkpoints/combo_imu_calibrator_lw_rp_h_ori_only_jerk_nopose_fulltrain_tb/best.pt',
        help='combo calibrator checkpoint for lw_rp_h',
    )
    parser.add_argument(
        '--compare-calibrator',
        action='store_true',
        help='show live poses with and without the combo calibrator side by side',
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    clock = Clock()

    if args.mocap:
        ckpt_path = "data/checkpoints/base_model_12combo.pth"
        net = load_model(ckpt_path)
        net.eval()
        raw_net = copy.deepcopy(net).eval() if args.compare_calibrator else None
        calibrator = load_combo_calibrator(args.calibrator, device)
        print('Mobileposer model loaded.')

    sensor = CalibratedHuaweiSensor(HuaweiDevices.device_ids)
    sensor.calibrate("walking_6dof")

    raw_accs, accs, oris, gyros, mags, pressures, ppgs, poses = [], [], [], [], [], [], [], []
    raw_poses = []
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ids = sensor.ids

    print(f"Using device IDs: {ids}")

    idx = 0
    I, z = torch.eye(3).to(device), torch.zeros(3).to(device)
    viewer_count = 2 if args.mocap and args.compare_calibrator else 1
    viewer_names = ['NoCalibrator', 'Calibrator'] if viewer_count == 2 else ['LiveDemo']
    with torch.no_grad(), MotionViewer(viewer_count, overlap=False, names=viewer_names) as viewer:
        while True:
            try:
                clock.tick(30)
                ori = torch.zeros(7, 3, 3).to(device)
                a = torch.zeros(7, 3).to(device)

                # device readings
                t, aS, aI, aM, RIS, RMB, gyro, mag, pressure, ppg = sensor.get()

                ori[ids] = RMB.to(device)
                a[ids] = aM.to(device)

                oris.append(ori.clone())
                accs.append(a.clone())
                raw_accs.append(aS)

                gyros.append(gyro)
                mags.append(mag)
                pressures.append(pressure)
                ppgs.append(ppg)

                if args.mocap:
                    calibrated_ori = apply_combo_calibrator(calibrator, a, ori)
                    calibrated_input = make_mocap_input(a, calibrated_ori)
                    pose = net.forward_frame(calibrated_input)
                    poses.append(pose)

                    zero_tran = np.array([0, 0, 0])
                    if args.compare_calibrator:
                        raw_input = make_mocap_input(a, ori)
                        raw_pose = raw_net.forward_frame(raw_input)
                        raw_poses.append(raw_pose)
                        viewer.update_all(
                            [raw_pose.cpu().numpy(), pose.cpu().numpy()],
                            [zero_tran, zero_tran],
                            render=False,
                        )
                    else:
                        viewer.update_all([pose.cpu().numpy()], [zero_tran], render=False)
                    viewer.render()

                idx += 1

                print('\r', clock.get_fps(), end='')

                if keyboard.is_pressed('q'):
                    break
            except Exception as e:
                print(f"Error occurred: {e}")
                print(traceback.format_exc())
                os._exit(0)
            except KeyboardInterrupt:
                print("Exiting...")
                os._exit(0)

    accs = torch.stack(accs)
    oris = torch.stack(oris)
    poses = torch.stack(poses) if poses else torch.empty(0)
    raw_poses = torch.stack(raw_poses) if raw_poses else torch.empty(0)
    raw_accs = torch.tensor(np.array(raw_accs))
    gyros = torch.tensor(np.array(gyros))
    mags = torch.tensor(np.array(mags))
    pressures = torch.tensor(np.array(pressures))
    ppgs = torch.tensor(np.array(ppgs))
    RMI, RSB, acc_bias = sensor.get_cali_matrices()

    print(f"raw_accs: {raw_accs.shape}, accs: {accs.shape}, oris: {oris.shape}, poses: {poses.shape}")
    if args.compare_calibrator:
        print(f"raw poses: {raw_poses.shape}")
    print(f"gyros: {gyros.shape}, mags: {mags.shape}, pressures: {pressures.shape}, ppgs: {ppgs.shape}")
    print(f"RMI: {RMI.shape}, RSB: {RSB.shape}, acc_bias: {acc_bias.shape}")
    print('Frames: %d' % accs.shape[0])

    data_name = f"{args.name}_{timestamp}.pt"
    sub_name = args.sub

    save_dir = os.path.join(paths.record_dir, sub_name)
    os.makedirs(save_dir, exist_ok=True)
    record = {
        'raw_acc': raw_accs,
        'acc': accs,
        'ori': oris,
        'gyro': gyros,
        'mag': mags,
        'pressure': pressures,
        'ppg': ppgs,
        'pose': poses,
        'RMI': RMI,
        'RSB': RSB,
        'acc_bias': acc_bias,
    }
    if args.compare_calibrator:
        record['pose_calibrated'] = poses
        record['pose_raw'] = raw_poses
    torch.save(record, os.path.join(save_dir, data_name))

    print('\rFinish.')
    os._exit(0)
