import torch
from pygame.time import Clock

import articulate as art
import os
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
    args = parser.parse_args()
    
    device = torch.device("cuda")
    clock = Clock()
    
    if args.mocap:
        ckpt_path = "data/checkpoints/base_model_12combo.pth"
        net = load_model(ckpt_path)
        net.eval()
        calibrator = load_combo_calibrator(args.calibrator, device)
        print('Mobileposer model loaded.')

    sensor = CalibratedHuaweiSensor(HuaweiDevices.device_ids)
    sensor.calibrate("walking_6dof")

    raw_accs, accs, oris, gyros, mags, pressures, ppgs, poses  = [], [], [], [], [], [], [], []
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ids = sensor.ids
    
    print(f"Using device IDs: {ids}")
    
    idx = 0
    I, z = torch.eye(3).to(device), torch.zeros(3).to(device)
    with torch.no_grad(), MotionViewer(1, overlap=False, names=['LiveDemo']) as viewer:
        while True:
            try:
                clock.tick(30)
                ori = torch.zeros(7, 3, 3).to(device)
                a   = torch.zeros(7, 3).to(device)
                    
                # device readings
                t, aS, aI, aM, RIS, RMB, gyro, mag, pressure, ppg = sensor.get()
                
                ori[ids] = RMB.to(device)
                a[ids]   = aM.to(device)
                
                
                oris.append(ori.clone())
                accs.append(a.clone())
                raw_accs.append(aS)
                
                gyros.append(gyro)
                mags.append(mag)
                pressures.append(pressure)
                ppgs.append(ppg)

                if args.mocap:
                    calibrator_input = build_imu_input(
                        a.unsqueeze(0),
                        ori.unsqueeze(0),
                    )[0, CALIBRATOR_COMBO]
                    _, pred_ori_combo = calibrator.forward_frame_windowed(calibrator_input)
                    ori[CALIBRATOR_COMBO] = pred_ori_combo
                    
                    ori = ori[:model_config.n_joints].view(model_config.n_joints, 3, 3)
                    a = a[:model_config.n_joints].view(model_config.n_joints, 3)

                    a = a / amass.acc_scale
                    
                    input = torch.cat([a.flatten(), ori.flatten()], dim=0).to("cuda")

                    pose = net.forward_frame(input)

                    poses.append(pose)
                    
                    pose = pose.cpu().numpy()      
                    
                    zero_tran = np.array([0, 0, 0])  
                    viewer.update_all([pose], [zero_tran], render=False)
                    viewer.render()
                
                idx += 1
                
                print('\r', clock.get_fps(), end='')
                
                if keyboard.is_pressed('q'):
                    break
            except Exception as e:
                print(f"Error occurred: {e}")
                print(traceback.format_exc())  # 打印完整的异常追踪信息
                os._exit(0)
            except KeyboardInterrupt:
                print("Exiting...")
                os._exit(0)
    
    accs = torch.stack(accs)
    oris = torch.stack(oris)
    poses = torch.stack(poses)
    raw_accs = torch.tensor(np.array(raw_accs))
    gyros = torch.tensor(np.array(gyros))
    mags = torch.tensor(np.array(mags))
    pressures = torch.tensor(np.array(pressures))
    ppgs = torch.tensor(np.array(ppgs))
    RMI, RSB, acc_bias = sensor.get_cali_matrices()

    print(f"raw_accs: {raw_accs.shape}, accs: {accs.shape}, oris: {oris.shape}, poses: {poses.shape}")
    print(f"gyros: {gyros.shape}, mags: {mags.shape}, pressures: {pressures.shape}, ppgs: {ppgs.shape}")
    print(f"RMI: {RMI.shape}, RSB: {RSB.shape}, acc_bias: {acc_bias.shape}")
    print('Frames: %d' % accs.shape[0])
    
    data_name = f"{args.name}_{timestamp}.pt"
    sub_name = args.sub
    
    save_dir = os.path.join(paths.record_dir, sub_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'raw_acc': raw_accs,
                'acc': accs,
                'ori': oris,
                'gyro': gyros,
                'mag': mags,
                'pressure': pressures,
                'ppg': ppgs,
                'pose': poses,
                'RMI': RMI,
                'RSB': RSB,
                'acc_bias': acc_bias
                }, os.path.join(save_dir, data_name))

    print('\rFinish.')
    os._exit(0)
        
