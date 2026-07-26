import os

import torch
from pathlib import Path
from enum import Enum, auto


class train_hypers:
    """Hyperparameters for training."""
    batch_size = 256
    num_workers = 8
    num_epochs = 60
    accelerator = "gpu"
    device = 0
    lr = 1e-3


class finetune_hypers:
    """Hyperparamters for finetuning."""
    batch_size = 32
    num_workers = 8
    num_epochs = 15
    accelerator = "gpu"
    device = 0
    lr = 5e-5


class paths:
    """Project paths.

    Paths are resolved from this file instead of the current working
    directory.  Dataset locations can be overridden without editing source
    code; see ``configs/paths.env.example``.
    """

    package_dir = Path(__file__).resolve().parent
    root_dir = package_dir.parent
    data_dir = Path(os.getenv("MOBILEPOSER_DATA_DIR", package_dir / "data")).expanduser().resolve()

    _legacy_dataset_root = Path("/root/autodl-tmp/dataset")
    _default_dataset_root = _legacy_dataset_root if _legacy_dataset_root.exists() else data_dir / "datasets"
    dataset_root = Path(os.getenv("MOBILEPOSER_DATASET_ROOT", _default_dataset_root)).expanduser().resolve()

    checkpoint = data_dir / "checkpoints"
    experiments_dir = data_dir / "experiments"
    paper_dir = data_dir / "paper"
    eval_output_dir = data_dir / "eval"
    video_output_dir = data_dir / "video"
    record_dir = data_dir / "records"
    dev_data = data_dir / "device_data"
    smpl_file = Path(os.getenv("MOBILEPOSER_SMPL_FILE", package_dir / "smpl/basicmodel_m.pkl")).expanduser().resolve()
    weights_file = checkpoint / "weights.pth"
    raw_dir = dataset_root / "raw"
    raw_amass = raw_dir / "AMASS"
    raw_dip = raw_dir / "DIP_IMU"
    raw_imuposer = raw_dir / "IMUPoser"
    raw_huawei = raw_dir / "Huawei"
    raw_huawei_new = raw_dir / "Huawei_new"
    raw_totalcapture_official = raw_dir / "TotalCapture" / "raw"
    calibrated_totalcapture = raw_dir / "TotalCapture" / "IMU"
    processed_datasets = dataset_root / "processed"
    eval_dir = processed_datasets / "eval"
    processed_totalcapture = eval_dir / "totalcapture.pt"

class model_config:
    """MobilePoser Model configurations."""
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # joint set
    n_joints = 7                        # active IMU locations used by the base 12-combo model
    n_imu = 12*n_joints                 # 84 (3 accel. axes + 3x3 orientation rotation matrix) * 7 IMU locations
    n_output_joints = 24                # 24 output joints
    n_pose_output = n_output_joints*6   # 144 pose output (24 output joints * 6D rotation matrix)

    # model config
    past_frames = 40
    future_frames = 5
    total_frames = past_frames + future_frames


class amass:
    """AMASS dataset information."""
    # device-location combinationsa
    combos_full = {
        # # # leaf device
        # 'lw_rw_lp_rp_h_feet': [0, 1, 2, 3, 4, 5, 6],
        # 'lw_rw_lp_rp_h': [0, 1, 2, 3, 4],
        # 'lw_rw_lp_rp': [0, 1, 2, 3],
        # 'lw_rw_h_feet': [0, 1, 4, 5, 6],
        # 'lw_rw_feet': [0, 1, 5, 6],
        
        # # five devices
        # 'lw_rp_h_feet': [0, 3, 4, 5, 6],
        # 'rw_rp_h_feet': [1, 3, 4, 5, 6],
        # 'lw_lp_h_feet': [0, 2, 4, 5, 6],
        # 'rw_lp_h_feet': [1, 2, 4, 5, 6],
        # # three devices
        'lw_rp_h': [0, 3, 4],
        # 'rw_rp_h': [1, 3, 4],
        # 'lw_lp_h': [0, 2, 4],
        # 'rw_lp_h': [1, 2, 4],
        # # two devices
        # 'lw_rp': [0, 3],
        # 'rw_rp': [1, 3],
        # 'lw_lp': [0, 2],
        # 'rw_lp': [1, 2],
        # # one device
        # 'lw': [0],
     }
    
    acc_scale = 30
    vel_scale = 2

    # left wrist, right wrist, left thigh, right thigh, head, pelvis
    all_imu_ids = [0, 1, 2, 3, 4, 5, 6]
    imu_ids = [0, 1, 2, 3]

    pred_joints_set = [*range(24)]
    joint_sets = [18, 19, 1, 2, 15, 0]
    ignored_joints = list(set(pred_joints_set) - set(joint_sets))


class datasets:
    """Dataset information."""
    # FPS of data
    fps = 30

    # DIP dataset
    dip_test = "dip_test.pt"
    dip_train = "dip_train.pt"

    # TotalCapture dataset
    totalcapture = "totalcapture.pt"

    # IMUPoser dataset
    imuposer = "imuposer.pt"
    imuposer_train = "imuposer_train.pt"
    imuposer_test = "imuposer_test.pt"
    
    # Huawei dataset
    huawei_train = "huawei_train.pt"
    huawei_test = "huawei_test.pt"
    huawei_new_calibrator_train = "huawei_new_calibrator_train.pt"
    huawei_new_calibrator_val = "huawei_new_calibrator_val.pt"
    huawei_new_calibrator_test = "huawei_new_calibrator_test.pt"

    # Test datasets
    test_datasets = {
        'dip': dip_test,
        'totalcapture': totalcapture,
        'imuposer': imuposer_test,
        'huawei': huawei_test,
    }

    # Finetune datasets
    finetune_datasets = {
        'dip': dip_train,
        'imuposer': imuposer_train,
        'huawei': huawei_train,
    }

    # AMASS datasets (add more as they become available in AMASS!)
    amass_datasets = ['ACCAD', 'BioMotionLab_NTroje', 'BMLhandball', 'BMLmovi', 'CMU', 
                      'DanceDB', 'DFaust_67', 'EKUT', 'Eyes_Japan_Dataset', 'HUMAN4D',
                      'HumanEva', 'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU',
                      'SSM_synced', 'TCD_handMocap', 'TotalCapture', 'Transitions_mocap']

    # Root-relative joint positions
    root_relative = False

    # Window length of IMU and Pose data 
    window_length = 125


class joint_set:
    """Joint sets configurations."""
    gravity_velocity = -0.018

    full = list(range(0, 24))
    reduced = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    n_full = len(full)
    n_ignored = len(ignored)
    n_reduced = len(reduced)

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]


class sensor: 
    """Sensor parameters."""
    device_ids = {
        'Left_phone': 0,
        'Left_watch': 1,
        'Left_headphone': 2,
        'Right_phone': 3,
        'Right_watch': 4
    }


class Devices(Enum):
    """Device IDs."""
    Left_Phone = auto()
    Left_Watch = auto()
    Right_Headphone = auto()
    Right_Phone = auto()
    Right_Watch = auto()

class HuaweiDevices:
    device_ids = {
        "Left_Watch": 0,
        # "Right_Watch": 1,
        # "Left_Phone": 2,
        "Right_Phone": 3,
        "Head": 4,
        # "Left_STag": 5,
        # "Right_STag": 6,
    }
    time_offsets = [0, 0, 0, 0, 0, 0, 0]
    BUFFER_SIZE = 50
