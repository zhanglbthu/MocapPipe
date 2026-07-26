import os
import numpy as np
import pickle
import torch
import random
from argparse import ArgumentParser
from tqdm import tqdm
import glob

from articulate.model import ParametricModel
from articulate import math
from config import paths, datasets


# specify target FPS
TARGET_FPS = 30

# left wrist, right wrist, left thigh, right thigh, head, left foot, right foot
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3365, 6765])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 7, 8])
body_model = ParametricModel(paths.smpl_file)


HUAWEI_NEW_DEFAULT_DEVICE_ORDER = [
    "Watch_left",
    "Watch_right",
    "Phone_left",
    "Phone_right",
    "Headset",
    "STag_left",
    "STag_right",
]

def _syn_acc(v, smooth_n=4):
    """Synthesize accelerations from vertex positions."""
    mid = smooth_n // 2
    scale_factor = TARGET_FPS ** 2 

    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * scale_factor for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))

    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * scale_factor / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def process_amass():
    def _foot_ground_probs(joint):
        """Compute foot-ground contact probabilities."""
        dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
        dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
        lfoot_contact = (dist_lfeet < 0.008).int()
        rfoot_contact = (dist_rfeet < 0.008).int()
        lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), lfoot_contact))
        rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), rfoot_contact))
        return torch.stack((lfoot_contact, rfoot_contact), dim=1)

    # enable skipping processed files
    try:
        processed = [fpath.name for fpath in (paths.processed_datasets).iterdir()]
    except FileNotFoundError:
        processed = []

    for ds_name in datasets.amass_datasets:
        # skip processed 
        if f"{ds_name}.pt" in processed:
            continue

        data_pose, data_trans, data_beta, length = [], [], [], []
        print("\rReading", ds_name)

        for npz_fname in tqdm(sorted(glob.glob(os.path.join(paths.raw_amass, ds_name, "*/*_poses.npz")))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate not in [120, 60, 59]:
                continue

            # enable downsampling
            step = max(1, round(framerate / TARGET_FPS))

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

        if len(data_pose) == 0:
            print(f"AMASS dataset, {ds_name} not supported")
            continue

        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))
        pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)

        # include the left and right index fingers in the pose
        pose[:, 23] = pose[:, 37]     # right hand 
        pose = pose[:, :24].clone()   # only use body + right and left fingers

        # align AMASS global frame with DIP
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(math.axis_angle_to_rotation_matrix(pose[:, 0])))

        print("Synthesizing IMU accelerations and orientations")
        b = 0
        out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, out_contact = [], [], [], [], [], [], []
        for i, l in tqdm(list(enumerate(length))):
            if l <= 12: b += l; print("\tdiscard one sequence with length", l); continue
            p = math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)

            out_pose.append(p.clone())  # N, 24, 3, 3
            out_tran.append(tran[b:b + l].clone())  # N, 3
            out_shape.append(shape[i].clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
            out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 7, 3
            out_contact.append(_foot_ground_probs(joint).clone()) # N, 2

            out_vrot.append(grot[:, ji_mask])  # N, 7, 3, 3
            b += l

        print("Saving...")
        data = {
            'joint': out_joint,
            'pose': out_pose,
            'shape': out_shape,
            'tran': out_tran,
            'acc': out_vacc,
            'ori': out_vrot,
            'contact': out_contact
        }
        data_path = paths.processed_datasets / f"{ds_name}.pt"
        torch.save(data, data_path)
        print(f"Synthetic AMASS dataset is saved at: {data_path}")

def process_totalcapture():
    """Preprocess TotalCapture dataset for testing."""

    inches_to_meters = 0.0254
    pos_file = 'gt_skel_gbl_pos.txt'
    ori_file = 'gt_skel_gbl_ori.txt'

    subjects = ['S1', 'S2', 'S3', 'S4', 'S5']

    # Load poses from processed AMASS dataset
    amass_tc = torch.load(os.path.join(paths.processed_datasets, "AMASS", "TotalCapture", "pose.pt"))
    tc_poses = {pose.shape[0]: pose for pose in amass_tc}

    processed, failed_to_process = [], []
    accs, oris, poses, trans = [], [], [], []
    for file in sorted(os.listdir(paths.calibrated_totalcapture)):
        if not file.endswith(".pkl") or ('s5' in file and 'acting3' in file) or not any(file.startswith(s.lower()) for s in subjects):
            continue

        data = pickle.load(open(os.path.join(paths.calibrated_totalcapture, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()
        acc = torch.from_numpy(data['acc']).float()

        # Load pose data from AMASS
        try: 
            name_split = file.split("_")
            subject, activity = name_split[0], name_split[1].split(".")[0]
            pose_npz = np.load(os.path.join(paths.raw_amass, "TotalCapture", subject, f"{activity}_poses.npz"))
            pose = torch.from_numpy(pose_npz['poses']).float().view(-1, 52, 3)
        except:
            failed_to_process.append(f"{subject}_{activity}")
            print(f"Failed to Process: {file}")
            continue

        pose = tc_poses[pose.shape[0]]
    
        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        # convert axis-angle to rotation matrix
        pose = math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3, 3

        processed.append(file)
    
    for subject_name in subjects:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official, subject_name))):
            if (subject_name == 'S5' and motion_name == 'acting3') or motion_name.startswith(".") or (f"{subject_name.lower()}_{motion_name}" in failed_to_process):
                continue   # no SMPL poses

            f = open(os.path.join(paths.raw_totalcapture_official, subject_name, motion_name, pos_file))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])   # N, 3

    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    # remove acceleration bias
    for iacc, pose, tran in zip(accs, poses, trans):
        pose = pose.view(-1, 24, 3, 3)
        _, _, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        vacc = _syn_acc(vert[:, vi_mask])
        for imu_id in range(6):
            for i in range(3):
                d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
                iacc[:, imu_id, i] += d

    data = {
        'acc': accs,
        'ori': oris,
        'pose': poses,
        'tran': trans
    }
    data_path = paths.eval_dir / "totalcapture.pt"
    torch.save(data, data_path)
    print("Preprocessed TotalCapture dataset is saved at:", paths.processed_totalcapture)

def process_dipimu(split="test"):
    """Preprocess DIP for finetuning and evaluation."""
    imu_mask = [7, 8, 9, 10, 0, 2]

    test_split = ['s_09', 's_10']
    train_split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
    subjects = train_split if split == "train" else test_split
     
    # left wrist, right wrist, left thigh, right thigh, head, pelvis
    vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
    ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

    # enable downsampling
    step = max(1, round(60 / TARGET_FPS))

    accs, oris, poses, trans, shapes, joints = [], [], [], [], [], []
    for subject_name in subjects:
        for motion_name in os.listdir(os.path.join(paths.raw_dip, subject_name)):
            try:
                path = os.path.join(paths.raw_dip, subject_name, motion_name)
                print(f"Processing: {subject_name}/{motion_name}")
                data = pickle.load(open(path, 'rb'), encoding='latin1')
                acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
                ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
                pose = torch.from_numpy(data['gt']).float()

                # fill nan with nearest neighbors
                for _ in range(4):
                    acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                    ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                    acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                    ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

                # enable downsampling
                acc = acc[6:-6:step].contiguous()
                ori = ori[6:-6:step].contiguous()
                pose = pose[6:-6:step].contiguous()

                shape = torch.ones((10))
                tran = torch.zeros(pose.shape[0], 3) # dip-imu does not contain translations
                if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                    accs.append(acc.clone())
                    oris.append(ori.clone())
                    trans.append(tran.clone())  
                    shapes.append(shape.clone()) # default shape
                    
                    # forward kinematics to get the joint position
                    p = math.axis_angle_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
                    grot, joint, vert = body_model.forward_kinematics(p, shape, tran, calc_mesh=True)
                    poses.append(p.clone())
                    joints.append(joint)
                else:
                    print(f"DIP-IMU: {subject_name}/{motion_name} has too much nan! Discard!")
            except Exception as e:
                print(f"Error processing the file: {path}.", e)


    print("Saving...")
    data = {
        'joint': joints,
        'pose': poses,
        'shape': shapes,
        'tran': trans,
        'acc': accs,
        'ori': oris,
    }
    data_path = paths.eval_dir / f"dip_{split}.pt"
    torch.save(data, data_path)
    print(f"Preprocessed DIP-IMU dataset is saved at: {data_path}")

def process_imuposer(split: str="train"):
    """Preprocess the IMUPoser dataset"""

    train_split = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
    test_split = ['P9', 'P10']
    subjects = train_split if split == "train" else test_split

    accs, oris, poses, trans = [], [], [], []
    for pid_path in sorted(paths.raw_imuposer.iterdir()):
        if pid_path.name not in subjects:
            continue

        print(f"Processing: {pid_path.name}")
        for fpath in sorted(pid_path.iterdir()):
            with open(fpath, "rb") as f: 
                fdata = pickle.load(f)
                
                acc = fdata['imu'][:, :5*3].view(-1, 5, 3)
                ori = fdata['imu'][:, 5*3:].view(-1, 5, 3, 3)
                pose = math.axis_angle_to_rotation_matrix(fdata['pose']).view(-1, 24, 3, 3)
                tran = fdata['trans'].to(torch.float32)
                
                 # align IMUPoser global fame with DIP
                rot = torch.tensor([[[-1, 0, 0], [0, 0, 1], [0, 1, 0.]]])
                pose[:, 0] = rot.matmul(pose[:, 0])
                tran = tran.matmul(rot.squeeze())

                # ensure sizes are consistent
                assert tran.shape[0] == pose.shape[0]

                accs.append(acc)    # N, 5, 3
                oris.append(ori)    # N, 5, 3, 3
                poses.append(pose)  # N, 24, 3, 3
                trans.append(tran)  # N, 3

    print(f"# Data Processed: {len(accs)}")
    data = {
        'acc': accs,
        'ori': oris,
        'pose': poses,
        'tran': trans
    }
    data_path = paths.eval_dir / f"imuposer_{split}.pt"
    torch.save(data, data_path)

def get_sorted_files(data_dir):
    # 获取所有以.pt结尾的文件
    pt_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    
    # 按照文件名前的数字部分排序
    pt_files.sort(key=lambda f: int(f.split('.')[0]))  # 通过文件名前的数字进行排序
    
    return pt_files

def process_huawei(split: str="train"):
    """Preprocess the IMUPoser dataset"""

    train_split = ['hyq_0402', 'yl_0403', 'lisha_0407', 'huohuo_0407', 'yinqi_0408', 'yanyu_0408', 'liran_0408']
    test_split = ['xinrui_0407']
    subjects = train_split if split == "train" else test_split

    accs, oris, poses, trans = [], [], [], []
    
    length = 2 * 60 * 30  # 2 minutes of data at 30 FPS
    for pid_path in sorted(paths.raw_huawei.iterdir()):
        if pid_path.name not in subjects:
            continue

        print(f"Processing: {pid_path.name}")
        for fpath in sorted(pid_path.glob("*.pt"), key=lambda x: int(x.stem)):
            print(f"\tProcessing: {fpath.name}")

            fdata = torch.load(fpath)
            
            acc  = fdata['aM'][:, :7].view(-1, 7, 3)
            ori  = fdata['RMB'][:, :7].view(-1, 7, 3, 3)
            pose = fdata['pose_gt'].view(-1, 24, 3, 3)
            tran = fdata['tran_gt'].view(-1, 3)

            # crop the last few frames which contain too much noise
            acc = acc[:length]
            ori = ori[:length]
            pose = pose[:length]
            tran = tran[:length]
            
            # ensure sizes are consistent
            assert tran.shape[0] == pose.shape[0]

            accs.append(acc)    # N, 7, 3
            oris.append(ori)    # N, 7, 3, 3
            poses.append(pose)  # N, 24, 3, 3
            trans.append(tran)  # N, 3

    print(f"# Data Processed: {len(accs)}")
    data = {
        'acc': accs,
        'ori': oris,
        'pose': poses,
        'tran': trans
    }
    data_path = paths.eval_dir / f"huawei_{split}.pt"
    torch.save(data, data_path)


def _split_items(items, train_ratio: float = 0.8, seed: int = 1234):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    train_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_ratio))))
    return {"train": shuffled[:train_count], "test": shuffled[train_count:]}


def _split_items_by_subject(items, train_ratio: float = 0.8, seed: int = 1234):
    """Split complete recording subjects, never windows or sequences."""
    subjects = sorted({item["source_subject"] for item in items})
    if len(subjects) < 2:
        raise ValueError("At least two Huawei subjects are required for train/validation splitting.")
    subject_split = _split_items(subjects, train_ratio=train_ratio, seed=seed)
    train_subjects = set(subject_split["train"])
    return {
        "train": [item for item in items if item["source_subject"] in train_subjects],
        "val": [item for item in items if item["source_subject"] not in train_subjects],
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(set(subjects) - train_subjects),
    }


def _pad_devices(tensor: torch.Tensor, target_devices: int, trailing_shape):
    if tensor.shape[1] == target_devices:
        return tensor
    pad_shape = (tensor.shape[0], target_devices - tensor.shape[1], *trailing_shape)
    return torch.cat([tensor, torch.zeros(*pad_shape, dtype=tensor.dtype)], dim=1)


def _valid_device_mask(sample: dict, seq_len: int, num_devices: int, observed_device_count: int):
    valid_mask = torch.zeros(seq_len, num_devices, dtype=torch.bool)
    valid_mask[:, :observed_device_count] = True
    if "synthetic_device_indices" in sample:
        synthetic = [int(i) for i in sample["synthetic_device_indices"]]
        valid_mask[:, synthetic] = False
    return valid_mask


def _collect_huawei_new_calibrator_items():
    items = []
    num_devices = len(HUAWEI_NEW_DEFAULT_DEVICE_ORDER)
    for pid_path in sorted(paths.raw_huawei_new.iterdir()):
        if not pid_path.is_dir():
            continue
        for fpath in sorted(pid_path.glob("*.pt"), key=lambda x: int(x.stem)):
            print(f"Processing calibrator source: {fpath.parent.name}/{fpath.name}")
            fdata = torch.load(fpath)

            input_acc = fdata["aM"].float().view(-1, fdata["aM"].shape[1], 3)
            input_ori = fdata["RMB"].float().view(-1, fdata["RMB"].shape[1], 3, 3)
            pose = fdata["pose_gt"].float().view(-1, 24, 3, 3)
            tran = fdata["tran_gt"].float().view(-1, 3)
            tran[:, 0].neg_()
            tran[:, 2].neg_()

            seq_len = min(input_acc.shape[0], input_ori.shape[0], pose.shape[0], tran.shape[0])
            input_acc = input_acc[:seq_len]
            input_ori = input_ori[:seq_len]
            pose = pose[:seq_len]
            tran = tran[:seq_len]

            grot, _, vert = body_model.forward_kinematics(pose=pose, tran=tran, calc_mesh=True)
            target_acc = _syn_acc(vert[:, vi_mask]).float()
            target_ori = grot[:, ji_mask].float()

            input_acc = _pad_devices(input_acc, num_devices, (3,))
            input_ori = _pad_devices(input_ori, num_devices, (3, 3))
            valid_mask = _valid_device_mask(fdata, seq_len, num_devices, fdata["aM"].shape[1])

            items.append(
                {
                    "input_acc": input_acc,
                    "input_ori": input_ori,
                    "target_acc": target_acc,
                    "target_ori": target_ori,
                    "valid_mask": valid_mask,
                    "pose": pose,
                    "tran": tran,
                    "device_names": fdata.get("device_order", HUAWEI_NEW_DEFAULT_DEVICE_ORDER),
                    "source_file": f"{fpath.parent.name}/{fpath.name}",
                    "source_domain": "huawei_new",
                    "source_subject": fpath.parent.name,
                }
            )
    return items


def _collect_imuposer_calibrator_items(split: str = "train"):
    items = []
    num_devices = len(HUAWEI_NEW_DEFAULT_DEVICE_ORDER)
    dataset_name = datasets.imuposer_train if split == "train" else datasets.imuposer_test
    fdata = torch.load(paths.eval_dir / dataset_name)
    for seq_idx, (input_acc, input_ori, pose, tran) in enumerate(
        zip(fdata["acc"], fdata["ori"], fdata["pose"], fdata["tran"]),
        start=1,
    ):
        print(f"Processing calibrator source: imuposer_{split}/{seq_idx}")
        input_acc = input_acc.float().view(-1, input_acc.shape[1], 3)
        input_ori = input_ori.float().view(-1, input_ori.shape[1], 3, 3)
        pose = pose.float().view(-1, 24, 3, 3)
        tran = tran.float().view(-1, 3)

        seq_len = min(input_acc.shape[0], input_ori.shape[0], pose.shape[0], tran.shape[0])
        input_acc = input_acc[:seq_len]
        input_ori = input_ori[:seq_len]
        pose = pose[:seq_len]
        tran = tran[:seq_len]

        grot, _, vert = body_model.forward_kinematics(pose=pose, tran=tran, calc_mesh=True)
        target_acc = _syn_acc(vert[:, vi_mask]).float()
        target_ori = grot[:, ji_mask].float()

        observed_count = input_acc.shape[1]
        input_acc = _pad_devices(input_acc, num_devices, (3,))
        input_ori = _pad_devices(input_ori, num_devices, (3, 3))
        valid_mask = torch.zeros(seq_len, num_devices, dtype=torch.bool)
        valid_mask[:, :observed_count] = True

        items.append(
            {
                "input_acc": input_acc,
                "input_ori": input_ori,
                "target_acc": target_acc,
                "target_ori": target_ori,
                "valid_mask": valid_mask,
                "pose": pose,
                "tran": tran,
                "device_names": HUAWEI_NEW_DEFAULT_DEVICE_ORDER,
                "source_file": f"imuposer_{split}/{seq_idx}",
                "source_domain": f"imuposer_{split}",
                "source_subject": f"imuposer_{split}",
            }
        )
    return items


def process_huawei_new_calibrator(split: str = "train", train_ratio: float = 0.8, seed: int = 1234):
    """Prepare paired real-to-synthetic IMU calibration data.

    Huawei subjects are split between train and validation.  IMUPoser train is
    used only for training; IMUPoser test remains untouched until final test.
    """
    huawei_split = _split_items_by_subject(
        _collect_huawei_new_calibrator_items(),
        train_ratio=train_ratio,
        seed=seed,
    )
    if split == "train":
        selected_items = huawei_split["train"] + _collect_imuposer_calibrator_items(split="train")
        split_meta = {
            "train_sources": ["huawei_new", "imuposer_train"],
            "validation_sources": ["huawei_new"],
            "test_sources": ["imuposer_test"],
            "subjects": huawei_split["train_subjects"],
            "seed": seed,
            "train_ratio": train_ratio,
        }
    elif split == "val":
        selected_items = huawei_split["val"]
        split_meta = {
            "train_sources": ["huawei_new", "imuposer_train"],
            "validation_sources": ["huawei_new"],
            "test_sources": ["imuposer_test"],
            "subjects": huawei_split["val_subjects"],
            "seed": seed,
            "train_ratio": train_ratio,
        }
    elif split == "test":
        selected_items = _collect_imuposer_calibrator_items(split="test")
        split_meta = {
            "train_sources": ["huawei_new", "imuposer_train"],
            "validation_sources": ["huawei_new"],
            "test_sources": ["imuposer_test"],
            "subjects": ["imuposer_test"],
            "seed": seed,
            "train_ratio": train_ratio,
        }
    else:
        raise ValueError(f"Unsupported split: {split}")

    input_accs, input_oris = [], []
    target_accs, target_oris, valid_masks = [], [], []
    poses, trans, device_names, source_files, source_domains, source_subjects = [], [], [], [], [], []

    for item in selected_items:
        input_accs.append(item["input_acc"])
        input_oris.append(item["input_ori"])
        target_accs.append(item["target_acc"])
        target_oris.append(item["target_ori"])
        valid_masks.append(item["valid_mask"])
        poses.append(item["pose"])
        trans.append(item["tran"])
        device_names.append(item["device_names"])
        source_files.append(item["source_file"])
        source_domains.append(item["source_domain"])
        source_subjects.append(item["source_subject"])

    print(f"# Calibrator sequences processed: {len(input_accs)}")
    data = {
        "input_acc": input_accs,
        "input_ori": input_oris,
        "target_acc": target_accs,
        "target_ori": target_oris,
        "valid_mask": valid_masks,
        "pose": poses,
        "tran": trans,
        "device_names": device_names,
        "source_files": source_files,
        "source_domains": source_domains,
        "source_subjects": source_subjects,
        "split_meta": {
            **split_meta,
            "num_sequences": len(selected_items),
            "num_huawei_sequences": sum(1 for x in source_domains if x == "huawei_new"),
            "num_imuposer_train_sequences": sum(1 for x in source_domains if x == "imuposer_train"),
            "num_imuposer_test_sequences": sum(1 for x in source_domains if x == "imuposer_test"),
            "unique_subjects": sorted(set(source_subjects)),
        },
    }
    data_path = paths.eval_dir / f"huawei_new_calibrator_{split}.pt"
    torch.save(data, data_path)
    print(f"Huawei_new calibrator dataset saved at: {data_path}")

def create_directories():
    paths.processed_datasets.mkdir(exist_ok=True, parents=True)
    paths.eval_dir.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="amass")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=1234)
    args = parser.parse_args()

    # create dataset directories
    create_directories()

    # process datasets
    if args.dataset == "amass":
        process_amass()
    elif args.dataset == "totalcapture":
        process_totalcapture()
    elif args.dataset == "imuposer":
        process_imuposer(split="train")
        process_imuposer(split="test")
    elif args.dataset == "dip":
        process_dipimu(split="train")
        process_dipimu(split="test")
    elif args.dataset == "huawei":
        process_huawei(split="train")
        process_huawei(split="test")
    elif args.dataset == "huawei_new_calibrator":
        process_huawei_new_calibrator(split="train", train_ratio=args.train_ratio, seed=args.split_seed)
        process_huawei_new_calibrator(split="val", train_ratio=args.train_ratio, seed=args.split_seed)
        process_huawei_new_calibrator(split="test", train_ratio=args.train_ratio, seed=args.split_seed)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
