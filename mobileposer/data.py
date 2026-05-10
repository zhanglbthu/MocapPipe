import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from typing import List
import random
import lightning as L
from tqdm import tqdm 

import articulate as art
from config import *
from utils import *
from helpers import *


class PoseDataset(Dataset):
    def __init__(self, fold: str='train', evaluate: str=None, finetune: str=None):
        super().__init__()
        self.fold = fold
        self.evaluate = evaluate
        self.finetune = finetune
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)
        self.combos = list(amass.combos_full.items())
        self.data = self._prepare_dataset()

    def _get_data_files(self, data_folder):
        if self.fold == 'train':
            return self._get_train_files(data_folder)
        elif self.fold == 'test':
            return self._get_test_files()
        else:
            raise ValueError(f"Unknown data fold: {self.fold}.")

    def _get_train_files(self, data_folder):
        if self.finetune:
            return [datasets.finetune_datasets[self.finetune]]
        else:
            return [x.name for x in data_folder.iterdir() if not x.is_dir()]

    def _get_test_files(self):
        return [datasets.test_datasets[self.evaluate]]

    def _prepare_dataset(self):
        data_folder = paths.processed_datasets / ('eval' if (self.finetune or self.evaluate) else '')
        data_files = self._get_data_files(data_folder)
        data = {key: [] for key in ['imu_inputs', 'pose_outputs', 'joint_outputs', 'tran_outputs', 'vel_outputs', 'foot_outputs']}
        
        for data_file in tqdm(data_files):
            file_data = torch.load(data_folder / data_file)
            self._process_file_data(file_data, data)
        return data

    def _process_file_data(self, file_data, data):
        accs, oris, poses, trans = file_data['acc'], file_data['ori'], file_data['pose'], file_data['tran']
        joints = file_data.get('joint', [None] * len(poses))
        foots = file_data.get('contact', [None] * len(poses))

        for acc, ori, pose, tran, joint, foot in zip(accs, oris, poses, trans, joints, foots):
            # if acc.shape[1] < 7; concat zeros to make it 7 (for 7 IMUs), and do the same for orientation
            # change acc shape from [N, 5, 3] to [N, 7, 3], and ori shape from [N, 5, 3, 3] to [N, 7, 3, 3]
            if acc.shape[1] < 7:
                acc = torch.cat([acc, torch.zeros(acc.shape[0], 7 - acc.shape[1], 3)], dim=1)
                ori = torch.cat([ori, torch.zeros(ori.shape[0], 7 - ori.shape[1], 3, 3)], dim=1)
            
            acc, ori = acc[:, :7]/amass.acc_scale, ori[:, :7] # change: select 7 IMUs
            pose_global, joint = self.bodymodel.forward_kinematics(pose=pose.view(-1, 216)) # convert local rotation to global
            pose = pose if self.evaluate else pose_global.view(-1, 24, 3, 3)                # use global only for training
            joint = joint.view(-1, 24, 3)
            self._process_combo_data(acc, ori, pose, joint, tran, foot, data)

    def _process_combo_data(self, acc, ori, pose, joint, tran, foot, data):
        for _, c in self.combos:
            # mask out layers for different subsets
            combo_acc = torch.zeros_like(acc)
            combo_ori = torch.zeros_like(ori)
            combo_acc[:, c] = acc[:, c]
            combo_ori[:, c] = ori[:, c]
            imu_input = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1) # [[N, 15], [N, 45]] => [N, 60] 

            data_len = len(imu_input) if self.evaluate else datasets.window_length
            
            for key, value in zip(['imu_inputs', 'pose_outputs', 'joint_outputs', 'tran_outputs'],
                                [imu_input, pose, joint, tran]):
                data[key].extend(torch.split(value, data_len))

            if not (self.evaluate or self.finetune): # do not finetune translation module
                self._process_translation_data(joint, tran, foot, data_len, data)

    def _process_translation_data(self, joint, tran, foot, data_len, data):
        root_vel = torch.cat((torch.zeros(1, 3), tran[1:] - tran[:-1]))
        vel = torch.cat((torch.zeros(1, 24, 3), torch.diff(joint, dim=0)))
        vel[:, 0] = root_vel
        data['vel_outputs'].extend(torch.split(vel * (datasets.fps / amass.vel_scale), data_len))
        data['foot_outputs'].extend(torch.split(foot, data_len))

    def __getitem__(self, idx):
        imu = self.data['imu_inputs'][idx].float()
        joint = self.data['joint_outputs'][idx].float()
        tran = self.data['tran_outputs'][idx].float()
        num_pred_joints = len(amass.pred_joints_set)
        pose = art.math.rotation_matrix_to_r6d(self.data['pose_outputs'][idx]).reshape(-1, num_pred_joints, 6)[:, amass.pred_joints_set].reshape(-1, 6*num_pred_joints)

        if self.evaluate or self.finetune:
            return imu, pose, joint, tran

        vel = self.data['vel_outputs'][idx].float()
        contact = self.data['foot_outputs'][idx].float()

        return imu, pose, joint, tran, vel, contact

    def __len__(self):
        return len(self.data['imu_inputs'])


class DiffusionPoseDataset(Dataset):
    """Dataset for DiffusionPoser-style sequence generation.

    Training follows the original DiffusionPoser setup: each sample contains
    the full clean motion state `x0`. Sensor-combination masks are constructed
    only during inpainting inference.
    """

    pose_dim = 24 * 6
    root_vel_dim = 3
    root_y_dim = 1
    contact_dim = 2

    def __init__(
        self,
        fold: str = 'train',
        evaluate: str = None,
        window_length: int = None,
        data_file_limit: int = None,
    ):
        super().__init__()
        self.fold = fold
        self.evaluate = evaluate
        self.window_length = window_length or datasets.window_length
        self.data_file_limit = data_file_limit
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)
        self.data = self._prepare_dataset()

    @property
    def state_dim(self):
        return (
            self.pose_dim
            + self.root_vel_dim
            + self.root_y_dim
            + self.contact_dim
        )

    @property
    def pose_slice(self):
        return slice(0, self.pose_dim)

    @property
    def root_vel_slice(self):
        start = self.pose_dim
        return slice(start, start + self.root_vel_dim)

    @property
    def root_y_slice(self):
        start = self.root_vel_slice.stop
        return slice(start, start + self.root_y_dim)

    @property
    def contact_slice(self):
        start = self.root_y_slice.stop
        return slice(start, start + self.contact_dim)

    def _get_data_files(self, data_folder):
        if self.fold == 'train':
            files = sorted([x.name for x in data_folder.iterdir() if not x.is_dir()])
            if self.data_file_limit is not None:
                files = files[:self.data_file_limit]
            return files
        if self.fold == 'test':
            return [datasets.test_datasets[self.evaluate]]
        raise ValueError(f"Unknown data fold: {self.fold}.")

    def _prepare_dataset(self):
        data_folder = paths.processed_datasets / ('eval' if self.evaluate else '')
        data_files = self._get_data_files(data_folder)
        data = {
            key: []
            for key in [
                'x0',
                'pose',
                'joint',
                'tran',
                'contact',
            ]
        }

        for data_file in tqdm(data_files, desc="Loading diffusion data files"):
            file_data = torch.load(data_folder / data_file)
            self._process_file_data(file_data, data, data_file)
        return data

    def _pad_imus(self, acc, ori):
        if acc.shape[1] < 7:
            acc = torch.cat([acc, torch.zeros(acc.shape[0], 7 - acc.shape[1], 3)], dim=1)
            ori = torch.cat([ori, torch.zeros(ori.shape[0], 7 - ori.shape[1], 3, 3)], dim=1)
        return acc[:, :7] / amass.acc_scale, ori[:, :7]

    def _get_global_pose_and_joint(self, pose):
        pose_global, joint = self.bodymodel.forward_kinematics(pose=pose.view(-1, 216))
        return pose_global.view(-1, 24, 3, 3), joint.view(-1, 24, 3)

    def _get_contact(self, contact, joint):
        if contact is not None:
            return contact.float()

        dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
        dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
        lfoot_contact = torch.cat((torch.zeros(1), (dist_lfeet < 0.008).float()))
        rfoot_contact = torch.cat((torch.zeros(1), (dist_rfeet < 0.008).float()))
        return torch.stack((lfoot_contact, rfoot_contact), dim=1)

    def _build_state(self, pose, tran, contact):
        pose_6d = art.math.rotation_matrix_to_r6d(pose).reshape(pose.shape[0], -1)
        root_vel = torch.cat((torch.zeros(1, 3), tran[1:] - tran[:-1]))
        root_vel = root_vel * (datasets.fps / amass.vel_scale)
        root_y = tran[:, 1:2]
        return torch.cat(
            [
                pose_6d,
                root_vel,
                root_y,
                contact,
            ],
            dim=1,
        )

    def _process_file_data(self, file_data, data, data_file=None):
        accs, oris, poses, trans = file_data['acc'], file_data['ori'], file_data['pose'], file_data['tran']
        foots = file_data.get('contact', [None] * len(poses))
        sequences = zip(accs, oris, poses, trans, foots)
        desc = f"Processing {data_file}" if data_file else "Processing diffusion sequences"

        for acc, ori, pose, tran, foot in tqdm(sequences, total=len(poses), desc=desc, leave=False):
            acc, ori = self._pad_imus(acc, ori)
            pose, joint = self._get_global_pose_and_joint(pose)
            contact = self._get_contact(foot, joint)
            x0 = self._build_state(pose, tran, contact)

            data_len = len(x0) if self.evaluate else self.window_length
            x0_chunks = torch.split(x0, data_len)
            pose_chunks = torch.split(pose, data_len)
            joint_chunks = torch.split(joint, data_len)
            tran_chunks = torch.split(tran, data_len)
            contact_chunks = torch.split(contact, data_len)

            for x_chunk, pose_chunk, joint_chunk, tran_chunk, contact_chunk in zip(
                x0_chunks,
                pose_chunks,
                joint_chunks,
                tran_chunks,
                contact_chunks,
            ):
                data['x0'].append(x_chunk)
                data['pose'].append(pose_chunk)
                data['joint'].append(joint_chunk)
                data['tran'].append(tran_chunk)
                data['contact'].append(contact_chunk)

    def __getitem__(self, idx):
        return {
            'x0': self.data['x0'][idx].float(),
            'pose': self.data['pose'][idx].float(),
            'joint': self.data['joint'][idx].float(),
            'tran': self.data['tran'][idx].float(),
            'contact': self.data['contact'][idx].float(),
        }

    def __len__(self):
        return len(self.data['x0'])


def pad_diffusion_seq(batch):
    """Pad variable-length diffusion samples."""
    tensor_keys = ['x0', 'pose', 'joint', 'tran', 'contact']
    out = {}

    for key in tensor_keys:
        sequences = [item[key] for item in batch]
        out[key] = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        out[f'{key}_lengths'] = [seq.shape[0] for seq in sequences]

    return out

def pad_seq(batch):
    """Pad sequences to same length for RNN."""
    def _pad(sequence):
        padded = nn.utils.rnn.pad_sequence(sequence, batch_first=True)
        lengths = [seq.shape[0] for seq in sequence]
        return padded, lengths

    inputs, poses, joints, trans = zip(*[(item[0], item[1], item[2], item[3]) for item in batch])
    inputs, input_lengths = _pad(inputs)
    poses, pose_lengths = _pad(poses)
    joints, joint_lengths = _pad(joints)
    trans, tran_lengths = _pad(trans)
    
    outputs = {'poses': poses, 'joints': joints, 'trans': trans}
    output_lengths = {'poses': pose_lengths, 'joints': joint_lengths, 'trans': tran_lengths}

    if len(batch[0]) > 5: # include velocity and foot contact, if available
        vels, foots = zip(*[(item[4], item[5]) for item in batch])

        # foot contact 
        foot_contacts, foot_contact_lengths = _pad(foots)
        outputs['foot_contacts'], output_lengths['foot_contacts'] = foot_contacts, foot_contact_lengths

        # root velocities
        vels, vel_lengths = _pad(vels)
        outputs['vels'], output_lengths['vels'] = vels, vel_lengths

    return (inputs, input_lengths), (outputs, output_lengths)


class PoseDataModule(L.LightningDataModule):
    def __init__(self, finetune: str = None):
        super().__init__()
        self.finetune = finetune
        self.hypers = finetune_hypers if self.finetune else train_hypers

    def setup(self, stage: str):
        if stage == 'fit':
            dataset = PoseDataset(fold='train', finetune=self.finetune)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        elif stage == 'test':
            self.test_dataset = PoseDataset(fold='test', finetune=self.finetune)

    def _dataloader(self, dataset):
        return DataLoader(
            dataset, 
            batch_size=self.hypers.batch_size, 
            collate_fn=pad_seq, 
            num_workers=self.hypers.num_workers, 
            shuffle=True, 
            drop_last=True
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)


class DiffusionPoseDataModule(L.LightningDataModule):
    def __init__(self, evaluate: str = None, train_data_file_limit: int = None):
        super().__init__()
        self.evaluate = evaluate
        self.train_data_file_limit = train_data_file_limit
        self.hypers = train_hypers

    def setup(self, stage: str):
        if stage == 'fit':
            if hasattr(self, 'train_dataset') and hasattr(self, 'val_dataset'):
                return
            dataset = DiffusionPoseDataset(fold='train', data_file_limit=self.train_data_file_limit)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        elif stage == 'test':
            if hasattr(self, 'test_dataset'):
                return
            self.test_dataset = DiffusionPoseDataset(fold='test', evaluate=self.evaluate)

    def _dataloader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.hypers.batch_size,
            collate_fn=pad_diffusion_seq,
            num_workers=self.hypers.num_workers,
            shuffle=shuffle,
            drop_last=True,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, shuffle=False)

    def get_normalization_stats(self):
        if hasattr(self, "_normalization_stats"):
            return self._normalization_stats

        if not hasattr(self, "train_dataset"):
            raise RuntimeError("Call setup('fit') before requesting normalization stats.")

        state_dim = self.train_dataset[0]["x0"].shape[-1]
        total_count = 0
        total_sum = torch.zeros(state_dim)
        total_sq_sum = torch.zeros(state_dim)

        for sample in tqdm(self.train_dataset, desc="Computing diffusion normalization stats"):
            x0 = sample["x0"].float().reshape(-1, state_dim)
            total_sum += x0.sum(dim=0)
            total_sq_sum += (x0 ** 2).sum(dim=0)
            total_count += x0.shape[0]

        if total_count == 0:
            raise RuntimeError("Normalization stats cannot be computed from an empty training dataset.")

        mean = total_sum / total_count
        var = total_sq_sum / total_count - mean ** 2
        std = torch.sqrt(var.clamp_min(1e-8))

        contact_slice = self.train_dataset.dataset.contact_slice
        mean[contact_slice] = 0.0
        std[contact_slice] = 1.0

        self._normalization_stats = {
            "mean": mean,
            "std": std.clamp_min(1e-6),
            "count": total_count,
        }
        return self._normalization_stats
