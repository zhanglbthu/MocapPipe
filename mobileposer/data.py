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
    def __init__(
        self,
        fold: str='train',
        evaluate: str=None,
        finetune: str=None,
        use_global_pose: bool=True,
        show_progress: bool=False,
    ):
        super().__init__()
        self.fold = fold
        self.evaluate = evaluate
        self.finetune = finetune
        self.use_global_pose = use_global_pose
        self.show_progress = show_progress
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)
        self.combos = list(amass.combos_full.items())
        print(f"[PoseDataset] using combos: {[c[0] for c in self.combos]}")
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
        print(f"[PoseDataset] preparing dataset from {len(data_files)} files in {data_folder}")
        data = {key: [] for key in ['imu_inputs', 'pose_outputs', 'joint_outputs', 'tran_outputs', 'vel_outputs', 'foot_outputs']}
        
        for file_idx, data_file in enumerate(tqdm(data_files), start=1):
            if self.show_progress:
                print(f"[PoseDataset] loading file {file_idx}/{len(data_files)}: {data_file}", flush=True)
            file_data = torch.load(data_folder / data_file)
            self._process_file_data(file_data, data)
            if self.show_progress:
                print(
                    f"[PoseDataset] finished {data_file}; accumulated windows={len(data['imu_inputs'])}",
                    flush=True,
                )
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
            if joint is None or self.use_global_pose:
                pose_global, joint = self.bodymodel.forward_kinematics(pose=pose.view(-1, 216))
                joint = joint.view(-1, 24, 3)
                if not self.evaluate and self.use_global_pose:
                    pose = pose_global.view(-1, 24, 3, 3)
            elif joint is not None:
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
    def __init__(self, finetune: str = None, use_global_pose: bool = True, show_progress: bool = False):
        super().__init__()
        self.finetune = finetune
        self.use_global_pose = use_global_pose
        self.show_progress = show_progress
        self.hypers = finetune_hypers if self.finetune else train_hypers

    def setup(self, stage: str):
        if stage == 'fit':
            dataset = PoseDataset(
                fold='train',
                finetune=self.finetune,
                use_global_pose=self.use_global_pose,
                show_progress=self.show_progress,
            )
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        elif stage == 'test':
            self.test_dataset = PoseDataset(
                fold='test',
                finetune=self.finetune,
                use_global_pose=self.use_global_pose,
                show_progress=self.show_progress,
            )

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

