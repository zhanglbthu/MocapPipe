import torch
import torch.nn as nn
import lightning as L

from config import *
from utils.model_utils import reduced_pose_to_full
import articulate as art
from models.rnn import RNN
from models.transformer import TransformerBackbone


class DirectPoserNet(L.LightningModule):
    """
    Inputs: IMU windows.
    Outputs: SMPL pose parameters without translation.

    The model predicts reduced global pose rotations directly from IMU features
    and converts them back to local SMPL rotations for evaluation.
    """

    def __init__(
        self,
        backbone="lstm",
        transformer_d_model=192,
        transformer_nhead=8,
        transformer_num_layers=6,
        transformer_dim_feedforward=768,
        transformer_dropout=0.4,
    ):
        super().__init__()

        self.C = model_config
        self.hypers = train_hypers
        self.backbone = backbone
        self.transformer_d_model = transformer_d_model
        self.transformer_nhead = transformer_nhead
        self.transformer_num_layers = transformer_num_layers
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.transformer_dropout = transformer_dropout

        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.C.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R
        self.pose = self._build_backbone()

        self.loss = nn.MSELoss()
        self.t_weight = 1e-5
        self.use_pos_loss = True

        self.num_past_frames = model_config.past_frames
        self.num_future_frames = model_config.future_frames
        self.num_total_frames = self.num_past_frames + self.num_future_frames
        self.imu = None

        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    def _build_backbone(self):
        if self.backbone == "lstm":
            return RNN(self.C.n_imu, joint_set.n_reduced * 6, 256, bidirectional=True)
        if self.backbone == "transformer":
            return TransformerBackbone(
                self.C.n_imu,
                joint_set.n_reduced * 6,
                d_model=self.transformer_d_model,
                nhead=self.transformer_nhead,
                num_layers=self.transformer_num_layers,
                dim_feedforward=self.transformer_dim_feedforward,
                dropout=self.transformer_dropout,
                max_len=max(datasets.window_length, self.C.total_frames),
            )
        raise ValueError(f"Unsupported backbone: {self.backbone}")

    def reset(self):
        self.imu = None

    def forward(self, batch, input_lengths=None):
        pred_pose, _, _ = self.pose(batch, input_lengths)
        return pred_pose

    def _reduced_global_to_full(self, reduced_pose):
        bsz, seqlen = reduced_pose.shape[:2]
        pose = art.math.r6d_to_rotation_matrix(reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        pose = reduced_pose_to_full(pose.view(bsz, seqlen, joint_set.n_reduced, 3, 3)).view(-1, 24, 3, 3)
        pred_pose = self.global_to_local_pose(pose)
        for ignore in joint_set.ignored:
            pred_pose[:, ignore] = torch.eye(3, device=pred_pose.device)
        pred_pose[:, 0] = pose[:, 0]
        return pred_pose.view(bsz, seqlen, 24, 3, 3)

    def _pose_loss(self, pred_pose, target_pose, target_joints):
        bsz, seqlen, _ = target_pose.shape
        pose_t = target_pose.view(bsz, seqlen, 24, 6)[:, :, joint_set.reduced].reshape(bsz, seqlen, -1)
        loss = self.loss(pred_pose, pose_t)
        loss += self.t_weight * self.compute_jerk_loss(pred_pose)

        if self.use_pos_loss:
            full_pose_p = self._reduced_global_to_full(pred_pose)
            joints_p = self.bodymodel.forward_kinematics(pose=full_pose_p.view(-1, 24, 3, 3))[1].view(bsz, seqlen, -1)
            loss += self.loss(joints_p, target_joints)

        return loss

    def shared_step(self, batch):
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, _ = outputs

        target_pose = outputs["poses"]
        bsz, seqlen, _ = target_pose.shape
        target_joints = outputs["joints"].view(bsz, seqlen, -1)

        pred_pose = self(imu_inputs, input_lengths)
        return self._pose_loss(pred_pose, target_pose, target_joints)

    def compute_jerk_loss(self, pred_pose):
        jerk = pred_pose[:, 3:, :] - 3 * pred_pose[:, 2:-1, :] + 3 * pred_pose[:, 1:-2, :] - pred_pose[:, :-3, :]
        l1_norm = torch.norm(jerk, p=1, dim=2)
        return l1_norm.sum(dim=1).mean()

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("training_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.training_step_loss.append(loss.item())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("validation_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.validation_step_loss.append(loss.item())
        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        inputs, _ = batch
        imu_inputs, input_lengths = inputs
        return self(imu_inputs, input_lengths)

    def on_train_epoch_end(self):
        self.epoch_end_callback(self.training_step_loss, loop_type="train")
        self.training_step_loss.clear()

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_step_loss, loop_type="val")
        self.validation_step_loss.clear()

    def epoch_end_callback(self, outputs, loop_type):
        average_loss = torch.mean(torch.Tensor(outputs))
        self.log(f"{loop_type}_loss", average_loss, prog_bar=True, batch_size=self.hypers.batch_size)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True)

    @torch.no_grad()
    def forward_frame(self, data):
        imu = data.repeat(self.num_total_frames, 1) if self.imu is None else torch.cat((self.imu[1:], data.view(1, -1)))
        reduced_pose = self(imu.unsqueeze(0), [self.num_total_frames])
        pose = self._reduced_global_to_full(reduced_pose[:, self.num_past_frames:self.num_past_frames + 1]).squeeze(0).squeeze(0)
        self.imu = imu
        return pose

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hypers.lr)
