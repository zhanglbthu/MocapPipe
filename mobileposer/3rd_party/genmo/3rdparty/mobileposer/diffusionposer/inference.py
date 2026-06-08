import torch

import articulate as art
from config import amass, datasets


class StateLayout:
    pose_dim = 24 * 6
    root_vel_dim = 3
    root_y_dim = 1
    contact_dim = 2
    state_dim = pose_dim + root_vel_dim + root_y_dim + contact_dim

    pose_slice = slice(0, pose_dim)
    root_vel_slice = slice(pose_dim, pose_dim + root_vel_dim)
    root_y_slice = slice(root_vel_slice.stop, root_vel_slice.stop + root_y_dim)
    contact_slice = slice(root_y_slice.stop, root_y_slice.stop + contact_dim)

    @classmethod
    def pose_joint_slice(cls, joint_id):
        start = joint_id * 6
        return slice(start, start + 6)


class DiffusionPoserInference:
    """DDIM inpainting and autoregressive inference for DiffusionPoser."""

    def __init__(self, model, num_steps=10):
        self.model = model
        self.num_steps = num_steps
        self.layout = StateLayout

    @property
    def device(self):
        return next(self.model.parameters()).device

    def inpaint(self, x_input, observed_mask, num_steps=None):
        """Fill unknown dimensions while clamping observed dimensions."""
        self.model.eval()
        steps = num_steps or self.num_steps
        total_steps = self.model.config.diffusion_steps
        timesteps = torch.linspace(total_steps - 1, 0, steps, device=x_input.device).long()

        x_input = self.model.normalize_state(x_input).unsqueeze(0)
        observed_mask = observed_mask.unsqueeze(0).bool()
        x_t = torch.randn_like(x_input)
        x_t = torch.where(observed_mask, self._known_noisy(x_input, timesteps[0]), x_t)

        for i, timestep in enumerate(timesteps):
            t = torch.full((1,), timestep.item(), device=x_input.device, dtype=torch.long)
            pred_x0 = self.model(x_t, t)

            if i == len(timesteps) - 1:
                x_t = pred_x0
            else:
                next_t = timesteps[i + 1]
                x_t = self._ddim_step(x_t, pred_x0, timestep, next_t)

            known = x_input if i == len(timesteps) - 1 else self._known_noisy(x_input, timesteps[min(i + 1, len(timesteps) - 1)])
            x_t = torch.where(observed_mask, known, x_t)

        return self.model.denormalize_state(x_t.squeeze(0))

    def autoregressive(self, x0, combo, window_length=None, num_steps=None, acc_obs=None, ori_obs=None):
        """Run sliding-window inpainting using ground-truth sparse IMU observations."""
        raise NotImplementedError("Autoregressive sparse-observation inference is disabled in the acc-free prior-only setup.")

    def state_to_pose(self, state):
        pose_6d = state[:, self.layout.pose_slice].contiguous().reshape(-1, 24, 6)
        return art.math.r6d_to_rotation_matrix(pose_6d).view(-1, 24, 3, 3)

    def state_to_tran(self, state):
        root_vel = state[:, self.layout.root_vel_slice] / (datasets.fps / amass.vel_scale)
        deterministic_enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = (
            torch.is_deterministic_algorithms_warn_only_enabled()
            if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
            else False
        )

        if deterministic_enabled and root_vel.is_cuda:
            torch.use_deterministic_algorithms(False)
            try:
                tran = torch.cumsum(root_vel, dim=0)
            finally:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
        else:
            tran = torch.cumsum(root_vel, dim=0)

        tran[:, 1:2] = state[:, self.layout.root_y_slice]
        return tran

    def _known_noisy(self, x0, timestep):
        alpha_bar = self.model.alpha_bars[timestep].view(1, 1, 1)
        noise = torch.randn_like(x0)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def _ddim_step(self, x_t, pred_x0, timestep, next_t):
        alpha_bar = self.model.alpha_bars[timestep]
        next_alpha_bar = self.model.alpha_bars[next_t]
        pred_noise = (x_t - alpha_bar.sqrt() * pred_x0) / (1.0 - alpha_bar).sqrt().clamp_min(1e-8)
        return next_alpha_bar.sqrt() * pred_x0 + (1.0 - next_alpha_bar).sqrt() * pred_noise
