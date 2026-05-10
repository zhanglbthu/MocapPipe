import torch
import lightning as L
import json
import shutil
import os
import subprocess
import sys
from argparse import ArgumentParser
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

from config import paths, train_hypers
from data import DiffusionPoseDataModule
from diffusionposer import DiffusionPoser, DiffusionPoserConfig, DiffusionPoserInference
from utils.file_utils import get_datestring, get_dir_number


torch.set_float32_matmul_precision('medium')


class UnconditionalSampleCallback(L.Callback):
    def __init__(
        self,
        every_n_epochs,
        num_steps,
        num_samples,
        seed,
        output_root,
        render=False,
        fps=30,
        keep_intermediates=False,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.seed = seed
        self.output_root = Path(output_root)
        self.render = render
        self.fps = fps
        self.keep_intermediates = keep_intermediates

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.every_n_epochs <= 0:
            return

        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return

        was_training = pl_module.training
        try:
            pl_module.eval()
            save_dir = self.output_root / "prior_samples" / f"epoch_{epoch:03d}"
            render_dir = self.output_root / "prior_samples_videos" / f"epoch_{epoch:03d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            summary = []
            for sample_idx in range(1, self.num_samples + 1):
                sample_seed = self.seed + sample_idx - 1
                pred_state, pose, tran = self._unconditional_sample(
                    model=pl_module,
                    num_steps=self.num_steps,
                    window_length=pl_module.config.window_length,
                    seed=sample_seed,
                )
                torch.save(
                    {
                        "pred_state": pred_state.cpu(),
                        "pose_p": pose.cpu(),
                        "pose_t": pose.cpu(),
                        "tran_p": tran.cpu(),
                        "tran_t": tran.cpu(),
                    },
                    save_dir / f"{sample_idx}.pt",
                )
                summary.append(
                    {
                        "sample_id": sample_idx,
                        "seed": sample_seed,
                        **self._motion_stats(pose.cpu(), tran.cpu()),
                    }
                )

            with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            print()
            print("-" * 50)
            print(
                f"Saved unconditional prior samples: epoch={epoch}, "
                f"num_samples={self.num_samples}, num_steps={self.num_steps}, dir={save_dir}"
            )
            print("-" * 50)
            print()

            if self.render:
                self._render(save_dir, render_dir)

            if not self.keep_intermediates:
                shutil.rmtree(save_dir, ignore_errors=True)
        finally:
            if was_training:
                pl_module.train()

    def _unconditional_sample(self, model, num_steps, window_length, seed):
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

        inference = DiffusionPoserInference(model, num_steps=num_steps)
        x_input = torch.zeros(window_length, inference.layout.state_dim, device=device)
        observed_mask = torch.zeros_like(x_input)
        pred_state = inference.inpaint(x_input, observed_mask, num_steps=num_steps)
        pose = inference.state_to_pose(pred_state)
        tran = inference.state_to_tran(pred_state)
        return pred_state, pose, tran

    @staticmethod
    def _motion_stats(pose, tran):
        pose_delta = pose[1:] - pose[:-1] if pose.shape[0] > 1 else torch.zeros_like(pose[:0])
        root_delta = tran[1:] - tran[:-1] if tran.shape[0] > 1 else torch.zeros_like(tran[:0])
        return {
            "num_frames": int(pose.shape[0]),
            "mean_pose_delta": float(pose_delta.abs().mean().item()) if pose_delta.numel() else 0.0,
            "mean_root_speed": float(root_delta.norm(dim=1).mean().item()) if root_delta.numel() else 0.0,
            "root_path_length": float(root_delta.norm(dim=1).sum().item()) if root_delta.numel() else 0.0,
        }

    def _render(self, save_dir, render_dir):
        env = os.environ.copy()
        cmd = [
            sys.executable,
            str((paths.root_dir / "visualize.py").resolve()),
            "--input-dir",
            str(save_dir),
            "--output-dir",
            str(render_dir),
            "--fps",
            str(self.fps),
        ]
        subprocess.run(cmd, check=True, cwd=paths.root_dir, env=env)

def build_config(args):
    return DiffusionPoserConfig(
        state_dim=args.state_dim,
        window_length=args.window_length,
        diffusion_steps=args.diffusion_steps,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        loss_simple_weight=args.loss_simple_weight,
        loss_vel_weight=args.loss_vel_weight,
        loss_fk_weight=args.loss_fk_weight,
        loss_drift_weight=args.loss_drift_weight,
        loss_slide_weight=args.loss_slide_weight,
    )


def build_checkpoint_path(args):
    root = Path(args.save_dir) if args.save_dir else paths.checkpoint
    root.mkdir(exist_ok=True, parents=True)
    run_dir = Path(args.run_name) if args.run_name else Path(str(get_dir_number(root)))
    checkpoint_path = root / run_dir / "diffusionposer"
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    return checkpoint_path


def build_callbacks(args, checkpoint_path):
    callbacks = []
    monitor_metric = "validation_step_loss" if args.monitor_metric == "auto" else args.monitor_metric

    if args.prior_sample_every_n_epochs > 0:
        callbacks.append(
            UnconditionalSampleCallback(
                every_n_epochs=args.prior_sample_every_n_epochs,
                num_steps=args.prior_sample_num_steps,
                num_samples=args.prior_sample_num_samples,
                seed=args.prior_sample_seed,
                output_root=checkpoint_path.parent,
                render=args.prior_sample_render,
                fps=args.prior_sample_fps,
                keep_intermediates=args.prior_sample_keep_intermediates,
            )
        )

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        save_top_k=args.save_top_k,
        mode="min",
        verbose=False,
        dirpath=checkpoint_path,
        save_weights_only=True,
        filename="{epoch}-{" + monitor_metric +":.4f}",
    )
    callbacks.append(checkpoint_callback)
    return callbacks, monitor_metric


def build_trainer(args, checkpoint_path):
    logger = TensorBoardLogger(
        save_dir=str(checkpoint_path.parent),
        name=checkpoint_path.name,
        version=get_datestring(),
    )
    callbacks, monitor_metric = build_callbacks(args, checkpoint_path)

    trainer_kwargs = {
        "fast_dev_run": args.fast_dev_run,
        "min_epochs": args.epochs,
        "max_epochs": args.epochs,
        "accelerator": args.accelerator,
        "logger": logger,
        "callbacks": callbacks,
        "deterministic": True,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
    }

    if args.accelerator != "cpu":
        trainer_kwargs["devices"] = [args.device]
    else:
        trainer_kwargs["devices"] = 1

    return L.Trainer(**trainer_kwargs), monitor_metric


def main():
    parser = ArgumentParser()
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--accelerator", type=str, default=train_hypers.accelerator)
    parser.add_argument("--device", type=int, default=train_hypers.device)
    parser.add_argument("--epochs", type=int, default=train_hypers.num_epochs)
    parser.add_argument("--batch-size", type=int, default=train_hypers.batch_size)
    parser.add_argument("--num-workers", type=int, default=train_hypers.num_workers)
    parser.add_argument("--lr", type=float, default=train_hypers.lr)
    parser.add_argument("--save-top-k", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--train-data-file-limit", type=int, default=None)
    parser.add_argument("--monitor-metric", type=str, default="auto")

    parser.add_argument("--state-dim", type=int, default=150)
    parser.add_argument("--window-length", type=int, default=125)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--loss-simple-weight", type=float, default=1.0)
    parser.add_argument("--loss-vel-weight", type=float, default=1.0)
    parser.add_argument("--loss-fk-weight", type=float, default=1.0)
    parser.add_argument("--loss-drift-weight", type=float, default=1.0)
    parser.add_argument("--loss-slide-weight", type=float, default=1.0)
    parser.add_argument("--prior-sample-every-n-epochs", type=int, default=0)
    parser.add_argument("--prior-sample-num-steps", type=int, default=30)
    parser.add_argument("--prior-sample-num-samples", type=int, default=1)
    parser.add_argument("--prior-sample-seed", type=int, default=1234)
    parser.add_argument("--prior-sample-render", action="store_true")
    parser.add_argument("--prior-sample-fps", type=int, default=30)
    parser.add_argument("--prior-sample-keep-intermediates", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    if args.fast_dev_run and args.train_data_file_limit is None:
        args.train_data_file_limit = 2
        print("fast-dev-run detected: limiting training dataset loading to first 2 processed files.")

    datamodule = DiffusionPoseDataModule(train_data_file_limit=args.train_data_file_limit)
    datamodule.hypers.batch_size = args.batch_size
    datamodule.hypers.num_workers = args.num_workers
    datamodule.setup("fit")

    config = build_config(args)
    model = DiffusionPoser(config)
    model.hypers.lr = args.lr
    model.hypers.batch_size = args.batch_size

    checkpoint_path = build_checkpoint_path(args)
    stats = datamodule.get_normalization_stats()
    model.set_normalization_stats(stats["mean"], stats["std"])
    torch.save(
        {
            "mean": stats["mean"],
            "std": stats["std"],
            "count": stats["count"],
        },
        checkpoint_path.parent / "normalization_stats.pt",
    )
    trainer, monitor_metric = build_trainer(args, checkpoint_path)

    print()
    print("-" * 50)
    print("Training Module: diffusionposer")
    print("Checkpoint Path:", checkpoint_path)
    print("Checkpoint Monitor:", monitor_metric)
    print("Normalization Frames:", stats["count"])
    print("Config:", config)
    print("-" * 50)
    print()

    try:
        trainer.fit(model, datamodule=datamodule)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
