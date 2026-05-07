import torch
import lightning as L
import random
import numpy as np
import json
import csv
from argparse import ArgumentParser
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
from tqdm import tqdm

from config import paths, train_hypers
from data import DiffusionPoseDataModule, DiffusionPoseDataset
from diffusionposer import DiffusionPoser, DiffusionPoserConfig, DiffusionPoserInference
from evaluate import PoseEvaluator
from utils.file_utils import get_datestring, get_dir_number


torch.set_float32_matmul_precision('medium')


BENCHMARK_METRIC_INDEX = {
    "benchmark_sip_error_deg": 0,
    "benchmark_angular_error_deg": 1,
    "benchmark_masked_angular_error_deg": 2,
    "benchmark_positional_error_cm": 3,
    "benchmark_masked_positional_error_cm": 4,
    "benchmark_mesh_error_cm": 5,
    "benchmark_jitter_error_100m_s3": 6,
    "benchmark_distance_error_cm": 7,
}


class DiffusionBenchmarkCallback(L.Callback):
    def __init__(
        self,
        dataset_name,
        combo,
        num_steps,
        indices,
        every_n_epochs,
        seed,
        window_length,
        output_root,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.combo = combo
        self.num_steps = num_steps
        self.indices = indices
        self.every_n_epochs = every_n_epochs
        self.seed = seed
        self.window_length = window_length
        self.output_root = Path(output_root)
        self.evaluator = PoseEvaluator()
        self.dataset = None

    def setup(self, trainer, pl_module, stage=None):
        if self.dataset is None:
            if not self.dataset_name:
                raise ValueError("benchmark_dataset is required for benchmark setup.")
            dataset = DiffusionPoseDataset(
                fold="test",
                evaluate=self.dataset_name,
                window_length=self.window_length,
            )
            self.dataset = self._build_subset(dataset, self.indices)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return

        if self.dataset is None:
            self.setup(trainer, pl_module)

        states = self._capture_rng_state()
        self._set_seed(self.seed)

        was_training = pl_module.training
        device = pl_module.device
        inference = DiffusionPoserInference(pl_module, num_steps=self.num_steps)
        errors = []
        per_sequence = []

        try:
            pl_module.eval()
            with torch.no_grad():
                sample_count = len(self.dataset)
                print(
                    f"Running benchmark inference: dataset={self.dataset_name}, combo={self.combo}, "
                    f"steps={self.num_steps}, indices={self.indices}"
                )
                for idx in tqdm(range(sample_count), desc="Benchmark inference", leave=False):
                    sample = self.dataset[idx]
                    x0 = sample["x0"].to(device)
                    pose_t = sample["pose"].to(device)
                    tran_t = sample["tran"].to(device)
                    acc_obs = sample["acc"].to(device)
                    ori_obs = sample["ori"].to(device)
                    pred_state = inference.autoregressive(
                        x0,
                        combo=self.combo,
                        num_steps=self.num_steps,
                        acc_obs=acc_obs,
                        ori_obs=ori_obs,
                    )
                    pose_p = inference.state_to_pose(pred_state)
                    tran_p = inference.state_to_tran(pred_state)
                    err = self.evaluator.eval(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)
                    errors.append(err)
                    per_sequence.append(
                        {
                            "sample_order": idx + 1,
                            "source_index": self.indices[idx],
                            "num_frames": int(pose_t.shape[0]),
                            "error": err.cpu(),
                            "pred_state": pred_state.cpu(),
                            "pose_p": pose_p.cpu(),
                            "pose_t": pose_t.cpu(),
                            "tran_p": tran_p.cpu(),
                            "tran_t": tran_t.cpu(),
                        }
                    )
        finally:
            self._restore_rng_state(states)
            if was_training:
                pl_module.train()

        if not errors:
            return

        summary = torch.stack(errors).mean(dim=0)
        angle = summary[1, 0]
        mesh = summary[5, 0]
        score = angle + mesh

        metrics = {
            "benchmark_score": score,
            "benchmark_angular_error_deg": angle,
            "benchmark_mesh_error_cm": mesh,
        }

        for name, metric_idx in BENCHMARK_METRIC_INDEX.items():
            metrics[name] = summary[metric_idx, 0]

        for name, value in metrics.items():
            pl_module.log(name, value, prog_bar=(name == "benchmark_score"), logger=True, sync_dist=False)

        print()
        print("-" * 50)
        print(
            f"Benchmark Epoch {epoch}: dataset={self.dataset_name}, combo={self.combo}, "
            f"steps={self.num_steps}, indices={self.indices}"
        )
        print(
            f"benchmark_score={float(score):.4f} | "
            f"angular={float(angle):.4f} | mesh={float(mesh):.4f}"
        )
        print("-" * 50)
        print()

        self._save_epoch_outputs(epoch, per_sequence, summary)

    @staticmethod
    def _build_subset(dataset, indices):
        zero_based = []
        for idx in indices:
            if idx < 1 or idx > len(dataset):
                raise IndexError(f"Benchmark index {idx} is out of range for dataset of size {len(dataset)}.")
            zero_based.append(idx - 1)
        return torch.utils.data.Subset(dataset, zero_based)

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _capture_rng_state():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng_state(states):
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.random.set_rng_state(states["torch"])
        if torch.cuda.is_available() and states["cuda"] is not None:
            torch.cuda.set_rng_state_all(states["cuda"])

    def _save_epoch_outputs(self, epoch, per_sequence, summary):
        save_dir = (
            self.output_root
            / "benchmark_eval"
            / self.dataset_name
            / self.combo
            / f"epoch_{epoch:03d}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        metric_names = [
            "sip_error_deg",
            "angular_error_deg",
            "masked_angular_error_deg",
            "positional_error_cm",
            "masked_positional_error_cm",
            "mesh_error_cm",
            "jitter_error_100m_s3",
            "distance_error_cm",
        ]

        for item in per_sequence:
            torch.save(
                {
                    "pred_state": item["pred_state"],
                    "pose_p": item["pose_p"],
                    "pose_t": item["pose_t"],
                    "tran_p": item["tran_p"],
                    "tran_t": item["tran_t"],
                },
                save_dir / f"{item['sample_order']}.pt",
            )

        with open(save_dir / "source_indices.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": self.dataset_name,
                    "combo": self.combo,
                    "epoch": epoch,
                    "benchmark_indices": self.indices,
                },
                f,
                indent=2,
            )

        with open(save_dir / "metrics_per_sequence.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_order", "source_index", "num_frames", *metric_names])
            for item in per_sequence:
                writer.writerow(
                    [
                        item["sample_order"],
                        item["source_index"],
                        item["num_frames"],
                        *[float(v) for v in item["error"][:, 0].tolist()],
                    ]
                )

        summary_payload = {
            name: {
                "mean": float(summary[idx, 0].item()),
                "std": float(summary[idx, 1].item()),
            }
            for idx, name in enumerate(metric_names)
        }
        with open(save_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)


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
    monitor_metric = args.monitor_metric

    if args.benchmark_dataset:
        callbacks.append(
            DiffusionBenchmarkCallback(
                dataset_name=args.benchmark_dataset,
                combo=args.benchmark_combo,
                num_steps=args.benchmark_num_steps,
                indices=parse_benchmark_indices(args.benchmark_indices),
                every_n_epochs=args.benchmark_every_n_epochs,
                seed=args.benchmark_seed,
                window_length=args.window_length,
                output_root=checkpoint_path.parent,
            )
        )
        if monitor_metric == "auto":
            monitor_metric = "benchmark_score"
    elif monitor_metric == "auto":
        monitor_metric = "validation_step_loss"

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


def parse_benchmark_indices(value):
    if value is None:
        return [1, 2, 3]
    if isinstance(value, list):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return [1, 2, 3]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


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

    parser.add_argument("--state-dim", type=int, default=171)
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

    parser.add_argument("--benchmark-dataset", type=str, default="imuposer")
    parser.add_argument("--benchmark-combo", type=str, default="lw_rw_lp_rp_h")
    parser.add_argument("--benchmark-num-steps", type=int, default=30)
    parser.add_argument("--benchmark-indices", type=str, default="1,2,3")
    parser.add_argument("--benchmark-every-n-epochs", type=int, default=1)
    parser.add_argument("--benchmark-seed", type=int, default=1234)
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
