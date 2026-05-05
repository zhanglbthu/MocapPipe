import torch
import lightning as L
from argparse import ArgumentParser
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

from config import paths, train_hypers
from data import DiffusionPoseDataModule
from diffusionposer import DiffusionPoser, DiffusionPoserConfig
from utils.file_utils import get_datestring, get_dir_number


torch.set_float32_matmul_precision('medium')


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
    )


def build_checkpoint_path(args):
    root = Path(args.save_dir) if args.save_dir else paths.checkpoint
    root.mkdir(exist_ok=True, parents=True)
    run_dir = Path(args.run_name) if args.run_name else Path(str(get_dir_number(root)))
    checkpoint_path = root / run_dir / "diffusionposer"
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    return checkpoint_path


def build_trainer(args, checkpoint_path):
    logger = TensorBoardLogger(
        save_dir=str(checkpoint_path.parent),
        name=checkpoint_path.name,
        version=get_datestring(),
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_step_loss",
        save_top_k=args.save_top_k,
        mode="min",
        verbose=False,
        dirpath=checkpoint_path,
        save_weights_only=True,
        filename="{epoch}-{validation_step_loss:.4f}",
    )

    trainer_kwargs = {
        "fast_dev_run": args.fast_dev_run,
        "min_epochs": args.epochs,
        "max_epochs": args.epochs,
        "accelerator": args.accelerator,
        "logger": logger,
        "callbacks": [checkpoint_callback],
        "deterministic": True,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
    }

    if args.accelerator != "cpu":
        trainer_kwargs["devices"] = [args.device]
    else:
        trainer_kwargs["devices"] = 1

    return L.Trainer(**trainer_kwargs)


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
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    datamodule = DiffusionPoseDataModule()
    datamodule.hypers.batch_size = args.batch_size
    datamodule.hypers.num_workers = args.num_workers

    config = build_config(args)
    model = DiffusionPoser(config)
    model.hypers.lr = args.lr
    model.hypers.batch_size = args.batch_size

    checkpoint_path = build_checkpoint_path(args)
    trainer = build_trainer(args, checkpoint_path)

    print()
    print("-" * 50)
    print("Training Module: diffusionposer")
    print("Checkpoint Path:", checkpoint_path)
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
