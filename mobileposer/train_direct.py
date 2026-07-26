from argparse import ArgumentParser
from pathlib import Path

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from config import paths, train_hypers
from data import PoseDataModule
from models.directposer import DirectPoserNet

def main():
    parser = ArgumentParser()
    parser.add_argument("--save-dir", type=str, default=str(paths.checkpoint / "directposer_amass"))
    parser.add_argument("--finetune-dataset", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--transformer-d-model", type=int, default=192)
    parser.add_argument("--transformer-nhead", type=int, default=8)
    parser.add_argument("--transformer-num-layers", type=int, default=6)
    parser.add_argument("--transformer-dim-feedforward", type=int, default=768)
    parser.add_argument("--transformer-dropout", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=train_hypers.batch_size)
    parser.add_argument("--num-workers", type=int, default=train_hypers.num_workers)
    parser.add_argument("--num-epochs", type=int, default=train_hypers.num_epochs)
    parser.add_argument("--accelerator", type=str, default=train_hypers.accelerator)
    parser.add_argument("--device", type=int, default=train_hypers.device)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    seed_everything(42, workers=True)

    train_hypers.batch_size = args.batch_size
    train_hypers.num_workers = args.num_workers
    train_hypers.num_epochs = args.num_epochs
    train_hypers.accelerator = args.accelerator
    train_hypers.device = args.device

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("begin training")
    datamodule = PoseDataModule(
        finetune=args.finetune_dataset,
        use_global_pose=True,
        show_progress=True,
    )
    model = DirectPoserNet(
        backbone=args.backbone,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_num_layers=args.transformer_num_layers,
        transformer_dim_feedforward=args.transformer_dim_feedforward,
        transformer_dropout=args.transformer_dropout,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_step_loss",
        save_top_k=3,
        mode="min",
        verbose=False,
        dirpath=save_dir,
        save_weights_only=True,
        filename="{epoch}-{validation_step_loss:.4f}",
    )
    logger = TensorBoardLogger(save_dir=str(save_dir), name="logs")

    trainer = L.Trainer(
        fast_dev_run=args.fast_dev_run,
        min_epochs=train_hypers.num_epochs,
        max_epochs=train_hypers.num_epochs,
        devices=[train_hypers.device],
        accelerator=train_hypers.accelerator,
        logger=logger,
        callbacks=[checkpoint_callback],
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)
    print(f"BEST_CHECKPOINT={checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
