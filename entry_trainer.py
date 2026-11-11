import argparse
import os
from pathlib import Path

import pytorch_lightning
import torch
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import CSVLogger

from data.data_handler import PolicyDataModule
from entry_model import ChessResNet

# =============================================== PARAMETERS =======================================================
JSONL_PATH = Path(__file__).resolve().parent / "data" / "games.jsonl"  # Output of data/create_dataset.py
FROM_JSONL = True  # Set True if using JSONL instead of .pt
BATCH_SIZE = 256  # Number of samples sent to the GPU at a time
LEARNING_RATE = 1e-3  # How much model adjusts; too high = unstable, too low = slow learning
EPOCHS = 1000  # Number of times to loop over the dataset MAX
NUM_WORKERS = max(0, min(4, (os.cpu_count() or 1) - 2))  # Allow zero workers on small machines
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"  # Saved checkpoints


# ==================================================================================================================

def entry_dataloader():
    data_module = PolicyDataModule(data_path=JSONL_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    return data_module


def entry_train(resume_ckpt=None, smoke_test=False):
    if not JSONL_PATH.is_file() or JSONL_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"Training data missing or empty: {JSONL_PATH}. Run python data/create_dataset.py first."
        )
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    data = (PolicyDataModule(data_path=JSONL_PATH, batch_size=8, num_workers=0)
            if smoke_test else entry_dataloader())

    # An intentionally tiny model is enough to verify the end-to-end training flow.
    model = (ChessResNet(num_blocks=1, num_channels=16, lr=LEARNING_RATE)
             if smoke_test else ChessResNet(lr=LEARNING_RATE))

    logger = CSVLogger(str(Path(__file__).resolve().parent / "logs"), name="entry_train")
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="step-{step:06d}",
        every_n_train_steps=500,
    )

    trainer = pytorch_lightning.Trainer(
        max_epochs=1 if smoke_test else EPOCHS,
        logger=logger,
        callbacks=[RichProgressBar(), checkpoint_callback],
        log_every_n_steps=50,
        enable_checkpointing=True,
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true"
    )

    trainer.fit(model, datamodule=data, ckpt_path=resume_ckpt)

    print(f"Training complete. Logs saved to: {logger.log_dir}")
    save_path = CHECKPOINT_DIR / ("chess_model_smoke.ckpt" if smoke_test else "chess_model_final.ckpt")
    trainer.save_checkpoint(str(save_path))
    print(f"\n✅ Training complete. Model saved to: {save_path}\n")


if __name__ == "__main__":
    import os

    if os.isatty(0):
        os.system('cls' if os.name == 'nt' else 'clear')
    print("== Starting Training ==")

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Path to .ckpt file to resume training from")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Train a tiny network for one epoch to verify the pipeline.")
    args = parser.parse_args()

    entry_train(resume_ckpt=args.resume, smoke_test=args.smoke_test)
