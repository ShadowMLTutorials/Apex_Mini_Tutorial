import argparse

import torch,  pytorch_lightning, os
from pytorch_lightning.loggers import CSVLogger
from entry_model import ChessResNet
from data.data_handler import PolicyDataModule
from pytorch_lightning.callbacks import RichProgressBar

# =============================================== PARAMETERS =======================================================
JSONL_PATH         = "data/converted_data.jsonl"   # Path to .jsonl dataset
FROM_JSONL         = True                          # Set True if using JSONL instead of .pt
BATCH_SIZE         = 2560                          # Number of samples sent to the GPU at a time
LEARNING_RATE      = 1e-3                          # How much model adjusts; too high = unstable, too low = slow learning
EPOCHS             = 1000                          # Number of times to loop over the dataset MAX
NUM_WORKERS        = min(4, os.cpu_count()-2)      # Number of parallel processes(should be no more than 4)
CHECKPOINT_DIR     = "checkpoints"                 # Sub folder to save models in
# ==================================================================================================================

def entry_dataloader():
    data_module = PolicyDataModule(data_path=JSONL_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    return data_module

def entry_train(resume_ckpt=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    data = entry_dataloader()

    model = ChessResNet(lr=LEARNING_RATE)

    logger = CSVLogger("logs", name="entry_train")
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="step-{step:06d}",
        every_n_train_steps=500,
    )

    trainer = pytorch_lightning.Trainer(
        max_epochs=EPOCHS,
        logger=logger,
        callbacks=[RichProgressBar(), checkpoint_callback],
        log_every_n_steps=50,
        enable_checkpointing=True,
        devices="auto",
        precision="16-mixed"
    )

    trainer.fit(model, datamodule=data, ckpt_path=resume_ckpt)

    print(f"Training complete. Logs saved to: {logger.log_dir}")
    save_path = os.path.join(CHECKPOINT_DIR, "chess_model_final.ckpt")
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Training complete. Model saved to: {save_path}\n")




if __name__ == "__main__":
    import os
    if os.isatty(0):
        os.system('cls' if os.name == 'nt' else 'clear')
    print("== Starting Training ==")

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Path to .ckpt file to resume training from")
    args = parser.parse_args()

    entry_train(resume_ckpt=args.resume)