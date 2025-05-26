import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner

from data.data_handler import PolicyDataModule
from entry_model import ChessResNet


def entry_dataloader():
    return PolicyDataModule(data_path="data/games.jsonl", batch_size=256, num_workers=4)


def get_lr(trainer, datamodule, model):
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        model,
        datamodule=datamodule,
        mode='linear',
        num_training=200,
        min_lr=1e-5,
        max_lr=.5,
        update_attr=True
    )
    return lr_finder


if __name__ == "__main__":
    trainer = Trainer(max_epochs=1, enable_checkpointing=False, logger=False, precision='16-mixed')
    model = ChessResNet(lr=0.001)
    lr_finder = get_lr(trainer, entry_dataloader(), model)

    suggested_lr = lr_finder.suggestion()
    print(f"\n🔍 Suggested LR: {suggested_lr:.6f}\n")

    # Plot
    fig = lr_finder.plot(suggest=True)
    plt.title("Learning Rate Finder")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
