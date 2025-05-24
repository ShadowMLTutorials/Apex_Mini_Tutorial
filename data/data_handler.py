# data_handler.py
# -------------------------------------------------------------
# This module defines a chess training dataset and a data module
# for PyTorch Lightning. It supports:
# - Loading training data from .jsonl (FEN + move index)
# - Converting FENs into tensor representations
# - Threaded, RAM-efficient line-by-line loading for large datasets
# Ideal for learners and hobbyists working with chess AI models.
# -------------------------------------------------------------

from torch.utils.data import DataLoader
import json, torch, pytorch_lightning

from utils import indexUCI
from utils.fen_to_tensor import convert_fen_to_tensor
from torch.utils.data import IterableDataset
import warnings
warnings.filterwarnings("ignore", message="Your `IterableDataset` has `__len__` defined.*")

# Loads chess data from .jsonl file line-by-line (on-demand)
class ChessDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self._length = None

    def _count_lines(self):
        if self._length is None:
            with open(self.file_path, 'r') as f:
                self._length = sum(1 for _ in f)
        return self._length

    def __len__(self):
        return self._count_lines()

    def parse_line(self, line):
        try:
            sample = json.loads(line)
            tensor_fen = convert_fen_to_tensor(sample['fen'])
            fen_tensor = torch.tensor(tensor_fen, dtype=torch.float32)
            try:
                move_tensor = torch.tensor(sample['index'], dtype=torch.long)
            except:
                move_tensor = torch.tensor(indexUCI.encode(sample['move']), dtype=torch.long)

            return fen_tensor, move_tensor
        except KeyError:
            return None

    def __iter__(self):
        with open(self.file_path, 'r') as f:
            for line in f:
                result = self.parse_line(line)
                if result is not None:
                    yield result

# Lightning DataModule that provides a PyTorch DataLoader
class PolicyDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, data_path, batch_size=64, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_set = ChessDataset(self.data_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.num_workers
        )
