# 🧠 Apex Chess Mini - Tutorial Version

A no-nonsense, beginner-friendly chess AI framework for hobbyists, learners, and tinkerers. Built to demystify machine
learning through a real, working chess model — no PhD required.

## 📌 Project Overview

This repo walks you through building and training your own chess-playing AI using PyTorch Lightning. The core model is a
lightweight ResNet policy network that learns to predict strong chess moves from real games.

> Whether you're a chess player curious about AI or a coder dipping your toes into ML — this is your entry point.

## 🔧 Features

* Train from PGN files using real chess games
* Line-by-line data loading for huge datasets
* Lightweight ResNet policy model
* FP16 training, GPU-friendly
* Fully scriptable CLI — no notebooks required
* Prepped for YouTube tutorials and Patreon community

## 📁 Directory Overview

```
logs/                     # Training logs
checkpoints/              # Saved model checkpoints
data/
├── games.pgn             # User-provided training games (not committed)
├── games.jsonl           # Generated training examples (not committed)
├── create_dataset.py     # PGN -> JSONL converter
├── data_handler.py       # Line-based dataset logic
utils/
├── fen_to_tensor.py      # FEN -> Tensor conversion
├── indexUCI.py           # Move indexing logic
entry_model.py            # ResNet model definition
entry_trainer.py          # Training runner
model_vs_self.py         # Have a trained checkpoint play self-games
Model_vs_SF.py           # Evaluate a trained checkpoint against Stockfish
requirements.txt
README.md
```

## 🏁 Getting Started

### 1. Prepare Your Dataset

Place your PGN dataset at `data/games.pgn`, then generate training examples:

```bash
python data/create_dataset.py
```

This creates `data/games.jsonl` with FEN positions, UCI moves, and move indices. Both the trainer and learning-rate finder read this file directly. No extra conversion step is required. The dataset is not included in the repository.

### 2. Train the Model

```bash
python entry_trainer.py
```

Trains the model using `data/games.jsonl` and saves a reusable Lightning checkpoint to `checkpoints/chess_model_final.ckpt`.

### 3. Evaluate the Model

Once training has produced `checkpoints/chess_model_final.ckpt`, choose an evaluation script:

```bash
python model_vs_self.py
python Model_vs_SF.py
```

The first generates self-play PGNs. The second plays the model against Stockfish and requires `STOCKFISH_PATH` in `Model_vs_SF.py` to point at an installed Stockfish executable. These are automated evaluation scripts, not a human-versus-AI interface.

## 📦 Requirements

Install all dependencies via:

```bash
pip install -r requirements.txt
```

## 💡 Project Goals

* Make ML easier to understand for those new to the space
* Reduce setup friction
* Use real tools (not toy frameworks)
* Grow a teaching community

## 💬 Community & Support

Need help? Join the [Discord](https://discord.gg/EbZJxnJyQr) and hop into the ML Lounge or Supporter channels.

---

📘 Need help? See [Getting Started Guide](getting_started.md)

## 📜 License

The repository's original source code is MIT licensed (see [LICENSE](LICENSE)). Dependencies retain their own licenses; in particular, `chess` (python-chess) is GPL-3.0-or-later. Review its conditions when distributing combined software or checking project eligibility.

## 🧪 Small-data smoke testing

A trained model and a large PGN are **not** required to check that the software runs. Use one of two test paths:

**Checkpoint-free self-play:** This creates a small *randomly initialized*, untrained network in memory and attempts one short self-play game. The moves will not reflect chess skill.

```bash
python model_vs_self.py --smoke-test
```

**End-to-end data and training test:** Lichess lets you export a bounded number of public games as PGN. The following example requests up to 50 games from a public account; this is only for local testing, and the data remains gitignored.

```bash
curl --fail --location -H "Accept: application/x-chess-pgn" \
  "https://lichess.org/api/games/user/DrNykterstein?max=50&clocks=false&evals=false&opening=false" \
  --output data/games.pgn

python data/create_dataset.py --smoke-test
python entry_trainer.py --smoke-test
python model_vs_self.py --checkpoint checkpoints/chess_model_smoke.ckpt --games 1 --max-moves 20
```

The dataset smoke-test flag relaxes the default Elo/checkmate filters. Training smoke mode runs one epoch with a small 1-block, 16-channel model and writes `checkpoints/chess_model_smoke.ckpt` instead of overwriting the regular checkpoint. Both the PGN and checkpoint remain excluded from Git.

If you have Stockfish installed, the engine comparison also accepts an untrained model or a saved checkpoint:

```bash
python Model_vs_SF.py --smoke-test --stockfish /path/to/stockfish
python Model_vs_SF.py --checkpoint checkpoints/chess_model_smoke.ckpt --stockfish /path/to/stockfish --games-per-level 1 --max-moves 20
```

These smoke tests verify the software pipeline, **not** playing strength or a meaningful Elo estimate. No pretrained checkpoint is included.
