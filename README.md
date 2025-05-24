# 🧠 Apex Chess Mini - Tutorial Version

A no-nonsense, beginner-friendly chess AI framework for hobbyists, learners, and tinkerers. Built to demystify machine learning through a real, working chess model — no PhD required.

## 📌 Project Overview

This repo walks you through building and training your own chess-playing AI using PyTorch Lightning. The core model is a lightweight ResNet policy network that learns to predict strong chess moves from real games.

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
├── create_dataset.py     # PGN -> JSONL converter
├── data_handler.py       # Line-based dataset logic
utils/
├── fen_to_tensor.py      # FEN -> Tensor conversion
├── indexUCI.py           # Move indexing logic
entry_model.py            # ResNet model definition
entry_trainer.py          # Training runner
entry_game.py             # Play against your model
requirements.txt
README.md
```

## 🏁 Getting Started

### 1. Prepare Your Dataset

Convert a PGN file into training data:

```bash
python create_dataset.py
```

This creates a `games.jsonl` file with FEN positions and move labels.

### 2. Train the Model

```bash
python entry_trainer.py
```

Trains the model using your dataset and saves a checkpoint to `/checkpoints`.

### 3. Evaluate the Model

```bash
python entry_game.py
```

Play against your model and test its understanding.

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

* Want more advanced content? Join the [Patreon](#)
* Questions? Open an issue or message on Discord (coming soon)

---

📘 Need help? See [Getting Started Guide](getting_started.md)
