# Getting Started with Apex Mini (Linux)

Welcome to your first step in building your own AI-powered chess engine. This guide walks you through setting up the environment, preparing your data, and running your first training session using the Apex Mini tutorial project.

> 🪟 **Using Windows?** [Click here for the Windows setup guide](getting_started_win.md)

## 🔧 Prerequisites

* Python 3.9 or newer (install via your package manager or `pyenv`)
* Git (usually pre-installed or `sudo apt install git`)
* NVIDIA GPU with CUDA support (minimum 6GB VRAM recommended)

  * A high-end CPU can be used but will be slow, expect days of training.
  * AMD GPUs are not supported for training but I will find a way in the future to incorporate AMD support

    * You still can run model inference, you just can't train a new model
* Terminal (bash, zsh, etc.)
* Python IDE recommended: PyCharm [Official Download](https://www.jetbrains.com/pycharm/download/)

## 📦 Step 1: Clone the Repo

```bash
git clone https://github.com/ShadowBLGaming/Apex_Mini_Tutorial.git
cd Apex_Mini_Tutorial
```

## 🐍 Step 2: Set Up Your Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 📁 Step 3: Prepare Your Dataset

You can:

* Use a PGN dataset from any source

  * I suggest the [Lichess Elite Database](https://database.nikonoel.fr/)
  * You only need 1 PGN, we can start with a 30-50mb file for smaller models
* Or generate your own PGN file from games you have played
* The PGN file should be saved in the data folder as "games.pgn"

  * Alternatively open `data/create_dataset.py` and adjust the filepath to where your PGN is located

Once you have your PGN file convert it to `.jsonl` with `create_dataset.py`:

* Open the script and modify the parameters at the top, such as:

  * Minimum and maximum move count per game
  * Filter only games that end in checkmate
  * Minimum ELO (White\_elo + Black\_elo / 2)

```bash
source .venv/bin/activate # if not already in venv
python data/create_dataset.py
```

## 🧠 Step 4: Train the Model

* Once the training data file has been generated, we are ready for training
* Navigate to the project folder in the terminal

```bash
source .venv/bin/activate # if not already in venv
python entry_trainer.py
```

To resume from a checkpoint:

```bash
python entry_trainer.py --resume checkpoints/(checkpoint_name).ckpt
```

## 💡 Tips

* Monitor logs inside the `logs/` directory
* Checkpoints are saved to `checkpoints/`
* Tune `BATCH_SIZE`, `LEARNING_RATE`, or number of epochs in `entry_trainer.py`

---

Need help? Join the [Discord](https://discord.gg/EbZJxnJyQr) and hop into the ML Lounge or Supporter channels.

Let’s build.

— Shadow
