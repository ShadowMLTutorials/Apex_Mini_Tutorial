# Getting Started with Apex Mini (Windows)

Welcome to your first step in building your own AI-powered chess engine. This guide walks you through setting up the
environment, preparing your data, and running your first training session using the Apex Mini tutorial project.

> 🐧 **Using Linux?** [Click here for the Linux setup guide](getting_started.md)

## 🔧 Prerequisites

* Python 3.9 or newer (Install from [python.org](https://www.python.org/downloads/))

    * Be sure to check "Add Python to PATH" during installation
* Git for Windows ([Download here](https://git-scm.com/download/win))

    * During setup, accept default options unless you know what you're doing
* NVIDIA GPU with CUDA support (minimum 6GB VRAM recommended)

    * High-end CPUs are usable but very slow (expect days of training)
    * AMD GPUs are not supported for training (for now)

        * You can still run inference
* Command Prompt or PowerShell
* Python IDE recommended: PyCharm [Official Download](https://www.jetbrains.com/pycharm/download/)

## 📦 Step 1: Clone the Repo

```cmd
git clone https://github.com/ShadowBLGaming/Apex_Mini_Tutorial.git
cd Apex_Mini_Tutorial
```

## 🐍 Step 2: Set Up Your Environment

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 📁 Step 3: Prepare Your Dataset

You can:

* Use a PGN dataset from any source

    * Suggested: [Lichess Elite Database](https://database.nikonoel.fr/)
    * Start with a 30-50MB PGN file for smaller models
* Or generate your own PGN from games you’ve played
* Save the PGN in the `data/` folder as `games.pgn`

    * Or modify the filepath in `data/create_dataset.py`

Then convert to `.jsonl` format:

* Modify the top of `create_dataset.py` to set:

    * Min/max move count per game
    * Require checkmate ending
    * Minimum ELO filter

```cmd
.venv\Scripts\activate  # if not already in venv
python data\create_dataset.py
```

## 🧠 Step 4: Train the Model

* With the dataset ready, launch training:

```cmd
.venv\Scripts\activate  # if not already in venv
python entry_trainer.py
```

To resume from a saved checkpoint:

```cmd
python entry_trainer.py --resume checkpoints\(checkpoint_name).ckpt
```

## 💡 Tips

* Training logs are saved in the `logs/` directory
* Checkpoints are stored in `checkpoints/`
* Edit `entry_trainer.py` to adjust batch size, learning rate, and more

---

Need help? Join the [Discord](https://discord.gg/EbZJxnJyQr) and hop into the ML Lounge or Supporter channels.

Let’s build.

— Shadow
