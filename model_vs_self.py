import random
from pathlib import Path

import chess
import chess.pgn
import torch

from entry_model import ChessResNet
from utils import indexUCI
from utils.fen_to_tensor import convert_fen_to_tensor

MODEL_PATH = Path(__file__).resolve().parent / "checkpoints" / "chess_model_final.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_MOVES = 300
NUM_GAMES = 20
TEMPERATURE = .5


def load_model(path):
    # Lightning checkpoints retain the model hyperparameters used for training.
    model = ChessResNet.load_from_checkpoint(str(path), map_location=DEVICE)
    model.to(DEVICE)
    model.eval()
    return model


def choose_move(model, board):
    x = convert_fen_to_tensor(board.fen())
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x).squeeze() / TEMPERATURE
        probs = torch.softmax(logits, dim=0)

    legal_moves = list(board.legal_moves)
    legal_uci = [m.uci() for m in legal_moves]
    legal_indices = [indexUCI.encode(uci) for uci in legal_uci if indexUCI.encode(uci) is not None]

    if not legal_indices:
        print("model still cheated")
        return random.choice(legal_moves)

    legal_probs = probs[legal_indices]
    legal_probs = legal_probs / legal_probs.sum()
    sampled_relative_idx = torch.multinomial(legal_probs, 1).item()
    chosen_index = legal_indices[sampled_relative_idx]
    chosen_uci = indexUCI.decode(chosen_index)
    return chess.Move.from_uci(chosen_uci)


def play_self_game(model):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    for _ in range(MAX_MOVES):
        if board.is_game_over():
            break

        move = choose_move(model, board)
        if move is None or not board.is_legal(move):
            break

        board.push(move)
        node = node.add_variation(move)

    game.headers["Event"] = "Self-Play"
    game.headers["Result"] = board.result()
    return game


def save_games(games, path="self_play_games.pgn"):
    with open(path, "w") as f:
        for game in games:
            print(game, file=f)


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    games = [play_self_game(model) for _ in range(NUM_GAMES)]
    save_games(games)
    print(f"✅ {NUM_GAMES} games saved to self_play_games.pgn")
