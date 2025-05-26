import torch
import chess
import chess.engine
from entry_model import ChessResNet
from utils.fen_to_tensor import convert_fen_to_tensor
from utils import indexUCI
import random

MODEL_PATH = "checkpoints/19m_test.ckpt"
STOCKFISH_PATH = "/home/shadowbl/Downloads/stockfish/src/stockfish"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_MOVES = 300
GAMES_PER_LEVEL = 25
TEMPERATURE = 0.1

SKILL_ELO_MAP = {0: 1320, 1: 1467, 2: 1608, 3: 1742, 4: 1923, 5: 2204, 6: 2363, 7: 2499, 8: 2596, 9: 2703,
                 10: 2788, 11: 2856, 12: 2923, 13: 2973, 14: 3025, 15: 3070, 16: 3111, 17: 3141, 18: 3170, 19: 3191
}

def load_model(path):
    model = ChessResNet(num_channels=208, num_blocks=20)
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.half().to(DEVICE)
    model.eval()
    return model


def choose_move(model, board):
    x = convert_fen_to_tensor(board.fen())
    x = torch.tensor(x, dtype=torch.float16).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x).squeeze() / TEMPERATURE
        probs = torch.softmax(logits, dim=0)

    legal_moves = list(board.legal_moves)
    legal_uci = [m.uci() for m in legal_moves]
    legal_indices = [indexUCI.encode(uci) for uci in legal_uci if indexUCI.encode(uci) is not None]
    legal_probs = probs[legal_indices]

    if torch.any(torch.isnan(legal_probs)) or torch.any(legal_probs < 0) or legal_probs.sum() == 0:
        print("Playing random move")
        return random.choice(legal_moves)

    legal_probs = legal_probs / legal_probs.sum()
    sampled_relative_idx = torch.multinomial(legal_probs, 1).item()
    chosen_index = legal_indices[sampled_relative_idx]
    chosen_uci = indexUCI.decode(chosen_index)
    return chess.Move.from_uci(chosen_uci)


def play_vs_stockfish(model, engine, skill_level):
    engine.configure({"Skill Level": skill_level})
    board = chess.Board()
    result = None

    for _ in range(MAX_MOVES):
        if board.is_game_over():
            result = board.result()
            break

        if board.turn == chess.WHITE:
            move = choose_move(model, board)
        else:
            result_info = engine.play(board, chess.engine.Limit(time=0.1))
            move = result_info.move

        if move is None or not board.is_legal(move):
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            break

        board.push(move)

    if result is None:
        result = board.result()

    return result


def evaluate_model(model):
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        skill_level = 0
        total_wins, total_draws, total_losses, total_games = 0, 0, 0, 0
        retry_flag = False

        while True:
            wins, draws, losses = 0, 0, 0
            for game_num in range(1, GAMES_PER_LEVEL + 1):
                result = play_vs_stockfish(model, engine, skill_level)
                total_games += 1
                if result == "1-0":
                    wins += 1
                    total_wins += 1
                elif result == "0-1":
                    losses += 1
                    total_losses += 1
                else:
                    draws += 1
                    total_draws += 1

                elo = SKILL_ELO_MAP.get(skill_level, "N/A")
                print(f"\rSkill: {skill_level} | Stockfish ELO: {elo} | Game: {game_num}/{GAMES_PER_LEVEL} | W/D/L: {wins}/{draws}/{losses} | Total W/D/L: {total_wins}/{total_draws}/{total_losses} ({total_games} games)", end="")

            print()  # Line break after each skill level

            if wins >= losses + draws:
                skill_level += 1
                retry_flag = False  # Reset retry flag if improvement
            elif losses >= wins + draws:
                elo = SKILL_ELO_MAP.get(skill_level, "N/A")
                print(f"\nModel defeated at Stockfish skill level {skill_level} (ELO ~{elo})")
                break
            else:
                if retry_flag:
                    print("\nPerformance is balanced again; evaluation halted.")
                    break
                else:
                    print("\nPerformance is balanced; retrying skill level.")
                    retry_flag = True



if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    evaluate_model(model)
