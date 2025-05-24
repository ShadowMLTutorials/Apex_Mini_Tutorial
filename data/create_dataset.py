# create_dataset.py
# -------------------------------------------------------------
# This script converts a PGN file into a filtered, deduplicated
# JSONL dataset suitable for training a chess policy network.
# It applies these filters:
# - Minimum average Elo (e.g., 2100+)
# - Only games ending in checkmate (optional)
# - Move count bounds (e.g., 20–100)
# It parallelizes processing and saves FEN + move + index triples.
# Ideal for beginners building their first ML-ready chess dataset.
# -------------------------------------------------------------

import io, chess.pgn, json, sys, os
from pathlib import Path
from random import shuffle
sys.path.append(str(Path(__file__).resolve().parent.parent))
from multiprocessing import Pool
from tqdm import tqdm
from utils.indexUCI import encode

# ===================================== USER SETTINGS =====================================
INPUT_PGN =    Path("games.pgn")               # Path to input PGN file
OUTPUT_JSONL = Path("games.jsonl")             # Path to save processed data
REQUIRE_MATE = True                            # Only keep games that end in checkmate
MIN_ELO =      2100                            # Minimum average Elo of players
MIN_MOVES =    20                              # Ignore games shorter than this
MAX_MOVES =    100                             # Ignore games longer than this
NUM_WORKERS =  min(36, os.cpu_count() - 2)     # Parallel workers
# =========================================================================================

fen_move_pairs = []
game_strings   = []

def should_include_game(headers):
    """Check if a game meets filtering criteria based on ELO and termination."""
    try:
        avg_elo = (int(headers["WhiteElo"]) + int(headers["BlackElo"])) / 2
        if avg_elo < MIN_ELO:
            return False # skip game for low elo

        if REQUIRE_MATE and "Normal" not in headers.get("Termination", ""):
            return False # skip game for no checkmate
    except:
        return False # skip game because header is missing elo or checkmate info
    return True

def find_valid_games():
    """Scan PGN file and collect games that meet the filtering criteria."""
    with open(INPUT_PGN, encoding='utf-8') as f:
        for game in tqdm(iter(lambda: chess.pgn.read_game(f), None), desc="Scanning games", unit="game"):
            if should_include_game(game.headers):
                game_strings.append(str(game))

def process_game(game_str):
    """Convert a single game string into a list of (FEN, move, index) tuples."""
    game = chess.pgn.read_game(io.StringIO(game_str))
    board = game.board()
    local_triplets = []
    for move_count, move in enumerate(game.mainline_moves(), start=1):
        if move_count < MIN_MOVES:
            board.push(move)
            continue
        if MAX_MOVES and move_count > MAX_MOVES:
            break
        fen = board.fen()
        uci = move.uci()
        try:
            idx = encode(uci)
            local_triplets.append((fen, uci, idx))
        except:
            continue
        board.push(move)
    return local_triplets

def remove_duplicates(triplets):
    """Filter out duplicate positions based on FEN."""
    seen = set()
    unique_triplets = []
    for fen, move, idx in triplets:
        fen_key = " ".join(fen.split()[:4])
        if fen_key not in seen:
            unique_triplets.append((fen, move, idx))
            seen.add(fen_key)
    return unique_triplets

def process_pgn():
    """Parallel processing of all valid games into FEN/move/index triplets and write to output."""
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(process_game, game_strings), total=len(game_strings), desc="Processing PGN"))

    for result in results:
        fen_move_pairs.extend(result)

    print("\n📦 Removing duplicate FENs for diverse training...")
    unique = remove_duplicates(fen_move_pairs)
    fen_move_pairs[:] = unique
    shuffle(fen_move_pairs)

    print(f"✅ Unique FEN/move/index samples retained: {len(fen_move_pairs)}")

    with open(OUTPUT_JSONL, 'w') as out_f:
        for fen, move, idx in tqdm(fen_move_pairs, desc="💾 Saving to JSONL"):
            json.dump({"fen": fen, "move": move, "index": idx}, out_f)
            out_f.write("\n")

if __name__ == "__main__":
    print("\n================= ♟ CHESS DATASET CREATOR ♟ =================")
    print(f"📥 Input PGN ..........: {INPUT_PGN}")
    print(f"💾 Output JSONL .......: {OUTPUT_JSONL}")
    print(f"🔍 Elo Filter .........: >= {MIN_ELO}")
    print(f"⏱ Move Range .........: {MIN_MOVES} to {MAX_MOVES}")
    if REQUIRE_MATE: print("✅ Only games ending in CHECKMATE will be included")
    print("============================================================\n")

    find_valid_games()
    process_pgn()
    print("\n🎉 Dataset creation complete! Your training data is ready.")
