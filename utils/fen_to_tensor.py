import numpy as np
import chess

def convert_fen_to_tensor(fen):
    """ Final tensor shape: [18, 8, 8]
     ------------------------------
     18 channels encode:
      - 0-5  : White pieces [P, N, B, R, Q, K]
      - 6-11 : Black pieces [p, n, b, r, q, k]
      - 12   : Side to move (1.0 if white, 0.0 if black)
      - 13   : White king  side castling rights
      - 14   : White queen side castling rights
      - 15   : Black king  side castling rights
      - 16   : Black queen side castling rights
      - 17   : En passant square (1.0 only on ep-square if available)
     8 x 8 grid:
       - First axis [0-7]: rows (rank 8 to rank 1, top to bottom)
       - Second axis [0-7]: columns (file a to h, left to right)"""

    board = chess.Board(fen)
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    # In each of these layers, a value of 1 at [row, col] means that piece is present at that board position.
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        tensor[piece_map[piece.symbol()], row, col] = 1

    # Side to move (white = 1.0, black = 0.0)
    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights
    tensor[13, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[14, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[15, :, :] = board.has_kingside_castling_rights(chess.BLACK)
    tensor[16, :, :] = board.has_queenside_castling_rights(chess.BLACK)

    # En passant square
    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        tensor[17, row, col] = 1

    return tensor
