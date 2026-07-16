import io
import chess.pgn
import time
import numpy as np
import chess
import torch

def board_to_tensor(board: chess.Board):
    tensor = np.zeros((14, 8, 8), dtype=np.int8)
    
    # chess.PAWN = 1, chess.KNIGHT = 2, etc.
    piece_indices = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Determine row and column index (0 to 7)
            row = chess.square_rank(square)
            col = chess.square_file(square)
            
            # Determine channel based on piece type and color
            channel = piece_indices[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6 # Shift by 6 for black pieces
                
            tensor[channel, row, col] = 1
            
    # Fill channel 12 with active color metadata
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1
        
    # channel 13 omitted for simplicity
    return tensor

def move_to_index(move):
    """Converts a chess.Move to an integer from 0 to 4095."""
    # move.from_square and move.to_square are integers from 0 to 63
    return (move.from_square * 64) + move.to_square



def process_games(path, max_positions=500000):
    X_array = np.zeros((max_positions, 14, 8, 8), dtype=np.int8) # X here is the input. Size [max_positions, 14, 8, 8]
    y_array = np.zeros(max_positions, dtype=np.int16) # Y here is the output. Size [max_positions]
    # X and y each correspond to each other. I.e. index 1 of X corresponds to index 1 of y.

    with open(path, encoding="utf-8") as pgn_file:
        positions_extracted = 0
        
        while positions_extracted < max_positions:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # End of file reached
                
            board = game.board()

            for move in game.mainline_moves():
                X_array[positions_extracted] = board_to_tensor(board)
                y_array[positions_extracted] = move_to_index(move)
                positions_extracted += 1
                board.push(move)
                
                if positions_extracted >= max_positions:
                    break
    
    X_array = X_array[:positions_extracted]
    y_array = y_array[:positions_extracted]
    
    print(f"Successfully processed {positions_extracted} moves.")
    
    print(X_array[0][0])
    print(y_array[0])
    
    output_filename = "chess_dataset_testdemo.npz"
    np.savez_compressed(output_filename, X=X_array, y=y_array)
    

if __name__ == "__main__":
    process_games("deep_learning_dataset_2026-06_620841games.pgn", max_positions=10000)
    