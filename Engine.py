import torch
import numpy as np
import chess

from Primary_ResNet import PrimaryModel
from Baseline import BaselineModel
from Data.tensor_parser import board_to_tensor, move_to_index

def get_best_legal_move(model, board, device):
    """
    Takes the board, asks the model for predictions, masks out illegal moves,
    and returns the best valid chess.Move.
    """
    tensor_int8 = board_to_tensor(board)
    
    tensor_gpu = torch.tensor(tensor_int8, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor_gpu) # [1, 4096]
        logits = outputs[0]         # [4096]

    # mask illegal moves
    masked_logits = torch.full_like(logits, float('-inf'))
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return None # game over (checkmate, stalemate)

    for move in legal_moves:
        idx = move_to_index(move)
        masked_logits[idx] = logits[idx]

    best_move_idx = torch.argmax(masked_logits).item()

    # convert back into a python-chess move object
    for move in legal_moves:
        if move_to_index(move) == best_move_idx:
            return move

    return None


# game loop
def play_game(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{device}")
    
    model = PrimaryModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    board = chess.Board()
    
    print("\n--- NEW GAME ---")
    print("you are playing White. CNN is playing Black.")
    print("enter moves in UCI format. 'quit' to quit.\n")
    
    while not board.is_game_over():
        print(board)
        print("\n")
        
        if board.turn == chess.WHITE:
            # Human's Turn
            user_input = input("your move: ").strip().lower()
            if user_input == 'quit':
                break
                
            try:
                move = chess.Move.from_uci(user_input)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move")
            except:
                print("Invalid format")
                
        else:
            print("Generating a move")
            ai_move = get_best_legal_move(model, board, device)
            
            if ai_move:
                print(f"CNN plays: {ai_move.uci()}")
                board.push(ai_move)
            else:
                print("no move found")
                break

    print("\nGame Over!")
    print(f"Result: {board.result()}")

if __name__ == "__main__":
    BEST_MODEL_WEIGHTS = "model_epoch_15.pth"
    play_game(BEST_MODEL_WEIGHTS)