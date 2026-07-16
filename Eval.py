import torch
import chess

from Baseline import BaselineModel
from Primary_ResNet import PrimaryModel
from Data.tensor_parser import board_to_tensor, move_to_index

def predict_top_3_legal_moves(model, board, device):
    """Passes the board to the model, masks illegal moves, and returns the top 3 choices."""
    tensor_int8 = board_to_tensor(board)
    tensor_gpu = torch.tensor(tensor_int8, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor_gpu)[0]

    masked_logits = torch.full_like(logits, float('-inf'))
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return ["Game Over"]

    for move in legal_moves:
        idx = move_to_index(move)
        masked_logits[idx] = logits[idx]

    k = min(3, len(legal_moves))
    topk_values, topk_indices = torch.topk(masked_logits, k)
    
    top_moves = []
    for i in range(k):
        best_idx = topk_indices[i].item()
        # Find the matching python-chess move object
        for move in legal_moves:
            if move_to_index(move) == best_idx:
                top_moves.append(move.uci())
                break
                
    return top_moves
    
def predict_top_3_moves(model, board, device):
    # This script does not remove illegal moves
    tensor_int8 = board_to_tensor(board)
    tensor_gpu = torch.tensor(tensor_int8, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor_gpu)[0]

    k=3
    topk_values, topk_indices = torch.topk(logits, k)
    
    top_moves = []
    for i in range(k):
        best_idx = topk_indices[i].item()
        
        # Reverse the index back to from/to squares
        from_square = best_idx // 64
        to_square = best_idx % 64
        
        # Construct the move object and get the UCI string
        raw_move = chess.Move(from_square, to_square)
        top_moves.append(raw_move.uci())  
    return top_moves

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}...\n")
    
    baseline = BaselineModel(hidden_size=1024)
    baseline.load_state_dict(torch.load("baseline/model_epoch_20.pth", map_location=device))
    baseline.to(device)
    
    resnet = PrimaryModel()
    resnet.load_state_dict(torch.load("model_epoch_15.pth", map_location=device))
    resnet.to(device)

    
    # test cases
    """
    puzzles = {
        "OPENING 1": "rn2k2r/1bqp1pp1/p3pn1p/8/2P5/b1N1PN2/3BBPPP/1R1Q1RK1 w kq - 0 15",
        "OPENING 2": "r1b1k2r/pp1p1pp1/1qn1p2p/2b5/N3P1nB/5N2/PPP2PPP/1R1QKB1R b Kkq - 5 9",
        "OPENING 3": "r1bqr1k1/ppp2pb1/5np1/8/1PP1N3/6P1/PB2PPBP/R2Q1RK1 b - - 0 15",
        "OPENING 4": "r2q1rk1/ppp1b1pn/2np3p/5b2/4BP2/1NN5/PPP3PP/R1BQ1R1K w - - 1 13",
        "OPENING 5": "r1b1k2r/pppp1Npp/2n5/2b1p3/2B3nq/6P1/PPPPQ2P/RNB2RK1 w kq - 1 9",
        
        "MIDGAME 1": "rn3rk1/p1p1npb1/7p/1B1q1b2/3P4/2P2QP1/PP1N3P/R1B2RK1 w - - 0 15",
        "MIDGAME 2": "2r2rk1/pp4pp/1q1Np3/8/1n4p1/5Q1P/PPP3P1/1R3R1K w - - 0 21",
        "MIDGAME 3": "5k2/pQ3p2/1b6/5b2/1P1P2Pr/P7/2r1BP2/1K1R4 b - - 2 28",
        "MIDGAME 4": "r3rnk1/ppq2p1p/2pb2p1/3P3R/2P1B3/5P2/PBQ2P1P/4R1K1 w - - 0 21",
        "MIDGAME 5": "r3k2r/2p2pp1/1pppb3/7p/4PPPq/P1PB4/1P4P1/R1BQ1RK1 b kq - 0 15",
        
        "ENDGAME 1": "7k/4q1p1/1p2Pp1p/p2Q4/P6P/3p1P2/5PK1/8 w - - 0 33",
        "ENDGAME 2": "3r1r2/1Q3p1k/p1p4p/4Pqp1/1P1P2R1/P5KP/3R2P1/8 b - - 2 35",
        "ENDGAME 3": "8/5Q1p/1p3p2/6k1/2P3P1/3p4/1q3PK1/8 w - - 0 39",
        "ENDGAME 4": "1R2B3/P3k1p1/3r4/8/4n1PK/7P/8/8 b - - 2 58",
        "ENDGAME 5": "2b2k2/7r/2p2P2/p1pqR1Q1/8/P2P3P/1PP3P1/7K b - - 6 35",
    }
    """
    
    puzzles = {
        
        "BLACK OPENING 1 Sicilian Defense, correct move b8c6": "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "BLACK OPENING 2 King's Indian Defense, correct move f8g7": "rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3",
        "BlACK OPENING 3 London System Response, correct move f8g7": "rnbqkb1r/pppppp1p/5np1/8/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 1 3",
        
        "WHITE OPENING 1 Ruy Lopez, correct move f1b5": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "WHITE OPENING 2 Queen's gambit declined Knight variation, correct move b1c3": "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
        "WHITE OPENING 3 Catalan Opening Closed variation, correct move f1g2": "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/6P1/PP2PP1P/RNBQKBNR w KQkq - 0 4",
        
        "Bishop 1, correct move f3a8": "r3k2r/N4ppp/1p2pn2/2q5/8/1Q2PB2/P4PPP/R4RK1 w - - 0 1",
        "Bishop 2, correct move a1f6": "7k/p4p1p/1p3p2/8/8/8/P4P1P/B5K1 w - - 0 30",
        "Bishop 3, correct move a6e2": "r1r3k1/3nBppp/b1qPp3/p7/1p1R3P/7R/PPP1Q1P1/2K2B2 b - - 0 1",
        
        
    }
    
    for name, fen in puzzles.items():
        print(f"--- {name} ---")
        board = chess.Board(fen)
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        # print(board) 
        
        baseline_preds = predict_top_3_moves(baseline, board, device)
        resnet_preds = predict_top_3_moves(resnet, board, device)
        
        print(f"Baseline Top 3: {baseline_preds}")
        print(f"Primary Top 3: {resnet_preds}")
        print("\n" + "="*40 + "\n")
    
    
    