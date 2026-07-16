import zstandard as zstd
import io
import chess.pgn
import time
import numpy as np
import chess

def extract_high_quality_games(zst_file_path, output_pgn_path, min_elo=2000, max_games=float('inf')):
    """
    Streams a Lichess .pgn.zst file, filters out low-quality games using 
    string parsing, saves smaller dataset.
    """
    print(f"Opening {zst_file_path} for streaming...")
    
    dctx = zstd.ZstdDecompressor()
    
    games_processed = 0
    games_saved = 0
    
    # State variables for string parsing
    buffer = []
    white_elo = 0
    black_elo = 0
    is_classical_rapid = False
    is_normal = False
    
    with open(zst_file_path, 'rb') as compressed_file:
        with dctx.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            with open(output_pgn_path, 'w', encoding='utf-8') as out_file:
                for line in text_stream:
                    # Every Lichess game always starts with the Event header
                    if line.startswith("[Event "):
                        if buffer and is_classical_rapid and is_normal and white_elo >= min_elo and black_elo >= min_elo:
                            out_file.write("".join(buffer))
                            games_saved += 1
                            
                            if games_saved >= max_games:
                                print(f"\nTarget reached! Successfully extracted {games_saved} high-quality games.")
                                return
                        
                        games_processed += 1
                        if games_processed % 100000 == 0:
                            print(f"Scanned {games_processed:,} games... Saved {games_saved:,} high-quality games.")
                            
                        buffer = [line]
                        white_elo = 0
                        black_elo = 0
                        is_classical_rapid = "Classical" in line or "Rapid" in line
                        is_normal = False
                        continue
                        
                    # Add current line to the buffer
                    buffer.append(line)
                    
                    if line.startswith("[Termination "):
                        is_normal = "Normal" in line
                    elif line.startswith("[WhiteElo "):
                        try:
                            # Splits '[WhiteElo "2500"]\n' -> grabs the 2500
                            white_elo = int(line.split('"')[1])
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith("[BlackElo "):
                        try:
                            black_elo = int(line.split('"')[1])
                        except (ValueError, IndexError):
                            pass

                # Flush the very last game in the file to the output
                if buffer and is_classical_rapid and is_normal and white_elo >= min_elo and black_elo >= min_elo:
                    out_file.write("".join(buffer))
                    games_saved += 1

    print(f"\nFinished scanning file. Total games scanned: {games_processed:,}. Total saved: {games_saved:,}")



if __name__ == "__main__":
    # Update this path
    INPUT_ZST = "lichess_db_standard_rated_2026-04.pgn.zst" 
    
    # Save output
    OUTPUT_FILE = "placeholder.pgn"
    
    start_time = time.time()
    
    # Set max_games to float('inf') to extract all games matching the criteria
    extract_high_quality_games(INPUT_ZST, OUTPUT_FILE, min_elo=2000, max_games=float('inf'))
    
    end_time = time.time()
    print(f"Extraction took {round(end_time - start_time, 2)} seconds.")
    
