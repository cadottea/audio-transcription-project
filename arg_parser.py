import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Whisper on a single file and convert results to HTML.")
    parser.add_argument("-f", "--file", required=True, help="Path to the audio file (.WAV)")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output files")
    parser.add_argument("-m", "--model", help="Path to the model file (optional). If not provided, user will select.")
    return parser.parse_args()