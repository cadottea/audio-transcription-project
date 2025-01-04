import os
import subprocess
import sys
from whisper import run_whisper  # Import the function from whisper.py
from model_selection import select_model  # Import model selection
from arg_parser import parse_arguments  # Import the argument parser if needed

def process_batch(audio_files, output_dir, model):
    """
    Processes multiple audio files sequentially, calling the whisper function for each one.
    """
    for audio_file in audio_files:
        run_whisper(audio_file, output_dir, model)  # Call run_whisper function for each file

def main():
    desktop = os.path.expanduser("~/Desktop")
    audio_dir = os.path.join(desktop, "transcribe")  # Folder containing .wav files
    output_dir = os.path.join(desktop, "output")    # Folder to save transcriptions

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all .WAV files from the Transcribe folder (case-insensitive)
    files = os.listdir(audio_dir)
    print(f"Files in the directory: {files}")

    audio_files = [os.path.join(audio_dir, f) for f in files if f.lower().endswith(".wav")]
    print(f"Filtered .wav files: {audio_files}")

    if not audio_files:
        print("No .wav files found in the specified directory.")
        exit(1)

    # Select the model
    model = select_model()  # Use the function from model_selection.py

    # Process all files in batch (sequentially)
    process_batch(audio_files, output_dir, model)

if __name__ == "__main__":
    main()