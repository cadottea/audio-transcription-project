import os
import subprocess
import sys
import json
from whisper import run_whisper  # Import the function from whisper.py
from model_selection import select_model  # Import model selection
from arg_parser import parse_arguments  # Import the argument parser if needed

def generate_json_from_txt(txt_file, output_json_file):
    """
    Convert the transcription from a .txt file into a .json file with simulated timestamps.
    
    Args:
        txt_file (str): The path to the transcription .txt file.
        output_json_file (str): The path to save the JSON file.
    """
    # Read the transcription from the .txt file
    with open(txt_file, "r") as file:
        lines = file.readlines()

    # Create the JSON structure
    transcription_data = []
    for i, line in enumerate(lines):
        transcription_data.append({
            "text": line.strip(),
            "start": i * 2.0,  # Simulating start time (2-second gap between lines)
            "end": (i + 1) * 2.0  # Simulating end time (2-second gap between lines)
        })

    # Write to .json file
    with open(output_json_file, "w") as json_file:
        json.dump(transcription_data, json_file, indent=4)

    print(f"Transcription JSON successfully created at: {output_json_file}")


def process_batch(audio_files, output_dir, model):
    """
    Processes multiple audio files sequentially, calling the whisper function for each one.
    Converts each .txt file into a .json file after Whisper transcription.
    """
    for audio_file in audio_files:
        # Run Whisper for the current audio file
        run_whisper(audio_file, output_dir, model)

        # Generate the corresponding .json file from the .txt file in the output directory
        base_name = os.path.basename(audio_file).replace(".WAV", "")
        output_txt_file = os.path.join(output_dir, base_name + ".txt")
        output_json_file = os.path.join(output_dir, base_name + ".json")

        # Check if the .txt file exists, then convert it to .json
        if os.path.exists(output_txt_file):
            generate_json_from_txt(output_txt_file, output_json_file)
        else:
            print(f"ERROR: {output_txt_file} not found. Skipping JSON conversion.")


def main():
    desktop = os.path.expanduser("~/Desktop")
    audio_dir = os.path.join(desktop, "transcribe")  # Folder containing .wav files
    output_dir = os.path.join(desktop, "output")    # Folder to save transcriptions

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all .WAV files from the Transcribe folder (case-insensitive)
    files = os.listdir(audio_dir)
    audio_files = [os.path.join(audio_dir, f) for f in files if f.lower().endswith(".wav")]

    if not audio_files:
        print("No .wav files found in the specified directory.")
        exit(1)

    # Select the model
    model = select_model()  # Use the function from model_selection.py

    # Process all files in batch (sequentially)
    process_batch(audio_files, output_dir, model)


if __name__ == "__main__":
    main()