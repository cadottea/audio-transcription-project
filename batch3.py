import os
import subprocess
import sys
import json
from whisper import run_whisper  # Import the function from whisper.py
from model_selection import select_model  # Import model selection
from arg_parser import parse_arguments  # Import the argument parser if needed

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# Assuming the model is pre-trained and available
model_name = "./wav2vec2-base-superb-er"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

processor = Wav2Vec2Processor.from_pretrained(model_name)


#from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import librosa

# Use this instead to load the model directly:

#emotion_classifier = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

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

def add_emotion_stamp_to_json(output_json_file, audio_file):
    """
    Adds emotion stamps to the .json file by processing each transcription line
    and assigning emotions using a pre-trained model (loaded locally).
    
    Args:
        output_json_file (str): The path to the .json file to be processed.
        audio_file (str): The audio file to extract the emotion from.
    """
    print(f"Processing emotion stamps for {output_json_file}...")
    
    # Load the transcription data from the JSON file
    with open(output_json_file, 'r') as f:
        transcription_data = json.load(f)

    # Process each line of the transcription and add the emotion
    for entry in transcription_data:
        # Extract the start and end times from the transcription entry
        start_time = entry["start"]
        end_time = entry["end"]

        # Load the full audio file (replace 'audio_file' with your actual audio file path)
        audio, sr = librosa.load(audio_file, sr=16000)  # Load audio at 16kHz sample rate

        # Extract the segment of audio corresponding to the transcription (based on start and end time)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = audio[start_sample:end_sample]  # Get the segment of the audio

        # Now, use the processor to extract features from the audio segment
        audio_features = processor(audio_segment, return_tensors="pt", padding=True)

        # Use the emotion classifier to predict the emotion based on the extracted audio features
        result = emotion_classifier(**audio_features)

        # Find the emotion with the highest score (assuming the model returns a sorted list of emotions)
        if result:
            emotion = result[0]['label']
        else:
            emotion = "neutral"  # Default to "neutral" if no emotion detected

        # Add the emotion to the current entry
        entry["emotion"] = emotion
    
        # Save the updated transcription data with emotion stamps back to the JSON file
        with open(output_json_file, 'w') as f:
            json.dump(transcription_data, f, indent=4)
    
        print(f"Emotion stamps successfully added to: {output_json_file}")


def process_batch(audio_files, output_dir, model):
    """
    Processes multiple audio files sequentially, calling Whisper for each one.
    Converts each .txt file into a .json file after Whisper transcription and adds emotion stamps.
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
            
            # After generating the .json file, process it with emotion stamps
            with open(output_json_file, "r") as json_file:
                transcription_data = json.load(json_file)
                
            # Process emotion stamps line-by-line
            for entry in transcription_data:
                start_time = entry["start"]
                end_time = entry["end"]
                
                # Load the full audio file (replace 'audio_file' with your actual audio file path)
                audio, sr = librosa.load(audio_file, sr=16000)  # Load audio at 16kHz sample rate

                # Extract the segment of audio corresponding to the transcription (based on start and end time)
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                audio_segment = audio[start_sample:end_sample]  # Get the segment of the audio
                
                # Now, use the processor to extract features from the audio segment
                audio_features = processor(audio_segment, return_tensors="pt", padding=True)

                result = emotion_classifier(**audio_features)
                
                # Add the emotion label with the highest score
                if result:
                    emotion = result[0]['label']
                else:
                    emotion = "neutral"
                
                entry["emotion"] = emotion
            
            # Save the updated transcription data with emotion stamps back to the JSON file
            with open(output_json_file, "w") as json_file:
                json.dump(transcription_data, json_file, indent=4)
                
            print(f"Emotion stamps successfully added to: {output_json_file}")
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