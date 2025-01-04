import os
import subprocess
import sys
import json
from split3 import split_audio  # Import the split function from split3.py
import librosa
import torch
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from model_selection import select_model

# Define the model and feature extractor for emotion recognition
emotion_model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(emotion_model_id, do_normalize=True)
id2label = emotion_model.config.id2label

# Function to preprocess audio for emotion recognition
def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    
    # Max length for the input (in terms of samples)
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))
    
    # Extract features using the feature extractor
    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    # Check the length of input features and pad to the required length (3000)
    input_length = inputs['input_features'].shape[-1]
    if input_length < 3000:
        padding = 3000 - input_length
        inputs['input_features'] = torch.nn.functional.pad(inputs['input_features'], (0, padding))

    return inputs

# Predict emotion
def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]
    
    return predicted_label

# Function to run Whisper for transcription (same as batch.py)
def run_whisper(audio_file, output_dir, model):
    if not os.path.isfile(audio_file):
        print(f"Error: Audio file '{audio_file}' does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    audio_basename = os.path.splitext(os.path.basename(audio_file))[0]
    txt_output_file = os.path.join(output_dir, f"{audio_basename}.txt")

    command = [
        "../build/bin/main", "-m", model, "-f", audio_file, "--output-txt",
        "--split-on-word", "--max-len", "50", "--word-thold", "0.5", "--print-colors"
    ]

    try:
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        colored_output = result.stdout

        # Output transcription directly to the terminal
        print("Transcription:\n", colored_output)

        with open(txt_output_file, 'w') as file:
            file.write(colored_output)
        print(f"Transcription saved to: {txt_output_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error: Whisper.cpp failed with return code {e.returncode}")
        sys.exit(1)

    return txt_output_file

# Process audio in chunks, transcribe, and analyze emotion
def process_audio_chunks(audio_file, model, output_dir, chunk_duration=5):
    intervals = []

    # Split audio file into parts using split3
    split_audio_dir = os.path.join(output_dir, "temp_split")
    split_audio(audio_file, split_audio_dir)  # Split the file into temp_split

    # Transcribe and process each chunk
    for split_file in os.listdir(split_audio_dir):
        split_file_path = os.path.join(split_audio_dir, split_file)

        # Run Whisper for transcription (if necessary for this chunk)
        transcription_file = run_whisper(split_file_path, output_dir, model)
        
        # Predict emotion for the current audio chunk
        emotion = predict_emotion(split_file_path, emotion_model, feature_extractor, id2label, max_duration=chunk_duration)
        print(f"Predicted Emotion: {emotion}")

        # Append emotion and transcription data to intervals
        intervals.append({
            "file": split_file,
            "emotion": emotion,
            "transcription": transcription_file
        })

    return intervals

def main():
    # Get .wav files from the "transcribe" folder on the desktop
    desktop = os.path.expanduser("~/Desktop")
    audio_dir = os.path.join(desktop, "transcribe")  # Folder containing .wav files
    output_dir = os.path.join(desktop, "output")    # Folder to save outputs

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the first .WAV file from the "transcribe" folder
    files = os.listdir(audio_dir)
    audio_files = [os.path.join(audio_dir, f) for f in files if f.lower().endswith(".wav")]

    if not audio_files:
        print("No .wav files found in the 'transcribe' directory.")
        exit(1)

    audio_file = audio_files[0]  # Get the first .wav file in the directory

    # Select model using the previous method
    model = select_model()

    # Process the audio in chunks and get intervals with transcription and emotion
    intervals = process_audio_chunks(audio_file, model, output_dir)

    # Save results to JSON
    output_json_path = os.path.join(output_dir, "emotion_output.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(intervals, json_file, indent=4)

    print(f"Emotion labels and transcriptions have been appended to the JSON file at {output_json_path}")

if __name__ == "__main__":
    main()